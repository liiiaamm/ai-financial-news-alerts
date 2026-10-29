"""
This module contains the code to train the BiLSTM sentiment analysis model.
"""

# Standard libraries
import json
import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Final

# Allow running from training/ or from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 3pps
import torch
import torch.nn as nn
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Own modules
from data.dataset import get_dataloaders, save_vocab_info
from models.bilstm_sa import BiLSTMSA


# Set device and seed
SEED: Final[int] = 42


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception as exc:
            print(f"[train_sa] CUDA detected but unusable ({exc}); using CPU.")
    return torch.device("cpu")


device: torch.device = _pick_device()
torch.manual_seed(SEED)


def train_loop(
    model: BiLSTMSA,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    clip: float,
    epoch: int,
    writer: SummaryWriter,
) -> float:
    """
    This function computes one training epoch for the SA model.
    """

    model.train()

    # Accumulate batch losses
    train_loss: float = 0.0
    n_batches:  int   = 0

    for batch in train_loader:
        tids:     torch.Tensor = batch["token_ids"].to(device)
        sa:       torch.Tensor = batch["sentiment"].to(device)
        mask:     torch.Tensor = batch["attention_mask"].to(device)

        optimizer.zero_grad()

        logits: torch.Tensor
        logits, _ = model(tids, mask)

        loss: torch.Tensor = criterion(logits, sa)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        train_loss += loss.item()
        n_batches  += 1

    train_loss /= n_batches

    # Write training metrics to TensorBoard
    writer.add_scalar("sa/train_loss", train_loss, epoch)
    writer.add_scalar("sa/lr", optimizer.param_groups[0]["lr"], epoch)

    return train_loss


def val_loop(
    model: BiLSTMSA,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    epoch: int,
    writer: SummaryWriter,
) -> tuple[float, float, float]:
    """
    This function computes the validation step for the SA model.
    """

    from sklearn.metrics import f1_score, accuracy_score

    model.eval()

    # Accumulate loss and predictions
    total_loss: float = 0.0
    n_batches:  int   = 0
    all_true:   list  = []
    all_pred:   list  = []

    with torch.no_grad():
        for batch in val_loader:
            tids: torch.Tensor = batch["token_ids"].to(device)
            sa:   torch.Tensor = batch["sentiment"].to(device)
            mask: torch.Tensor = batch["attention_mask"].to(device)

            logits: torch.Tensor
            logits, _ = model(tids, mask)

            total_loss += criterion(logits, sa).item()
            n_batches  += 1

            all_pred.extend(logits.argmax(dim=-1).cpu().tolist())
            all_true.extend(sa.cpu().tolist())

    val_loss: float = total_loss / max(n_batches, 1)
    val_acc:  float = float(accuracy_score(all_true, all_pred))
    val_f1:   float = float(f1_score(all_true, all_pred, average="macro",
                                     labels=list(range(len(set(all_true))))))

    # Write validation metrics to TensorBoard
    writer.add_scalar("sa/val_macro_f1", val_f1, epoch)

    model.train()

    return val_loss, val_acc, val_f1


def main() -> None:
    """
    This function is the main program for training the SA model.
    """

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    # Load configuration
    with open(args.config) as f:
        config: dict = yaml.safe_load(f)

    # Load data and vocabulary
    train_loader, val_loader, test_loader, vocab_info = get_dataloaders(config)

    word2idx: dict = vocab_info["word2idx"]
    sa2idx:   dict = vocab_info["sa2idx"]
    idx2sa:   dict = {v: k for k, v in sa2idx.items()}
    num_cls:  int  = len(sa2idx)

    # Compute inverse frequency class weights to handle class imbalance
    counts: Counter = Counter()
    total:  int     = 0
    for batch in train_loader:
        for lbl in batch["sentiment"].tolist():
            counts[lbl] += 1
            total       += 1

    weights: torch.Tensor = torch.tensor(
        [total / (num_cls * counts.get(c, 1)) for c in range(num_cls)],
        dtype=torch.float32,
    ).to(device)

    # Instantiate model
    model: BiLSTMSA = BiLSTMSA(
        vocab_size=len(word2idx),
        embedding_dim=config["model"]["embedding_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_classes=num_cls,
        embedding_matrix=vocab_info["embedding_matrix"],
        dropout=config["model"]["dropout"],
        idx2sa=idx2sa,
    ).to(device)

    # Training hyperparameters
    lr:       float = config["sa"]["lr"]
    epochs:   int   = config["sa"]["epochs"]
    clip:     float = config["sa"]["clip_grad"]
    patience: int   = config["sa"]["patience"]

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    # Checkpoint and TensorBoard setup
    ckpt_dir:  Path = Path(config["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path: Path = ckpt_dir / "sa_best.pt"
    save_vocab_info(vocab_info, ckpt_dir / "vocab_info.pt")

    m: dict   = config["model"]
    s: dict   = config["sa"]
    run_name: str = f"SA_hd{m['hidden_dim']}_nl{m['num_layers']}_do{m['dropout']}_lr{s['lr']}_ep{s['epochs']}"
    writer: SummaryWriter = SummaryWriter(log_dir=f"runs/{run_name}")
    print(f"TensorBoard: runs/{run_name}  |  device: {device}")

    # Training loop with early stopping
    best_f1:    float      = -1.0
    no_improve: int        = 0
    history:    list[dict] = []

    for epoch in tqdm(range(1, epochs + 1), desc="SA", unit="epoch"):
        train_loss: float = train_loop(model, train_loader, criterion, optimizer, clip, epoch, writer)
        val_loss:   float
        val_acc:    float
        val_f1:     float
        val_loss, val_acc, val_f1 = val_loop(model, val_loader, criterion, epoch, writer)
        scheduler.step(val_f1)

        history.append({
            "epoch":        epoch,
            "train_loss":   round(train_loss, 4),
            "val_loss":     round(val_loss,   4),
            "val_macro_f1": round(val_f1,     4),
        })

        if val_f1 > best_f1:
            best_f1    = val_f1
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            tqdm.write(f"  epoch {epoch:02d}  train={train_loss:.4f}  val_f1={val_f1:.4f}  saved")
        else:
            no_improve += 1
            if no_improve >= patience:
                tqdm.write(f"  Early stopping at epoch {epoch}.")
                break

    # Save results and close writer
    writer.close()
    Path(config["paths"]["results"]).mkdir(parents=True, exist_ok=True)
    with open(Path(config["paths"]["results"]) / "sa_training.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val macro F1: {best_f1:.4f}  |  checkpoint: {ckpt_path}")

    return None


if __name__ == "__main__":
    main()
