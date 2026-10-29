"""
This module contains the code to train the BiLSTM-CRF NER model.
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
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Own modules
from data.dataset import get_dataloaders, save_vocab_info
from models.bilstm_ner import BiLSTMNER


# Set device and seed
SEED: Final[int] = 42


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception as exc:
            print(f"[train_ner] CUDA detected but unusable ({exc}); using CPU.")
    return torch.device("cpu")


device: torch.device = _pick_device()
torch.manual_seed(SEED)


def _to_iob2(tag: str) -> str:
    """
    This function converts an internal tag string to IOB2 format.
    """

    if tag == "O":
        return "O"

    entity: str
    pos: str
    entity, pos = tag.split("_")

    return f"{pos}-{entity}"


def train_loop(
    model: BiLSTMNER,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    clip: float,
    epoch: int,
    writer: SummaryWriter,
) -> float:
    """
    This function computes one training epoch for the NER model.
    """

    model.train()

    # Accumulate batch losses
    train_loss: float = 0.0
    n_batches:  int   = 0

    for batch in train_loader:
        tids: torch.Tensor = batch["token_ids"].to(device)
        tags: torch.Tensor = batch["ner_tags"].to(device)
        mask: torch.Tensor = batch["attention_mask"].to(device)

        optimizer.zero_grad()

        loss: torch.Tensor = model.compute_loss(tids, mask, tags)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        train_loss += loss.item()
        n_batches  += 1

    train_loss /= n_batches

    # Write training loss to TensorBoard
    writer.add_scalar("ner/train_loss", train_loss, epoch)
    writer.add_scalar("ner/lr", optimizer.param_groups[0]["lr"], epoch)

    return train_loss


def val_loop(
    model: BiLSTMNER,
    val_loader: torch.utils.data.DataLoader,
    idx2tag: dict,
    epoch: int,
    writer: SummaryWriter,
) -> tuple[float, float]:
    """
    This function computes the validation step for the NER model.
    """

    from seqeval.metrics import f1_score

    model.eval()

    # Accumulate loss and predictions
    total_loss: float     = 0.0
    n_batches:  int       = 0
    all_true:   list[list] = []
    all_pred:   list[list] = []

    with torch.no_grad():
        for batch in val_loader:
            tids: torch.Tensor = batch["token_ids"].to(device)
            tags: torch.Tensor = batch["ner_tags"].to(device)
            mask: torch.Tensor = batch["attention_mask"].to(device)

            total_loss += model.compute_loss(tids, mask, tags).item()
            n_batches  += 1

            preds: torch.Tensor = model.predict(tids, mask).cpu()
            for i, length in enumerate(batch["length"]):
                all_true.append([_to_iob2(idx2tag.get(t, "O")) for t in tags.cpu()[i, :length].tolist()])
                all_pred.append([_to_iob2(idx2tag.get(p, "O")) for p in preds[i, :length].tolist()])

    val_loss: float = total_loss / max(n_batches, 1)
    val_f1:   float = float(f1_score(all_true, all_pred, average="micro"))

    # Write validation metrics to TensorBoard
    writer.add_scalar("ner/val_loss", val_loss, epoch)
    writer.add_scalar("ner/val_f1",   val_f1,   epoch)

    model.train()

    return val_loss, val_f1


def main() -> None:
    """
    This function is the main program for training the NER model.
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
    tag2idx:  dict = vocab_info["tag2idx"]
    idx2tag:  dict = {v: k for k, v in tag2idx.items()}
    num_tags: int  = len(tag2idx)

    # Compute log-inverse frequency class weights to handle tag imbalance
    tag_counts: Counter = Counter()
    total_toks: int     = 0
    for batch in train_loader:
        for tags_seq, mask_seq in zip(batch["ner_tags"], batch["attention_mask"]):
            for tag, m in zip(tags_seq.tolist(), mask_seq.tolist()):
                if m and tag != -100:
                    tag_counts[tag] += 1
                    total_toks      += 1

    log_w: torch.Tensor = torch.tensor(
        [np.log(total_toks / (num_tags * max(tag_counts.get(t, 1), 1))) for t in range(num_tags)],
        dtype=torch.float32,
    ).to(device)

    # Instantiate model and register class weights
    model: BiLSTMNER = BiLSTMNER(
        vocab_size=len(word2idx),
        embedding_dim=config["model"]["embedding_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_tags=num_tags,
        embedding_matrix=vocab_info["embedding_matrix"],
        dropout=config["model"]["dropout"],
    ).to(device)
    model.register_buffer("class_log_weights", log_w)

    # Training hyperparameters
    lr:       float = config["ner"]["lr"]
    epochs:   int   = config["ner"]["epochs"]
    clip:     float = config["ner"]["clip_grad"]
    patience: int   = config["ner"]["patience"]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    # Checkpoint and TensorBoard setup
    ckpt_dir:  Path = Path(config["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path: Path = ckpt_dir / "ner_best.pt"
    save_vocab_info(vocab_info, ckpt_dir / "vocab_info.pt")

    m: dict    = config["model"]
    n: dict    = config["ner"]
    run_name:  str = f"NER_hd{m['hidden_dim']}_nl{m['num_layers']}_do{m['dropout']}_lr{n['lr']}_ep{n['epochs']}"
    writer: SummaryWriter = SummaryWriter(log_dir=f"runs/{run_name}")
    print(f"TensorBoard: runs/{run_name}  |  device: {device}")

    # Training loop with early stopping
    best_f1:    float      = -1.0
    no_improve: int        = 0
    history:    list[dict] = []

    for epoch in tqdm(range(1, epochs + 1), desc="NER", unit="epoch"):
        train_loss: float = train_loop(model, train_loader, optimizer, clip, epoch, writer)
        val_loss:   float
        val_f1:     float
        val_loss, val_f1 = val_loop(model, val_loader, idx2tag, epoch, writer)
        scheduler.step(val_f1)

        history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_loss,   4),
            "val_f1":     round(val_f1,     4),
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
    with open(Path(config["paths"]["results"]) / "ner_training.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val F1: {best_f1:.4f}  |  checkpoint: {ckpt_path}")

    return None


if __name__ == "__main__":
    main()
