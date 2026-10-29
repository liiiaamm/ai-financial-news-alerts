"""
This module contains the code to train the joint NER+SA SharedBiLSTMEncoder.
"""

# Standard libraries
import json
import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Final, Optional

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
from models.shared_encoder import SharedBiLSTMEncoder


# Set device and seed
SEED: Final[int] = 42


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception as exc:
            print(f"[train_joint] CUDA detected but unusable ({exc}); using CPU.")
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


def val_loop(
    encoder: SharedBiLSTMEncoder,
    val_loader: torch.utils.data.DataLoader,
    vocab_info: dict,
) -> tuple[float, float]:
    """
    This function computes the validation step for the joint model.
    """

    from seqeval.metrics import f1_score as ner_f1_fn
    from sklearn.metrics import f1_score as sa_f1_fn

    idx2tag: dict = {v: k for k, v in vocab_info["tag2idx"].items()}
    idx2sa:  dict = {v: k for k, v in vocab_info["sa2idx"].items()}

    encoder.eval()

    # Accumulate predictions for both tasks
    ner_true: list = []
    ner_pred: list = []
    sa_true:  list = []
    sa_pred:  list = []

    with torch.no_grad():
        for batch in val_loader:
            tids: torch.Tensor = batch["token_ids"].to(device)
            mask: torch.Tensor = batch["attention_mask"].to(device)

            ner_preds: torch.Tensor = encoder.predict_ner(tids, mask).cpu()
            sa_labels: list[str]    = encoder.predict_sa(tids, mask)

            for i, length in enumerate(batch["length"]):
                ner_true.append([_to_iob2(idx2tag.get(t, "O")) for t in batch["ner_tags"][i, :length].tolist()])
                ner_pred.append([_to_iob2(idx2tag.get(p, "O")) for p in ner_preds[i, :length].tolist()])

            sa_true.extend(batch["sentiment"].tolist())
            sa_pred.extend([vocab_info["sa2idx"].get(p, 0) for p in sa_labels])

    f1_ner: float = float(ner_f1_fn(ner_true, ner_pred, average="micro"))
    f1_sa:  float = float(sa_f1_fn(sa_true, sa_pred, average="macro",
                                    labels=list(idx2sa.keys()), zero_division=0))

    encoder.train()

    return f1_ner, f1_sa


def train_one(
    config: dict,
    vocab_info: dict,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    lambda_ner: float,
    lambda_sa: float,
    writer: Optional[SummaryWriter] = None,
    writer_prefix: str = "joint",
) -> SharedBiLSTMEncoder:
    """
    This function trains a SharedBiLSTMEncoder with given lambda values.
    """

    sa2idx: dict = vocab_info["sa2idx"]
    idx2sa: dict = {v: k for k, v in sa2idx.items()}

    # Compute SA class weights from training distribution
    counts: Counter = Counter()
    total:  int     = 0
    for batch in train_loader:
        for lbl in batch["sentiment"].tolist():
            counts[lbl] += 1
            total       += 1

    sa_weights: torch.Tensor = torch.tensor(
        [total / (len(sa2idx) * counts.get(c, 1)) for c in range(len(sa2idx))],
        dtype=torch.float32,
    )

    # Instantiate shared encoder
    encoder: SharedBiLSTMEncoder = SharedBiLSTMEncoder(
        vocab_size=len(vocab_info["word2idx"]),
        embedding_dim=config["model"]["embedding_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_tags=len(vocab_info["tag2idx"]),
        num_sa_classes=len(sa2idx),
        embedding_matrix=vocab_info["embedding_matrix"],
        dropout=config["model"]["dropout"],
        sa_class_weights=sa_weights,
        idx2sa=idx2sa,
    ).to(device)

    # Training hyperparameters
    lr:       float = config["joint"]["lr"]
    epochs:   int   = config["joint"]["epochs"]
    patience: int   = config["joint"]["patience"]

    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    best_avg:   float             = 0.0
    no_improve: int               = 0
    best_state: Optional[dict]    = None

    print(f"\nJoint training: λ_NER={lambda_ner}, λ_SA={lambda_sa}  |  device={device}")

    for epoch in tqdm(range(1, epochs + 1), desc=f"Joint λ={lambda_ner:.1f}", unit="epoch"):
        encoder.train()

        # Accumulate combined loss over batches
        total_loss: float = 0.0
        n_batches:  int   = 0

        for batch in train_loader:
            tids:  torch.Tensor = batch["token_ids"].to(device)
            mask:  torch.Tensor = batch["attention_mask"].to(device)
            ner_t: torch.Tensor = batch["ner_tags"].to(device)
            sa_t:  torch.Tensor = batch["sentiment"].to(device)

            optimizer.zero_grad()

            ner_l: torch.Tensor
            sa_l:  torch.Tensor
            ner_l, sa_l, _ = encoder(tids, mask, task="both")

            loss: torch.Tensor
            loss, _, _ = encoder.combined_loss(ner_l, sa_l, ner_t, sa_t, lambda_ner, lambda_sa)
            loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_loss: float = total_loss / max(n_batches, 1)

        # Validation and scheduler step
        f1_ner: float
        f1_sa:  float
        f1_ner, f1_sa = val_loop(encoder, val_loader, vocab_info)
        avg: float = (f1_ner + f1_sa) / 2
        scheduler.step(avg)

        if writer is not None:
            writer.add_scalar(f"{writer_prefix}/train_loss", avg_loss, epoch)
            writer.add_scalar(f"{writer_prefix}/val_f1_avg", avg,      epoch)

        if avg > best_avg:
            best_avg   = avg
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
            tqdm.write(f"  epoch {epoch:02d}  loss={avg_loss:.4f}  ner={f1_ner:.4f}  sa={f1_sa:.4f}  avg={avg:.4f}  saved")
        else:
            no_improve += 1
            if no_improve >= patience:
                tqdm.write(f"  Early stopping at epoch {epoch}.")
                break

    # Restore best weights
    encoder.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    encoder.eval()

    return encoder


def main() -> None:
    """
    This function is the main program for training the joint NER+SA model.
    """

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",         default="configs/config.yaml")
    parser.add_argument("--skip-ablation",  action="store_true")
    args = parser.parse_args()

    # Load configuration
    with open(args.config) as f:
        config: dict = yaml.safe_load(f)

    # Load data and vocabulary
    train_loader, val_loader, test_loader, vocab_info = get_dataloaders(config)

    # TensorBoard setup
    m: dict   = config["model"]
    j: dict   = config["joint"]
    run_name: str = (
        f"Joint_hd{m['hidden_dim']}_nl{m['num_layers']}_do{m['dropout']}"
        f"_lner{j['lambda_ner']}_lsa{j['lambda_sa']}_lr{j['lr']}_ep{j['epochs']}"
    )
    writer: SummaryWriter = SummaryWriter(log_dir=f"runs/{run_name}")
    print(f"TensorBoard: runs/{run_name}")

    # Train joint model with config lambda values
    encoder: SharedBiLSTMEncoder = train_one(
        config, vocab_info, train_loader, val_loader,
        lambda_ner=j["lambda_ner"], lambda_sa=j["lambda_sa"],
        writer=writer, writer_prefix="joint",
    )

    # Save checkpoint
    ckpt_dir:  Path = Path(config["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path: Path = ckpt_dir / "joint_best.pt"
    save_vocab_info(vocab_info, ckpt_dir / "vocab_info.pt")
    torch.save(encoder.state_dict(), ckpt_path)
    print(f"\nCheckpoint saved: {ckpt_path}")

    # Evaluate on test set
    f1_ner: float
    f1_sa:  float
    f1_ner, f1_sa = val_loop(encoder, test_loader, vocab_info)
    print(f"Test  NER F1={f1_ner:.4f}  SA macro F1={f1_sa:.4f}")

    # Lambda ablation study
    if not args.skip_ablation:
        print("\n--- Lambda ablation study ---")
        ablation: list[dict] = []

        for lam in [0.3, 0.5, 0.7, 1.0]:
            enc: SharedBiLSTMEncoder = train_one(
                config, vocab_info, train_loader, val_loader,
                lambda_ner=lam, lambda_sa=1.0 - lam,
                writer=writer, writer_prefix=f"ablation_lam{lam}",
            )
            f_ner: float
            f_sa:  float
            f_ner, f_sa = val_loop(enc, test_loader, vocab_info)
            ablation.append({
                "lambda_ner": lam,
                "lambda_sa":  1.0 - lam,
                "ner_f1":     round(f_ner, 4),
                "sa_f1":      round(f_sa,  4),
            })
            print(f"  λ_NER={lam:.1f}  NER={f_ner:.4f}  SA={f_sa:.4f}")

        # Save ablation results
        Path(config["paths"]["results"]).mkdir(parents=True, exist_ok=True)
        with open(Path(config["paths"]["results"]) / "joint_hyperparameter_search.json", "w") as f:
            json.dump(ablation, f, indent=2)

    writer.close()
    print("\nDone.")

    return None


if __name__ == "__main__":
    main()
