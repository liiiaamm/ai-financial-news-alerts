"""
This module contains the code to evaluate the NER and SA models interactively.
"""

# Standard libraries
import argparse
from pathlib import Path
from typing import Final

# 3pps
import torch
import yaml

# Own modules
from data.dataset import (
    get_dataloaders,
    load_vocab_info,
    extract_entities,
    enrich_ner_tags_with_years,
)
from models.bilstm_ner import BiLSTMNER
from models.bilstm_sa import BiLSTMSA
from models.shared_encoder import SharedBiLSTMEncoder


# Set device


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception as exc:
            print(f"[evaluate] CUDA detected but unusable ({exc}); using CPU.")
    return torch.device("cpu")


device: torch.device = _pick_device()

# Tag id to string mapping
TAG_ID_TO_NAME: Final[dict] = {
    0: "O",
    1: "PER_B", 2: "PER_I",
    3: "LOC_B", 4: "LOC_I",
    5: "ORG_B", 6: "ORG_I",
    7: "YEAR_B", 8: "YEAR_I",
}

# Terminal colour codes for entity types
_COLOURS: Final[dict] = {
    "ORG":  "\033[94m",
    "PER":  "\033[92m",
    "LOC":  "\033[93m",
    "YEAR": "\033[95m",
}
_RESET: Final[str] = "\033[0m"

# Built-in example sentences for each task
NER_EXAMPLES: Final[list] = [
    "Apple CEO Tim Cook announced record quarterly revenue of 143 billion dollars",
    "The Federal Reserve in Washington raised interest rates for the third time this year",
    "Goldman Sachs and JPMorgan reported strong earnings despite market volatility",
    "Elon Musk sold Tesla shares worth 5 billion dollars in 2023",
    "The European Central Bank in Frankfurt signalled further rate cuts in 2024",
    "Markets in New York and London fell sharply after the announcement",
]

SA_EXAMPLES: Final[list] = [
    "Apple reports record losses in Q3 amid supply chain disruptions",
    "Markets rally on strong earnings data from tech sector",
    "The company maintained stable revenue despite economic uncertainty",
    "Investors flee as central bank raises interest rates unexpectedly",
    "Clinton mocked for emoji tweet about market recovery",
    "This year marks the second consecutive year of revenue growth",
]


def _encode(
    text: str,
    word2idx: dict,
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    This function encodes a raw text string into padded token ids and a mask.
    """

    tokens: list[str] = text.lower().split()
    unk:    int        = word2idx.get("<UNK>", 1)

    ids:    list[int] = [word2idx.get(t, unk) for t in tokens]
    length: int       = min(len(ids), max_len)
    padded: list[int] = ids[:max_len] + [0] * max(0, max_len - len(ids))

    token_ids: torch.Tensor = torch.tensor([padded], dtype=torch.long).to(device)
    mask:      torch.Tensor = torch.zeros(1, max_len, dtype=torch.bool).to(device)
    mask[0, :length] = True

    return token_ids, mask, length


def _load_model(
    config: dict,
    vocab_info: dict,
    task: str,
    use_joint: bool,
) -> torch.nn.Module:
    """
    This function loads the appropriate model checkpoint for the given task.
    """

    idx2sa:   dict = {v: k for k, v in vocab_info["sa2idx"].items()}
    ckpt_dir: Path = Path(config["paths"]["checkpoints"])

    # Select model class and checkpoint based on task and mode
    if use_joint:
        model: torch.nn.Module = SharedBiLSTMEncoder(
            vocab_size=len(vocab_info["word2idx"]),
            embedding_dim=config["model"]["embedding_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            num_tags=len(vocab_info["tag2idx"]),
            num_sa_classes=len(vocab_info["sa2idx"]),
            embedding_matrix=vocab_info["embedding_matrix"],
            dropout=config["model"]["dropout"],
            idx2sa=idx2sa,
        ).to(device)
        ckpt: Path = ckpt_dir / "joint_best.pt"
        name: str  = "SharedBiLSTMEncoder (joint)"

    elif task == "ner":
        model = BiLSTMNER(
            vocab_size=len(vocab_info["word2idx"]),
            embedding_dim=config["model"]["embedding_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            num_tags=len(vocab_info["tag2idx"]),
            embedding_matrix=vocab_info["embedding_matrix"],
            dropout=config["model"]["dropout"],
        ).to(device)
        ckpt = ckpt_dir / "ner_best.pt"
        name = "BiLSTMNER"

    else:
        model = BiLSTMSA(
            vocab_size=len(vocab_info["word2idx"]),
            embedding_dim=config["model"]["embedding_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            num_classes=len(vocab_info["sa2idx"]),
            embedding_matrix=vocab_info["embedding_matrix"],
            dropout=config["model"]["dropout"],
            idx2sa=idx2sa,
        ).to(device)
        ckpt = ckpt_dir / "sa_best.pt"
        name = "BiLSTMSA"

    model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
    model.eval()
    print(f"Loaded {name} from {ckpt}  |  device: {device}\n")

    return model


def run_ner(
    model: torch.nn.Module,
    sentence: str,
    vocab_info: dict,
    max_len: int,
) -> None:
    """
    This function runs NER inference on a sentence and prints the tagged output.
    """

    tokens:   list[str] = sentence.split()
    word2idx: dict      = vocab_info["word2idx"]
    idx2tag:  dict      = {v: k for k, v in vocab_info["tag2idx"].items()}
    unk:      int       = word2idx.get("<UNK>", 1)

    # Encode and pad token sequence
    ids:    list[int] = [word2idx.get(t.lower(), unk) for t in tokens]
    length: int       = min(len(ids), max_len)
    padded: list[int] = ids[:max_len] + [0] * max(0, max_len - len(ids))

    token_ids: torch.Tensor = torch.tensor([padded], dtype=torch.long).to(device)
    mask:      torch.Tensor = torch.zeros(1, max_len, dtype=torch.bool).to(device)
    mask[0, :length] = True

    # Run NER inference and enrich year tags
    with torch.no_grad():
        if hasattr(model, "predict_ner"):
            preds: list = model.predict_ner(token_ids, mask).cpu()[0, :length].tolist()
        else:
            preds = model.predict(token_ids, mask).cpu()[0, :length].tolist()

    tags:          list[str] = [idx2tag.get(p, "O") for p in preds]
    inv_map:       dict      = {v: k for k, v in TAG_ID_TO_NAME.items()}
    tag_ids:       list[int] = [inv_map.get(t, 0) for t in tags]
    enriched:      list[int] = enrich_ner_tags_with_years(tokens[:length], tag_ids)
    entities:      list[dict]= extract_entities(tokens[:length], enriched)
    enriched_tags: list[str] = [TAG_ID_TO_NAME.get(t, "O") for t in enriched]

    # Print aligned token / tag table
    sep: str = "-" * 60
    print(sep)
    print(f"  Sentence : {sentence}\n")

    col_w:   list[int] = [max(len(tok), len(tag)) + 1 for tok, tag in zip(tokens[:length], enriched_tags)]
    tok_row: str       = "  "
    tag_row: str       = "  "
    for tok, tag, w in zip(tokens[:length], enriched_tags, col_w):
        colour: str = next((_COLOURS[e] for e in _COLOURS if e in tag), _RESET)
        tok_row += f"{tok:<{w}}"
        tag_row += colour + f"{tag:<{w}}" + _RESET

    print("  Tokens:"); print(tok_row)
    print("  Tags:");   print(tag_row)
    print()

    # Print extracted entities
    if entities:
        print("  Entities:")
        for ent in entities:
            colour = _COLOURS.get(ent["type"], _RESET)
            print(f"    {colour}{ent['text']:25s}{_RESET}  [{ent['type']}]  tokens {ent['start']}–{ent['end']}")
    else:
        print("  No named entities detected.")

    print(sep)

    return None


def run_sa(
    model: torch.nn.Module,
    sentence: str,
    vocab_info: dict,
    max_len: int,
) -> None:
    """
    This function runs SA inference on a sentence and prints probabilities and attention.
    """

    word2idx: dict      = vocab_info["word2idx"]
    tokens:   list[str] = sentence.lower().split()
    idx2sa:   dict      = {v: k for k, v in vocab_info["sa2idx"].items()}

    # Encode sentence and count OOV tokens
    token_ids: torch.Tensor
    mask:      torch.Tensor
    length:    int
    token_ids, mask, length = _encode(sentence, word2idx, max_len)
    oov: int = sum(1 for t in tokens if t not in word2idx)

    # Run SA inference
    with torch.no_grad():
        if isinstance(model, SharedBiLSTMEncoder):
            _, sa_logits, _ = model(token_ids, mask, task="sa")
            probs:     torch.Tensor = torch.softmax(sa_logits, dim=-1)[0]
            pred_idx:  int          = int(probs.argmax())
            label:     str          = idx2sa.get(pred_idx, "neutral")
            per_class: dict         = {idx2sa[i]: round(float(probs[i]),          4) for i in range(len(probs))}
            raw_logits:dict         = {idx2sa[i]: round(float(sa_logits[0, i]),   3) for i in range(sa_logits.shape[1])}
            top_attn:  list         = []
        else:
            label, per_class, raw = model.predict_proba(token_ids, mask)
            raw_logits = {k: round(raw[i], 3) for i, k in enumerate(per_class)}
            _, attn_weights = model(token_ids, mask)
            attn:     list[float] = attn_weights[0, :length].cpu().tolist()
            top_attn  = sorted(zip(tokens[:length], attn), key=lambda x: x[1], reverse=True)[:5]
            top_attn  = [(tok, round(w, 4)) for tok, w in top_attn]

    # Print prediction and probabilities
    sep: str = "-" * 60
    print(sep)
    print(f"  Sentence   : {sentence}")
    print(f"  Length     : {length}  |  OOV: {oov} ({100 * oov / max(len(tokens), 1):.1f}%)\n")
    print(f"  Prediction : {label.upper()}  (confidence: {per_class[label]:.4f})\n")

    print("  Class probabilities:")
    for cls, prob in sorted(per_class.items(), key=lambda x: x[1], reverse=True):
        bar:    str = "#" * int(prob * 30)
        marker: str = " <--" if cls == label else ""
        print(f"    {cls:>10} : {prob:.4f}  {bar}{marker}")

    print("\n  Raw logits:")
    for cls, logit in sorted(raw_logits.items(), key=lambda x: x[1], reverse=True):
        print(f"    {cls:>10} : {logit:+.3f}")

    if top_attn:
        print("\n  Top attention tokens:")
        for tok, w in top_attn:
            print(f"    {tok:>15} : {w:.4f}  {'#' * int(w * 40)}")

    print(sep)

    return None


def main() -> None:
    """
    This function is the main program for interactive model evaluation.
    """

    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate NER and/or SA models")
    parser.add_argument("--task",      choices=["ner", "sa", "both"], default="both")
    parser.add_argument("--text",      default=None,                  help="Single sentence to evaluate")
    parser.add_argument("--config",    default="configs/config.yaml")
    parser.add_argument("--use-joint", dest="use_joint", action="store_true", default=False)
    args = parser.parse_args()

    # Load configuration and vocabulary
    with open(args.config) as f:
        config: dict = yaml.safe_load(f)

    ckpt_dir: Path = Path(config["paths"]["checkpoints"])
    vocab_path: Path = ckpt_dir / "vocab_info.pt"
    if vocab_path.exists():
        print(f"Loading vocabulary from {vocab_path}...")
        vocab_info = load_vocab_info(vocab_path)
    else:
        print("Loading vocabulary...")
        _, _, _, vocab_info = get_dataloaders(config)
    max_len: int = config["data"]["max_seq_len"]

    # Run NER evaluation
    if args.task in ("ner", "both"):
        model_ner: torch.nn.Module = _load_model(config, vocab_info, "ner", args.use_joint)
        sentences: list[str]       = [args.text] if args.text else NER_EXAMPLES
        for sent in sentences:
            run_ner(model_ner, sent, vocab_info, max_len)

    # Run SA evaluation
    if args.task in ("sa", "both"):
        model_sa: torch.nn.Module = _load_model(config, vocab_info, "sa", args.use_joint)
        sentences = [args.text] if args.text else SA_EXAMPLES
        for sent in sentences:
            run_sa(model_sa, sent, vocab_info, max_len)

    return None


if __name__ == "__main__":
    main()
