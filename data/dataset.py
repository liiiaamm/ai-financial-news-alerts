"""
dataset.py  —  Dataset construction and PyTorch data utilities.

Pipeline:
    1. build_dataset()   : download FiNER-ORD, run FinBERT, enrich with YEAR
                           tags / entities / alerts, save to JSONL.
    2. get_dataloaders() : read JSONL, build vocab, load GloVe, return
                          train / val / test DataLoaders.

Run directly to build the dataset:
    python data/dataset.py
"""

import json
import re
import numpy as np
import torch
from collections import Counter
from pathlib import Path
from typing import Any, Final, Optional
from torch.utils.data import DataLoader, Dataset


# ── Constants ─────────────────────────────────────────────────────────────────

SEED: Final[int] = 42

TAG_ID_TO_NAME: Final[dict] = {
    0: "O",
    1: "PER_B",
    2: "PER_I",
    3: "LOC_B",
    4: "LOC_I",
    5: "ORG_B",
    6: "ORG_I",
    7: "YEAR_B",
    8: "YEAR_I",
}

ENTITY_PRIORITY: Final[list] = ["ORG", "PER", "LOC"]
YEAR_PATTERN: Final = re.compile(r"^(19|20)\d{2}$")

# Boilerplate strings that appear in FiNER-ORD but carry no useful signal
_NOISE_EXACT: Final[frozenset] = frozenset(
    {
        "Loading ...",
        "Learn more ...",
        "Back to work ...",
        "They signal that many ...",
        "Share this :",
        "Other Stories Follow Us",
        "Most Popular",
        "TAGS",
        "Bank accounts",
        "Economics",
        "Related Articles",
        "Related articles",
        "...",
        "…",
        "Broker CBA",
    }
)
_NOISE_PREFIXES: Final[tuple] = (
    "Tip : Use comma",
    "All rights reserved",
    "Distributed by AllAfrica",
    "AllAfrica publishes",
    "Publishers named above each report",
    "Articles and commentaries that identify allAfrica.com",
    "We Fools do n't all hold the same opinions",
    "Click here to get your copy",
    "See also delay times for other exchanges",
)


# ── NER helpers ───────────────────────────────────────────────────────────────


def tag_to_name(tag: Any) -> str:
    if isinstance(tag, int):
        return TAG_ID_TO_NAME.get(tag, "O")
    return str(tag)


def is_year_token(token: str) -> bool:
    return bool(YEAR_PATTERN.match(token))


def enrich_ner_tags_with_years(tokens: list[str], ner_tags: list[Any]) -> list[int]:
    """Replace O tags on 4-digit year tokens with YEAR_B / YEAR_I (ids 7/8).

    FiNER-ORD has no YEAR label, so the model often misclassifies years
    that appear near location names as LOC or ORG.
    """
    inverse = {v: k for k, v in TAG_ID_TO_NAME.items()}
    enriched = []
    prev_year = False

    for tok, raw in zip(tokens, ner_tags):
        tag = raw if isinstance(raw, int) else inverse.get(str(raw), 0)
        if is_year_token(tok):
            enriched.append(8 if prev_year else 7)
            prev_year = True
        else:
            enriched.append(tag)
            prev_year = False

    return enriched


def smart_join_tokens(tokens: list[str]) -> str:
    """Reconstruct a readable span from BIO tokens, handling punctuation."""
    if not tokens:
        return ""

    no_space_before = {".", ",", ":", ";", "!", "?", "%", ")", "]", "}", "'s"}
    no_space_after = {"(", "[", "{", "$", "€", "£", '"'}
    attach_both = {"-", "/"}

    text = tokens[0]
    for tok in tokens[1:]:
        if tok in no_space_before:
            text += tok
        elif text.endswith(tuple(no_space_after)):
            text += tok
        elif tok in attach_both or text.endswith(tuple(attach_both)):
            text += tok
        else:
            text += " " + tok

    return text.replace('" ', '"').replace(" 's", "'s").strip()


def extract_entities(tokens: list[str], ner_tags: list[Any]) -> list[dict]:
    """Parse a BIO tag sequence into a list of entity dicts (PER/ORG/LOC/YEAR)."""
    entities, current_tokens, current_type, start = [], [], None, None

    def flush(end):
        nonlocal current_tokens, current_type, start
        if current_tokens and current_type:
            entities.append(
                {
                    "text": smart_join_tokens(current_tokens),
                    "type": current_type,
                    "start": start,
                    "end": end,
                }
            )
        current_tokens, current_type, start = [], None, None

    for i, (tok, raw) in enumerate(zip(tokens, ner_tags)):
        tag = tag_to_name(raw)
        if tag == "O":
            flush(i - 1)
            continue

        try:
            ent_type, bio = tag.split("_")
        except ValueError:
            flush(i - 1)
            continue

        if ent_type not in {"PER", "ORG", "LOC", "YEAR"}:
            flush(i - 1)
            continue

        if bio == "B":
            flush(i - 1)
            current_tokens, current_type, start = [tok], ent_type, i
        elif bio == "I":
            if current_type == ent_type and current_tokens:
                current_tokens.append(tok)
            else:
                flush(i - 1)
                current_tokens, current_type, start = [tok], ent_type, i
        else:
            flush(i - 1)

    flush(len(tokens) - 1)
    return entities


def deduplicate_entities(entities: list[dict]) -> list[dict]:
    seen, result = set(), []
    for ent in entities:
        key = (ent["text"], ent["type"])
        if key not in seen:
            seen.add(key)
            result.append(ent)
    return result


def choose_main_entity(entities: list[dict]) -> Optional[dict]:
    """Select the most prominent entity: ORG > PER > LOC; longest text wins ties."""
    for ent_type in ENTITY_PRIORITY:
        candidates = [e for e in entities if e["type"] == ent_type]
        if candidates:
            return sorted(candidates, key=lambda x: (-len(x["text"]), x["start"]))[0]
    return None


def extract_years(tokens: list[str], ner_tags: list[int]) -> list[str]:
    return [
        tok
        for tok, tag in zip(tokens, ner_tags)
        if tag in (7, 8) and is_year_token(tok)
    ]


# ── Alert building ────────────────────────────────────────────────────────────

_ALERT_TEMPLATES: Final[dict] = {
    "ORG": {
        "positive": (
            "org_positive",
            "Positive business outlook: {e} mentioned positively.",
        ),
        "negative": ("org_negative", "Reputation risk: {e} mentioned negatively."),
        "neutral": (
            "org_neutral",
            "Watchlist update: {e} referenced in a neutral context.",
        ),
    },
    "PER": {
        "positive": (
            "per_positive",
            "Positive reputation signal: {e} mentioned positively.",
        ),
        "negative": ("per_negative", "Reputation risk: {e}."),
        "neutral": (
            "per_neutral",
            "Watchlist update: {e} referenced in a neutral context.",
        ),
    },
    "LOC": {
        "positive": ("loc_positive", "Economic opportunity related to {e}."),
        "negative": ("loc_negative", "Risk signal related to {e}."),
        "neutral": (
            "loc_neutral",
            "Regional update: {e} referenced in a neutral context.",
        ),
    },
}

_GENERIC_TEMPLATES: Final[dict] = {
    "positive": ("generic_positive", "Positive market signal detected in the article."),
    "negative": ("generic_negative", "Potential risk signal detected in the article."),
    "neutral": ("generic_neutral", "Neutral market update detected in the article."),
}


def build_alert(main_entity: Optional[dict], sentiment: str) -> dict[str, str]:
    s = (sentiment or "neutral").lower()
    s = s if s in ("positive", "negative", "neutral") else "neutral"

    if main_entity is None:
        alert_type, alert_text = _GENERIC_TEMPLATES[s]
        return {"alert_type": alert_type, "alert_text": alert_text}

    e = main_entity["text"]
    etype = main_entity["type"]

    if etype in _ALERT_TEMPLATES:
        alert_type, tmpl = _ALERT_TEMPLATES[etype][s]
        return {"alert_type": alert_type, "alert_text": tmpl.format(e=e)}

    # fallback for unexpected entity types
    alert_type, alert_text = _GENERIC_TEMPLATES[s]
    return {"alert_type": alert_type, "alert_text": alert_text}


def _format_entities(entities: list[dict]) -> str:
    if not entities:
        return "none"
    return "; ".join(f"{e['text']} ({e['type']})" for e in entities)


def _build_model_input(
    sentence, sentiment, sentiment_score, entities, caption, years
) -> str:
    score = "null" if sentiment_score is None else f"{float(sentiment_score):.6f}"
    return (
        f"TEXT: {sentence}\n"
        f"CAPTION: {caption or 'none'}\n"
        f"ENTITIES: {_format_entities(entities)}\n"
        f"YEARS: {', '.join(years) if years else 'none'}\n"
        f"SENTIMENT: {sentiment}\n"
        f"SENTIMENT_SCORE: {score}"
    )


def _enrich_row(row: dict) -> dict:
    tokens = row["tokens"]
    enriched = enrich_ner_tags_with_years(tokens, row["ner_tags"])
    entities = deduplicate_entities(extract_entities(tokens, enriched))
    types = sorted({e["type"] for e in entities})
    main = choose_main_entity(entities)
    years = extract_years(tokens, enriched)
    sent = row.get("sentiment", "neutral")
    alert = build_alert(main, sent)

    return {
        **row,
        "ner_tags": enriched,
        "entities": entities,
        "entity_types_present": types,
        "has_org": "ORG" in types,
        "has_per": "PER" in types,
        "has_loc": "LOC" in types,
        "main_entity": main["text"] if main else None,
        "main_entity_type": main["type"] if main else None,
        "alert_type": alert["alert_type"],
        "alert_text": alert["alert_text"],
        "model_input": _build_model_input(
            row.get("sentence", " ".join(tokens)),
            sent,
            row.get("sentiment_score"),
            entities,
            None,
            years,
        ),
        "model_target": alert["alert_text"],
    }


# ── Dataset construction ──────────────────────────────────────────────────────


def _is_noise(sentence: str, tokens: list[str]) -> bool:
    s = sentence.strip()
    if re.match(r"^[^a-zA-Z]+$", s):
        return True
    if len(tokens) <= 3 and any(re.match(r"https?://|www\.", t) for t in tokens):
        return True
    if s in _NOISE_EXACT or any(s.startswith(p) for p in _NOISE_PREFIXES):
        return True
    if len(tokens) <= 2 and not any(t.isalpha() and len(t) > 3 for t in tokens):
        return True
    return False


def _run_finbert(sentences: list[str]) -> list[str]:
    from transformers import pipeline as hf_pipeline

    device = 0 if torch.cuda.is_available() else -1
    finbert = hf_pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        device=device,
        truncation=True,
        max_length=512,
    )
    return [r["label"].lower() for r in finbert(sentences, batch_size=64)]


def build_dataset(out_path: Path) -> None:
    """Download FiNER-ORD, label with FinBERT, enrich and save to JSONL."""
    from datasets import load_dataset as hf_load

    print("Downloading FiNER-ORD...")
    try:
        finer = hf_load("gtfintechlab/finer-ord-bio")
    except TypeError as e:
        print(f"Error loading dataset: {e}")
        print(
            "This often indicates an incompatibility with the 'datasets' library version."
        )
        print(
            "Please try updating the 'datasets' library: pip install --upgrade datasets"
        )
        raise  # Re-raise the exception after providing guidance

    samples, gid = [], 0
    for split in finer:
        for ex in finer[split]:
            pairs = [(t, g) for t, g in zip(ex["tokens"], ex["tags"]) if t is not None]
            if not pairs:
                gid += 1
                continue
            tokens, tags = zip(*pairs)
            samples.append(
                {
                    "id": f"{split}_{gid}",
                    "split": split,
                    "tokens": list(tokens),
                    "ner_tags": list(tags),
                    "sentence": " ".join(tokens),
                    "sentiment": None,
                    "sentiment_score": None,
                }
            )
            gid += 1

    print(f"Running FinBERT on {len(samples)} samples...")
    labels = _run_finbert([s["sentence"] for s in samples])
    for s, lbl in zip(samples, labels):
        s["sentiment"] = lbl

    before = len(samples)
    samples = [s for s in samples if not _is_noise(s["sentence"], s["tokens"])]
    print(f"Noise filter: {before - len(samples)} removed, {len(samples)} kept")

    print("Enriching with YEAR tags, entities and rule-based alerts...")
    samples = [_enrich_row(s) for s in samples]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Saved {len(samples)} samples → {out_path}")


# ── Vocabulary and GloVe ──────────────────────────────────────────────────────


def build_vocab(sentences: list[list[str]], min_freq: int = 2) -> dict[str, int]:
    freq = Counter(tok.lower() for toks in sentences for tok in toks)
    word2idx: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for word in sorted(w for w, c in freq.items() if c >= min_freq):
        if word not in word2idx:
            word2idx[word] = len(word2idx)
    return word2idx


def load_glove(
    glove_path: str, word2idx: dict[str, int], embed_dim: int = 100
) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    matrix = rng.normal(0, 0.1, (len(word2idx), embed_dim)).astype(np.float32)
    matrix[0] = 0.0  # <PAD> → zero vector

    hits = 0
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if parts[0] in word2idx:
                matrix[word2idx[parts[0]]] = np.array(parts[1:], dtype=np.float32)
                hits += 1

    print(
        f"Vocab: {len(word2idx):,}  |  GloVe coverage: {hits:,}/{len(word2idx):,} ({100*hits/len(word2idx):.1f}%)"
    )
    return matrix


def save_vocab_info(vocab_info: dict, out_path: Path) -> None:
    """Persist vocabulary metadata used to build/checkpoint models."""

    payload = {
        "word2idx": vocab_info["word2idx"],
        "tag2idx": vocab_info["tag2idx"],
        "sa2idx": vocab_info["sa2idx"],
        "embedding_matrix": torch.as_tensor(
            vocab_info["embedding_matrix"], dtype=torch.float32
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)


def load_vocab_info(path: Path) -> dict:
    """Load persisted vocabulary metadata from disk."""

    payload = torch.load(path, map_location="cpu")
    embedding = payload["embedding_matrix"]
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.numpy()
    return {
        "word2idx": payload["word2idx"],
        "tag2idx": payload["tag2idx"],
        "sa2idx": payload["sa2idx"],
        "embedding_matrix": embedding,
    }


# ── PyTorch Dataset ───────────────────────────────────────────────────────────


class NERSADataset(Dataset):

    def __init__(self, samples, word2idx, tag2idx, sa2idx, max_len=128):
        self.samples = samples
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.sa2idx = sa2idx
        self.max_len = max_len
        self.unk = word2idx.get("<UNK>", 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        tokens = s["tokens"]
        length = min(len(tokens), self.max_len)

        token_ids = torch.zeros(self.max_len, dtype=torch.long)
        for i, tok in enumerate(tokens[: self.max_len]):
            token_ids[i] = self.word2idx.get(tok.lower(), self.unk)

        ner_tags = torch.full((self.max_len,), -100, dtype=torch.long)
        for i, tag in enumerate(s["ner_tags"][: self.max_len]):
            ner_tags[i] = tag if isinstance(tag, int) else self.tag2idx.get(tag, 0)

        lbl = s["sentiment"]
        sentiment = torch.tensor(
            self.sa2idx.get(lbl, 0) if isinstance(lbl, str) else lbl,
            dtype=torch.long,
        )

        mask = torch.zeros(self.max_len, dtype=torch.bool)
        mask[:length] = True

        return {
            "token_ids": token_ids,
            "ner_tags": ner_tags,
            "sentiment": sentiment,
            "attention_mask": mask,
            "length": length,
        }


def _collate(batch):
    return {
        "token_ids": torch.stack([b["token_ids"] for b in batch]),
        "ner_tags": torch.stack([b["ner_tags"] for b in batch]),
        "sentiment": torch.stack([b["sentiment"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "length": [b["length"] for b in batch],
    }


# ── DataLoader factory ────────────────────────────────────────────────────────


def get_dataloaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    path = Path(config["paths"]["finer_ord_raw"])
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\nRun: python data/dataset.py"
        )

    with open(path, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]

    word2idx = build_vocab(
        [s["tokens"] for s in samples], config["data"]["min_word_freq"]
    )

    glove_path = config["paths"]["glove"]
    if not Path(glove_path).exists():
        raise FileNotFoundError(f"GloVe file not found: {glove_path}")

    embedding_matrix = load_glove(
        glove_path, word2idx, config["model"]["embedding_dim"]
    )
    tag2idx = {t: i for i, t in enumerate(config["ner"]["tags"])}
    sa2idx = {c: i for i, c in enumerate(config["sa"]["classes"])}

    vocab_info = {
        "word2idx": word2idx,
        "tag2idx": tag2idx,
        "sa2idx": sa2idx,
        "embedding_matrix": embedding_matrix,
    }

    max_len = config["data"]["max_seq_len"]
    batch_size = config["data"]["batch_size"]
    num_workers = config["data"]["num_workers"]

    splits: dict[str, list] = {"train": [], "validation": [], "test": []}
    for s in samples:
        key = s.get("split", "train")
        if key in splits:
            splits[key].append(s)

    def _loader(data, shuffle):
        ds = NERSADataset(data, word2idx, tag2idx, sa2idx, max_len)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=_collate,
        )

    loaders = (
        _loader(splits["train"], True),
        _loader(splits["validation"], False),
        _loader(splits["test"], False),
    )

    print(
        f"Split sizes — train: {len(splits['train'])}  "
        f"val: {len(splits['validation'])}  test: {len(splits['test'])}"
    )

    return *loaders, vocab_info


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    out_path = Path(config["paths"]["finer_ord_raw"])
    if out_path.exists():
        print(f"Dataset already exists at {out_path}. Delete it to rebuild.")
    else:
        build_dataset(out_path)
