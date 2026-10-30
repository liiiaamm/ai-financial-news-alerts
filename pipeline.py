"""
This module contains the end-to-end financial news alert generation pipeline.
"""

# stdlib
import os
import json
import argparse
import concurrent.futures
import re
from typing import Optional, Final

# 3pps
import torch
import yaml
import httpx
from pathlib import Path
from openai import OpenAI

# own modules
from data.dataset import (
    get_dataloaders,
    load_vocab_info,
    extract_entities,
    enrich_ner_tags_with_years,
)
from models.bilstm_ner import BiLSTMNER
from models.bilstm_sa import BiLSTMSA
from models.shared_encoder import SharedBiLSTMEncoder
from models.deplot import DePlotCaptioner
from models.local_llm import get_pipeline as load_qwen


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


# Split punctuation from words, as in the FiNER-ORD training data. Splitting on whitespace
# alone leaves tokens like "year," or "NASDAQ." that are not in the vocabulary.
_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*|[A-Za-z]+(?:\.[A-Za-z]+)+\.?|\w+(?:['’]\w+)*|'s|[^\w\s]")


def tokenize(text: str) -> list[str]:
    """Tokenise raw text the way the training data was tokenised."""
    return _TOKEN_RE.findall(text)


def _pick_device() -> torch.device:
    """Return a CUDA device only when it is actually usable."""

    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception as exc:
            print(f"[pipeline] CUDA detected but unusable ({exc}); using CPU.")
    return torch.device("cpu")


# System prompt for the LLM-based alert generation stage
_ALERT_SYSTEM_PROMPT: Final[str] = (
    "You are a financial news alert generator. "
    "Given named entities, a sentiment class, a confidence score, and an optional chart caption, "
    "write exactly one alert sentence (max 50 words).\n\n"
    "Follow these examples precisely:\n\n"
    "---\n"
    "Entities (2 total):\n"
    "  - Apple [ORG]\n"
    "  - Tim Cook [PER]\n"
    "Sentiment class: negative\n"
    "Confidence     : 0.92\n"
    'Image caption  : "Revenue dropped 12% in Q3 2025."\n'
    "Alert: Apple and Tim Cook face a negative outlook after revenue fell 12% in Q3 2025.\n\n"
    "---\n"
    "Entities (3 total):\n"
    "  - United States [LOC]\n"
    "  - Fed [ORG]\n"
    "  - Powell [PER]\n"
    "Sentiment class: neutral\n"
    "Confidence     : 0.93\n"
    'Image caption  : "Interest rates held at 5.25%."\n'
    "Alert: The outlook for the United States, the Fed, and Powell remains neutral, with interest rates held at 5.25%.\n\n"
    "---\n"
    "Entities (2 total):\n"
    "  - NYSE [ORG]\n"
    "  - NASDAQ [ORG]\n"
    "Sentiment class: negative\n"
    "Confidence     : 0.65\n"
    "Alert: NYSE and NASDAQ may face downside pressure, although the signal is only moderately confident.\n\n"
    "---\n"
    "Entities: none detected\n"
    "Sentiment class: positive\n"
    "Confidence     : 0.48\n"
    "Alert: There are tentative signs of a positive development, although the signal remains inconclusive and no named entities were identified.\n\n"
    "---\n"
    "Now write only the alert for the input below. "
    "Include every entity listed. State the sentiment class. Output only the alert text."
)


def load_config(path: str = "configs/config.yaml") -> dict:
    """
    Loads the pipeline configuration from a YAML file.

    Args:
        path: Path to the configuration file. Defaults to "configs/config.yaml".

    Returns:
        Dictionary containing the parsed configuration values.
    """

    with open(path) as f:
        return yaml.safe_load(f)


def load_vocab_for_inference(config: dict) -> dict:
    """Load persisted vocab metadata when available, else rebuild from dataset."""

    ckpt_dir = Path(config["paths"]["checkpoints"])
    vocab_path = ckpt_dir / "vocab_info.pt"
    if vocab_path.exists():
        print(f"[pipeline] Loading vocabulary from {vocab_path}")
        return load_vocab_info(vocab_path)

    print(
        "[pipeline] No persisted vocab_info.pt found; rebuilding vocabulary from dataset."
    )
    _, _, _, vocab_info = get_dataloaders(config)
    return vocab_info


def load_ner_model(config: dict, vocab_info: dict, device: torch.device) -> BiLSTMNER:
    """
    Loads the trained NER model from its checkpoint.

    Args:
        config: Configuration dictionary with model and path settings.
        vocab_info: Vocabulary metadata including token/tag mappings and
            the pretrained embedding matrix.
        device: Torch device where the model will run.

    Returns:
        BiLSTMNER model loaded from checkpoint and set to eval mode.
    """

    # Instantiate model from config and vocab metadata
    model = BiLSTMNER(
        vocab_size=len(vocab_info["word2idx"]),
        embedding_dim=config["model"]["embedding_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_tags=len(vocab_info["tag2idx"]),
        embedding_matrix=vocab_info["embedding_matrix"],
        dropout=config["model"]["dropout"],
    ).to(device)

    # Load checkpoint weights
    ckpt = Path(config["paths"]["checkpoints"]) / "ner_best.pt"
    state_dict = torch.load(ckpt, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"[pipeline] NER model loaded from {ckpt}")
    return model


def load_sa_model(config: dict, vocab_info: dict, device: torch.device) -> BiLSTMSA:
    """
    Loads the trained sentiment analysis model from its checkpoint.

    Args:
        config: Configuration dictionary with model and path settings.
        vocab_info: Vocabulary metadata including token/sentiment label
            mappings and the pretrained embedding matrix.
        device: Torch device where the model will run.

    Returns:
        BiLSTMSA model loaded from checkpoint and set to eval mode.
    """

    idx2sa = {v: k for k, v in vocab_info["sa2idx"].items()}

    # Instantiate model from config and vocab metadata
    model = BiLSTMSA(
        vocab_size=len(vocab_info["word2idx"]),
        embedding_dim=config["model"]["embedding_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_classes=len(vocab_info["sa2idx"]),
        embedding_matrix=vocab_info["embedding_matrix"],
        dropout=config["model"]["dropout"],
        idx2sa=idx2sa,
    ).to(device)

    # Load checkpoint weights
    ckpt = Path(config["paths"]["checkpoints"]) / "sa_best.pt"
    state_dict = torch.load(ckpt, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"[pipeline] SA model loaded from {ckpt}")
    return model


def load_joint_model(
    config: dict, vocab_info: dict, device: torch.device
) -> SharedBiLSTMEncoder:
    """
    Loads the trained joint NER and sentiment model from its checkpoint.

    Args:
        config: Configuration dictionary with model and path settings.
        vocab_info: Vocabulary metadata including token, tag and sentiment
            label mappings, and the pretrained embedding matrix.
        device: Torch device where the model will run.

    Returns:
        SharedBiLSTMEncoder model loaded from checkpoint and set to eval mode.
    """

    idx2sa = {v: k for k, v in vocab_info["sa2idx"].items()}

    # Instantiate shared encoder from config and vocab metadata
    model = SharedBiLSTMEncoder(
        vocab_size=len(vocab_info["word2idx"]),
        embedding_dim=config["model"]["embedding_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_tags=len(vocab_info["tag2idx"]),
        num_sa_classes=len(vocab_info["sa2idx"]),
        embedding_matrix=vocab_info["embedding_matrix"],
        dropout=config["model"]["dropout"],
        idx2sa=idx2sa,
    ).to(device)

    # Load checkpoint weights
    ckpt = Path(config["paths"]["checkpoints"]) / "joint_best.pt"
    state_dict = torch.load(ckpt, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"[pipeline] Joint model loaded from {ckpt}")
    return model


def predict_ner(
    model,
    tokens: list[str],
    vocab_info: dict,
    config: dict,
    device: torch.device,
) -> list[dict]:
    """
    Runs named entity recognition on a tokenised input sequence.

    Args:
        model: Loaded NER model (BiLSTMNER or SharedBiLSTMEncoder).
        tokens: List of input tokens.
        vocab_info: Vocabulary metadata including token and tag mappings.
        config: Configuration dictionary with data preprocessing settings.
        device: Torch device used for inference.

    Returns:
        List of entity dicts, each with text, type and token span boundaries.
    """

    word2idx = vocab_info["word2idx"]
    idx2tag = {v: k for k, v in vocab_info["tag2idx"].items()}
    max_len = config["data"]["max_seq_len"]
    unk = word2idx.get("<UNK>", 1)

    # Encode and pad token sequence
    ids = [word2idx.get(t.lower(), unk) for t in tokens]
    length = min(len(ids), max_len)
    padded = ids[:max_len] + [0] * max(0, max_len - len(ids))

    token_ids = torch.tensor([padded], dtype=torch.long).to(device)
    mask = torch.zeros(1, max_len, dtype=torch.bool).to(device)
    mask[0, :length] = True

    # Run NER inference
    with torch.no_grad():
        if hasattr(model, "predict_ner"):
            preds = model.predict_ner(token_ids, mask).cpu()[0, :length].tolist()
        else:
            preds = model.predict(token_ids, mask).cpu()[0, :length].tolist()

    # Postprocess: enrich year tags and extract entity spans
    inv_map = {v: k for k, v in TAG_ID_TO_NAME.items()}
    tag_ids = [inv_map.get(idx2tag.get(p, "O"), 0) for p in preds]
    enriched = enrich_ner_tags_with_years(tokens[:length], tag_ids)

    return extract_entities(tokens[:length], enriched)


def predict_sa(
    model,
    text: str,
    vocab_info: dict,
    config: dict,
    device: torch.device,
) -> tuple[str, float]:
    """
    Runs sentiment analysis on raw input text.

    Args:
        model: Loaded SA model (BiLSTMSA or SharedBiLSTMEncoder).
        text: Raw input text to analyse.
        vocab_info: Vocabulary metadata including token and sentiment mappings.
        config: Configuration dictionary with data preprocessing settings.
        device: Torch device used for inference.

    Returns:
        Tuple of (predicted sentiment label, confidence score).
    """

    word2idx = vocab_info["word2idx"]
    max_len = config["data"]["max_seq_len"]
    unk = word2idx.get("<UNK>", 1)

    # Encode and pad token sequence
    tokens = [t.lower() for t in tokenize(text)]
    ids = [word2idx.get(t, unk) for t in tokens]
    length = min(len(ids), max_len)
    padded = ids[:max_len] + [0] * max(0, max_len - len(ids))

    token_ids = torch.tensor([padded], dtype=torch.long).to(device)
    mask = torch.zeros(1, max_len, dtype=torch.bool).to(device)
    mask[0, :length] = True

    # Run SA inference
    with torch.no_grad():
        if hasattr(model, "predict_sa"):
            idx2sa = {v: k for k, v in vocab_info["sa2idx"].items()}
            _, sa_logits, _ = model(token_ids, mask, task="sa")
            probs = torch.softmax(sa_logits, dim=-1)
            pred_idx = int(probs.argmax(dim=-1).item())
            return idx2sa.get(pred_idx, "neutral"), float(probs[0, pred_idx].item())
        else:
            label, prob_dict, _ = model.predict_proba(token_ids, mask)
            return label, prob_dict[label]


def _build_llm_prompt(
    entities: list[dict],
    sentiment: str,
    score: float,
    caption: Optional[str],
) -> str:
    """
    Builds the user message for the LLM alert generation prompt.

    Args:
        entities: List of extracted entity dicts.
        sentiment: Predicted sentiment label.
        score: Confidence score for the sentiment prediction.
        caption: Optional image caption. Defaults to None.

    Returns:
        Formatted prompt string.
    """

    if entities:
        ent_lines = "\n".join(f"  - {e['text']} [{e['type']}]" for e in entities)
        ent_block = f"Entities ({len(entities)} total):\n{ent_lines}"
    else:
        ent_block = "Entities: none detected"

    lines = [
        ent_block,
        f"Sentiment class: {sentiment}",
        f"Confidence     : {score:.2f}",
    ]
    if caption:
        lines.append(f'Image caption  : "{caption[:200]}"')
    lines.append("Alert:")

    return "\n".join(lines)


def _llm_alert(
    entities: list[dict],
    sentiment: str,
    score: float,
    caption: Optional[str],
    config: dict,
) -> Optional[str]:
    """
    Generates an alert text using a language model.

    Calls either a local Qwen model or the HuggingFace inference API depending
    on the configuration. Returns None if generation fails.

    Args:
        entities: List of extracted entity dicts.
        sentiment: Predicted sentiment label.
        score: Confidence score for the sentiment prediction.
        caption: Optional image caption. Defaults to None.
        config: Configuration dictionary with LLM generation settings.

    Returns:
        Generated alert text string, or None if generation failed.
    """

    alert_cfg = config.get("alert", {})
    model_id = alert_cfg.get("llm_model", "Qwen/Qwen2.5-1.5B-Instruct").split(":")[0]
    max_tok = int(alert_cfg.get("llm_max_tokens", 80))
    temp = float(alert_cfg.get("llm_temperature", 0.3))
    cache_dir = alert_cfg.get("local_model_dir", None)
    use_local = bool(alert_cfg.get("use_local_llm", True))

    messages = [
        {"role": "system", "content": _ALERT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _build_llm_prompt(entities, sentiment, score, caption),
        },
    ]

    try:
        if use_local:
            # Call local Qwen model loaded as a singleton
            from models.local_llm import generate as llm_generate

            return llm_generate(
                messages,
                model_id=model_id,
                cache_dir=cache_dir,
                max_new_tokens=max_tok,
                temperature=temp,
            )
        else:
            # Call HuggingFace inference API
            hf_token = os.environ.get("HF_TOKEN", "").strip()
            if not hf_token:
                print("[pipeline] HF_TOKEN not set — cannot call HuggingFace API.")
                return None
            client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=hf_token,
                http_client=httpx.Client(),
            )
            completion = client.chat.completions.create(
                model=alert_cfg.get("llm_model", "Qwen/Qwen2.5-1.5B-Instruct"),
                messages=messages,
                max_tokens=max_tok,
                temperature=temp,
            )
            return completion.choices[0].message.content.strip()

    except Exception as exc:
        print(f"[pipeline] LLM error: {exc}")
        return None


def _rule_based(
    entities: list[dict],
    sentiment: str,
    caption: Optional[str],
) -> dict:
    """
    Generates a rule-based alert from entities and sentiment.

    Args:
        entities: List of extracted entity dicts.
        sentiment: Predicted sentiment label.
        caption: Optional image caption. Defaults to None.

    Returns:
        Alert dictionary with alert_text, alert_type, entities_used,
        sentiment, confidence and mode fields.
    """

    sentiment = (sentiment or "neutral").lower()

    orgs = [e for e in entities if e.get("type") == "ORG"]
    pers = [e for e in entities if e.get("type") == "PER"]
    locs = [e for e in entities if e.get("type") == "LOC"]
    multi = len(entities) > 1

    def first(lst: list) -> str:
        return lst[0]["text"] if lst else ""

    def ent_list(lst: list) -> str:
        return ", ".join(f"{e['text']} [{e['type']}]" for e in lst)

    # Match entity type + sentiment to alert template
    if orgs and sentiment == "negative":
        atype, text = (
            "REPUTATION_RISK",
            f"REPUTATION RISK: {first(orgs)} mentioned NEGATIVELY.",
        )
    elif pers and sentiment == "negative":
        atype, text = (
            "REPUTATION_RISK",
            f"REPUTATION RISK: {first(pers)} mentioned NEGATIVELY.",
        )
    elif orgs and sentiment == "positive":
        atype, text = (
            "ECONOMIC_UPDATE",
            f"POSITIVE UPDATE: {first(orgs)} reported POSITIVELY.",
        )
    elif pers and sentiment == "positive":
        atype, text = (
            "ECONOMIC_UPDATE",
            f"POSITIVE UPDATE: {first(pers)} mentioned in positive context.",
        )
    elif locs and sentiment == "negative":
        atype, text = (
            "GEOPOLITICAL_RISK",
            f"GEOPOLITICAL RISK: negative news involving {first(locs)}.",
        )
    elif locs and sentiment == "positive":
        atype, text = (
            "ECONOMIC_UPDATE",
            f"POSITIVE DEVELOPMENT: good news from {first(locs)}.",
        )
    elif multi and sentiment == "negative":
        atype, text = (
            "RISK_ALERT",
            f"RISK ALERT: negative sentiment. Entities: {ent_list(entities)}.",
        )
    elif multi and sentiment == "positive":
        atype, text = (
            "POSITIVE_ALERT",
            f"POSITIVE ALERT: positive sentiment. Entities: {ent_list(entities)}.",
        )
    elif not entities and sentiment == "negative":
        atype, text = (
            "SENTIMENT_ALERT",
            "NEGATIVE SENTIMENT DETECTED. No named entities identified.",
        )
    elif not entities and sentiment == "positive":
        atype, text = (
            "SENTIMENT_ALERT",
            "POSITIVE SENTIMENT DETECTED. No named entities identified.",
        )
    elif entities and sentiment == "neutral":
        atype, text = "INFO", f"INFO: neutral content. Entities: {ent_list(entities)}."
    else:
        atype, text = (
            "NEUTRAL",
            "No significant alert. Content appears neutral with no named entities.",
        )

    # Append image context if a caption is available
    if caption:
        text += f" [Image context: {caption[:80]}]"

    return {
        "alert_text": text,
        "alert_type": atype,
        "entities_used": [e["text"] for e in entities],
        "sentiment": sentiment,
        "confidence": 1.0,
        "mode": "rule_based",
    }


def generate_alert(
    entities: list[dict],
    sentiment: str,
    score: float,
    caption: Optional[str],
    config: dict,
) -> dict:
    """
    Generates the final alert using LLM or rule-based mode.

    Tries LLM generation when enabled; falls back to rule-based if it fails.

    Args:
        entities: List of extracted entity dicts.
        sentiment: Predicted sentiment label.
        score: Confidence score for the sentiment prediction.
        caption: Optional image caption. Defaults to None.
        config: Configuration dictionary with alert generation settings.

    Returns:
        Alert dictionary with alert_text, alert_type and metadata fields.
    """

    if config.get("alert", {}).get("use_llm", False):
        # Attempt LLM generation; fall back to rule-based on failure
        llm_text = _llm_alert(entities, sentiment, score, caption, config)
        if llm_text:
            return {
                "alert_text": llm_text,
                "alert_type": "LLM_ALERT",
                "entities_used": [e["text"] for e in entities],
                "sentiment": sentiment,
                "confidence": round(float(score), 4),
                "mode": "llm",
            }
        print("[pipeline] LLM unavailable — falling back to rule-based alert.")

    return _rule_based(entities, sentiment, caption)


def run(
    text: str,
    image_path: Optional[str],
    config: dict,
    ner_model,
    sa_model,
    captioner: Optional[DePlotCaptioner],
    vocab_info: dict,
    device: torch.device,
) -> dict:
    """
    Runs the full alert generation pipeline on a single input sample.

    NER and SA are executed in parallel threads. Image captioning and alert
    generation follow sequentially.

    Args:
        text: Input financial news text.
        image_path: Path to an optional chart image. Defaults to None.
        config: Configuration dictionary for the pipeline.
        ner_model: Loaded NER model or joint encoder.
        sa_model: Loaded SA model, or None when using the joint encoder.
        captioner: Loaded DePlotCaptioner instance, or None if disabled.
        vocab_info: Vocabulary metadata for preprocessing and decoding.
        device: Torch device used for inference.

    Returns:
        Dictionary with entities, sentiment, sentiment_score, caption,
        alert and models_io fields.
    """

    tokens = tokenize(text)

    # Run NER and SA in parallel to reduce latency
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        fut_ner = executor.submit(
            predict_ner, ner_model, tokens, vocab_info, config, device
        )
        fut_sa = executor.submit(
            predict_sa,
            sa_model if sa_model else ner_model,
            text,
            vocab_info,
            config,
            device,
        )
        entities = fut_ner.result()
        sentiment, score = fut_sa.result()

    # Image captioning: DePlot extracts table, Qwen generates narrative caption
    deplot_table: Optional[str] = None
    caption: Optional[str] = None
    if captioner is not None and image_path is not None:
        deplot_table, caption = captioner.caption(image_path)

    # Generate alert from NER + SA + optional caption
    alert = generate_alert(entities, sentiment, score, caption, config)

    # Build per-model input/output trace for logging and evaluation
    models_io: dict = {
        "ner": {"input": text, "output": entities},
        "sa": {
            "input": text,
            "output": {"sentiment": sentiment, "score": round(float(score), 4)},
        },
    }
    if deplot_table is not None:
        models_io["deplot"] = {"input": str(image_path), "output": deplot_table}
        models_io["qwen_captioning"] = {"input": deplot_table, "output": caption}
    if alert.get("mode") == "llm":
        alert_input = {
            "entities": [f"{e['text']} [{e['type']}]" for e in entities]
            or ["none detected"],
            "sentiment": sentiment,
            "score": round(float(score), 4),
        }
        if caption:
            alert_input["caption"] = caption[:200]
        models_io["qwen_alert"] = {"input": alert_input, "output": alert["alert_text"]}

    return {
        "entities": entities,
        "sentiment": sentiment,
        "sentiment_score": round(float(score), 4),
        "caption": caption,
        "alert": alert,
        "models_io": models_io,
    }


def main() -> None:
    """
    Entry point for the command-line pipeline.

    Parses arguments, loads config and models, runs the pipeline and
    prints the final alert result to stdout.
    """

    # Allow online model download by default when cache is missing.
    # Respect an explicit user override (e.g. TRANSFORMERS_OFFLINE=1).
    if os.environ.get("TRANSFORMERS_OFFLINE") is None:
        os.environ["TRANSFORMERS_OFFLINE"] = "0"

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Financial news alert generation pipeline"
    )
    parser.add_argument("--text", required=True, help="Input news text")
    parser.add_argument("--image", default=None, help="Path to chart image (optional)")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--use-joint", dest="use_joint", action="store_true", default=False
    )
    parser.add_argument(
        "--use-deplot", dest="use_deplot", action="store_true", default=False
    )
    parser.add_argument("--use-llm", dest="use_llm", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

    # Load config and apply CLI overrides
    config = load_config(args.config)
    config.setdefault("pipeline", {})
    config.setdefault("alert", {})
    if args.use_joint:
        config["pipeline"]["use_joint"] = True
    if args.use_llm:
        config["alert"]["use_llm"] = True
    if args.use_deplot:
        config.setdefault("captioning", {})["use_deplot"] = True

    use_joint = config["pipeline"].get("use_joint", False)
    device = _pick_device()

    if args.verbose:
        print(f"  text      : {args.text[:80]}")
        print(f"  image     : {args.image}")
        print(f"  use_joint : {use_joint}")
        print(f"  use_llm   : {config['alert'].get('use_llm', False)}")

    # Load vocabulary metadata
    print("[pipeline] Loading vocabulary...")
    vocab_info = load_vocab_for_inference(config)

    # Load NER and SA models (separate or joint)
    if use_joint:
        ner_model = load_joint_model(config, vocab_info, device)
        sa_model = None
    else:
        ner_model = load_ner_model(config, vocab_info, device)
        sa_model = load_sa_model(config, vocab_info, device)

    # Load image captioner if requested (must happen before ThreadPoolExecutor)
    captioner: Optional[DePlotCaptioner] = None
    if args.use_deplot and args.image is not None:
        cap_cfg = config.get("captioning", {})
        captioner = DePlotCaptioner(
            deplot_model_id=cap_cfg.get("deplot_model", "google/deplot"),
            qwen_model_id=cap_cfg.get("qwen_model", "Qwen/Qwen2.5-1.5B-Instruct"),
            max_table_tokens=int(cap_cfg.get("deplot_max_table_tokens", 512)),
            max_caption_tokens=int(cap_cfg.get("deplot_max_caption_tokens", 200)),
            temperature=float(cap_cfg.get("deplot_temperature", 0.3)),
        )
        captioner.load_models()

    # Run full pipeline
    result = run(
        text=args.text,
        image_path=args.image,
        config=config,
        ner_model=ner_model,
        sa_model=sa_model,
        captioner=captioner,
        vocab_info=vocab_info,
        device=device,
    )

    # Print results
    alert = result["alert"]
    print(f"\nEntities  : {result['entities']}")
    print(f"Sentiment : {result['sentiment']}  (score={result['sentiment_score']:.4f})")
    print(f"Caption   : {result['caption']}")
    print(f"Alert type: {alert['alert_type']}")
    print(f"Alert text: {alert['alert_text']}")
    print(
        f"Confidence: {alert.get('confidence', 0):.2f}  (mode: {alert.get('mode', 'unknown')})"
    )

    if args.verbose:
        print("\nModels I/O:")
        print(json.dumps(result["models_io"], indent=2, ensure_ascii=False))

    return None


if __name__ == "__main__":
    main()
