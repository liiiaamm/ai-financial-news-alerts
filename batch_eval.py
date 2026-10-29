"""
This module contains the code to run the pipeline over all samples in input/.
"""

# Standard libraries
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Final, Optional

# 3pps
import torch

# Own modules
from data.dataset import get_dataloaders, load_vocab_info
from models.deplot import DePlotCaptioner
from pipeline import load_config, load_joint_model, load_ner_model, load_sa_model, run


# Supported image extensions for chart captioning
IMAGE_EXTS: Final[set] = {".png", ".jpg", ".jpeg"}


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception as exc:
            print(f"[batch_eval] CUDA detected but unusable ({exc}); using CPU.")
    return torch.device("cpu")


def discover_samples(input_dir: Path) -> list[dict]:
    """
    This function discovers one sample per immediate subdirectory of input_dir.
    """

    samples: list[dict] = []

    if not input_dir.exists():
        return samples

    # Sort subdirectories for reproducible ordering
    subdirs: list[Path] = sorted(
        [p for p in input_dir.iterdir() if p.is_dir()],
        key=lambda p: (p.name.lower(), p.name),
    )

    for subdir in subdirs:
        item_id:    str       = subdir.name
        text_path:  str | None = None
        image_path: str | None = None
        errors:     list[str] = []

        txt_files: list[Path] = sorted(subdir.glob("*.txt"))
        img_files: list[Path] = sorted(
            p for p in subdir.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )

        # Validate text file presence
        if len(txt_files) == 0:
            errors.append("no .txt file found in sample folder")
        elif len(txt_files) > 1:
            errors.append(
                f"ambiguous: {len(txt_files)} .txt files found, expected 1 "
                f"({', '.join(f.name for f in txt_files)})"
            )
        else:
            text_path = str(txt_files[0])

        # Validate image file presence (0 images is allowed → text-only run)
        if len(img_files) > 1:
            errors.append(
                f"ambiguous: {len(img_files)} image files found, expected at most 1 "
                f"({', '.join(f.name for f in img_files)})"
            )
        elif len(img_files) == 1:
            image_path = str(img_files[0])

        samples.append({
            "item_id":    item_id,
            "text_path":  text_path,
            "image_path": image_path,
            "error":      "; ".join(errors) if errors else None,
        })

    return samples


def main() -> None:
    """
    This function is the main program for batch evaluation over the input directory.
    """

    # Parse arguments
    parser = argparse.ArgumentParser(description="Batch evaluation over input/ samples")
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--input-dir",  default="input",                 help="Root folder with one subfolder per sample")
    parser.add_argument("--output",     default="results/batch_eval.json")
    parser.add_argument("--use-joint",  dest="use_joint",  action="store_true", default=False)
    parser.add_argument("--use-deplot", dest="use_deplot", action="store_true", default=False)
    parser.add_argument("--use-llm",    dest="use_llm",    action="store_true", default=False)
    args = parser.parse_args()

    # Load configuration and apply CLI overrides
    config: dict = load_config(args.config)
    config.setdefault("pipeline",   {})
    config.setdefault("alert",      {})
    config.setdefault("captioning", {})

    if args.use_joint:  config["pipeline"]["use_joint"]      = True
    if args.use_llm:    config["alert"]["use_llm"]           = True
    if args.use_deplot: config["captioning"]["use_deplot"]   = True

    device: torch.device = _pick_device()
    print(f"[batch_eval] Device: {device}")

    # Discover samples in the input directory
    input_dir: Path       = Path(args.input_dir)
    samples:   list[dict] = discover_samples(input_dir)

    if not samples:
        print(f"[batch_eval] No sample subfolders found in '{input_dir}'. Exiting.")
        sys.exit(1)

    print(f"[batch_eval] Found {len(samples)} sample(s):")
    for s in samples:
        status: str = f"text={Path(s['text_path']).name if s['text_path'] else 'MISSING'}"
        status     += f"  image={Path(s['image_path']).name if s['image_path'] else 'none'}"
        if s["error"]:
            status += f"  ERROR: {s['error']}"
        print(f"  [{s['item_id']}]  {status}")

    # Load vocabulary and models
    ckpt_dir: Path = Path(config["paths"]["checkpoints"])
    vocab_path: Path = ckpt_dir / "vocab_info.pt"
    if vocab_path.exists():
        print(f"[batch_eval] Loading vocabulary from {vocab_path}")
        vocab_info = load_vocab_info(vocab_path)
    else:
        print("[batch_eval] Loading vocabulary and dataloaders...")
        _, _, _, vocab_info = get_dataloaders(config)

    use_joint: bool = config["pipeline"].get("use_joint", False)
    if use_joint:
        ner_model = load_joint_model(config, vocab_info, device)
        sa_model  = None
    else:
        ner_model = load_ner_model(config, vocab_info, device)
        sa_model  = load_sa_model(config, vocab_info, device)

    # Load image captioner if requested
    captioner: Optional[DePlotCaptioner] = None
    if args.use_deplot:
        cap_cfg:   dict      = config.get("captioning", {})
        captioner = DePlotCaptioner(
            deplot_model_id=cap_cfg.get("deplot_model",              "google/deplot"),
            qwen_model_id=cap_cfg.get("qwen_model",                  "Qwen/Qwen2.5-1.5B-Instruct"),
            max_table_tokens=int(cap_cfg.get("deplot_max_table_tokens", 512)),
            max_caption_tokens=int(cap_cfg.get("deplot_max_caption_tokens", 200)),
            temperature=float(cap_cfg.get("deplot_temperature",         0.3)),
        )
        captioner.load_models()

    # Run pipeline over each sample
    results:     list[dict] = []
    total_start: float      = time.time()

    for i, sample in enumerate(samples, 1):
        item_id:         str       = sample["item_id"]
        text_path:       str | None = sample["text_path"]
        image_path:      str | None = sample["image_path"]
        discovery_error: str | None = sample["error"]

        print(f"\n[batch_eval] ({i}/{len(samples)}) {item_id}")

        flags: dict = {
            "use_joint":  args.use_joint,
            "use_deplot": args.use_deplot,
            "use_llm":    args.use_llm,
        }

        # Skip samples with discovery-level errors
        if discovery_error:
            print(f"  SKIP — {discovery_error}")
            results.append({"item_id": item_id, "flags": flags, "models_io": {}, "error": discovery_error})
            continue

        text: str = Path(text_path).read_text(encoding="utf-8").strip()
        if not text:
            err: str = "text file is empty"
            print(f"  SKIP — {err}")
            results.append({"item_id": item_id, "flags": flags, "models_io": {}, "error": err})
            continue

        # Use image only when deplot is enabled
        effective_image: str | None = image_path if args.use_deplot else None
        if args.use_deplot and image_path is None:
            print(f"  [warn] No image found — running text-only for {item_id}")

        t0: float = time.time()
        try:
            output: dict = run(
                text=text,
                image_path=effective_image,
                config=config,
                ner_model=ner_model,
                sa_model=sa_model,
                captioner=captioner,
                vocab_info=vocab_info,
                device=device,
            )
            elapsed_ms: int = int((time.time() - t0) * 1000)

            record: dict = {
                "item_id":    item_id,
                "flags":      flags,
                "models_io":  output["models_io"],
                "sentiment":  output["sentiment"],
                "score":      output["sentiment_score"],
                "entities":   output["entities"],
                "caption":    output.get("caption"),
                "alert_type": output["alert"]["alert_type"],
                "alert_text": output["alert"]["alert_text"],
                "alert_mode": output["alert"].get("mode", "rule_based"),
                "error":      None,
            }

            print(f"  sentiment : {record['sentiment']} (score={record['score']})")
            print(f"  alert     : [{record['alert_type']}] {record['alert_text']}")
            print(f"  time      : {elapsed_ms} ms")

        except Exception as exc:
            elapsed_ms = int((time.time() - t0) * 1000)
            print(f"  ERROR: {exc}")
            record = {
                "item_id":    item_id,
                "flags":      flags,
                "models_io":  {},
                "sentiment":  None,
                "score":      None,
                "entities":   None,
                "caption":    None,
                "alert_type": None,
                "alert_text": None,
                "alert_mode": None,
                "error":      str(exc),
            }

        results.append(record)

    # Build and save final report
    total_elapsed: float     = round(time.time() - total_start, 2)
    successful:    list[dict] = [r for r in results if r.get("error") is None]

    report: dict = {
        "run_info": {
            "timestamp":   datetime.now().isoformat(timespec="seconds"),
            "use_joint":   args.use_joint,
            "use_deplot":  args.use_deplot,
            "use_llm":     args.use_llm,
            "total_items": len(samples),
            "successful":  len(successful),
            "failed":      len(samples) - len(successful),
        },
        "results": results,
    }

    output_path: Path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[batch_eval] Done. {len(successful)}/{len(samples)} successful in {total_elapsed}s.")
    print(f"[batch_eval] Results saved to: {output_path}")

    return None


if __name__ == "__main__":
    main()
