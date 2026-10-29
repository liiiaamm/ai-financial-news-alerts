"""Quick runner for custom news experiments.

Reads TEXT and IMAGE_PATH from experiments/news.py and forwards them to pipeline.py
with convenient flags.
"""

from __future__ import annotations

import argparse
import importlib.util
import shlex
import subprocess
import sys
from pathlib import Path


def load_news_module(news_path: Path):
    spec = importlib.util.spec_from_file_location("experiments.news", news_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {news_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pipeline using experiments/news.py")
    parser.add_argument("--use-joint", action="store_true", default=False)
    parser.add_argument("--use-deplot", action="store_true", default=False)
    parser.add_argument("--use-llm", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    news_path = repo_root / "experiments" / "news.py"
    news = load_news_module(news_path)

    text = str(getattr(news, "TEXT", "")).strip()
    image_value = getattr(news, "IMAGE_PATH", None)
    image_path = str(image_value).strip() if image_value else None

    if not text:
        print("[run_news] ERROR: TEXT in experiments/news.py is empty.")
        return 2

    command = [
        sys.executable,
        "pipeline.py",
        "--text",
        text,
        "--config",
        args.config,
    ]

    if args.use_joint:
        command.append("--use-joint")
    if args.use_llm:
        command.append("--use-llm")
    if args.verbose:
        command.append("--verbose")

    if args.use_deplot:
        if not image_path:
            print(
                "[run_news] ERROR: --use-deplot requires IMAGE_PATH in experiments/news.py"
            )
            return 2
        candidates = [
            (repo_root / image_path).resolve(),
            (news_path.parent / image_path).resolve(),
        ]
        full_image_path = next((p for p in candidates if p.exists()), None)
        if full_image_path is None:
            print(
                "[run_news] ERROR: image not found. Tried: "
                + ", ".join(str(p) for p in candidates)
            )
            return 2
        try:
            image_arg = str(full_image_path.relative_to(repo_root))
        except ValueError:
            image_arg = str(full_image_path)
        command.extend(["--image", image_arg, "--use-deplot"])
    elif image_path:
        command.extend(["--image", image_path])

    print("[run_news] Running:", flush=True)
    print(shlex.join(command), flush=True)
    print("", flush=True)
    return subprocess.run(command, cwd=repo_root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
