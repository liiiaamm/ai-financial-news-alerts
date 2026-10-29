# Quickstart (Pretrained Inference)

Use this guide if you cloned the repository with pretrained artifacts and want to test inference in a few minutes.

If you want to rebuild dataset and retrain everything from scratch, use `Reproducibility_README.md` instead.

---

## What this quickstart covers

- Fast inference with existing checkpoints and vocabulary metadata
- Demo with an existing sample (`input/p1/apple.txt`)
- Demo with your own custom news text via `experiments/news.py`
- Optional image captioning (`--use-deplot`) and optional LLM alert mode (`--use-llm`)

## What this quickstart does not cover

- Dataset generation
- Model training
- Full end-to-end reproducibility pipeline

---

## 1) Minimal setup (2-5 minutes)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv alert_generator --python 3.12
source alert_generator/bin/activate
uv pip install -r requirements.txt
```

If your CUDA stack is detected but unusable (for example `no kernel image is available`), run on CPU:

```bash
export CUDA_VISIBLE_DEVICES=""
```

---

## 2) Check required pretrained files

Quickstart inference expects these files to exist:

```bash
ls checkpoints/ner_best.pt checkpoints/sa_best.pt checkpoints/joint_best.pt checkpoints/vocab_info.pt
ls data/glove.6B.100d.txt
```

If any of them is missing, use `Reproducibility_README.md` to regenerate artifacts.

---

## 3) Scenario A - run an existing sample from the repo

Use the `p1` Apple sample text and image:

```bash
TEXT="$(cat input/p1/apple.txt)"
IMG="input/p1/apple.png"
```

### Pipeline flags (combinable)

- `--use-joint`: use the joint shared encoder instead of separate NER + SA models.
- `--use-deplot`: enable image captioning from `--image`.
- `--use-llm`: use LLM-based alert generation.
- `--verbose`: print model I/O details.

Important behavior:

- `--use-deplot` only has effect if you pass `--image <path>`.
- `--use-llm` works with or without image/deplot, it requires a HuggingFace token with inference access. \
If `HF_TOKEN` is missing or invalid, `--use-llm` falls back cleanly to rule-based alerts.
```bash
export HF_TOKEN="hf_your_token_here"
```

### Performance note:

- Text-only runs are usually quick.
- Runs with `--use-deplot` can take much longer, especially on first use (model load/download + caption generation).
- Combining `--use-deplot` and `--use-llm` is the slowest quickstart mode.

### Command template

```bash
python pipeline.py --text "$TEXT" [--image "$IMG"] [--use-joint] [--use-deplot] [--use-llm] [--verbose]
```

### Suggested combinations

```bash
# Baseline (rule-based, separate models)
python pipeline.py --text "$TEXT"

# Rule-based + joint + image captioning
python pipeline.py --text "$TEXT" --image "$IMG" --use-joint --use-deplot

# LLM + image captioning
export HF_TOKEN="hf_your_token_here"
python pipeline.py --text "$TEXT" --image "$IMG" --use-deplot --use-llm

# Full combo
python pipeline.py --text "$TEXT" --image "$IMG" --use-joint --use-deplot --use-llm --verbose
```

### Expected output (high level)

Each run prints:

- `Entities`: detected NER entities
- `Sentiment`: class + confidence score
- `Alert type` and `Alert text`: final generated alert

---

## 4) Scenario B - run your own news text

You do not need to edit core scripts.

### B.1 Edit `experiments/news.py`

Set your text and optional image path:

```python
TEXT = "Your new financial headline or paragraph here."
IMAGE_PATH = None
# IMAGE_PATH = "experiments/my_chart.png"
```

### B.2 Run with helper script

`experiments/run_news.py` supports the same flags as Scenario A (`--use-joint`, `--use-deplot`, `--use-llm`, `--verbose`).

```bash
# Rule-based (default)
python -m experiments.run_news

# Example with image captioning + LLM + joint model
python -m experiments.run_news --use-joint --use-deplot --use-llm --verbose
```

Equivalent direct invocation also works:

```bash
python experiments/run_news.py --use-joint --use-deplot --use-llm --verbose
```

Notes:

- `--use-deplot` requires a valid `IMAGE_PATH` in `experiments/news.py`.
- If `IMAGE_PATH` is set but `--use-deplot` is not used, the image is passed but ignored for captioning.
- Without `HF_TOKEN`, `--use-llm` falls back to rule-based mode.

---

## 5) Troubleshooting (quick)

- **CUDA error (`no kernel image is available`)**
  - Run with CPU fallback: `export CUDA_VISIBLE_DEVICES=""`
- **Missing checkpoints or vocab**
  - Run the full reproducibility flow from `Reproducibility_README.md`

---

## 6) Next step

When quick inference is working, move to `Reproducibility_README.md` for full dataset + training + evaluation reproduction.
