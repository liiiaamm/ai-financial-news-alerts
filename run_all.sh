#!/usr/bin/env bash
# run_all.sh
#
# Full reproducibility script for the Financial News Alert System.
# Runs everything from scratch in order:
#   dataset build -> NER training -> SA training -> joint training
#   -> model evaluation -> all 8 pipeline combinations -> batch evaluation
#
# Usage:
#   source alert_generator/bin/activate   # activate your Python environment first
#   bash run_all.sh
#   # optional logging:
#   # bash run_all.sh | tee results/run_all.log
#
# The LLM combinations (5-8) download ~3 GB of Qwen weights on first run.
# Output is printed to the terminal. Use tee if you want a log file.

set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] uv is required but not found."
  echo "[setup] Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

if [ ! -x "alert_generator/bin/python" ]; then
  echo "[setup] Creating virtual environment: alert_generator (Python 3.12)"
  uv venv alert_generator --python 3.12
else
  echo "[setup] Reusing existing virtual environment: alert_generator"
fi

echo "[setup] Installing/updating dependencies from requirements.txt"
uv pip install --python alert_generator/bin/python -r requirements.txt

PYTHON="${PYTHON:-alert_generator/bin/python}"
HF_TOKEN="${HF_TOKEN:-}"
TEXT=$(cat input/p1/apple.txt | tr -d '\r')
IMAGE="input/p1/apple.png"
mkdir -p results

echo "========================================================"
echo "  Financial News Alert System -- full reproducibility run"
echo "  $(date)"
echo "  Python : $($PYTHON --version 2>&1)"
echo "========================================================"

# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------

echo ""
echo "--------------------------------------------------------"
echo " [1/7] Building dataset"
echo "--------------------------------------------------------"
$PYTHON data/dataset.py

# ---------------------------------------------------------------------------
# 2. Train NER
# ---------------------------------------------------------------------------

echo ""
echo "--------------------------------------------------------"
echo " [2/7] Training NER model"
echo "--------------------------------------------------------"
rm -f checkpoints/ner_best.pt
$PYTHON training/train_ner.py

# ---------------------------------------------------------------------------
# 3. Train SA
# ---------------------------------------------------------------------------

echo ""
echo "--------------------------------------------------------"
echo " [3/7] Training Sentiment Analysis model"
echo "--------------------------------------------------------"
rm -f checkpoints/sa_best.pt
$PYTHON training/train_sa.py

# ---------------------------------------------------------------------------
# 4. Train Joint
# ---------------------------------------------------------------------------

echo ""
echo "--------------------------------------------------------"
echo " [4/7] Training Joint shared encoder  (with lambda ablation)"
echo "--------------------------------------------------------"
rm -f checkpoints/joint_best.pt
$PYTHON training/train_joint.py

# ---------------------------------------------------------------------------
# 5. Evaluate models
# ---------------------------------------------------------------------------

echo ""
echo "--------------------------------------------------------"
echo " [5/7] Evaluating models"
echo "--------------------------------------------------------"

echo ""
echo "  NER -- built-in examples"
$PYTHON evaluate.py --task ner

echo ""
echo "  SA -- built-in examples"
$PYTHON evaluate.py --task sa

echo ""
echo "  Both tasks -- sample text (separate models)"
$PYTHON evaluate.py --task both --text "$TEXT"

echo ""
echo "  Both tasks -- sample text (joint model)"
$PYTHON evaluate.py --task both --text "$TEXT" --use-joint

# ---------------------------------------------------------------------------
# 6. Pipeline combinations (8/8)
# ---------------------------------------------------------------------------

echo ""
echo "========================================================"
echo " [6/7] Pipeline combinations -- input/p1"
echo "========================================================"

echo ""
echo "  [1/8] Rule-based  |  separate models  |  no image"
$PYTHON pipeline.py --text "$TEXT"

echo ""
echo "  [2/8] Rule-based  |  separate models  |  image given, no --use-deplot (image ignored)"
$PYTHON pipeline.py --text "$TEXT" --image "$IMAGE"

echo ""
echo "  [3/8] Rule-based  |  separate models  |  DePlot caption"
$PYTHON pipeline.py --text "$TEXT" --image "$IMAGE" --use-deplot

echo ""
echo "  [4/8] Rule-based  |  joint encoder    |  no image"
$PYTHON pipeline.py --text "$TEXT" --use-joint

echo ""
echo "  [5/8] LLM alert   |  separate models  |  no image"
HF_TOKEN="$HF_TOKEN" $PYTHON pipeline.py --text "$TEXT" --use-llm

echo ""
echo "  [6/8] LLM alert   |  separate models  |  DePlot caption"
HF_TOKEN="$HF_TOKEN" $PYTHON pipeline.py --text "$TEXT" --image "$IMAGE" --use-deplot --use-llm

echo ""
echo "  [7/8] LLM alert   |  joint encoder    |  DePlot caption"
HF_TOKEN="$HF_TOKEN" $PYTHON pipeline.py --text "$TEXT" --image "$IMAGE" --use-joint --use-deplot --use-llm

echo ""
echo "  [8/8] LLM alert   |  joint encoder    |  DePlot caption  |  verbose"
HF_TOKEN="$HF_TOKEN" $PYTHON pipeline.py --text "$TEXT" --image "$IMAGE" --use-joint --use-deplot --use-llm --verbose

# ---------------------------------------------------------------------------
# 7. Batch evaluation over all 10 samples
# ---------------------------------------------------------------------------

echo ""
echo "========================================================"
echo " [7/7] Batch evaluation -- all input/ samples"
echo "========================================================"

echo ""
echo "  Rule-based  |  separate models"
$PYTHON batch_eval.py --output results/batch_eval.json

echo ""
echo "  Rule-based  |  joint encoder"
$PYTHON batch_eval.py --use-joint --output results/batch_eval_joint.json

echo ""
echo "  LLM alert   |  joint encoder  |  DePlot caption"
HF_TOKEN="$HF_TOKEN" $PYTHON batch_eval.py --use-joint --use-deplot --use-llm --output results/batch_eval_full.json

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

echo ""
echo "========================================================"
echo " Done -- $(date)"
echo " Results saved to: results/"
echo "========================================================"
