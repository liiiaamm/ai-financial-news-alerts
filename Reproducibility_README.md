# Financial News Alert System

Full reproducibility guide for rebuilding the complete project pipeline:
environment setup, GloVe download, dataset build, training, evaluation, and full scripted run.

For a project overview, see `README.md`.
For a fast inference-only path with existing checkpoints, see `Quickstart_README.md`.

---

## Requirements

- Python 3.11 or 3.12 (tested)
- ~5 GB disk space (GloVe embeddings + model checkpoints)
- Internet connection for the first run (HuggingFace dataset + FinBERT + Qwen weights)
- A CUDA-capable GPU is recommended but not required (CPU-only is supported)

---

## 1. Set up the environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv alert_generator --python 3.12
source alert_generator/bin/activate          # Windows: venv\Scripts\activate
uv pip install -r requirements.txt
```

If your machine has an unsupported CUDA setup (for example `no kernel image is available for execution on the device`), force CPU execution:

```bash
export CUDA_VISIBLE_DEVICES=""
```

---

## 2. Download GloVe embeddings

The NER and SA models use 100-dimensional GloVe vectors (~340 MB unzipped).

```bash
cd data/
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip glove.6B.100d.txt
rm glove.6B.zip
cd ..
```

If unzip fails, try running one of these two commands:

```bash
sudo apt-get update && sudo apt-get install -y unzip && unzip glove.6B.zip glove.6B.100d.txt
python -m zipfile -e glove.6B.zip ./
```

---

## 3. Build the dataset

Downloads [FiNER-ORD](https://huggingface.co/datasets/gtfintechlab/finer-ord) from HuggingFace, runs FinBERT (`ProsusAI/finbert`) over every sentence to generate silver sentiment labels, and writes the result to `data/finer_ord_with_alerts.jsonl`.

```bash
python -m data.dataset
```

Equivalent command to use if the first one fails:

```bash
python data/dataset.py
```

Expected time: 5-10 minutes with a GPU; 15-25 minutes on CPU only.

---

## 4. Train the models

Each script reads `configs/config.yaml` and saves its checkpoint to `checkpoints/`.

```bash
# Step 1 -- NER model (~20 min)
python training/train_ner.py

# Step 2 -- Sentiment Analysis model (~25 min)
python training/train_sa.py

# Step 3 -- Joint shared encoder + lambda ablation study (~30 min)
python training/train_joint.py
```

Training curves (loss, F1, learning rate) are logged to TensorBoard:

```bash
tensorboard --logdir runs/
```

Checkpoints are saved to:

```
checkpoints/ner_best.pt
checkpoints/sa_best.pt
checkpoints/joint_best.pt
```

---

## 5. Evaluate the models

```bash
python evaluate.py --task ner               # NER examples with colored entity tags
python evaluate.py --task sa                # SA examples with probabilities and attention
python evaluate.py --task both              # both tasks on built-in examples
python evaluate.py --task both --use-joint  # same with the joint shared encoder
python evaluate.py --task both --text "Apple reports record losses in Q3."
```

---

## 6. Run the pipeline

### Rule-based alert generation

```bash
python pipeline.py --text "Apple reports record quarterly losses amid supply chain disruptions."
```

With a chart image:

```bash
python pipeline.py \
  --text "Apple reports record quarterly losses amid supply chain disruptions." \
  --image input/p1/apple.png \
  --use-deplot
```

With the joint shared encoder:

```bash
python pipeline.py --text "..." --use-joint
```

### LLM-based alert generation

Feeds the NER tags, sentiment score, and optional image caption to Qwen 2.5 to produce a more natural alert. Requires a HuggingFace token with inference access.

```bash
export HF_TOKEN="hf_your_token_here"
python pipeline.py --text "Apple reports record losses." --use-llm
```

On first run, the Qwen weights (~3 GB) are downloaded to `~/.cache/huggingface/hub`. Subsequent runs load from cache.

If `HF_TOKEN` is missing or invalid, the pipeline falls back to rule-based alert generation and prints a warning.

### Full combination

```bash
python pipeline.py \
  --text "Tesla stock drops after disappointing earnings report." \
  --image input/p1/apple.png \
  --use-joint \
  --use-deplot \
  --use-llm \
  --verbose
```

For additional quick combinations of these flags, see `Quickstart_README.md`.

---

## 7. Batch evaluation

Runs the pipeline over all samples in `input/` and saves a JSON report.

```bash
python batch_eval.py                                                  # rule-based, separate models
python batch_eval.py --use-joint                                      # joint encoder
python batch_eval.py --use-joint --use-deplot --use-llm               # full pipeline
```

If `HF_TOKEN` is not available, LLM runs degrade gracefully to rule-based alerts.

---

## 8. Full reproducibility

To run the full pipeline in one command:

```bash
bash run_all.sh
```

This builds the dataset, trains all three models, evaluates them, runs all pipeline combinations, and saves batch-evaluation results to `results/`.

Important: if `data/finer_ord_with_alerts.jsonl` already exists, dataset generation is skipped by `data/dataset.py`. Delete that file first if you want a strict rebuild from scratch.

---

## 9. Project structure

```
.
|- configs/
|  |- config.yaml              # all hyperparameters and paths
|- data/
|  |- dataset.py               # dataset builder (FiNER-ORD + FinBERT) and PyTorch DataLoaders
|  |- glove.6B.100d.txt        # GloVe embeddings (downloaded manually)
|- models/
|  |- bilstm_ner.py            # BiLSTM + CRF for NER
|  |- bilstm_sa.py             # BiLSTM with attention for sentiment analysis
|  |- shared_encoder.py        # shared BiLSTM encoder (joint model)
|  |- deplot.py                # DePlot + Qwen chart captioner
|  |- local_llm.py             # singleton wrapper for the local Qwen model
|  `- captioning_CNN_LSTM.py   # CNN + LSTM captioner (experimental, not used in pipeline)
|- input/                      # 10 sample pairs (text + chart image)
|- checkpoints/                # saved model weights (created after training)
|- results/                    # evaluation results and training history
|- runs/                       # TensorBoard logs
|- training/
|  |- train_ner.py             # BiLSTM-CRF NER training
|  |- train_sa.py              # BiLSTM SA training
|  `- train_joint.py           # joint shared encoder training + lambda ablation
|- pipeline.py                 # end-to-end inference script
|- batch_eval.py               # batch evaluation over input/ samples
|- evaluate.py                 # interactive model evaluation (NER, SA, joint)
|- run_all.sh                  # full reproducibility script
|- requirements.txt
```

---

## Notes

**Why F1 and not accuracy?**
The dataset is class-imbalanced: most tokens are tagged `O` in NER, and the neutral class dominates in sentiment analysis. Accuracy would be misleadingly high even for a model that never predicts minority classes. Entity-level F1 via seqeval also penalizes partial matches - a model must predict the full span correctly to get credit. Macro F1 for sentiment treats all three classes equally regardless of frequency.

**CNN + LSTM captioner (`captioning_CNN_LSTM.py`)**
This model was built and trained from scratch as an academic exercise to understand sequence-to-sequence image captioning. In practice, the DePlot + Qwen pipeline produces significantly better captions without any fine-tuning, so the CNN+LSTM version is kept for reference only and is not used in the default pipeline.

**Reproducibility**
All random seeds are fixed to 42 across all training scripts. Results may vary slightly between CPU and GPU runs due to floating-point non-determinism in cuDNN operations.
