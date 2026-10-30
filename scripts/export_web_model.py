"""Export the joint NER + sentiment model for in-browser inference.

Writes web/model/weights.bin (float16, concatenated tensors), web/model/manifest.json
(name, shape and offset of each tensor) and web/model/vocab.json.

    python scripts/export_web_model.py
"""
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "model"
KEEP = [
    "embedding.weight",
    *[f"lstm.{p}_l{l}{s}" for l in (0, 1) for s in ("", "_reverse") for p in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")],
    "ner_head.weight", "ner_head.bias", "W_attention.weight", "sa_head.weight", "sa_head.bias",
]


def main():
    state = torch.load(ROOT / "checkpoints" / "joint_best.pt", map_location="cpu", weights_only=False)
    vocab = torch.load(ROOT / "checkpoints" / "vocab_info.pt", map_location="cpu", weights_only=False)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, blobs, offset = [], [], 0
    for name in KEEP:
        arr = state[name].numpy().astype(np.float16)
        manifest.append({"name": name, "shape": list(arr.shape), "offset": offset})
        blobs.append(arr.tobytes())
        offset += arr.size
    (OUT / "weights.bin").write_bytes(b"".join(blobs))
    idx2tag = {v: k for k, v in vocab["tag2idx"].items()}
    idx2sa = {v: k for k, v in vocab["sa2idx"].items()}
    (OUT / "manifest.json").write_text(json.dumps({
        "dtype": "float16", "tensors": manifest,
        "tags": [idx2tag[i] for i in range(len(idx2tag))],
        "sentiments": [idx2sa[i] for i in range(len(idx2sa))],
        "max_len": 128, "unk": vocab["word2idx"]["<UNK>"],
    }))
    words = sorted(vocab["word2idx"], key=vocab["word2idx"].get)
    (OUT / "vocab.json").write_text(json.dumps(words, ensure_ascii=False, separators=(",", ":")))
    print(f"{offset:,} parameters, {(OUT / 'weights.bin').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
