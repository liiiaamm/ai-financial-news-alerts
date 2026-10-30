"""Collect the 10 demo samples and their offline outputs (chart caption, LLM alert) for the web demo.

    python scripts/build_web_samples.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TITLES = {
    "p1": "Apple's record quarter", "p2": "The US primary credit rate", "p3": "Facebook's IPO",
    "p4": "The Fed's growth outlook", "p5": "Seven decades of the S&P 500", "p6": "Share prices, rich and emerging",
    "p7": "The world's 100 largest banks", "p8": "Euro against the dollar", "p9": "The Treasury yield curve",
    "p10": "Americans' savings rate",
}


def main():
    results = {r["item_id"]: r for r in json.loads((ROOT / "results" / "batch_eval_full.json").read_text())["results"]}
    samples = []
    for i in range(1, 11):
        sid = f"p{i}"
        folder = ROOT / "input" / sid
        txt = next(folder.glob("*.txt"))
        png = next(folder.glob("*.png"), None)
        r = results.get(sid, {})
        samples.append({
            "id": sid,
            "title": TITLES[sid],
            "text": txt.read_text(encoding="utf-8").strip(),
            "chart": f"input/{sid}/{png.name}" if png else None,
            "caption": r.get("caption"),
        })
    out = ROOT / "web" / "samples.json"
    out.write_text(json.dumps(samples, ensure_ascii=False, indent=1))
    print(f"{len(samples)} samples -> {out}")


if __name__ == "__main__":
    main()
