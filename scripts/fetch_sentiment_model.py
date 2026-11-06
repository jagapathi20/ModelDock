"""Download the main sentiment model from the Hugging Face Hub into a local directory.

    uv run python -m scripts.fetch_sentiment_model --out artifacts/sentiment-v1

Runners only ever load from a local directory (see worker/runners/base.py).
This script is the one place that talks to the Hub. In Stage 5 the same
directory gets packaged and uploaded to object storage as version 1.
"""

import argparse
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"


def fetch(model_id: str, out_dir: Path, revision: str | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision)
    tokenizer.save_pretrained(out_dir)
    model.save_pretrained(out_dir)  # writes model.safetensors + config.json
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument(
        "--revision",
        default=None,
        help="Hub commit hash to pin. Pinning makes the artifact reproducible.",
    )
    p.add_argument("--out", type=Path, default=Path("artifacts/sentiment-v1"))
    args = p.parse_args()

    out = fetch(args.model_id, args.out, args.revision)
    total = 0
    for f in sorted(out.iterdir()):
        size = f.stat().st_size
        total += size
        print(f"  {f.name:<28} {size / 1e6:8.2f} MB")
    print(f"saved {args.model_id} to {out} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
