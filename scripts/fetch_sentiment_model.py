"""Download the main sentiment model from the Hugging Face Hub into a local directory.

    uv run python -m scripts.fetch_sentiment_model --out artifacts/sentiment-v1

Runners only ever load from a local directory (see worker/runners/base.py).
This script is the one place that talks to the Hub. (If the model registry is
built, future work in C1.23, this directory becomes version 1 in object
storage; see P2.)

The download is pinned to a Hub commit (``DEFAULT_REVISION``). Without a pin,
``from_pretrained`` takes whatever ``main`` points to on the day you run it, so
two machines, or the same machine a month apart, could hold different weights
under the same "sentiment v1" name, and benchmark results and cached answers
(D10) would silently stop being comparable.
"""

import argparse
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
# Full 40-character commit hash: branch names move, and short hashes are not
# accepted by the Hub everywhere. This is the head of `main` as of 2026-11-21
# ("Change loaded mode to correct on that this model card refers to (#26)").
DEFAULT_REVISION = "714eb0fa89d2f80546fda750413ed43d93601a13"


def fetch(model_id: str, out_dir: Path, revision: str = DEFAULT_REVISION) -> Path:
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
        default=DEFAULT_REVISION,
        help="Hub commit hash to download (default: the pinned one). Pinning makes "
        "the artifact reproducible; change it only together with the model version.",
    )
    p.add_argument("--out", type=Path, default=Path("artifacts/sentiment-v1"))
    args = p.parse_args()

    out = fetch(args.model_id, args.out, args.revision)
    total = 0
    for f in sorted(out.iterdir()):
        size = f.stat().st_size
        total += size
        print(f"  {f.name:<28} {size / 1e6:8.2f} MB")
    print(f"saved {args.model_id}@{args.revision} to {out} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
