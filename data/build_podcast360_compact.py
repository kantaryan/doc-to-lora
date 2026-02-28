import argparse
import json
import os
import random
from pathlib import Path

from datasets import Dataset


def load_jsonl(path: str):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def mk_prompts(title: str):
    title = title or "this episode"
    return [
        f"Give a concise summary of {title}.",
        f"List the top 5 key insights from {title}.",
        f"What practical actions can a listener take based on {title}?",
        f"What claims or opinions are made in {title}, and what evidence is cited?",
        f"Extract notable names, companies, and tools mentioned in {title}.",
        f"What are the strongest disagreements or uncertainties discussed in {title}?",
    ]


def build_samples(rows):
    samples = []
    for row in rows:
        transcript = (row.get("transcript") or "").strip()
        if not transcript:
            continue
        title = (row.get("title") or row.get("episode_title") or "").strip()
        samples.append(
            {
                "episode_id": row.get("episode_id") or row.get("id") or "",
                "title": title,
                "context": transcript,
                "prompts": mk_prompts(title),
                # kept for compatibility with compact format loaders
                "responses": [],
            }
        )
    return samples


def save_split(base_out: Path, split: str, rows: list[dict]):
    split_dir = base_out / split
    split_dir.mkdir(parents=True, exist_ok=True)
    ds = Dataset.from_list(rows)
    out_file = split_dir / "ds.parquet"
    ds.to_parquet(str(out_file))
    print(f"Saved {len(rows)} rows to {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_jsonl",
        required=True,
        help="Path to JSONL with at least {title, transcript} columns (360 episodes)",
    )
    parser.add_argument(
        "--out_dir",
        default="data/raw_datasets/podcast360_compact",
        help="Output directory for compact parquet splits",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.84)
    parser.add_argument("--val_ratio", type=float, default=0.08)
    args = parser.parse_args()

    random.seed(args.seed)

    rows = load_jsonl(args.input_jsonl)
    samples = build_samples(rows)
    random.shuffle(samples)

    n = len(samples)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    train_rows = samples[:n_train]
    val_rows = samples[n_train : n_train + n_val]
    test_rows = samples[n_train + n_val :]

    out_dir = Path(args.out_dir)
    save_split(out_dir, "train", train_rows)
    save_split(out_dir, "validation", val_rows)
    save_split(out_dir, "test", test_rows)


if __name__ == "__main__":
    main()
