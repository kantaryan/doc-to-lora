import argparse
import json
import random
import re
from pathlib import Path

from datasets import Dataset


_TS_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_SPACES_RE = re.compile(r"[ \t]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")


def load_jsonl(path: str):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_transcript(raw: str, strip_timestamps: bool = True) -> str:
    txt = raw.replace("\r\n", "\n").replace("\r", "\n").strip()
    if strip_timestamps:
        txt = _TS_RE.sub("", txt)
    txt = _SPACES_RE.sub(" ", txt)
    txt = re.sub(r"\n +", "\n", txt)
    txt = _MULTI_NL_RE.sub("\n\n", txt)
    return txt.strip()


def extract_transcript(row: dict) -> str:
    # Preferred flat fields
    for k in ["transcript", "notes_transcript", "content", "episode_transcript"]:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # Segment-style payload from notes exports
    segments = row.get("segments") or row.get("transcript_segments") or []
    if isinstance(segments, list) and segments:
        lines = []
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            speaker = (seg.get("speaker") or seg.get("speaker_name") or "").strip()
            text = (seg.get("text") or seg.get("content") or "").strip()
            if not text:
                continue
            if speaker:
                lines.append(f"{speaker}: {text}")
            else:
                lines.append(text)
        if lines:
            return "\n".join(lines)

    return ""


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


def build_samples(rows, strip_timestamps: bool = True):
    samples = []
    for row in rows:
        transcript_raw = extract_transcript(row)
        transcript = normalize_transcript(
            transcript_raw, strip_timestamps=strip_timestamps
        )
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
    parser.add_argument(
        "--keep_timestamps",
        action="store_true",
        help="Keep inline timestamps (default strips HH:MM[:SS] patterns)",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    rows = load_jsonl(args.input_jsonl)
    samples = build_samples(rows, strip_timestamps=not args.keep_timestamps)
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
