# Podcast-360 Doc-to-LoRA pipeline

This runbook is engineered for a corpus of ~360 podcast episode transcripts.

## Why this pipeline fits this repo
- `train.py` expects tokenized samples with `context + prompts (+responses/logprobs)` and supports packed training.
- `data/self_generate_qa.py` can generate self-supervised QA/logprob tuples and save them as parquet under `data/raw_datasets/self_gen/...`.
- `src/ctx_to_lora/data/processing.py` already loads parquet patterns from `train_ds_names` when entries end with `.parquet`.

## 1) Build compact dataset from transcripts
Input format supports both:
- flat field rows: `{episode_id?, title?, transcript}`
- notes-like segment rows: `{episode_id?, title?, segments:[{speaker,text,...}]}`

```bash
uv run python data/build_podcast360_compact.py \
  --input_jsonl data/raw_datasets/podcast360/transcripts.jsonl \
  --out_dir data/raw_datasets/podcast360_compact
```

By default, timestamp tokens like `00:12:33` are stripped during normalization.
Use `--keep_timestamps` if those markers matter for your downstream tasks.

This creates:
- `data/raw_datasets/podcast360_compact/train/ds.parquet`
- `data/raw_datasets/podcast360_compact/validation/ds.parquet`
- `data/raw_datasets/podcast360_compact/test/ds.parquet`

## 2) Generate self-supervised QA/logprob training data

```bash
uv run python data/self_generate_qa.py \
  --vllm_model google/gemma-2-2b-it \
  --glob_pattern "data/raw_datasets/podcast360_compact/*/ds.parquet" \
  --split train \
  --closed_qa_prob 1.0 \
  --temp 0.0 \
  --do_truncate
```

Note: `--glob_pattern` mode writes output with split duplicated in path (as currently implemented), e.g.:
`.../podcast360_compact/train/train/*.parquet`.

## 3) Train Doc-to-LoRA pilot

```bash
uv run accelerate launch --config_file accelerate_config.yaml \
  --num_processes=1 train.py \
  configs/podcast_exp/podcast360_gemma2_2b_l2l.yaml
```

Scale `--num_processes` to available GPUs.

## 4) Evaluate quality/latency baselines
- Full context baseline: `run_eval.py --model_name_or_path ... --add_ctx_to_input`
- Base no context baseline: `run_eval.py --model_name_or_path ... --remove_context`
- Trained D2L checkpoint: `run_eval.py --checkpoint_path ...`

Record:
- EM/F1 by question type,
- generation latency per query,
- VRAM peaks.

## 5) Decision gate
Proceed only if:
- D2L no-context quality is close to full-context baseline on held-out episodes,
- repeated-query latency improves materially,
- hallucinations do not spike on negative controls.
