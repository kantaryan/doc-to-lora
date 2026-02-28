# Engineering Plan: Doc-to-LoRA on 360 Podcast Transcripts

## Objective
Train and evaluate a practical Doc-to-LoRA setup that internalizes podcast episode knowledge, so downstream QA can run without repeatedly injecting full transcripts in context.

## Codebase-grounded assumptions
This plan is based on current repository implementation details and the provided sample transcript note format (speaker turns + occasional timestamps):
- training entrypoint: `train.py`
- dataset loading/tokenization: `src/ctx_to_lora/data/processing.py`
- preprocessing behavior: `src/ctx_to_lora/data/preprocessing_fn.py`
- self-supervised data generation: `data/self_generate_qa.py`
- argument/config schema: `src/ctx_to_lora/configs.py`

## Key implementation constraints discovered
1. `get_ds_kwargs()` resolves `.parquet` patterns under `data/raw_datasets/`.
2. `self_generate_qa.py` requires `prompts` per context for generation.
3. Compact/self-gen datasets bypass heavy preprocessing and are expected to already have `context/prompts/...` structure.
4. HyperLoRA path in `train.py` enforces sequence packing (`use_sequence_packing=True`).

## End-to-end plan

## Phase A — Data construction from transcripts
Input: 360 episodes as JSONL. Supported fields now include:
- flat transcript: `transcript` / `notes_transcript` / `content` / `episode_transcript`
- segment arrays: `segments` / `transcript_segments` with `speaker` + `text`

Steps:
1. Build compact dataset with one row per episode:
   - `context`: normalized transcript text
   - `prompts`: 6+ episode-focused QA prompts
2. Normalize transcript artifacts from notes exports:
   - strip `HH:MM[:SS]` timestamps by default,
   - collapse noisy whitespace/newline bursts,
   - preserve speaker prefixes when available.
3. Split into train/validation/test (84/8/8 default).
4. Persist as parquet in `data/raw_datasets/podcast360_compact/{split}/ds.parquet`.

Implemented helper:
- `data/build_podcast360_compact.py`

## Phase B — Self-supervised QA/logprob generation
Use `data/self_generate_qa.py` in `--glob_pattern` mode to generate training tuples/logprobs from transcript contexts.

Why this matters:
- It aligns with existing D2L training data shape used by this repo.
- Avoids manual labeling for hundreds of episodes.

## Phase C — HyperLoRA training
Use config:
- `configs/podcast_exp/podcast360_gemma2_2b_l2l.yaml`

Core choices:
- Base model: `google/gemma-2-2b-it`
- LoRA rank: 8
- Target module: `down_proj`
- KL loss + per-context averaging enabled
- packed training lengths: 4096/4096
- initial pilot budget: 12k steps

## Phase D — Evaluation protocol
Evaluate three regimes on held-out episodes:
1. base model without context,
2. base model with full transcript context,
3. D2L internalized adapter (no raw transcript at query time).

Report:
- EM/F1 by question type,
- latency/query and memory usage,
- qualitative error taxonomy (hallucination, omission, temporal confusion, speaker attribution errors).

## Resource requirements (pilot)
- 1x 24GB GPU minimum (slower), 48GB preferred.
- 32GB RAM, ~60GB free disk.
- Python 3.11 + `uv` environment from repo.

## Risks specific to podcast transcripts
1. Transcript noise (ASR errors) can poison factual internalization.
2. Episodes with multiple speakers increase attribution mistakes.
3. Long episodes may require stronger chunking and rank budget.
4. Prompt quality strongly affects generated QA supervision quality.

Mitigations:
- normalize transcripts (timestamps/noise cleanup),
- include speaker-aware prompts,
- test rank/chunk ablations,
- add negative-control prompts in eval set.

## Go / No-Go criteria
Go if all are true:
- D2L within 10% EM/F1 of full-context baseline,
- repeated-query latency is materially better than full-context mode,
- no substantial increase in hallucination on held-out episodes.

Else iterate:
- adjust prompt set, rank, chunk lengths,
- increase QA generation diversity,
- run second ablation cycle before scale-up.
