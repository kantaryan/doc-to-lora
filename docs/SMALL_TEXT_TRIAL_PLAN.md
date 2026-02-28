# Doc-to-LoRA Small-Text Pilot Plan (Senior Research Workflow)

## Goal
Run a reproducible pilot that tests whether Doc-to-LoRA can internalize a small custom text bundle into adapters that answer correctly **without passing the original document at query time**.

## Success Criteria
- Accuracy: Doc-to-LoRA within 10% EM/F1 of full-context baseline on factual QA.
- Efficiency: adapter generation + no-context inference lower average latency for repeated Q/A than long-context baseline.
- Stability: no major hallucination increase on negative-control questions.

## Environment Requirements

### Hardware
- Minimum: 1x NVIDIA GPU with 24 GB VRAM, 32 GB RAM, 40 GB disk.
- Preferred: 48+ GB VRAM for faster experiments and larger batch.

### Software
- Linux (Ubuntu 22.04+)
- Python 3.11
- CUDA-compatible PyTorch
- `uv` (project package/runtime manager)
- Hugging Face account + token (for model/checkpoint download)

### Project setup
```bash
uv sync
uv run huggingface-cli login
```

---

## Dataset Design (small but diagnostic)

Create `data/pilot_small_text/`:
- `docs/` with 3-5 documents (total 5k-20k tokens)
- `qa_eval.jsonl` with 40-80 labeled questions

Question mix:
- 50% direct fact retrieval
- 30% compositional/multi-hop in same document
- 20% hard negatives (plausible but false)

Schema per QA row:
```json
{"id":"q1","doc_id":"doc_a","question":"...","answer":"...","evidence_span":"...","type":"fact|multihop|negative"}
```

---

## Experiment Matrix

## Baselines
1. **Base-NoContext**: base model only.
2. **Full-Context**: question + full relevant document in prompt.
3. **RAG**: retrieve top-k chunks + question.
4. **Doc-to-LoRA**: generated adapter + question (no document text at query time).

## Ablations (pilot-size)
- LoRA rank: 4, 8
- Chunk size: 256, 512
- Chunk overlap: 32, 64

---

## Step-by-Step Execution

1. Prepare corpus and QA labels in `data/pilot_small_text/`.
2. Run baseline inference scripts and store metrics in `runs/pilot_small_text/baselines/`.
3. Generate teacher targets (full-context outputs/logits) for training tuples.
4. Train hypernetwork with small config (short schedule, strict checkpointing).
5. Generate adapters per document and run no-context QA.
6. Evaluate EM/F1, latency, VRAM; compare against baselines.
7. Produce report with failure taxonomy and go/no-go recommendation.

---

## Suggested Config (first pass)
- LoRA rank: 8
- Epochs: 3
- LR: 1e-4
- Effective batch size: 8 (via grad accumulation)
- Max input length: 1024 (chunked)
- Seed set: [13, 42] for variance check

---

## Operational Notes
- Keep every run reproducible: commit config, seed, model hash, checkpoint path.
- Store prompt templates and scoring scripts under version control.
- Record both aggregate and per-question-type metrics.

---

## Risks and Mitigations
- Overfitting small corpus -> use held-out QA and negatives.
- Under-capacity adapter -> test higher rank or chunk composition.
- Benchmark leakage -> separate train docs and eval-only docs where possible.
- Cost creep -> cap pilot to fixed GPU-hours and fixed matrix.

---

## Deliverables
- `runs/pilot_small_text/metrics_summary.json`
- `runs/pilot_small_text/latency_profile.csv`
- `runs/pilot_small_text/error_analysis.md`
- final recommendation: proceed / iterate / stop
