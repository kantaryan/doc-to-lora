# Pilot Runbook: Small-Text Trial

## 0) One-time fork bootstrap (already done)
Source fork target: `kantaryan/doc-to-lora`

## 1) Local bootstrap
```bash
git clone https://github.com/kantaryan/doc-to-lora.git
cd doc-to-lora
uv sync
uv run huggingface-cli login
```

## 2) Prepare pilot dataset
```bash
mkdir -p data/pilot_small_text/docs
# Add your text files into data/pilot_small_text/docs
# Add QA file: data/pilot_small_text/qa_eval.jsonl
```

## 3) Baseline runs
```bash
# Keep command templates; adjust to repository CLI/config style
python run_eval.py --config configs/eval/base_no_context.yaml
python run_eval.py --config configs/eval/full_context.yaml
python run_eval.py --config configs/eval/rag.yaml
```

## 4) Train pilot hypernetwork
```bash
python train.py --config configs/train/pilot_small_text.yaml
```

## 5) Evaluate Doc-to-LoRA mode
```bash
python run_eval.py --config configs/eval/doc_to_lora_pilot.yaml
```

## 6) Export report artifacts
```bash
mkdir -p runs/pilot_small_text
# Save metrics, latency, and errors in this folder
```

## 7) Decision gate
Proceed only if all are true:
- EM/F1 close to full-context baseline (<=10% absolute gap)
- repeated-query latency improvement is measurable
- no major hallucination increase on negative controls
