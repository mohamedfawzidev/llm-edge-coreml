# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Academic research paper: *"Optimizing LLM Inference on Edge Devices: A Comparative Study of Quantization Techniques using CoreML"*. Benchmarks 2 LLMs × 3 quantization levels (FP16, INT8, INT4) on a MacBook Pro M1 Pro, measuring latency, model size, memory, and MMLU accuracy.

See `docs/Experiment_Guide.md` for the full step-by-step protocol.

## Models

| Model | Source | FP16 acquisition |
|---|---|---|
| Phi-4 Mini Instruct (3.8B) | `AlonBBar/phi4mini-task-assistant-coreml` | Direct download (`phi4mini-float16.mlpackage`) |
| Mistral 7B Instruct v0.3 | `apple/mistral-coreml` | Direct download (`StatefulMistral7BInstructFP16.mlpackage`) |

Both models are acquired as pre-converted FP16 CoreML packages — manual PyTorch-to-CoreML conversion fails for these architectures due to unsupported RoPE operations in standard coremltools tracing. Mistral 7B uses a **stateful KV-cache** architecture (Apple WWDC 2024) — `cto.linear_quantize_weights` is compatible with it.

## Environment

```bash
# Python 3.12 required — coremltools 8.3 has no wheel for Python 3.13+
python3.12 -m venv coreml-env
source coreml-env/bin/activate
pip install coremltools==8.3 transformers==4.44.0 torch torchvision huggingface_hub datasets numpy
```

Neither model requires a HuggingFace login — both are publicly accessible.

## Directory Layout

```
models_fp16/          # FP16 CoreML .mlpackage files (gitignored)
models_int8/          # INT8 quantized .mlpackage files (gitignored)
models_int4/          # INT4 quantized .mlpackage files (gitignored)
results/              # Benchmark output JSON files (tracked)
docs/                 # Paper draft and experiment guide (tracked)
```

## Scripts

**Downloads (Phases 1–2):**
- `download_phi4.py` — downloads `phi4mini-float16.mlpackage` → `models_fp16/phi4-mini-fp16.mlpackage`
- `download_mistral.py` — downloads `StatefulMistral7BInstructFP16.mlpackage` → `models_fp16/mistral-7b-fp16.mlpackage`

**Quantization — one script per model per level:**
- `quantize_phi4_int8.py` / `quantize_phi4_int4.py`
- `quantize_mistral_int8.py` / `quantize_mistral_int4.py`

**Evaluation (Phase 4):**
- `sample_mmlu.py` — samples 200 MMLU questions → `mmlu_200.json`
- `evaluate_mmlu.py` — runs all 6 model-quantization combinations → `results/mmlu_accuracy.json`

## Quantization Pattern

INT8 uses `per_channel` granularity; INT4 uses `per_block` with `block_size=32` (AWQ-style):

```python
# INT8
cto.OpLinearQuantizerConfig(mode='linear_symmetric', dtype='int8', granularity='per_channel')
# INT4
cto.OpLinearQuantizerConfig(mode='linear_symmetric', dtype='int4', granularity='per_block', block_size=32)
```

## Benchmarking

Latency and memory: Swift macOS command-line app (`LLMBenchmark`) built in Xcode. Loads each `.mlpackage` with `config.computeUnits = .all`. Memory measured via Xcode Instruments (Allocations template).

MMLU accuracy: `evaluate_mmlu.py` loads each `.mlpackage` via `ct.models.MLModel()` and scores 200 4-choice questions. Results written to `results/mmlu_accuracy.json`.
