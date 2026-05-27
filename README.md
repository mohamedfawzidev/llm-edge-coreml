# LLM Edge Inference on Apple Silicon — CoreML Quantization Study

**Paper:** *Optimizing LLM Inference on Edge Devices: A Comparative Study of Quantization Techniques using CoreML*  
**Author:** Mohamed Mostafa Fawzi Ahmed — Cairo University, Faculty of Graduate Studies for Statistical Researches  
**Device:** MacBook Pro 14-inch (2021) · Apple M1 Pro · 16 GB unified memory · macOS Tahoe 26

---

## Overview

This repository contains all scripts and results for a systematic benchmarking study comparing FP16, INT8, and INT4 quantization applied to two open-source LLMs deployed via Apple CoreML on Apple Silicon. Metrics measured: inference latency (tok/s), model size on disk (MB), peak memory usage (GB), and MMLU accuracy (%).

---

## Key Results

| Model | Quantization | Disk Size (MB) | Peak Memory (GB) | Latency (tok/s) | MMLU (%) |
|---|---|---|---|---|---|
| Phi-4 Mini | FP16 | 7,673 | 16.26 | 3.93 | — |
| Phi-4 Mini | INT8 | 3,840 | 24.55 | 3.30 | — |
| Phi-4 Mini | INT4 | 2,159 | 24.60 | 3.02 | — |
| Mistral 7B | FP16 | 13,826 | 13.54 † | — | — |
| Mistral 7B | INT8 | 6,917 | 6.79 † | — | 51.1 |
| Mistral 7B | INT4 | 3,890 | 3.72 † | — | 50.6 |

† Load-time only — Mistral 7B inference is blocked on macOS (stateful KV-cache API is iOS-only).  
Phi-4 Mini MMLU not reported — CoreML Python API exposes single-token inference only, giving random-chance accuracy.  
See `docs/Key_Findings.md` for detailed explanations of both limitations.

---

## Setup

**Python 3.12 is required** — coremltools 8.3 has no wheel for Python 3.13+.

```bash
python3.12 -m venv coreml-env
source coreml-env/bin/activate
pip install coremltools==8.3 transformers==4.44.0 torch torchvision huggingface_hub datasets numpy
```

No Hugging Face login is required — both models are publicly accessible.

---

## Reproducing the Experiments

### Phase 1 — Phi-4 Mini

```bash
# Download FP16 baseline (~7.6 GB)
python3 download_phi4.py

# Quantize
python3 quantize_phi4_int8.py   # → models_int8/phi4-mini-int8.mlpackage (~3.8 GB)
python3 quantize_phi4_int4.py   # → models_int4/phi4-mini-int4.mlpackage (~2.2 GB)
```

### Phase 2 — Mistral 7B

```bash
# Download FP16 baseline (~13.8 GB)
python3 download_mistral.py

# Quantize
python3 quantize_mistral_int8.py   # → models_int8/mistral-7b-int8.mlpackage (~6.9 GB)
python3 quantize_mistral_int4.py   # → models_int4/mistral-7b-int4.mlpackage (~3.9 GB)
```

### Phase 3 — Latency & Memory Benchmarking

Open `LLMBenchmark/LLMBenchmark.xcodeproj` in Xcode. Set `MODEL_INDEX` in `main.swift` to select the model (0–5), then run. The app prints peak `phys_footprint` memory and median tokens/sec. Run once per model with a 60-second cooldown between runs.

### Phase 4 — MMLU Accuracy

```bash
# Sample 200 questions (run once)
python3 sample_mmlu.py

# Evaluate Mistral INT8 and INT4
python3 evaluate_mmlu.py
# Results → results/mmlu_accuracy.json
```

---

## Repository Structure

```
├── download_phi4.py            # Downloads Phi-4 Mini FP16 CoreML package
├── download_mistral.py         # Downloads Mistral 7B FP16 CoreML package
├── quantize_phi4_int8.py       # Quantizes Phi-4 Mini → INT8
├── quantize_phi4_int4.py       # Quantizes Phi-4 Mini → INT4
├── quantize_mistral_int8.py    # Quantizes Mistral 7B → INT8
├── quantize_mistral_int4.py    # Quantizes Mistral 7B → INT4
├── sample_mmlu.py              # Samples 200 MMLU questions → mmlu_200.json
├── evaluate_mmlu.py            # Runs MMLU evaluation on Mistral INT8/INT4
├── mmlu_200.json               # The exact 200-question test set used
├── LLMBenchmark/               # Swift macOS app — latency & memory benchmarking
│   └── LLMBenchmark/
│       └── main.swift          # Full benchmark source (phys_footprint memory + tok/s)
├── models_fp16/                # FP16 .mlpackage files (gitignored — download locally)
├── models_int8/                # INT8 .mlpackage files (gitignored — generate locally)
├── models_int4/                # INT4 .mlpackage files (gitignored — generate locally)
├── results/
│   └── mmlu_accuracy.json      # Recorded MMLU accuracy results
└── docs/
    ├── Key_Findings.md                     # Detailed findings including unexpected results
    ├── Experiment_Guide.md                 # Step-by-step reproduction protocol
    ├── Benchmark_Reproducibility.md        # Swift benchmark code and methodology
    └── Optimizing LLM Inference on Edge Devices_...md   # Full paper draft
```

---

## Notable Findings

**Quantization increases inference memory on CPU+GPU path.** When the Apple Neural Engine is not engaged, CoreML dequantizes weights to FP32 at runtime, holding both the compressed source weights and the FP32 compute buffer simultaneously. For Phi-4 Mini, this causes INT8 and INT4 to consume ~24.5 GB — 51% more than FP16 (16.3 GB) — despite having 50–72% smaller files on disk.

**Mistral 7B stateful inference is iOS-only.** The `MLModel.newState()` API required by Apple's stateful KV-cache architecture is explicitly unavailable on macOS. Inference latency for Mistral 7B cannot be measured on macOS without a separate non-stateful model conversion.

**INT4 accuracy degradation is minimal.** Mistral 7B INT8 (51.1%) and INT4 (50.6%) differ by only 0.5 percentage points on 200 MMLU questions, confirming negligible additional accuracy cost from INT4 vs INT8 at 7B scale.

---

## Citation

```bibtex
@article{fawzi2026llmedge,
  title   = {Optimizing LLM Inference on Edge Devices: A Comparative Study of
             Quantization Techniques using CoreML},
  author  = {Ahmed, Mohamed Mostafa Fawzi},
  year    = {2026},
  url     = {https://github.com/mohamedfawzidev/llm-edge-coreml}
}
```
