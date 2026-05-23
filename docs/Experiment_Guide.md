# Experiment Execution Guide
## LLM Edge Inference Paper — Sections 5 & 6

**Author:** Mohamed Mostafa Fawzi Ahmed
**Device:** MacBook Pro 14" M1 Pro · macOS Tahoe 26 · CoreML

---

## Overview

You will produce **6 `.mlpackage` files** (2 models × 3 quantization levels), then benchmark all 6.

Each of Phases 1–2 fully completes one model before moving to the next.

---

## Phase 0 — Environment Setup

> Complete this once before anything else.

### 0.1 Install Python Dependencies

```bash
# Python 3.12 required — coremltools 8.3 has no wheel for Python 3.13+
python3.12 -m venv coreml-env
source coreml-env/bin/activate
pip install coremltools==8.3 transformers==4.44.0 torch torchvision huggingface_hub datasets numpy
```

**✅ Expected result:** All packages install without errors.

---

### 0.2 Install Xcode

- Open the Mac App Store and install **Xcode 16** (free)
- After install, open Xcode once to accept the license agreement
- Go to Xcode → Settings → Platforms and install **macOS platform**

---

## Phase 1 — Phi-4 Mini (3.8B)

> ⚠️ Keep MacBook on power. Make sure venv is active: `source coreml-env/bin/activate`

### 1.1 Download FP16

```bash
python3 download_phi4.py
```

**✅** `models_fp16/phi4-mini-fp16.mlpackage` — ~7.6 GB

---

### 1.2 Quantize to INT8

```bash
python3 quantize_phi4_int8.py
```

**✅** `models_int8/phi4-mini-int8.mlpackage` — ~3.8 GB · Time: 5–15 min

---

### 1.3 Quantize to INT4

```bash
python3 quantize_phi4_int4.py
```

**✅** `models_int4/phi4-mini-int4.mlpackage` — ~1.9 GB · Time: 10–25 min

---

### 1.4 Record Phi-4 Mini Sizes

| Quantization | Expected Size | Your Recorded Size |
|---|---|---|
| FP16 | ~7,600 MB | 7,673 MB |
| INT8 | ~3,800 MB | 3,840 MB |
| INT4 | ~1,900 MB | 2,159 MB |

---

## Phase 2 — Mistral 7B

### 2.1 Download FP16

```bash
python3 download_mistral.py
```

**✅** `models_fp16/mistral-7b-fp16.mlpackage` — ~14 GB

---

### 2.2 Quantize to INT8

```bash
python3 quantize_mistral_int8.py
```

**✅** `models_int8/mistral-7b-int8.mlpackage` — ~7 GB · Time: 5–15 min

---

### 2.3 Quantize to INT4

```bash
python3 quantize_mistral_int4.py
```

**✅** `models_int4/mistral-7b-int4.mlpackage` — ~3.5 GB · Time: 10–25 min

---

### 2.4 Record Mistral 7B Sizes

| Quantization | Expected Size | Your Recorded Size |
|---|---|---|
| FP16 | ~14,000 MB | _______ MB |
| INT8 | ~7,000 MB | _______ MB |
| INT4 | ~3,500 MB | _______ MB |

---

## Phase 3 — Latency & Memory Benchmarking

You now have all 6 `.mlpackage` files. Create a Swift app to measure inference speed and memory.

### 3.1 Create Xcode Project

- Open Xcode → File → New → Project
- Choose **macOS → Command Line Tool**
- Product Name: **LLMBenchmark** | Language: **Swift**
- Save in `~/Developer/personal-projects/llm-edge-coreml`

### 3.2 Add Models to Xcode Project

- Right-click the project in the file navigator → **Add Files to LLMBenchmark**
- Add all 6 `.mlpackage` files
- Make sure **Copy items if needed** is NOT checked (files are large)

### 3.3 Benchmark Swift Code

Replace `main.swift` with:

```swift
import CoreML
import Foundation

let OUTPUT_TOKEN_COUNT = 100
let RUNS_PER_MODEL = 3

func benchmark(modelURL: URL, modelName: String) {
    print("\n=== \(modelName) ===")
    let config = MLModelConfiguration()
    config.computeUnits = .all

    guard let model = try? MLModel(contentsOf: modelURL, configuration: config) else {
        print("ERROR: Could not load \(modelName)")
        return
    }

    var latencies: [Double] = []
    for run in 1...RUNS_PER_MODEL {
        let start = Date()
        // model.prediction(input: ...)  ← add your inference call here
        let elapsed = Date().timeIntervalSince(start)
        let tokensPerSec = Double(OUTPUT_TOKEN_COUNT) / elapsed
        latencies.append(tokensPerSec)
        print("  Run \(run): \(String(format: "%.2f", tokensPerSec)) tokens/sec")
    }
    let median = latencies.sorted()[RUNS_PER_MODEL / 2]
    print("  Median: \(String(format: "%.2f", median)) tokens/sec")
}
```

### 3.4 Measure Memory with Xcode Instruments

- Product → **Profile** (Cmd+I) → select **Allocations**
- Run the app and note the **peak memory** while each model runs

### 3.5 Fill In Latency & Memory Results

| Model | Quantization | Latency (tok/s) | Peak Memory (GB) |
|---|---|---|---|
| Phi-4 Mini | FP16 | _______ | _______ |
| Phi-4 Mini | INT8 | _______ | _______ |
| Phi-4 Mini | INT4 | _______ | _______ |
| Mistral 7B | FP16 | _______ | _______ |
| Mistral 7B | INT8 | _______ | _______ |
| Mistral 7B | INT4 | _______ | _______ |

> ✅ Restart the app and wait 60 seconds between each model run.

---

## Phase 4 — MMLU Accuracy Evaluation

### 4.1 Sample the Dataset

```bash
python3 sample_mmlu.py
```

**✅** `mmlu_200.json` created with 200 questions.

---

### 4.2 Run Evaluation

```bash
python3 evaluate_mmlu.py
```

**✅** `results/mmlu_accuracy.json` — Expected range: FP16 60–75%, INT8 59–74%, INT4 57–72%.

---

### 4.3 Fill In MMLU Accuracy Results

| Model | FP16 (%) | INT8 (%) | INT4 (%) |
|---|---|---|---|
| Phi-4 Mini | _______ | _______ | _______ |
| Mistral 7B | _______ | _______ | _______ |

---

## Phase 5 — Write Section 5 & 6

Once all tables are filled in, send Claude your results in this format:

```
Phi-4 Mini FP16:  latency=X tok/s, size=X MB, memory=X GB, MMLU=X%
Phi-4 Mini INT8:  latency=X tok/s, size=X MB, memory=X GB, MMLU=X%
Phi-4 Mini INT4:  latency=X tok/s, size=X MB, memory=X GB, MMLU=X%
Mistral 7B FP16:  latency=X tok/s, size=X MB, memory=X GB, MMLU=X%
Mistral 7B INT8:  latency=X tok/s, size=X MB, memory=X GB, MMLU=X%
Mistral 7B INT4:  latency=X tok/s, size=X MB, memory=X GB, MMLU=X%
```

Claude will write Section 5 (Results & Discussion) and Section 6 (Conclusion) from your numbers.

---

## Master Checklist

| Done | Task | Script | Est. Time |
|---|---|---|---|
| ☐ | Phase 0 — Python 3.12 env + packages | — | 30 min |
| ☐ | Phase 0 — Xcode 16 | — | 20 min |
| ☐ | Phase 1 — Download Phi-4 Mini FP16 | `download_phi4.py` | 30 min |
| ☐ | Phase 1 — Quantize Phi-4 Mini INT8 | `quantize_phi4_int8.py` | 15 min |
| ☐ | Phase 1 — Quantize Phi-4 Mini INT4 | `quantize_phi4_int4.py` | 25 min |
| ☐ | Phase 2 — Download Mistral 7B FP16 | `download_mistral.py` | 60 min |
| ☐ | Phase 2 — Quantize Mistral INT8 | `quantize_mistral_int8.py` | 15 min |
| ☐ | Phase 2 — Quantize Mistral INT4 | `quantize_mistral_int4.py` | 25 min |
| ☐ | Phase 3 — Latency benchmarks (all 6) | Xcode | 2 hrs |
| ☐ | Phase 3 — Memory benchmarks (all 6) | Xcode Instruments | 90 min |
| ☐ | Phase 4 — Sample MMLU | `sample_mmlu.py` | 20 min |
| ☐ | Phase 4 — Run MMLU evaluation (all 6) | `evaluate_mmlu.py` | 2.5 hrs |
| ☐ | Phase 5 — Send results to Claude → Sections 5 & 6 | — | 1.5 hrs |
| ☐ | Final — Proofread + verify references | — | 2 hrs |

---

## Troubleshooting

### Model Loads But Produces Garbage Output
- Use the instruction-tuned variant (name ends in `-Instruct`)
- For Phi-4 Mini: wrap prompts in `<|user|>` and `<|assistant|>` tags

### INT4 Quantization Takes Too Long
- Normal — run overnight with MacBook plugged in

### Mistral 7B FP16 Crashes (OOM)
- Note it as `OOM` in your results table — this is a valid finding for the paper

---

> **When all phases are complete — send your filled tables to Claude and we write Sections 5 & 6 together.**
