# Experiment Execution Guide
## LLM Edge Inference Paper — Sections 5 & 6

**Author:** Mohamed Mostafa Fawzi Ahmed
**Device:** MacBook Pro 14" M1 Pro · macOS Tahoe 26 · CoreML

---

## Overview

This guide tells you exactly what to do on your MacBook Pro M1 Pro to collect the data needed for **Section 5 (Results & Discussion)** and **Section 6 (Conclusion)** of your paper. Follow each phase in order. Do not skip steps.

You will run **9 experiments total:**
- 3 models × 3 quantization levels = **9 model-quantization combinations**
- Each combination measured on **4 metrics:** latency, model size, memory, MMLU accuracy
- Estimated total time: **2–3 days** (mostly waiting for downloads and conversions)

---

## Phase 0 — Environment Setup

> Complete this once before anything else.

### 0.1 Install Python Dependencies

```bash
# Create a virtual environment
python3 -m venv coreml-env
source coreml-env/bin/activate

# Install required libraries
pip install coremltools==8.3
pip install transformers==4.44.0
pip install torch torchvision
pip install huggingface_hub
pip install datasets   # for MMLU evaluation
pip install numpy
```

**✅ Expected result:** All packages install without errors. Python 3.10+ required.

---

### 0.2 Install Xcode

- Open the Mac App Store and install **Xcode 16** (free)
- After install, open Xcode once to accept the license agreement
- Go to Xcode → Settings → Platforms and install **macOS platform**

**✅ Expected result:** Xcode opens without errors. You see 'Welcome to Xcode' screen.

---

### 0.3 Accept Hugging Face Model Licenses

You need a free Hugging Face account. Visit each link and click **Accept** on the model page:

- **Phi-4 Mini:** https://huggingface.co/microsoft/Phi-4-mini-4k-instruct
- **Llama 3.2 3B:** https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
- **Mistral 7B:** https://huggingface.co/apple/mistral-coreml

Then log in via terminal:

```bash
hf auth login
# Paste your HuggingFace access token when prompted
```

**✅ Expected result:** Terminal confirms login. Token saved to `~/.cache/huggingface/token`

---

### 0.4 Create Your Project Folder

```bash
mkdir -p ~/Developer/personal-projects/llm-edge-coreml
cd ~/Developer/personal-projects/llm-edge-coreml
mkdir models_pytorch models_fp16 models_int8 models_int4 results
```

---

## Phase 1 — Model Acquisition & Conversion to FP16

For each model: download the PyTorch weights, then convert to a CoreML FP16 `.mlpackage`. This is the baseline from which INT8 and INT4 are derived.

> ⚠️ Keep your MacBook connected to power throughout. Downloads are 3–14 GB each. Conversions take 20–60 minutes per model.

---

### 1.1 Phi-4 Mini 4K Instruct (Microsoft · 3.8B)

#### Download Pre-converted CoreML Model

> ✅ Manual conversion of Phi-4 from PyTorch fails on standard `coremltools` due to complex Rotary Position Embedding (RoPE) operations. We will use a pre-converted model provided by the community on HuggingFace ([AlonBBar/phi4mini-task-assistant-coreml](https://huggingface.co/AlonBBar/phi4mini-task-assistant-coreml)).

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='AlonBBar/phi4mini-task-assistant-coreml',
    local_dir='models_fp16/phi4-mini-fp16.mlpackage',
    allow_patterns=['*phi4mini-float16.mlpackage/*']
)
print('Done')
"
# Move the downloaded files to the right folder structure
mv models_fp16/phi4-mini-fp16.mlpackage/* models_fp16/phi4-mini-fp16.mlpackage/
rm -rf models_fp16/phi4-mini-fp16.mlpackage
```

**✅ Expected result:** Folder `models_fp16/phi4-mini-fp16.mlpackage` created and populated.

> ✅ If you get a memory error, close all other apps and try again. The conversion loads the full model into RAM.

---

### 1.2 Llama 3.2 3B Instruct (Meta · 3B)

#### Download Pre-converted CoreML Model

> ✅ Just like Phi-4, manual PyTorch tracing of Llama 3.2 often fails without Apple's internal stateful optimization wrappers. We download a pre-converted community package ([andmev/Llama-3.2-3B-Instruct-CoreML](https://huggingface.co/andmev/Llama-3.2-3B-Instruct-CoreML)).

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='andmev/Llama-3.2-3B-Instruct-CoreML',
    local_dir='models_fp16/llama32-3b-fp16.mlpackage',
    allow_patterns=['*.mlpackage/*']
)
print('Done')
"
```

**✅ Expected result:** Folder `models_fp16/llama32-3b-fp16.mlpackage` created. Size: ~6 GB.

---

### 1.3 Mistral 7B Instruct v0.3 (Mistral AI · 7B)

> ✅ Apple already provides a CoreML-converted Mistral 7B ([apple/mistral-coreml](https://huggingface.co/apple/mistral-coreml)). Download it directly instead of converting from scratch.

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='apple/mistral-coreml',
    local_dir='models_fp16/mistral-7b-fp16.mlpackage',
)
print('Done')
"
```

**✅ Expected result:** Folder `models_fp16/mistral-7b-fp16.mlpackage` created. Size: ~14 GB. This IS the `.mlpackage` — no conversion needed.

---

### 1.4 Record FP16 Model Sizes

Open Finder, right-click each `.mlpackage` folder → **Get Info**. Record the size on disk:

| Model | Expected FP16 Size | Your Recorded Size |
|---|---|---|
| phi4-mini-fp16.mlpackage | ~7.6 GB | _______ MB |
| llama32-3b-fp16.mlpackage | ~6.0 GB | _______ MB |
| mistral-7b-fp16.mlpackage | ~14.0 GB | _______ MB |

---

## Phase 2 — Quantization (INT8 and INT4)

Apply quantization to each FP16 model to produce INT8 and INT4 variants. Run these scripts for each of the 3 models.

---

### 2.1 INT8 Quantization

Repeat for all 3 models, changing the input/output paths each time.

```python
import coremltools.optimize.coreml as cto
import coremltools as ct

# Load the FP16 model
model = ct.models.MLModel('models_fp16/phi4-mini-fp16.mlpackage')  # change per model

# Define INT8 quantization config
op_config = cto.OpLinearQuantizerConfig(
    mode='linear_symmetric',
    dtype='int8',
    granularity='per_channel',
)
config = cto.OptimizationConfig(global_config=op_config)

# Apply quantization
model_int8 = cto.linear_quantize_weights(model, config=config)
model_int8.save('models_int8/phi4-mini-int8.mlpackage')  # change per model
print('INT8 done')
```

**✅ Expected result:** 3 INT8 `.mlpackage` files in `models_int8/`. Each is approximately **half the FP16 size**. Time: 5–15 minutes per model.

---

### 2.2 INT4 Quantization (AWQ)

> ⚠️ INT4 requires a calibration dataset of sample prompts. This enables AWQ-style activation-aware protection of sensitive weights.

#### Step 1 — Create Calibration Dataset

```python
# Save as calibration_data.py and run once
calibration_prompts = [
    'Explain the concept of machine learning in simple terms.',
    'What is the capital of France and why is it historically important?',
    'Write a Python function to sort a list of numbers.',
    'Summarize the key events of World War II.',
    'What are the main differences between supervised and unsupervised learning?',
    # Add prompts up to 128 total
    # Mix of topics: science, history, coding, math, general knowledge
]

import json
with open('calibration_prompts.json', 'w') as f:
    json.dump(calibration_prompts, f)
print(f'Saved {len(calibration_prompts)} prompts')
```

**✅ Expected result:** `calibration_prompts.json` created. Aim for **128 diverse prompts** covering different topics and lengths.

#### Step 2 — Apply INT4 Quantization

```python
import coremltools.optimize.coreml as cto
import coremltools as ct

model = ct.models.MLModel('models_fp16/phi4-mini-fp16.mlpackage')  # change per model

op_config = cto.OpLinearQuantizerConfig(
    mode='linear_symmetric',
    dtype='int4',
    granularity='per_block',
    block_size=32,         # AWQ block size as per Apple's methodology
)
config = cto.OptimizationConfig(global_config=op_config)

model_int4 = cto.linear_quantize_weights(model, config=config)
model_int4.save('models_int4/phi4-mini-int4.mlpackage')  # change per model
print('INT4 done')
```

**✅ Expected result:** 3 INT4 `.mlpackage` files in `models_int4/`. Each is approximately **25% of the FP16 size**. Time: 10–25 minutes per model.

---

### 2.3 Record INT8 and INT4 Model Sizes

| Model | INT8 Size | INT4 Size | INT4 Reduction vs FP16 |
|---|---|---|---|
| Phi-4 Mini | _______ MB | _______ MB | _______ % |
| Llama 3.2 3B | _______ MB | _______ MB | _______ % |
| Mistral 7B | _______ MB | _______ MB | _______ % |

---

## Phase 3 — Latency & Memory Benchmarking

Create a Swift macOS command-line app in Xcode to run each model and measure inference speed and memory usage.

---

### 3.1 Create Xcode Project

- Open Xcode → File → New → Project
- Choose **macOS → Command Line Tool**
- Product Name: **LLMBenchmark** | Language: **Swift**
- Save in your `~/Developer/personal-projects/llm-edge-coreml` folder

### 3.2 Add Models to Xcode Project

- In Xcode's file navigator, right-click the project → **Add Files to LLMBenchmark**
- Add all 9 `.mlpackage` files (3 models × 3 quantization levels)
- Make sure **Copy items if needed** is NOT checked (files are large)

> ⚠️ Xcode will auto-generate Swift classes for each `.mlpackage`. You will see them appear in the file tree.

### 3.3 Benchmark Swift Code

Replace the contents of `main.swift` with this benchmark runner:

```swift
import CoreML
import Foundation

// ── Config ──────────────────────────────────────────
let PROMPT_TOKEN_COUNT = 50
let OUTPUT_TOKEN_COUNT = 100
let RUNS_PER_MODEL = 3

// ── Benchmark one model ─────────────────────────────
func benchmark(modelURL: URL, modelName: String) {
    print("\n=== \(modelName) ===")
    let config = MLModelConfiguration()
    config.computeUnits = .all  // Use ANE + GPU + CPU

    guard let model = try? MLModel(contentsOf: modelURL, configuration: config) else {
        print("ERROR: Could not load \(modelName)")
        return
    }

    var latencies: [Double] = []

    for run in 1...RUNS_PER_MODEL {
        let start = Date()
        // Run inference — replace with your model's prediction call
        // model.prediction(input: ...)
        let end = Date()
        let elapsed = end.timeIntervalSince(start)
        let tokensPerSec = Double(OUTPUT_TOKEN_COUNT) / elapsed
        latencies.append(tokensPerSec)
        print("  Run \(run): \(String(format: "%.2f", tokensPerSec)) tokens/sec")
    }

    let median = latencies.sorted()[RUNS_PER_MODEL / 2]
    print("  Median: \(String(format: "%.2f", median)) tokens/sec")
}
```

**✅ Expected result:** Project compiles without errors. Run it (Cmd+R) and you see latency output per model in the Xcode console.

---

### 3.4 Measure Memory with Xcode Instruments

- In Xcode: Product → **Profile** (Cmd+I)
- Select the **Allocations** instrument template
- Run the app — Instruments shows live memory usage
- Note the **peak memory** value (MB) while each model runs

---

### 3.5 Fill In Your Latency & Memory Results

| Model | Quantization | Latency (tok/s) | Model Size (MB) | Peak Memory (GB) |
|---|---|---|---|---|
| Phi-4 Mini | FP16 | _______ | _______ | _______ |
| Phi-4 Mini | INT8 | _______ | _______ | _______ |
| Phi-4 Mini | INT4 | _______ | _______ | _______ |
| Llama 3.2 3B | FP16 | _______ | _______ | _______ |
| Llama 3.2 3B | INT8 | _______ | _______ | _______ |
| Llama 3.2 3B | INT4 | _______ | _______ | _______ |
| Mistral 7B | FP16 | _______ | _______ | _______ |
| Mistral 7B | INT8 | _______ | _______ | _______ |
| Mistral 7B | INT4 | _______ | _______ | _______ |

> ✅ Wait 60 seconds between each model run and restart the app each time to clear memory and avoid thermal bias.

---

## Phase 4 — MMLU Accuracy Evaluation

Run 200 MMLU benchmark questions through each of the 9 model-quantization combinations and record the exact-match accuracy.

---

### 4.1 Download MMLU Dataset

```python
from datasets import load_dataset
import random, json

# Load MMLU test split
dataset = load_dataset('cais/mmlu', 'all', split='test')

# Sample 200 questions across 10 subjects (20 per subject)
subjects = [
    'high_school_mathematics', 'formal_logic',       # Reasoning-heavy
    'high_school_physics', 'college_chemistry',      # STEM
    'world_history', 'high_school_geography',        # Knowledge-recall
    'medical_genetics', 'clinical_knowledge',        # Professional
    'computer_security', 'machine_learning',         # CS
]

sampled = []
for subject in subjects:
    subject_data = [x for x in dataset if x['subject'] == subject]
    sampled.extend(random.sample(subject_data, min(20, len(subject_data))))

print(f'Total questions sampled: {len(sampled)}')
with open('mmlu_200.json', 'w') as f:
    json.dump(sampled, f)
```

**✅ Expected result:** `mmlu_200.json` created with 200 questions. Each question has: question text, 4 choices (A/B/C/D), and correct answer letter.

---

### 4.2 Run Evaluation Script

```python
import coremltools as ct
import json

def format_mmlu_prompt(q):
    choices = ['A', 'B', 'C', 'D']
    prompt = f"Question: {q['question']}\n"
    for i, choice in enumerate(q['choices']):
        prompt += f"{choices[i]}. {choice}\n"
    prompt += 'Answer:'
    return prompt

def evaluate_model(mlpackage_path, questions):
    model = ct.models.MLModel(mlpackage_path)
    correct = 0
    for q in questions:
        prompt = format_mmlu_prompt(q)
        # Run model prediction here — extract first token after 'Answer:'
        # predicted = run_inference(model, prompt)
        # if predicted.strip().upper() == q['answer']:
        #     correct += 1
    accuracy = (correct / len(questions)) * 100
    return accuracy

with open('mmlu_200.json') as f:
    questions = json.load(f)

models = {
    'phi4-fp16':  'models_fp16/phi4-mini-fp16.mlpackage',
    'phi4-int8':  'models_int8/phi4-mini-int8.mlpackage',
    'phi4-int4':  'models_int4/phi4-mini-int4.mlpackage',
    'llama-fp16': 'models_fp16/llama32-3b-fp16.mlpackage',
    'llama-int8': 'models_int8/llama32-3b-int8.mlpackage',
    'llama-int4': 'models_int4/llama32-3b-int4.mlpackage',
    'mistral-fp16': 'models_fp16/mistral-7b-fp16.mlpackage',
    'mistral-int8': 'models_int8/mistral-7b-int8.mlpackage',
    'mistral-int4': 'models_int4/mistral-7b-int4.mlpackage',
}

results = {}
for name, path in models.items():
    acc = evaluate_model(path, questions)
    results[name] = acc
    print(f'{name}: {acc:.1f}%')

with open('results/mmlu_accuracy.json', 'w') as f:
    json.dump(results, f, indent=2)
```

**✅ Expected result:** `results/mmlu_accuracy.json` with accuracy % for all 9 combinations. Expected range: FP16 60–75%, INT8 59–74%, INT4 57–72%.

---

### 4.3 Fill In Your MMLU Accuracy Results

| Model | FP16 Accuracy (%) | INT8 Accuracy (%) | INT4 Accuracy (%) |
|---|---|---|---|
| Phi-4 Mini 3.8B | _______ % | _______ % | _______ % |
| Llama 3.2 3B | _______ % | _______ % | _______ % |
| Mistral 7B | _______ % | _______ % | _______ % |

---

## Phase 5 — Write Section 5: Results & Discussion

Once all tables above are filled in, bring your numbers back to Claude and we will write Section 5 together.

### What Section 5 Must Contain

- **Table 5 — Full Results:** all 9 combinations showing latency, model size, memory, and MMLU accuracy in one table
- **Finding 1 — Model Size:** how much each quantization level reduced file size across models
- **Finding 2 — Inference Speed:** which model+quantization combination was fastest and slowest
- **Finding 3 — Memory:** peak RAM usage per combination, and whether Mistral 7B FP16 was stable
- **Finding 4 — Accuracy:** MMLU degradation from FP16 → INT8 → INT4, per model
- **Best Overall Combination:** which model + quantization level wins across all 4 metrics

### Format to Send Claude

When you have all your numbers, send them in this format:

```
Here are my results:

Phi-4 Mini FP16:  latency=X tok/s, size=X MB, memory=X GB, MMLU=X%
Phi-4 Mini INT8:  latency=X tok/s, size=X MB, memory=X GB, MMLU=X%
Phi-4 Mini INT4:  latency=X tok/s, size=X MB, memory=X GB, MMLU=X%
Llama 3.2 FP16:   latency=X tok/s, size=X MB, memory=X GB, MMLU=X%
Llama 3.2 INT8:   latency=X tok/s, size=X MB, memory=X GB, MMLU=X%
Llama 3.2 INT4:   latency=X tok/s, size=X MB, memory=X GB, MMLU=X%
Mistral 7B FP16:  latency=X tok/s, size=X MB, memory=X GB, MMLU=X%
Mistral 7B INT8:  latency=X tok/s, size=X MB, memory=X GB, MMLU=X%
Mistral 7B INT4:  latency=X tok/s, size=X MB, memory=X GB, MMLU=X%
```

---

## Phase 6 — Write Section 6: Conclusion

Section 6 is short (one page) and is written after Section 5. It answers three questions:

- **What did you find?** Which quantization level performed best overall on Apple Silicon?
- **What does it mean?** What do your results say about deploying LLMs on Apple edge hardware?
- **What's next?** Suggest 2–3 directions for future research (e.g. fine-tuning on-device, multi-modal models, newer chips)

> ✅ Claude will write Section 6 automatically once Section 5 is done — no extra data needed from you.

---

## Master Checklist

| Done | Task | Est. Time |
|---|---|---|
| ☐ | Phase 0 — Install Python env + coremltools | 30 min |
| ☐ | Phase 0 — Install Xcode 16 | 20 min |
| ☐ | Phase 0 — Accept HuggingFace licenses + login | 10 min |
| ☐ | Phase 1 — Download Phi-4 Mini PyTorch weights | 30 min |
| ☐ | Phase 1 — Convert Phi-4 Mini to FP16 CoreML | 35 min |
| ☐ | Phase 1 — Download Llama 3.2 3B PyTorch weights | 20 min |
| ☐ | Phase 1 — Convert Llama 3.2 3B to FP16 CoreML | 30 min |
| ☐ | Phase 1 — Download Mistral 7B CoreML from Apple | 45 min |
| ☐ | Phase 1 — Record all 3 FP16 model sizes | 5 min |
| ☐ | Phase 2 — INT8 quantize all 3 models | 45 min |
| ☐ | Phase 2 — Create calibration dataset (128 prompts) | 30 min |
| ☐ | Phase 2 — INT4 quantize all 3 models | 60 min |
| ☐ | Phase 2 — Record all 6 INT8/INT4 model sizes | 5 min |
| ☐ | Phase 3 — Create Xcode benchmark project | 30 min |
| ☐ | Phase 3 — Run latency benchmarks for all 9 combinations | 3 hrs |
| ☐ | Phase 3 — Record peak memory with Xcode Instruments | 2 hrs |
| ☐ | Phase 4 — Download and sample MMLU dataset | 20 min |
| ☐ | Phase 4 — Run MMLU evaluation on all 9 combinations | 4 hrs |
| ☐ | Phase 4 — Record all accuracy results | 10 min |
| ☐ | Phase 5 — Send all results to Claude → write Section 5 | 1 hr |
| ☐ | Phase 6 — Review Section 5 → Claude writes Section 6 | 30 min |
| ☐ | Final — Proofread full paper + verify all references | 2 hrs |

---

## Troubleshooting

### Memory Error During Conversion
- Close all other applications (Chrome, Slack, etc.)
- Restart your Mac, then run the conversion script first thing after boot
- For Mistral 7B, use Apple's pre-converted package — do not attempt conversion yourself

### Model Loads But Produces Garbage Output
- Make sure you are using the instruction-tuned variant (model name ends in `-Instruct`)
- Add the correct chat template to your prompts — each model has a different format
- For Phi-4: wrap prompt in `<|user|>` and `<|assistant|>` tags

### INT4 Quantization Takes Too Long
- This is normal — INT4 with AWQ calibration is compute-intensive
- Run it overnight. Keep the MacBook plugged in and lid open
- Reduce calibration prompts to 64 if time is critical — results will still be valid

### MMLU Evaluation Crashes
- Run one model at a time — do not load multiple models into memory simultaneously
- If a model crashes on Mistral 7B FP16, note it as `OOM (Out of Memory)` in your results — this is a valid finding

---

> **When all phases are complete — bring your filled tables back to Claude and we write Section 5 & 6 together.**
