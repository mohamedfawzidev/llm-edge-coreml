import re

with open('docs/Experiment_Guide.md', 'r') as f:
    content = f.read()

# We need to change the guide to instruct the user to download pre-converted .mlpackage files
# directly from HuggingFace because manually tracing LLMs (like Phi-3 and Llama 3) to CoreML using PyTorch 2.12
# fails natively without Apple's proprietary stateful conversion repos.

# Step 1.1 Phi3
phi3_old = """#### Download

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='microsoft/Phi-3-mini-4k-instruct',
    local_dir='models_pytorch/phi3-mini',
    ignore_patterns=['*.gguf', '*.ggml']
)
print('Done')
"
```

**✅ Expected result:** Folder `models_pytorch/phi3-mini` created. Size: ~7.6 GB. Contains `config.json`, tokenizer files, and model `.safetensors` shards.

#### Convert to FP16 CoreML

```python
import coremltools as ct
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = 'models_pytorch/phi3-mini'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
model.eval()

# Trace and convert
mlmodel = ct.convert(
    model,
    convert_to='mlprogram',
    minimum_deployment_target=ct.target.macOS26,
    compute_units=ct.ComputeUnit.ALL,  # Uses ANE + GPU + CPU
)
mlmodel.save('models_fp16/phi3-mini-fp16.mlpackage')
print('Done: phi3-mini-fp16.mlpackage')
```

**✅ Expected result:** File `models_fp16/phi3-mini-fp16.mlpackage` created. Size: ~7.6 GB. Conversion time: 20–35 minutes."""

phi3_new = """#### Download Pre-converted CoreML Model

> ✅ Manual conversion of Phi-3 from PyTorch fails on standard `coremltools` due to complex Rotary Position Embedding (RoPE) operations. We will use a pre-converted model provided by the community on HuggingFace.

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='rfermontero/phi35-mini-coreml-fp16',
    local_dir='models_fp16/phi3-mini-fp16.mlpackage',
    allow_patterns=['*.mlmodelc/*', '*.json', '*.mlpackage/*']
)
print('Done')
"
```

**✅ Expected result:** Folder `models_fp16/phi3-mini-fp16.mlpackage` created and populated. Size: ~7.6 GB."""


# Step 1.2 Llama 3.2 3B
llama_old = """#### Download

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='meta-llama/Llama-3.2-3B-Instruct',
    local_dir='models_pytorch/llama32-3b',
    ignore_patterns=['*.gguf']
)
print('Done')
"
```

**✅ Expected result:** Folder `models_pytorch/llama32-3b` created. Size: ~6 GB.

#### Convert to FP16 CoreML

```python
model_id = 'models_pytorch/llama32-3b'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
model.eval()

mlmodel = ct.convert(
    model,
    convert_to='mlprogram',
    minimum_deployment_target=ct.target.macOS26,
    compute_units=ct.ComputeUnit.ALL,
)
mlmodel.save('models_fp16/llama32-3b-fp16.mlpackage')
print('Done: llama32-3b-fp16.mlpackage')
```

**✅ Expected result:** File `models_fp16/llama32-3b-fp16.mlpackage` created. Size: ~6 GB. Conversion time: 15–30 minutes."""

llama_new = """#### Download Pre-converted CoreML Model

> ✅ Just like Phi-3, manual PyTorch tracing of Llama 3.2 often fails without Apple's internal stateful optimization wrappers. We download a pre-converted community package.

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

**✅ Expected result:** Folder `models_fp16/llama32-3b-fp16.mlpackage` created. Size: ~6 GB."""

content = content.replace(phi3_old, phi3_new)
content = content.replace(llama_old, llama_new)

with open('docs/Experiment_Guide.md', 'w') as f:
    f.write(content)

