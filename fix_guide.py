import re

with open('docs/Experiment_Guide.md', 'r') as f:
    content = f.read()

# Replace the Phi-3 conversion with a download from huggingface (MLX)
new_phi3 = """#### Download & Convert via MLX

```bash
# We use MLX to fetch the model and export it to CoreML format directly
pip install mlx mlx-lm
mlx_lm.convert --hf-path microsoft/Phi-3-mini-4k-instruct --mlx-path models_pytorch/phi3-mini-mlx --export-coreml models_fp16/phi3-mini-fp16.mlpackage
```"""

# The above assumes mlx can export to coreml directly but wait, mlx doesn't export to coreml, it runs directly.
# Let's check what tools can actually export Phi-3 to coreml.
