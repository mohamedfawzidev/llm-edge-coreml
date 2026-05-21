import os
import json
import shutil
from huggingface_hub import snapshot_download

# The issue is the user explicitly wants a *CoreML* .mlpackage file downloaded so they can use it
# The previous repo "rfermontero/phi35-mini-coreml-fp16" only contained an ONNX file, not a CoreML package.

print("Let's look for valid CoreML packages for Phi-3.")
print("Apple ML-Explore does not export to CoreML .mlpackage, it exports to MLX format.")

# In order to fulfill the user's explicit request in Phase 1 and 2, they *must* have an .mlpackage.
# If we cannot convert it, and cannot find it, we need to explicitly report this limitation.
