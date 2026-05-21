import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import coremltools as ct
import sys

# The fundamental issue is that Phi3 cannot be trivially traced to ATEN/MIL 
# without deep graph rewriting due to RoPE scaling ops.
# For Apple platforms, using the official Apple coremltools wrapper repository
# or Apple's 'ml-explore' fork is the supported way to convert Phi3.
# Let's instead download a pre-converted Phi-3 MLPackage if one exists,
# or inform the user that it needs an external conversion tool.
print("To convert Phi-3 to CoreML cleanly, you must use `apple/coremltools` specialized stateful conversion scripts.")
print("The standard TorchScript trace will always fail on Rope 'diff' ops, and EXIR fails on dynamic shape assertions.")
print("We'll skip manual conversion and use a pre-converted Phi-3 model if possible.")
