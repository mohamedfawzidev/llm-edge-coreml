import re

with open('docs/Experiment_Guide.md', 'r') as f:
    content = f.read()

# Since we couldn't find a single unified .mlpackage for Phi-3 FP16, and the previous repo was just an ONNX file,
# we have to instruct the user to use mlx instead for Phi-3 (or we tell them how to get it).
# Apple provides Mistral, and community has Llama.
# For Phi-3, there is NO public complete .mlpackage for CoreML. 

# Let's adjust the guide. If we strictly need CoreML to measure ANE:
# Apple has Llama and Mistral. We can replace Phi-3 with a model Apple officially provides in their ML-Explore
# Or we can just use the HuggingFace `apple/coreml-mistral` and maybe another officially supported model.
# Since the paper already says Phi-3, we need to provide a working solution for Phi-3.
# Let's provide the exact instructions for mlx-lm conversion to mlx since CoreML conversion fails natively.
# Oh wait, the user's paper is *specifically* about CoreML. "A Comparative Study of Quantization Techniques using CoreML".
# So they *must* use CoreML.

# Let's write a python script that actually uses ml-explore or the apple/coremltools
# stateful converter from source, because that's the only way to convert Phi3 to CoreML.

