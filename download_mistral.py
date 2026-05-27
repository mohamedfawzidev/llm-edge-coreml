import os
import shutil
from huggingface_hub import snapshot_download

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

print("Downloading Mistral 7B FP16 CoreML from Apple (~14 GB)...")
snapshot_download(
    repo_id='apple/mistral-coreml',
    local_dir='models_fp16/_mistral_tmp',
    allow_patterns=['StatefulMistral7BInstructFP16.mlpackage/*']
)
shutil.move('models_fp16/_mistral_tmp/StatefulMistral7BInstructFP16.mlpackage', 'models_fp16/mistral-7b-fp16.mlpackage')
shutil.rmtree('models_fp16/_mistral_tmp')
print("Done → models_fp16/mistral-7b-fp16.mlpackage")
