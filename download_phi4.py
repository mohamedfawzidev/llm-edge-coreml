import os
import shutil
from huggingface_hub import snapshot_download

print("Downloading Phi-4 Mini FP16 CoreML (~7.6 GB)...")
snapshot_download(
    repo_id='AlonBBar/phi4mini-task-assistant-coreml',
    local_dir='models_fp16/_phi4_tmp',
    allow_patterns=['phi4mini-float16.mlpackage/*']
)
dst = 'models_fp16/phi4-mini-fp16.mlpackage'
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.move('models_fp16/_phi4_tmp/phi4mini-float16.mlpackage', dst)
shutil.rmtree('models_fp16/_phi4_tmp')
print("Done → models_fp16/phi4-mini-fp16.mlpackage")
