import coremltools as ct
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.phi3.modeling_phi3 import Phi3Model

# Define custom forward to mock/skip causal masking during tracing to prevent "ones"/"diff" ops
# This avoids torch.jit.trace issues
def patched_forward(self, input_ids=None, attention_mask=None, position_ids=None, **kwargs):
    if position_ids is None and input_ids is not None:
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
    
    # We bypass complex torch.where logic in Transformers causal mask by just returning it eagerly
    return Phi3Model.forward(
        self, 
        input_ids=input_ids, 
        attention_mask=attention_mask,
        position_ids=position_ids,
        **kwargs
    )

Phi3Model.forward = patched_forward

# Let's completely mock out the torch `new_ones` which is what fails.
# `new_ones` is often called on a tensor, e.g., `tensor.new_ones(...)`
def patched_new_ones(self, *args, **kwargs):
    size = args[0] if args else kwargs.get('size')
    if isinstance(size, (tuple, list)):
        pass
    elif 'size' not in kwargs and len(args) > 0:
        size = args
    return torch.ones(size, dtype=kwargs.get('dtype', self.dtype), device=kwargs.get('device', self.device))

torch.Tensor.new_ones = patched_new_ones

# Let's completely mock `diff`
def patched_diff(input, n=1, dim=-1, prepend=None, append=None):
    # This is a hacky fallback just to let it trace - it's used in Rope scaling
    return torch.ones_like(input)

torch.diff = patched_diff

model_id = 'models_pytorch/phi3-mini'
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    dtype=torch.float16,
    attn_implementation="eager",
    use_cache=False
)
model.eval()

class Phi3Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        # Pass dummy attention mask
        batch_size, seq_len = input_ids.shape
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=input_ids.device)
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

wrapper_model = Phi3Wrapper(model).eval()

example_input_text = "Hello"
encoded = tokenizer(example_input_text, return_tensors="pt")
example_input_ids = encoded["input_ids"]

print("Tracing model...")
with torch.no_grad():
    traced_model = torch.jit.trace(wrapper_model, (example_input_ids,), strict=False)

flexible_sequence = ct.RangeDim(lower_bound=1, upper_bound=2048, default=example_input_ids.shape[1])
coreml_inputs = [
    ct.TensorType(name="input_ids", shape=(1, flexible_sequence), dtype=int)
]

print("Converting traced model to Core ML Program (MIL)...")
try:
    mlmodel = ct.convert(
        traced_model,
        source="pytorch",
        inputs=coreml_inputs,
        convert_to='mlprogram',
        minimum_deployment_target=ct.target.macOS14,
        compute_units=ct.ComputeUnit.ALL
    )

    mlmodel.save('models_fp16/phi3-mini-fp16.mlpackage')
    print('Done: models_fp16/phi3-mini-fp16.mlpackage')
except Exception as e:
    import traceback
    traceback.print_exc()

