import coremltools as ct
import coremltools.optimize.coreml as cto

op_config = cto.OpLinearQuantizerConfig(
    mode='linear_symmetric',
    dtype='int4',
    granularity='per_block',
    block_size=32,
)
config = cto.OptimizationConfig(global_config=op_config)

print('Loading models_fp16/mistral-7b-fp16.mlpackage ...')
model = ct.models.MLModel('models_fp16/mistral-7b-fp16.mlpackage')
print('Quantizing to INT4 ...')
model_int4 = cto.linear_quantize_weights(model, config=config)
model_int4.save('models_int4/mistral-7b-int4.mlpackage')
print('Done → models_int4/mistral-7b-int4.mlpackage')
