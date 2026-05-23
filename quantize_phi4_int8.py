import coremltools as ct
import coremltools.optimize.coreml as cto

op_config = cto.OpLinearQuantizerConfig(
    mode='linear_symmetric',
    dtype='int8',
    granularity='per_channel',
)
config = cto.OptimizationConfig(global_config=op_config)

print('Loading models_fp16/phi4-mini-fp16.mlpackage ...')
model = ct.models.MLModel('models_fp16/phi4-mini-fp16.mlpackage')
print('Quantizing to INT8 ...')
model_int8 = cto.linear_quantize_weights(model, config=config)
model_int8.save('models_int8/phi4-mini-int8.mlpackage')
print('Done → models_int8/phi4-mini-int8.mlpackage')
