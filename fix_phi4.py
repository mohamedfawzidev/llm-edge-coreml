# Let's fix the guide to make sure the HuggingFace URL is exactly right
with open('docs/Experiment_Guide.md', 'r') as f:
    guide = f.read()

# Make sure to replace any leftover phi-3 occurrences 
guide = guide.replace('models_fp16/phi4-mini-fp16.mlpackage/phi4mini-float16.mlpackage', 'models_fp16/phi4-mini-fp16.mlpackage')
guide = guide.replace('models_int4/phi4-mini-int4.mlpackage', 'models_int4/phi4-mini-int4.mlpackage')
guide = guide.replace('models_int8/phi4-mini-int8.mlpackage', 'models_int8/phi4-mini-int8.mlpackage')

with open('docs/Experiment_Guide.md', 'w') as f:
    f.write(guide)
