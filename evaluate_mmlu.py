import coremltools as ct
import json

def format_mmlu_prompt(q):
    choices = ['A', 'B', 'C', 'D']
    prompt = f"Question: {q['question']}\n"
    for i, choice in enumerate(q['choices']):
        prompt += f"{choices[i]}. {choice}\n"
    prompt += 'Answer:'
    return prompt

def evaluate_model(mlpackage_path, questions):
    model = ct.models.MLModel(mlpackage_path)
    correct = 0
    for q in questions:
        prompt = format_mmlu_prompt(q)
        # Run model prediction here — extract first token after 'Answer:'
        # predicted = run_inference(model, prompt)
        # if predicted.strip().upper() == q['answer']:
        #     correct += 1
    accuracy = (correct / len(questions)) * 100
    return accuracy

with open('mmlu_200.json') as f:
    questions = json.load(f)

models = {
    'phi4-fp16':    'models_fp16/phi4-mini-fp16.mlpackage',
    'phi4-int8':    'models_int8/phi4-mini-int8.mlpackage',
    'phi4-int4':    'models_int4/phi4-mini-int4.mlpackage',
    'llama-fp16':   'models_fp16/llama32-3b-fp16.mlpackage',
    'llama-int8':   'models_int8/llama32-3b-int8.mlpackage',
    'llama-int4':   'models_int4/llama32-3b-int4.mlpackage',
    'mistral-fp16': 'models_fp16/mistral-7b-fp16.mlpackage',
    'mistral-int8': 'models_int8/mistral-7b-int8.mlpackage',
    'mistral-int4': 'models_int4/mistral-7b-int4.mlpackage',
}

results = {}
for name, path in models.items():
    print(f'Evaluating {name}...')
    acc = evaluate_model(path, questions)
    results[name] = acc
    print(f'  {name}: {acc:.1f}%')

with open('results/mmlu_accuracy.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSaved → results/mmlu_accuracy.json')
