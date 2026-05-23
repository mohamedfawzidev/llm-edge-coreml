from datasets import load_dataset
import random
import json

random.seed(42)

print("Loading MMLU dataset...")
dataset = load_dataset('cais/mmlu', 'all', split='test')

subjects = [
    'high_school_mathematics', 'formal_logic',
    'high_school_physics', 'college_chemistry',
    'world_history', 'high_school_geography',
    'medical_genetics', 'clinical_knowledge',
    'computer_security', 'machine_learning',
]

sampled = []
for subject in subjects:
    subject_data = [x for x in dataset if x['subject'] == subject]
    sampled.extend(random.sample(subject_data, min(20, len(subject_data))))

print(f'Total questions sampled: {len(sampled)}')
with open('mmlu_200.json', 'w') as f:
    json.dump(sampled, f)
print('Saved → mmlu_200.json')
