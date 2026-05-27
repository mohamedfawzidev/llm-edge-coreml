import os
# Must be set before importing transformers to prevent tokenizer parallelism
# from causing a memory spike when coremltools forks a subprocess.
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import gc
import json
import numpy as np
import coremltools as ct
from transformers import AutoTokenizer

TOKENIZER_IDS = {
    'phi4':    'microsoft/phi-4',
    'mistral': 'mistralai/Mistral-7B-Instruct-v0.3',
}

# Only Mistral models are evaluated here via CoreML.
#
# Phi-4 Mini cannot be evaluated via Python CoreML on 16 GB RAM:
#   - Single-token mode (the only mode that compiles) has no KV-cache context,
#     giving ~25% random-chance accuracy.
#   - Full-sequence mode OOMs during CoreML compilation.
# Phi-4 accuracy is measured separately via MLX (see evaluate_mmlu_phi4_mlx.py).
#
# Mistral 7B FP16 (14 GB) also exceeds 16 GB RAM — skip it.
MODELS = {
    'mistral-int4': 'models_int4/mistral-7b-int4.mlpackage',
    'mistral-int8': 'models_int8/mistral-7b-int8.mlpackage',
}


def make_prompt(q):
    text = f"Question: {q['question']}\n"
    for label, choice in zip('ABCD', q['choices']):
        text += f"{label}. {choice}\n"
    return text + "Answer:"


def answer_token_ids(tokenizer):
    """One representative token ID per choice A/B/C/D."""
    ids = []
    for label in 'ABCD':
        for candidate in (f' {label}', label, f'\n{label}'):
            toks = tokenizer.encode(candidate, add_special_tokens=False)
            if toks:
                ids.append(toks[-1])
                break
    assert len(ids) == 4, f"Could not resolve all 4 answer token IDs, got {ids}"
    return ids


def inspect_model(model):
    """
    Returns:
      main_input      – name of the primary token-ID input feature
      output_key      – name of the logits output feature
      has_causal_mask – True for Mistral-style models (single-token + KV-cache state)
                        False for Phi-4-style models (full-sequence, no explicit state)

    Phi-4 (AlonBBar): inputs=['input_ids'], accepts full sequences, make_state() throws.
    Mistral (Apple):  inputs=['inputIds','causalMask'], single-token, make_state() works.
    The causalMask flag is the reliable discriminator between the two architectures.
    """
    spec = model.get_spec()
    input_names  = [f.name for f in spec.description.input]
    output_names = [f.name for f in spec.description.output]

    main_input = next(
        (n for n in input_names if any(k in n.lower() for k in ('input', 'ids', 'token'))),
        input_names[0],
    )
    output_key = next(
        (n for n in output_names if 'logit' in n.lower()),
        output_names[0],
    )
    has_causal_mask = 'causalMask' in input_names

    print(f"  inputs={input_names}  outputs={output_names}  has_causal_mask={has_causal_mask}")
    return main_input, output_key, has_causal_mask


def predict_last_logits_mistral(model, main_input, output_key, token_ids):
    """Mistral: token-by-token with per-question state reset + growing causal mask."""
    state = model.make_state()
    for pos, tid in enumerate(token_ids):
        out = model.predict(
            {
                main_input:   np.array([[tid]], dtype=np.int32),
                'causalMask': np.zeros((1, 1, 1, pos + 1), dtype=np.float16),
            },
            state=state,
        )
    logits = out[output_key]
    if logits.ndim >= 3:
        return logits[0, -1, :]
    if logits.ndim == 2:
        return logits[-1, :]
    return logits.ravel()


def predict_last_logits_phi4(path, main_input, output_key, token_ids):
    """
    Phi-4: reload model for each question so any implicit KV-cache state is
    reset, then feed tokens one at a time (single-token mode is all the spec
    supports without triggering a full-sequence compilation OOM).
    """
    model = ct.models.MLModel(path)
    for tid in token_ids:
        out = model.predict({main_input: np.array([[tid]], dtype=np.int32)})
    del model
    gc.collect()

    logits = out[output_key]
    if logits.ndim >= 3:
        return logits[0, -1, :]
    if logits.ndim == 2:
        return logits[-1, :]
    return logits.ravel()


def evaluate_model(path, questions, tokenizer, ans_ids):
    # Load once to inspect the spec; Mistral keeps the instance for all questions.
    model = ct.models.MLModel(path)
    main_input, output_key, has_causal_mask = inspect_model(model)

    if not has_causal_mask:
        # Phi-4: drop the persistent instance — we reload per question.
        del model
        gc.collect()

    correct = 0
    for i, q in enumerate(questions):
        token_ids = tokenizer.encode(make_prompt(q), add_special_tokens=False)
        try:
            if has_causal_mask:
                logits = predict_last_logits_mistral(
                    model, main_input, output_key, token_ids
                )
            else:
                logits = predict_last_logits_phi4(
                    path, main_input, output_key, token_ids
                )
            scores = [float(logits[tid]) for tid in ans_ids]
            pred   = int(np.argmax(scores))
        except Exception as exc:
            print(f"  [Q{i}] error: {exc}")
            pred = 0

        if pred == q['answer']:
            correct += 1

        if (i + 1) % 10 == 0:
            print(f"  Q{i+1}/{len(questions)} — running accuracy: {correct/(i+1)*100:.1f}%",
                  flush=True)

    if has_causal_mask:
        del model
        gc.collect()
    return correct / len(questions) * 100


# ── main ──────────────────────────────────────────────────────────────────────

with open('mmlu_200.json') as f:
    questions = json.load(f)

print('Loading tokenizers ...')
tokenizers = {k: AutoTokenizer.from_pretrained(v) for k, v in TOKENIZER_IDS.items()}
ans_ids    = {k: answer_token_ids(tok) for k, tok in tokenizers.items()}
print(f"  phi4    A/B/C/D: {ans_ids['phi4']}")
print(f"  mistral A/B/C/D: {ans_ids['mistral']}")

os.makedirs('results', exist_ok=True)

results_path = 'results/mmlu_accuracy.json'
try:
    with open(results_path) as f:
        results = json.load(f)
    print(f'\nResuming from {results_path}: already have {list(results.keys())}')
except FileNotFoundError:
    results = {}

for name, path in MODELS.items():
    if name in results:
        print(f'\nSkipping {name} (already scored: {results[name]:.1f}%)')
        continue

    family = 'phi4' if 'phi4' in name else 'mistral'
    print(f'\nEvaluating {name} ...')
    try:
        acc = evaluate_model(path, questions, tokenizers[family], ans_ids[family])
        results[name] = round(acc, 1)
        print(f'  {name}: {acc:.1f}%')
    except Exception as exc:
        results[name] = None
        print(f'  {name}: FAILED — {exc}')

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  (saved partial results → {results_path})')

print('\nFinal results:')
for name, acc in results.items():
    tag = f'{acc:.1f}%' if acc is not None else 'FAILED'
    print(f'  {name}: {tag}')
print(f'\nSaved → {results_path}')
