"""Writes tiny random Llama-shaped checkpoints and the logits mlx-lm computes for them.

No model is downloaded: the weights are noise, which is what makes any wiring
mistake in the Rust program show up in the numbers.
"""
import json
import pathlib
import sys

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import _get_classes

TOKENS = [3, 1, 4, 1, 5, 9, 2, 6]
HIDDEN, HEADS, VOCAB = 16, 2, 32
CASES = {
    "llama": {"num_key_value_heads": 2, "tie_word_embeddings": False},
    "llama3": {
        "model_type": "llama", "num_key_value_heads": 2, "tie_word_embeddings": False, "max_position_embeddings": 128,
        "rope_scaling": {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 32},
    },
    "mistral": {"num_key_value_heads": 1, "tie_word_embeddings": False},
    "qwen3": {"num_key_value_heads": 1, "head_dim": 8, "tie_word_embeddings": True, "max_position_embeddings": 64, "rope_theta": 1000.0},
}

root = pathlib.Path(sys.argv[1])
for family, extra in CASES.items():
    config = {
        "model_type": extra.pop("model_type", family), "hidden_size": HIDDEN, "intermediate_size": 32, "vocab_size": VOCAB,
        "num_hidden_layers": 2, "num_attention_heads": HEADS, "rms_norm_eps": 1e-5,
        **extra,
    }
    model_cls, args_cls = _get_classes(config)
    model = model_cls(args_cls.from_dict(config))
    rng = np.random.default_rng(sum(map(ord, family)))
    weights = {}
    for name, value in tree_flatten(model.parameters()):
        shape = value.shape
        weights[name] = mx.array(rng.normal(0.0, 0.6, shape).astype(np.float32))
    model.load_weights(list(weights.items()))

    out = root / family
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "model.safetensors"), weights)
    (out / "config.json").write_text(json.dumps(config, indent=1) + "\n")

    cache = make_prompt_cache(model)
    logits = []
    for token in TOKENS:
        step = model(mx.array([[token]]), cache=cache)[0, -1]
        mx.eval(step)
        logits.append([round(float(v), 6) for v in step.tolist()])
    (out / "reference.json").write_text(json.dumps({"tokens": TOKENS, "logits": logits}) + "\n")
    print(family, sorted(weights)[:3], "...", len(weights))
