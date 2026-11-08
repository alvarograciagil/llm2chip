"""Writes reference tokens from MLX for comparison with the Rust executor."""
import json
import sys

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

model_dir, prompt_text, count, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
model, tokenizer = load(model_dir)
prompt = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
ids = tokenizer.encode(prompt, add_special_tokens=False)

cache = make_prompt_cache(model)
logits = model(mx.array(ids)[None], cache=cache)[0, -1]
mx.eval(logits)
top = mx.argsort(-logits)[:10].tolist()
first = {"top": top, "values": [logits[i].item() for i in top]}

generated = []
token = int(mx.argmax(logits).item())
for _ in range(count):
    generated.append(token)
    if token in (248046, 248044):
        break
    logits = model(mx.array([[token]]), cache=cache)[0, -1]
    mx.eval(logits)
    token = int(mx.argmax(logits).item())

json.dump({"prompt": ids, "first": first, "generated": generated}, open(out_path, "w"))
print(generated, tokenizer.decode(generated))
