# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["torch", "transformers", "safetensors", "numpy"]
# ///
"""What Transformers makes of a sentence with OuteTTS and DAC: the reference the Rust ones
are checked against.

    uv run tools/golden/speak.py OUTETTS_DIR DAC_DIR "text" out.json out.wav

OuteTTS writes codes greedily, with its repetition penalty (1.1 over the last 64 tokens),
from the prompt without a speaker. DAC's decoder is Transformers' `DacModel`, with the
weights `tools/convert/dac.py` wrote, renamed to its layout. The JSON holds the prompt's
tokens, the generated ones, the scores of the first steps, the two codebooks' codes, the
waveform and, for each Snake, the largest |α·x| it saw: what decides the sine's table.
"""
import json
import re
import sys
import wave

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, DacConfig, DacModel

STEPS, LIMIT, PENALTY, WINDOW = 4, 600, 1.1, 64

tts_dir, dac_dir, text, out_json, out_wav = sys.argv[1:6]
tokenizer = AutoTokenizer.from_pretrained(tts_dir)
model = AutoModelForCausalLM.from_pretrained(tts_dir, torch_dtype=torch.float32).eval()
prompt = f"<|im_start|>\n<|text_start|>{text}<|text_end|>\n<|audio_start|>\n"
ids = tokenizer(prompt, add_special_tokens=False).input_ids
end = tokenizer.convert_tokens_to_ids("<|audio_end|>")
tokens, scores = list(ids), []
with torch.no_grad():
    past = None
    step_input = torch.tensor([tokens])
    for _ in range(LIMIT):
        output = model(input_ids=step_input, past_key_values=past, use_cache=True)
        past = output.past_key_values
        logits = output.logits[0, -1].clone()
        if len(scores) < STEPS:
            scores.append(logits.tolist())
        for t in set(tokens[-WINDOW:]):
            logits[t] = logits[t] / PENALTY if logits[t] > 0 else logits[t] * PENALTY
        token = int(logits.argmax())
        tokens.append(token)
        if token in (end, tokenizer.eos_token_id):
            break
        step_input = torch.tensor([[token]])
generated = tokens[len(ids):]
pieces = tokenizer.convert_ids_to_tokens(generated)
c1 = [int(m.group(1)) for p in pieces if (m := re.fullmatch(r"<\|c1_(\d+)\|>", p))]
c2 = [int(m.group(1)) for p in pieces if (m := re.fullmatch(r"<\|c2_(\d+)\|>", p))]
frames = min(len(c1), len(c2))
codes = [c1[:frames], c2[:frames]]

# Original DAC names to Transformers' DacModel ones.
rename = []
rename.append((r"^decoder\.model\.0\.", "decoder.conv1."))
for b in range(1, 5):
    rename.append((rf"^decoder\.model\.{b}\.block\.0\.", f"decoder.block.{b - 1}.snake1."))
    rename.append((rf"^decoder\.model\.{b}\.block\.1\.", f"decoder.block.{b - 1}.conv_t1."))
    for u in range(3):
        for i, part in enumerate(["snake1", "conv1", "snake2", "conv2"]):
            rename.append((rf"^decoder\.model\.{b}\.block\.{u + 2}\.block\.{i}\.", f"decoder.block.{b - 1}.res_unit{u + 1}.{part}."))
rename.append((r"^decoder\.model\.5\.", "decoder.snake1."))
rename.append((r"^decoder\.model\.6\.", "decoder.conv2."))
state = {}
for name, value in load_file(f"{dac_dir}/model.safetensors").items():
    for pattern, replacement in rename:
        name = re.sub(pattern, replacement, name)
    state[name] = value
settings = json.load(open(f"{dac_dir}/config.json"))
config = DacConfig(
    hidden_size=settings["latent_dim"], decoder_hidden_size=settings["decoder_dim"], upsampling_ratios=settings["decoder_rates"],
    downsampling_ratios=list(reversed(settings["decoder_rates"])), n_codebooks=settings["n_codebooks"],
    codebook_size=settings["codebook_size"], codebook_dim=settings["codebook_dim"], sampling_rate=settings["sampling_rate"],
)
dac = DacModel(config).eval()
own = dac.state_dict()
missing = [k for k in own if (k.startswith("decoder.") or ".codebook." in k or ".out_proj." in k) and k not in state]
unknown = [k for k in state if k not in own]
assert not missing and not unknown, (missing[:5], unknown[:5])
dac.load_state_dict(state, strict=False)

reach = {}
def watch(name):
    def hook(module, inputs, output):
        x = inputs[0]
        reach[name] = max(reach.get(name, 0.0), float((module.alpha * x).abs().max()))
    return hook
for name, module in dac.named_modules():
    if name.endswith("snake1") or name.endswith("snake2"):
        module.register_forward_hook(watch(name))
with torch.no_grad():
    z = dac.quantizer.from_codes(torch.tensor([codes]))[0]
    audio = dac.decoder(z)[0, 0].numpy()

with wave.open(out_wav, "wb") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(settings["sampling_rate"])
    f.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
json.dump({"prompt": ids, "generated": generated, "scores": scores, "codes": codes, "audio": audio.tolist(), "snake_reach": reach}, open(out_json, "w"))
print(f"{len(generated)} tokens, {frames} frames, {len(audio) / settings['sampling_rate']:.2f} s; largest |αx| {max(reach.values()):.1f}")
