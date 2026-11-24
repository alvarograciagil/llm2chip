# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["torch", "transformers==5.24.0", "safetensors", "numpy"]
# ///
"""What Transformers' streaming Moonshine makes of a clip: the reference the Rust one is checked against.

    uv run tools/golden/moonshine_streaming.py MODEL_DIR clip.wav out.json

The clip is 16 kHz mono 16-bit PCM, cut to whole 20 ms (320 samples), as the chip takes it.
The JSON holds the encoder's frames, the greedy transcription's tokens and, for each of its
first steps, the scores of every token.
"""
import json
import sys
import wave

import numpy as np
import torch
from transformers import MoonshineStreamingForConditionalGeneration

STEPS = 4

model_dir, clip, out = sys.argv[1:4]
with wave.open(clip, "rb") as f:
    assert f.getframerate() == 16000 and f.getnchannels() == 1 and f.getsampwidth() == 2, "16 kHz mono 16-bit PCM"
    samples = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
samples = samples[: len(samples) // 320 * 320]

model = MoonshineStreamingForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32).eval()
config = model.config
values = torch.from_numpy(samples)[None, :]
mask = torch.ones_like(values, dtype=torch.long)
with torch.no_grad():
    encoded = model.model.encoder(values, attention_mask=mask)
    frames = encoded.last_hidden_state[0]
    limit = int(len(samples) * 6 / 16000) + 1
    tokens = [config.decoder_start_token_id]
    scores = []
    for _ in range(limit):
        logits = model(input_values=values, attention_mask=mask, decoder_input_ids=torch.tensor([tokens])).logits[0, -1]
        if len(scores) < STEPS:
            scores.append(logits.tolist())
        token = int(logits.argmax())
        if token == config.eos_token_id:
            break
        tokens.append(token)

json.dump({"samples": len(samples), "frames": frames.tolist(), "tokens": tokens[1:], "scores": scores}, open(out, "w"))
print(f"{frames.shape[0]} frames, tokens {tokens[1:]}")
