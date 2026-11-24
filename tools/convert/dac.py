# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["torch", "safetensors", "numpy"]
# ///
"""Converts a Descript Audio Codec checkpoint (`.pth`) into a folder llm2chip reads: the
quantizer's codebooks and output projections and the decoder, in safetensors, with every
weight normalisation folded into a plain weight, and a config.json with its settings.

    uv run tools/convert/dac.py weights_24khz_1.5kbps_v1.0.pth OUT_DIR

The checkpoint is a PyTorch pickle: it is read with `weights_only=True`, which loads its
tensors and plain values and refuses to run anything else in it. The encoder is left out:
the chip only turns codes into sound.
"""
import json
import pathlib
import sys

import torch
from safetensors.torch import save_file

source, out = sys.argv[1], pathlib.Path(sys.argv[2])
checkpoint = torch.load(source, map_location="cpu", weights_only=True)
state, kwargs = checkpoint["state_dict"], checkpoint["metadata"]["kwargs"]

tensors = {}
for name, value in state.items():
    if not (name.startswith("decoder.") or ".codebook." in name or ".out_proj." in name):
        continue
    if name.endswith(".weight_v"):
        base = name[: -len(".weight_v")]
        g = state[base + ".weight_g"]
        # torch's weight_norm over dimension 0: each slice along it has norm g.
        norm = value.flatten(1).norm(dim=1).reshape(g.shape)
        tensors[base + ".weight"] = (g * value / norm).contiguous()
    elif not name.endswith(".weight_g"):
        tensors[name] = value.contiguous()

out.mkdir(parents=True, exist_ok=True)
save_file({k: v.float() for k, v in tensors.items()}, str(out / "model.safetensors"))
config = {
    "model_type": "dac",
    "sampling_rate": kwargs["sample_rate"],
    "latent_dim": kwargs["latent_dim"],
    "decoder_dim": kwargs["decoder_dim"],
    "decoder_rates": kwargs["decoder_rates"],
    "n_codebooks": kwargs["n_codebooks"],
    "codebook_size": kwargs["codebook_size"],
    "codebook_dim": kwargs["codebook_dim"],
}
(out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
print(f"{len(tensors)} tensors to {out}")
