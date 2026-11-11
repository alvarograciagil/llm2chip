"""Turns a llama2.c checkpoint (the raw `.bin` of float32 weights and its `tok*.bin`
tokenizer) into a Hugging Face Llama folder: config.json, model.safetensors and
tokenizer.json. Only the standard library: the `.bin` is plain numbers, no pickle.

    python3 tools/convert/llama2c.py stories260K.bin tok512.bin OUT_DIR [--affine BITS]

With `--affine`, every matrix is stored as MLX stores quantised checkpoints: codes of BITS
bits packed in 32-bit words, with a float16 scale and bias for each group of columns.
"""

import json
import struct
import sys
from array import array
from pathlib import Path


def read_floats(data, offset, count):
    values = array("f")
    values.frombytes(data[offset:offset + 4 * count])
    return values, offset + 4 * count


def rotary_to_halves(rows, n_heads, dim_out, dim_in):
    """llama2.c rotates consecutive pairs; Hugging Face rotates the two halves of each head.
    The same permutation as transformers' convert_llama_weights_to_hf."""
    head = dim_out // n_heads
    out = array("f", bytes(4 * dim_out * dim_in))
    for h in range(n_heads):
        for i in range(head // 2):
            for which in range(2):
                src = h * head + 2 * i + which
                dst = h * head + which * (head // 2) + i
                out[dst * dim_in:(dst + 1) * dim_in] = rows[src * dim_in:(src + 1) * dim_in]
    return out


def half(value):
    return struct.unpack("<H", struct.pack("<e", value))[0]


def affine(values, rows, cols, bits):
    """MLX's affine quantisation: each group gets the scale and bias that map its codes
    0..2^bits-1 onto its range. Groups of 32 or 64 columns when they divide the row,
    otherwise the largest divisor that keeps the codes in whole words."""
    group = next(g for g in (64, 32, 16) + tuple(range(cols, 0, -1)) if cols % g == 0 and (g * bits) % 32 == 0 or g == cols)
    top = (1 << bits) - 1
    words, scales, biases = array("I"), array("H"), array("H")
    for r in range(rows):
        row = values[r * cols:(r + 1) * cols]
        codes = []
        for g in range(0, cols, group):
            part = row[g:g + group]
            low, high = min(part), max(part)
            scale = (high - low) / top or 1e-8
            # The float16 the chip keeps is what the codes must be measured against.
            scale16 = struct.unpack("<e", struct.pack("<e", scale))[0] or 1e-8
            bias16 = struct.unpack("<e", struct.pack("<e", low))[0]
            scales.append(half(scale16))
            biases.append(half(bias16))
            codes.extend(min(top, max(0, round((v - bias16) / scale16))) for v in part)
        per_word = 32 // bits
        for w in range(0, cols, per_word):
            word = 0
            for k, code in enumerate(codes[w:w + per_word]):
                word |= code << (k * bits)
            words.append(word)
    return group, words, scales, biases


def write_safetensors(path, tensors):
    header, blobs, offset = {}, [], 0
    for name, (shape, values) in tensors.items():
        raw = values.tobytes()
        dtype = {"f": "F32", "I": "U32", "H": "F16"}[values.typecode]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    text = json.dumps(header).encode()
    text += b" " * (-len(text) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(text)) + text + b"".join(blobs))


def quantise(tensors, bits):
    """Every matrix in MLX's affine format; vectors stay as they are."""
    result = {}
    for name, (shape, values) in tensors.items():
        if len(shape) != 2 or not name.endswith(".weight"):
            result[name] = (shape, values)
            continue
        rows, cols = shape
        group, words, scales, biases = affine(values, rows, cols, bits)
        stem = name[: -len(".weight")]
        result[stem + ".weight"] = ([rows, cols * bits // 32], words)
        result[stem + ".scales"] = ([rows, cols // group], scales)
        result[stem + ".biases"] = ([rows, cols // group], biases)
    return result


def convert_weights(path, out, bits=None):
    data = Path(path).read_bytes()
    dim, hidden, layers, heads, kv_heads, vocab, seq = struct.unpack("7i", data[:28])
    shared = vocab > 0
    vocab = abs(vocab)
    head = dim // heads
    kv_dim = kv_heads * head
    offset = 28
    take = lambda count: read_floats(data, offset, count)
    tensors = {}
    embed, offset = take(vocab * dim)
    tensors["model.embed_tokens.weight"] = ([vocab, dim], embed)
    per_layer = {}
    for name, size in [("rms_att", dim), ("wq", dim * dim), ("wk", dim * kv_dim), ("wv", dim * kv_dim), ("wo", dim * dim), ("rms_ffn", dim), ("w1", hidden * dim), ("w2", dim * hidden), ("w3", hidden * dim)]:
        per_layer[name] = []
        for _ in range(layers):
            values, offset = take(size)
            per_layer[name].append(values)
    final, offset = take(dim)
    offset += 4 * seq * head  # the precomputed rotations, which nothing here needs
    for layer in range(layers):
        p = f"model.layers.{layer}."
        tensors[p + "input_layernorm.weight"] = ([dim], per_layer["rms_att"][layer])
        tensors[p + "self_attn.q_proj.weight"] = ([dim, dim], rotary_to_halves(per_layer["wq"][layer], heads, dim, dim))
        tensors[p + "self_attn.k_proj.weight"] = ([kv_dim, dim], rotary_to_halves(per_layer["wk"][layer], kv_heads, kv_dim, dim))
        tensors[p + "self_attn.v_proj.weight"] = ([kv_dim, dim], per_layer["wv"][layer])
        tensors[p + "self_attn.o_proj.weight"] = ([dim, dim], per_layer["wo"][layer])
        tensors[p + "post_attention_layernorm.weight"] = ([dim], per_layer["rms_ffn"][layer])
        tensors[p + "mlp.gate_proj.weight"] = ([hidden, dim], per_layer["w1"][layer])
        tensors[p + "mlp.down_proj.weight"] = ([dim, hidden], per_layer["w2"][layer])
        tensors[p + "mlp.up_proj.weight"] = ([hidden, dim], per_layer["w3"][layer])
    tensors["model.norm.weight"] = ([dim], final)
    if shared:
        tensors["lm_head.weight"] = ([vocab, dim], embed)
    else:
        classifier, offset = take(vocab * dim)
        tensors["lm_head.weight"] = ([vocab, dim], classifier)
    if bits:
        tensors = quantise(tensors, bits)
    write_safetensors(out / "model.safetensors", tensors)
    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": dim,
        "intermediate_size": hidden,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "vocab_size": vocab,
        "max_position_embeddings": seq,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    return vocab


def convert_tokenizer(path, out, vocab_size):
    """llama2.c's tokenizer is SentencePiece BPE: pieces with a score, merged greedily by
    the score of what they make. As a Hugging Face BPE, each piece that splits into two
    pieces of the vocabulary is a merge, ranked by its score."""
    data = Path(path).read_bytes()
    offset = 4
    pieces, scores = [], []
    for _ in range(vocab_size):
        score, length = struct.unpack("fi", data[offset:offset + 8])
        offset += 8
        pieces.append(data[offset:offset + length].decode("utf-8", errors="replace"))
        scores.append(score)
        offset += length
    # llama2.c stores the special tokens and the byte fallbacks by their text.
    names = []
    for i, piece in enumerate(pieces):
        if i == 0:
            names.append("<unk>")
        elif i == 1:
            names.append("<s>")
        elif i == 2:
            names.append("</s>")
        elif piece.startswith("<0x") and piece.endswith(">") and len(piece) == 6:
            names.append(piece)
        else:
            names.append(piece.replace(" ", "▁"))
    vocab = {name: i for i, name in enumerate(names)}
    merges = []
    for i, name in enumerate(names):
        if i < 3 or name.startswith("<0x") or len(name) < 2:
            continue
        for cut in range(1, len(name)):
            left, right = name[:cut], name[cut:]
            if left in vocab and right in vocab:
                merges.append((scores[i], vocab[left], vocab[right], left, right))
    merges.sort(key=lambda m: (-m[0], m[1], m[2]))
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [{"id": i, "content": names[i], "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True} for i in range(3)],
        "normalizer": {"type": "Sequence", "normalizers": [{"type": "Prepend", "prepend": "▁"}, {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}]},
        "pre_tokenizer": None,
        "post_processor": {"type": "TemplateProcessing", "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}], "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}], "special_tokens": {"<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]}}},
        "decoder": {"type": "Sequence", "decoders": [{"type": "Replace", "pattern": {"String": "▁"}, "content": " "}, {"type": "ByteFallback"}, {"type": "Fuse"}, {"type": "Strip", "content": " ", "start": 1, "stop": 0}]},
        "model": {"type": "BPE", "dropout": None, "unk_token": "<unk>", "continuing_subword_prefix": None, "end_of_word_suffix": None, "fuse_unk": True, "byte_fallback": True, "ignore_merges": False, "vocab": vocab, "merges": [[m[3], m[4]] for m in merges]},
    }
    (out / "tokenizer.json").write_text(json.dumps(tokenizer, ensure_ascii=False))


def main():
    weights, tokenizer, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    bits = int(sys.argv[sys.argv.index("--affine") + 1]) if "--affine" in sys.argv else None
    out.mkdir(parents=True, exist_ok=True)
    vocab = convert_weights(weights, out, bits)
    convert_tokenizer(tokenizer, out, vocab)
    print(f"{out}: config.json, model.safetensors, tokenizer.json")


if __name__ == "__main__":
    main()
