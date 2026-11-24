# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["transformers", "tokenizers"]
# ///
"""The token ids of OuteTTS 1.0's prompt in a speaker's voice, as its package builds the
text and Transformers' tokenizer cuts it: what `SpeechVocabulary::prompt_with` must give.

    uv run tools/golden/speaker_prompt.py OUTETTS_DIR speaker.json "Text."
"""
import json
import sys

from transformers import AutoTokenizer

tts_dir, speaker_path, text = sys.argv[1:4]
speaker = json.load(open(speaker_path))
spoken = speaker["text"].strip()
joined = f"{spoken} {text.strip()}" if spoken[-1] in ".?!" else f"{spoken}. {text.strip()}"
words = []
for w in speaker["words"]:
    f = w["features"]
    feats = f"<|energy_{f['energy']}|><|spectral_centroid_{f['spectral_centroid']}|><|pitch_{f['pitch']}|>"
    pairs = "".join(f"<|c1_{a}|><|c2_{b}|>" for a, b in zip(w["c1"], w["c2"]))
    words.append(f"<|word_start|>{w['word']}<|features|><|t_{w['duration']:.2f}|>{feats}<|code|>{pairs}<|word_end|>")
prompt = f"<|im_start|>\n<|text_start|>{joined}<|text_end|>\n<|audio_start|>\n" + "\n".join(words) + "\n<|word_start|>"
print(json.dumps(AutoTokenizer.from_pretrained(tts_dir)(prompt, add_special_tokens=False).input_ids))
