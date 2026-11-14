# Exact arithmetic against floating point

Agreement on the most likely token between the integer machine and floating point, with real weights. The integer machine matches the gates bit for bit on a tiny model at 12, 16, 18 and 20 bits.

| Model | Vector bits | Matches |
|---|---|---|
| Qwen3.8-27B | 16 | 29 of 30 |
| Qwen3.8-27B | 14 | 11 of 12 |
| Qwen3.6-35B-A3B (MoE) | 16 | 11 of 12 |
| Qwen3.6-35B-A3B (MoE) | 18 | 11 of 12 |

Command: `cargo run --release -p sim --example exact -- <folder> <tokens>`, with `BITS` to change the width.
