# Quality of each weight format

Agreement with floating point over 16 generated tokens with Qwen3.8-27B (the Optimized-Speed-FP16 folder), requantizing each tensor to the given format and running the rest in floating point.

Limits of this measurement:

- The starting point is already quantized to 4 bits (8 in some tensors). Narrow formats start from an error that is already there, and MXFP4 or NVFP4 cannot be fully judged without the original weights.
- With 16 tokens, each match is worth 6.25 points. The mean KL separates similar formats better: that is why `mlx4` (100 %) is not better than `mlx6` (93.8 %).
- `mlx6` and `mlx8` were measured with groups of 32, and the app's presets use 64. `gguf3` also uses 32 against Q3_K's 16. That is why those three rows do not appear in the app's table.

| format | top-1 | mean KL | s/token |
|---|---|---|---|
| q2 | 0.0 % | 11.9131 | 5.4 |
| q3 | 0.0 % | 13.0815 | 5.3 |
| q4 | 0.0 % | 13.1163 | 5.8 |
| q5 | 0.0 % | 14.4340 | 5.6 |
| q6 | 43.8 % | 0.9755 | 5.7 |
| q7 | 87.5 % | 0.1267 | 5.8 |
| q8 | 87.5 % | 0.0177 | 5.8 |
| q10 | 100.0 % | 0.0008 | 7.3 |
| q12 | 100.0 % | 0.0001 | 6.7 |
| q16 | 100.0 % | 0.0000 | 8.9 |
| q24 | 100.0 % | 0.0000 | 6.1 |
| q32 | 100.0 % | 0.0000 | 6.1 |
| mlx2 | 31.2 % | 2.2172 | 7.6 |
| mlx3 | 87.5 % | 0.1466 | 7.4 |
| mlx4 | 100.0 % | 0.0048 | 2.0 |
| mlx5 | 87.5 % | 0.0056 | 8.8 |
| mlx6 | 93.8 % | 0.0009 | 9.9 |
| mlx8 | 100.0 % | 0.0000 | 9.8 |
| gguf3 | 81.2 % | 0.4703 | 7.9 |
| gguf4 | 87.5 % | 0.0478 | 7.3 |
| gguf8 | 100.0 % | 0.0001 | 8.9 |
| fp8 | 100.0 % | 0.0066 | 9.4 |
| fp16 | 100.0 % | 0.0000 | 8.5 |
| bf16 | 100.0 % | 0.0000 | 8.5 |
| fp24 | 100.0 % | 0.0000 | 8.5 |
| fp32 | 100.0 % | 0.0000 | 8.5 |
| mxfp4 | 93.8 % | 0.0648 | 9.0 |
| mxfp6 | 93.8 % | 0.0054 | 9.0 |
| mxfp8 | 100.0 % | 0.0066 | 9.0 |
| posit8 | 93.8 % | 0.0123 | 10.7 |
| posit16 | 100.0 % | 0.0000 | 10.7 |
| log8 | 81.2 % | 0.1800 | 37.8 |
| log16 | 100.0 % | 0.0000 | 26.7 |
| book4 | 43.8 % | 2.1889 | 50.9 |
| book8 | 56.2 % | 1.2565 | 68.7 |

Command: `cargo run --release -p sim --example formats -- <folder> 16 <formats...>`

## Measured with 64 tokens

With more tokens, each match is worth 1.6 points. The codebooks were measured after changing their initialization to values spread over the whole range.

| format | top-1 | mean KL | s/token |
|---|---|---|---|
| nvfp4 | 90.6 % | 0.0510 | 4.2 |
| mxfp4 | 93.8 % | 0.1073 | 2.7 |
| fp8 | 95.3 % | 0.0052 | 3.2 |
| q8 | 93.8 % | 0.0265 | 3.0 |
| mlx4 | 93.8 % | 0.0053 | 1.3 |
| q6 | 65.6 % | 0.6837 | 2.8 |
| posit8 | 93.8 % | 0.0099 | 3.1 |
| book4 | 57.8 % | 1.3339 | 4.2 |
| book8 | 73.4 % | 0.8838 | 4.6 |
