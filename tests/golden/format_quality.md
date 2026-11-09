# Calidad de cada formato de pesos

Coincidencia con la coma flotante en 16 tokens generados con Qwen3.8-27B (carpeta Optimized-Speed-FP16), recuantizando cada tensor al formato indicado y ejecutando el resto en coma flotante.

Límites de esta medida:

- El punto de partida ya está cuantizado a 4 bits (8 en algunos tensores). Los formatos estrechos parten de un error ya presente, y MXFP4 o NVFP4 no se pueden juzgar del todo sin pesos originales.
- Con 16 tokens, cada coincidencia vale 6,25 puntos. La KL media distingue mejor entre formatos parecidos: por eso `mlx4` (100 %) no es mejor que `mlx6` (93,8 %).
- `mlx6` y `mlx8` se midieron con grupos de 32, y las presets de la app usan 64. `gguf3` también usa 32 frente a los 16 de Q3_K. Por eso esas tres filas no aparecen en la tabla de la app.

| formato | top-1 | KL media | s/token |
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

Comando: `cargo run --release -p sim --example formats -- <carpeta> 16 <formatos...>`
