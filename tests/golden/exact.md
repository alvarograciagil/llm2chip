# Aritmética exacta frente a coma flotante

Acuerdo del token más probable entre la máquina de enteros y la coma flotante, con los pesos reales. La máquina de enteros coincide bit a bit con las puertas en un modelo diminuto a 12, 16, 18 y 20 bits.

| Modelo | Bits de vector | Coincidencias |
|---|---|---|
| Qwen3.8-27B | 16 | 29 de 30 |
| Qwen3.8-27B | 14 | 11 de 12 |
| Qwen3.6-35B-A3B (MoE) | 16 | 11 de 12 |
| Qwen3.6-35B-A3B (MoE) | 18 | 11 de 12 |

Comando: `cargo run --release -p sim --example exact -- <carpeta> <tokens>`, con `BITS` para cambiar el ancho.
