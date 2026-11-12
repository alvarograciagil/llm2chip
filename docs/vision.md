# Imágenes en el chip

Qwen3.5 lee imágenes con un codificador propio (un ViT) que convierte cada imagen en tokens que el modelo de lenguaje lee en lugar de `<|image_pad|>`. Este documento describe ese codificador tal como lo reproduce llm2chip, lo que ya funciona y cómo se llevará al circuito.

## El codificador de Qwen3.8-27B

Sale de `vision_config` y de `model-vision.safetensors` (pesos en BF16, unos 460 M de parámetros).

| Paso | Qué hace | Tamaño en Qwen3.8-27B |
|---|---|---|
| Preprocesado | Redimensiona a lados múltiplos de 32 px (entre 256×256 y 16 M de píxeles), normaliza a [−1, 1] y corta parches de 16×16 en dos fotogramas iguales | 1 536 valores por parche |
| Embedding de parche | Una convolución 3D que equivale a una matriz por parche, con sesgo | 1 536 → 1 152 |
| Posición aprendida | Una tabla de 48×48 interpolada bilinealmente a la rejilla de la imagen | 1 152 por parche |
| 27 bloques | LayerNorm, atención bidireccional con RoPE 2D (la mitad de las frecuencias sigue la fila y la otra mitad la columna), LayerNorm y MLP con GELU tanh; todo con sesgos | 16 cabezas de 72 |
| Fusión | LayerNorm y, por cada 2×2 parches vecinos, 4 608 → 4 608 → 5 120 con GELU | 1 token por 4 parches |

Los parches van ordenados por bloques de 2×2, así que cada token del modelo de lenguaje es la concatenación de cuatro parches consecutivos.

En el modelo de lenguaje, cada token tiene tres posiciones (tiempo, fila, columna) para la RoPE: de las 32 frecuencias rotadas, las de índice 1, 4, 7… (hasta 33) siguen la fila, las de índice 2, 5, 8… (hasta 30) la columna y el resto el tiempo (`mrope_section` [11, 11, 10], intercalado). El texto tiene las tres iguales, por eso los modelos sin imágenes no notan la diferencia. Tras una imagen de h×w tokens que empieza en la posición p, todos sus tokens tienen tiempo p, filas p…p+h−1 y columnas p…p+w−1, y el texto sigue en p + max(h, w).

## Lo que ya funciona

- **Referencia en coma flotante** (`crates/sim/src/vision.rs`): preprocesado, codificador completo y fusión, contra los pesos reales.
- **Modelo de lenguaje con imágenes** (`Engine::step_embedding`, `set_mrope`, `continue_text_at`): los tokens de imagen entran como vectores con sus tres posiciones, y el texto siguiente continúa donde toca.
- **Con la aritmética del chip** (`ExactEngine`, lo mismo): cada vector se carga en el formato del die donde iría la fila del embedding, y su rotación es una fila propia de la tabla de RoPE para esa posición de la caché, compuesta con las de sus tres posiciones (tiempo, fila y columna) según las secciones de M-RoPE. En el chip, el anfitrión escribe esas filas en la tabla que el die ya lee por posición, igual que trae las filas del embedding; el texto que sigue a la imagen se rota desde su propia posición. Un test lo compara con la coma flotante en un modelo de juguete con toda la cabeza rotada.
- **Ejemplo** `cargo run --release -p sim --example see -- <modelo> rojo`: dibuja una figura de color, se la enseña al Qwen real y le pregunta de qué color es.
- **Estimador**: con «Incluir visión», el área cuenta todos los pesos del codificador (bloques, embeddings y fusión) y aparece el tiempo de procesar una imagen de 512×512 en los multiplicadores de un die.

## Cómo se llevará al circuito

Casi todo se compone con operaciones que el die ya tiene:

1. **Embedding de parche y fusión**: productos matriz-vector en las celdas, con sus sesgos (`AddWeight`) y GELU por tabla (`FN_GELU_TANH`, `FN_GELU`).
2. **LayerNorm**: la media es una suma con el registro de unos, escalada; después, la misma norma que RMSNorm sobre el vector centrado, y el sesgo.
3. **Atención bidireccional**: la misma atención del modelo de lenguaje sobre un almacén de claves y valores de la imagen, pero sin máscara causal. Exige recorrer el codificador capa a capa: primero todas las claves y valores de un bloque, después todas las consultas. El firmware lo ordena con dos rutinas por bloque (`VIT_QKV` y `VIT_RESTO`) y un índice de parche, y guarda el estado de cada parche entre bloques en un almacén como el de la caché KV.
4. **RoPE 2D**: el firmware escribe para cada parche su fila y su columna; las dos tablas de cosenos y senos ya existen para la RoPE del texto.
5. **M-RoPE en el texto**: hecho en la aritmética del chip, con filas de rotación escritas por el anfitrión para las posiciones de la imagen (ver arriba).
6. **Entrada de imagen**: la rutina de token del modelo de lenguaje acepta como entrada un registro en lugar del embedding, igual que ya acepta la salida de la cabeza MTP.

Lo que falta para hacerlo en puertas es el orden capa a capa del firmware y las rutinas del codificador; la aritmética ya está.

## Estado, etapa a etapa

Cada etapa se comprueba contra la anterior en un codificador de juguete (`make_vision_checkpoint` en las pruebas: ancho 8 en 2 cabezas de 4, 2 bloques, parches de 2×2 píxeles en 2 fotogramas, 2×2 parches por token de ancho 8, el del modelo de lenguaje de juguete), con las pruebas de `crates/sim/tests/vision.rs`.

| Etapa | Qué | Estado |
|---|---|---|
| 0 | Codificador de juguete; la referencia en coma flotante lo ejecuta | Hecha |
| 1 | `compile_vision`: el codificador como un programa con sus rutinas (embedding de parche, las dos de cada bloque y la fusión) y tres operaciones nuevas: `LayerNorm` y leer y guardar la fila de un parche (`LoadRow`, `StoreRow`) | Hecha |
| 2 | `ExactVision`: las rutinas en la aritmética exacta del die, en el orden del firmware; difiere de la coma flotante menos de un 0,5 % y el modelo de lenguaje elige el mismo token con sus tokens que con los de la referencia | Hecha |
| 3 | Cada rutina en puertas, bit a bit igual que la máquina exacta: un microprograma con una entrada por rutina (`routines_microprogram`) y el anfitrión escribiendo píxeles, posiciones y filas de rotación (`rotation_writes`) | Hecha |
| 4 | El firmware las ordena; simulador de instrucciones frente a puertas, de la imagen al texto | Pendiente |
| 5 | La rutina de token del modelo de lenguaje toma la imagen como entrada, de punta a punta | En la máquina exacta: `attach_vision` mete el codificador en el programa del modelo (sus búferes y ranuras de claves con los del modelo, su propia tabla de rotación, la 2), y una entrada del microcódigo lee el token desde la salida del codificador en lugar del embedding. Da los mismos logits que darle al modelo los tokens del codificador aparte. Falta en puertas, con el firmware |

Cómo se ejecuta: el anfitrión escribe por cada parche sus píxeles y su posición aprendida ya interpolada, y la rotación de su fila y columna en la tabla que el die lee por posición. La rutina de embedding guarda cada parche en un almacén de filas. Cada bloque pasa dos veces por todos los parches: la primera guarda claves, valores y consulta; la segunda atiende sobre todos (con la posición en el último parche) y sigue con el resto del bloque. La fusión lee cuatro filas consecutivas del almacén como un vector.

Los almacenes guardan sus filas con un exponente fijo, como la caché de claves (`STATE_EXPONENT`, valores por debajo de 64): en un codificador real, con estados mayores, habrá que darles el suyo.

## Dibujar con Janus-Pro

Janus-Pro-1B también genera imágenes: su modelo de lenguaje escribe 576 tokens de un libro de códigos (una rejilla de 24 × 24), guiado lejos de lo que dibujaría sin el texto, y un decodificador convolucional los convierte en píxeles. `cargo run --release -p sim --example draw -- <carpeta> "<descripción>" salida.bmp [temperatura] [semilla]`.

Con `EXACTO=16` (y en la aplicación, con el modelo cargado en la aritmética del chip), corren en la aritmética entera del die las dos copias del modelo de lenguaje, con texto y sin él, y sus dos cabezas de imagen. El compilador añade al programa de Janus una cabeza de imagen (`ImageHead`) con dos listas de operaciones:

- **Puntuaciones**: del estado final, `proj_out` con su sesgo, GELU y `vision_head` con su sesgo, una puntuación por cada uno de los 16 384 códigos del libro.
- **Entrada**: el código leído de `generation_embeddings` como la fila de un embedding, `fc1` con su sesgo, GELU y `hidden_layers.0` con su sesgo, el vector que entra en lugar del embedding del siguiente token. Es el mismo para las dos copias, así que se calcula una vez.

En el anfitrión quedan la mezcla de la guía (dos contextos, dos cachés), el muestreo y el decodificador de píxeles. Un test con un Llama diminuto y cabezas aleatorias comprueba que las puntuaciones y las entradas salen del chip como en coma flotante, dentro del 2 %.

Con «a red apple on a wooden table», semilla 7 y temperatura 1, la coma flotante tarda 387 s y la aritmética del chip 543 s, y las dos dibujan una manzana roja sobre madera. No son la misma imagen: con temperatura 1, una diferencia mínima en un logit cambia qué token se elige, y desde ahí el dibujo sigue otro camino. Un test comprueba que el estado oculto que lee la cabeza de imagen sale del chip igual que en coma flotante, dentro del 2 %.
