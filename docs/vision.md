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
| 4 | El firmware las ordena; simulador de instrucciones frente a puertas, de la imagen al texto | Hecha: ver «El firmware con una imagen» |
| 5 | La rutina de token del modelo de lenguaje toma la imagen como entrada, de punta a punta | En la máquina exacta: `attach_vision` mete el codificador en el programa del modelo (sus búferes y ranuras de claves con los del modelo, su propia tabla de rotación, la 2), y una entrada del microcódigo lee el token desde la salida del codificador en lugar del embedding. Da los mismos logits que darle al modelo los tokens del codificador aparte. En puertas, con el anfitrión ordenando las rutinas, un mismo die codifica la imagen y el modelo la lee, con los logits bit a bit iguales a la máquina exacta |

Cómo se ejecuta: el anfitrión escribe por cada parche sus píxeles y su posición aprendida ya interpolada, y la rotación de su fila y columna en la tabla que el die lee por posición. La rutina de embedding guarda cada parche en un almacén de filas. Cada bloque pasa dos veces por todos los parches: la primera guarda claves, valores y consulta; la segunda atiende sobre todos (con la posición en el último parche) y sigue con el resto del bloque. El último bloque aplica ya la LayerNorm de la fusión, que es de cada parche por separado, y la fusión lee cuatro filas consecutivas del almacén y las junta en un vector.

### Exponentes por fila

En el codificador real, el estado de los parches crece de unas pocas unidades a casi 300 a partir del bloque 10, y tras el último bloque llega a 4 828 en un canal: el canal atípico de los ViT. Las consultas, claves y valores no pasan de 14. Un exponente fijo para todo el almacén no sirve: por debajo de 64 satura el canal atípico (error del 440 %), y por debajo de 8 192 se lleva la precisión del resto de filas (51 % a 16 bits).

Por eso cada fila de un almacén guarda su propio exponente, como lo tiene cada registro: `StoreRow` copia las mantisas tal cual y anota el exponente del vector en un registro aparte, en la posición de la fila; `LoadRow` las copia de vuelta y recupera el exponente. Son dos operaciones nuevas del juego (`save_exponent` y `restore_exponent`), un movimiento del ALU escalar entre el registro de exponentes y un elemento, sin hardware nuevo. Cuando la fusión junta cuatro filas, se alinean al mayor de sus exponentes; por eso la LayerNorm de la fusión va antes, en el último bloque: así una fila con el canal atípico no ahoga los valores pequeños de sus vecinas.

Con el codificador real de Qwen3.8-27B y una imagen de 256×256 (256 parches, 64 tokens), los tokens del die difieren de la coma flotante un 1,5 % con vectores de 16 bits y un 0,9 % con 20 bits (`cargo run --release -p sim --example see_exact -- <modelo> [bits]`). Con todo en la aritmética del die, codificador y modelo de lenguaje, el Qwen3.8-27B ve el disco rojo y responde «Rojo» (`--example see -- <modelo> rojo "<pregunta>" 3 chip`, unos 8 minutos).

### En la app

Con «Aritmética del die», un modelo con codificador lo lleva dentro del chip (`attach_vision`, hasta 256 parches; una imagen mayor se reduce hasta caber). La imagen pasa por el codificador en esa aritmética y el modelo lee sus tokens desde la salida del codificador; la app indica dónde se codificó. En coma flotante, el anfitrión la codifica como antes. Con «El chip completo», el firmware real recibe la imagen por el puerto del anfitrión y ordena sus rutinas (ver abajo); el die, que la app simula como un motor de software, codifica y lee la imagen igual que en «Aritmética del die». Con la cabeza MTP no se admiten imágenes.

## El firmware con una imagen

Un chip que lleva el codificador lo dice en su descripción (`INFO_VISION`, `add_vision`): cuántos parches admite, el token que la imagen sustituye, los bloques, dónde van los píxeles, la posición y la fila de rotación de cada parche, y dónde está la tabla de rotaciones del texto. El CPU puede escribir la memoria vectorial: escribe la dirección en `REG_VECTOR_ADDRESS` y el valor en `REG_VECTOR_DATA`, que entra por el puerto de carga del die un ciclo después, con el die parado y sin logits en camino. Solo lo tiene un chip con visión (`DieFeatures::host_writes`); los de solo texto no cambian.

El protocolo, antes del prompt (`vision_chip::picture_bytes`, cada número en 4 bytes little-endian):

1. Número de parches (0 si no hay imagen), filas y columnas de la rejilla.
2. Por cada parche: el exponente y los elementos de sus píxeles, ya normalizados y en el formato del die; lo mismo de su posición aprendida, ya interpolada; y su fila de la tabla de rotaciones del codificador.

Es la parte del anfitrión, como la de un controlador de cámara. El firmware escribe cada parche y ejecuta su embedding, luego cada bloque dos veces sobre todos los parches, y espera el prompt. Cuando llega el token de la imagen, por cada token de imagen ejecuta la fusión, compone su rotación M-RoPE y lo lee en lugar de un embedding.

**Rotaciones sin coma flotante en el CPU.** La fila de rotación de un token de imagen en (tiempo p, fila p + r, columna p + c) toma cada frecuencia de la fila de la tabla de la posición que le toca: la de p, la de p + r o la de p + c, según las secciones de M-RoPE. El firmware la compone copiando esos elementos con el controlador de carga desde la imagen del chip, donde la tabla está entera y nunca se escribe. El texto que sigue va max(h, w) posiciones después del inicio de la imagen, no h × w: antes de cada token en el hueco s de la caché, el firmware trae desde la imagen la fila s − (h × w − max(h, w)). Como la fuente es siempre la imagen, reescribir filas de la tabla del die no estropea las que se leen después.

Se comprueba en `crates/sim/tests/vision_chip.rs`, con el modelo de juguete y una imagen de 8 × 4 parches (2 × 4 tokens, filas distintas de columnas), texto antes y después y tres tokens generados:

- El firmware, sobre la máquina exacta del die, escribe los mismos tokens que el anfitrión ordenando las mismas rutinas y escribiendo las rotaciones calculadas desde las tres posiciones.
- El chip entero en puertas da los mismos bytes en los mismos ciclos que el simulador de instrucciones (prueba lenta, 1,3 millones de ciclos en minuto y medio).
- El firmware, sobre el die de software que usa la app, escribe los mismos tokens con los mismos logits que el anfitrión llamando a `step_image` con las tres posiciones.

Las pruebas detectan una rotación M-RoPE mal compuesta o un texto sin desplazar: los logits cambian.

**Pesos sin cuantizar en las celdas.** Los pesos del codificador real vienen en BF16 y los del de juguete en coma flotante. Sin un formato de pesos, una matriz así entra en las celdas como una mantisa del ancho de los vectores por peso (16 bits), y las celdas deben guardar pesos de ese ancho. Antes, unas celdas de 8 bits los truncaban en silencio y solo la máquina exacta multiplicaba bien; ahora el `Layout` da un error que pide un formato de pesos o celdas más anchas. El modelo de lenguaje no lo nota porque llega ya cuantizado.

## Dibujar con Janus-Pro

Janus-Pro-1B también genera imágenes: su modelo de lenguaje escribe 576 tokens de un libro de códigos (una rejilla de 24 × 24), guiado lejos de lo que dibujaría sin el texto, y un decodificador convolucional los convierte en píxeles. `cargo run --release -p sim --example draw -- <carpeta> "<descripción>" salida.bmp [temperatura] [semilla]`.

Con `EXACTO=16` (y en la aplicación, con el modelo cargado en la aritmética del chip), corren en la aritmética entera del die las dos copias del modelo de lenguaje, con texto y sin él, y sus dos cabezas de imagen. El compilador añade al programa de Janus una cabeza de imagen (`ImageHead`) con dos listas de operaciones:

- **Puntuaciones**: del estado final, `proj_out` con su sesgo, GELU y `vision_head` con su sesgo, una puntuación por cada uno de los 16 384 códigos del libro.
- **Entrada**: el código leído de `generation_embeddings` como la fila de un embedding, `fc1` con su sesgo, GELU y `hidden_layers.0` con su sesgo, el vector que entra en lugar del embedding del siguiente token. Es el mismo para las dos copias, así que se calcula una vez.

En el anfitrión quedan la mezcla de la guía (dos contextos, dos cachés), el muestreo y el decodificador de píxeles. Un test con un Llama diminuto y cabezas aleatorias comprueba que las puntuaciones y las entradas salen del chip como en coma flotante, dentro del 2 %.

Con «a red apple on a wooden table», semilla 7 y temperatura 1, la coma flotante tarda 387 s y la aritmética del chip 543 s, y las dos dibujan una manzana roja sobre madera. No son la misma imagen: con temperatura 1, una diferencia mínima en un logit cambia qué token se elige, y desde ahí el dibujo sigue otro camino. Un test comprueba que el estado oculto que lee la cabeza de imagen sale del chip igual que en coma flotante, dentro del 2 %.

## Retocar imágenes con Show-o

Janus-Pro dibuja desde un texto, pero no edita. Show-o (`showlab/show-o`, 1 300 M de parámetros) sí: es un único transformador, Phi-1.5, que lee texto y rellena imágenes hechas de códigos de MAGVIT-v2 (`showlab/magvitv2`), un tokenizador convolucional. Para retocar una zona, se enmascaran sus códigos y el transformador los rellena en unos pasos. En cada uno se queda con los códigos de los que está más seguro, guiado por el texto y alejado de lo que haría sin él.

`crates/sim/src/showo.rs` lo reproduce en coma flotante, tal como lo ejecuta el código original.

- **MAGVIT-v2**: un codificador convolucional (bloques residuales con GroupNorm y swish, cuatro reducciones a la mitad y atención en el centro) lleva una imagen de 256 × 256 a 16 × 16 vectores de 13 valores. Cada valor da un bit según su signo, y los 13 bits forman un código de 8 192. El decodificador hace el camino inverso. Codificar y decodificar una imagen da unos 30 dB de PSNR (`crates/sim/tests/showo.rs`).
- **Phi-1.5 con los cambios de Show-o**:
  - atención y MLP en paralelo sobre la misma entrada normalizada;
  - LayerNorm de cada cabeza de consultas y claves antes de rotar;
  - la rotación sobre la mitad de cada cabeza;
  - una atención mixta: el texto, causal; la imagen, entera en ambos sentidos y sobre todo el texto.

  El texto se calcula una vez y su caché sirve para todos los pasos.
- **El relleno**:
  1. En cada paso, cada código desconocido se sortea de las probabilidades del modelo, con la guía `(1 + g)·con texto − g·sin texto`.
  2. Siguen enmascarados los menos seguros, con un ruido de Gumbel cuya temperatura decrece paso a paso. Su número sigue un coseno.
  3. La zona a retocar pasa a la rejilla de códigos con un muestreo bicúbico en sus centros y un umbral de 0,5, como en el original.

Con 16 pasos, una imagen tarda unos 4 minutos en el procesador:

- desde «a red apple on a wooden table», Show-o dibuja la manzana sobre la mesa;
- retocando la zona de la manzana con «a green apple on a wooden table» y guía 5, la manzana pasa a verde y la mesa y el fondo se quedan como estaban.

```
cargo run --release -p sim --example inpaint -- <carpeta> retocar <imagen.bmp|.ppm> x0,y0,x1,y1 "<texto>" salida.bmp [pasos] [guía] [semilla]
cargo run --release -p sim --example inpaint -- <carpeta> dibujar "<texto>" salida.bmp [pasos] [guía] [semilla]
cargo run --release -p sim --example inpaint -- <carpeta> reconstruir <imagen> salida.bmp
```

La carpeta lleva el `pytorch_model.safetensors` y el `config.json` de Show-o. Dentro van `magvitv2` (sus pesos y su configuración) y `phi`, con el tokenizador de Phi-1.5.

### En la app

La pantalla **Imágenes** usa el Show-o que encuentre en las carpetas de modelos:

- **Sin imagen**, dibuja una nueva a partir del texto.
- **Con imagen**, se recorta su cuadrado central y se arrastra sobre ella para marcar la zona que cambia. La zona debe cubrir al menos un código, 1/16 del lado.
- **Seguir retocando** toma el resultado como nueva imagen de partida, para cambiar otra zona.

En el panel se eligen los pasos, la guía y la semilla; con la misma semilla, el mismo texto y la misma zona dan el mismo resultado.

El motor lo ejecuta en segundo plano (`/api/retouch`), un trabajo a la vez, con su progreso paso a paso, y se puede detener entre pasos. Corre en el anfitrión, fuera del chip.

### En la aritmética del chip

`crates/sim/src/showo_chip.rs` lleva el transformador de Show-o a las operaciones del chip y lo ejecuta en la máquina exacta, en el orden en que se las da el anfitrión:

- **El texto** entra como tokens normales, cada uno viendo los anteriores. El relleno a la izquierda no se envía: nada lo lee.
- **La imagen** se ve entera en los dos sentidos, así que pasa cada capa en dos vueltas sobre sus filas, como el codificador de visión:
  1. cada fila guarda su clave y su valor en la caché de la capa, detrás de los del texto, y su consulta;
  2. cada fila atiende sobre el texto y todas las filas de la imagen y hace el resto de la capa.

  La atención y la MLP van en paralelo sobre la misma entrada normalizada, que en la segunda vuelta se vuelve a calcular.
- **Las piezas de Phi** son operaciones que el chip ya tenía:
  - LayerNorm con sesgo, también por cabeza en consultas y claves;
  - la RoPE sobre la mitad de cada cabeza;
  - GELU con tangente;
  - sesgos en todas las proyecciones.
- **Las puntuaciones** salen de las filas de los códigos de la salida, con sus sesgos.
- **MAGVIT-v2** sigue en coma flotante en el anfitrión.

Comprobado con una pasada sobre la misma imagen tapada, frente a la coma flotante:

| Modelo | Diferencia mayor | Mejor código igual |
|---|---|---|
| Show-o de juguete (2 capas) | 0,06 % | 8 de 8 |
| Show-o real, vectores de 16 bits | 1,04 % | 61 de 64 |

Con el modelo real, según el formato de los pesos (`cargo run --release -p sim --example showo_formats -- <carpeta>`):

| Pesos | Diferencia mayor | Mejor código igual | KL media (nats) | Probabilidad del mejor del chip, frente al mejor | Segundos por pasada |
|---|---|---|---|---|---|
| enteros de 16 bits | 1,04 % | 61 de 64 | 0,0007 | 99,9 % | 156 |
| afín de 8 bits, grupos de 64 | 1,32 % | 61 de 64 | 0,0010 | 99,9 % | 77 |
| afín de 4 bits, grupos de 32 | 7,36 % | 37 de 64 | 0,066 | 90,5 % | 73 |

En coma flotante, una pasada tarda 6 s. Al muestreador le importa toda la distribución, no solo el mejor código, y la divergencia KL dice que con 8 bits apenas se mueve. Con 4 bits ya se nota.

Se ejecuta con `crates/sim/tests/showo_chip.rs`. La prueba del modelo real es lenta y se activa aparte. Con la misma semilla, el muestreo de MaskGIT se separa en cuanto un código difiere, así que la comparación se hace pasada a pasada y no sobre la imagen final.

En la pantalla Imágenes, la casilla «En la aritmética del chip» hace todo el retoque así, con pesos afines de 8 bits. Con guía son dos pasadas por paso, de unos 80 s cada una: un retoque de 4 pasos tardó 21 minutos con los enteros de 16 bits.

### MAGVIT-v2 en la aritmética del chip

`crates/sim/src/magvit_chip.rs` lleva el codificador y el decodificador de MAGVIT-v2 a las operaciones del chip, también sobre la máquina exacta guiada desde el anfitrión.

- **Los mapas** son almacenes de filas, una por píxel con todos sus canales, con un borde de filas a cero. Así, las ocho vecinas de un píxel son filas a distancias fijas, sea cual sea el píxel. Una convolución 3 × 3 es un producto de las nueve filas puestas en fila (`LoadRows`) por el núcleo reordenado igual. La reducción a la mitad lee las vecinas píxel sí, píxel no, y el borde hace de relleno.
- **GroupNorm** necesita la media y la dispersión de cada grupo en todo el mapa:
  1. cada píxel escribe sus sumas por grupo y las de sus cuadrados, con un producto por una matriz de unos en los acumuladores anchos de las celdas;
  2. esas filas se suman por parejas, un nivel cada vez, hasta que una tiene los totales;
  3. la operación `GroupNorm` normaliza cada píxel con ellos.
- **La atención del centro** va en dos pasadas sobre los píxeles, como la imagen de Show-o.
- **Duplicar el lado** copia cada fila a sus cuatro sitios.

| Pieza | Juguete | MAGVIT-v2 real | Tiempo real |
|---|---|---|---|
| Decodificador | 62,5 dB de PSNR frente a la coma flotante | 49,6 dB | 195 s |
| Codificador | 208 de 208 bits iguales | 3 306 de 3 328 bits (99,3 %), 236 de 256 códigos | 254 s |

En el codificador, los valores cercanos a cero pueden salir con el otro signo, así que algunos bits cambian.

En la pantalla Imágenes, «También MAGVIT-v2 en el chip» hace todo el retoque en la aritmética del chip.

### En las puertas del die

El Show-o de juguete corre en las puertas del die, rutina a rutina, y da las mismas puntuaciones que la máquina exacta del mismo die, bit a bit: 3,1 millones de ciclos (`show_o_s_routines_on_the_die_s_gates_give_the_exact_machine_s_scores`).

Al probarlo salió un fallo: la tabla de embeddings se dimensionaba con el vocabulario de la salida. En Show-o, la salida tiene 8 192 códigos y la tabla, 58 498 filas. Ahora el programa dice las filas de su tabla (`embedding_rows`).

Pendiente:
- el firmware del RV32 guiando a Show-o y el bucle de MaskGIT en la CPU del chip;
- MAGVIT en puertas.
