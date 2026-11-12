# Hacia la fabricación

Qué está verificado del chip, qué reglas debe cumplir la placa y qué falta para fabricarlo.

## Lo que está verificado

| Qué | Cómo | Prueba |
|---|---|---|
| El chip entero (CPU, firmware, cargador, die, muestreador) escribe lo mismo, en los mismos ciclos, a nivel de instrucciones y de puertas | Modelos de juguete, de texto y con imagen | `crates/sim/tests/chip.rs`, `crates/sim/tests/vision_chip.rs` |
| Un simulador ajeno, Icarus Verilog, da lo mismo que el nuestro con el netlist exportado | El chip entero con su firmware, puerta a puerta; con `LLM2CHIP_ICARUS=1` | `text_goes_through_the_whole_chip_in_icarus_verilog` (unos 40 min), `the_whole_chip_reads_a_picture_in_icarus_verilog` (horas) |
| Un modelo real, stories260K, por el chip a nivel de instrucciones | El firmware tokeniza «Once upon a time» y escribe «, there was», lo mismo que la máquina del die dirigida desde el anfitrión y que la coma flotante | `a_real_model_writes_the_same_text_through_the_whole_chip` |
| Bloques, die y celdas en Verilog frente al simulador de puertas | Icarus Verilog, ciclo a ciclo | `crates/netlist/tests` |
| Síntesis y rutado para FPGA | Yosys y nextpnr (ECP5), Gowin y Cyclone IV por síntesis | `docs/fpga.md` |

### Fallo abierto: stories260K en puertas

En puertas, el die de stories260K tal como lo construye el ejemplo FPGA no calcula. Tras guardar el primer token en la caché, el almacén de claves de la capa 0 sigue vacío, la atención da cero y todos los logits salen a cero, así que el chip escribe solo el token 0. La máquina exacta del mismo die lo hace bien. Los modelos de juguete no lo muestran, ni con 8 celdas ni con atención agrupada.

Descartado:
- el diccionario de microcódigo y la tabla de embedding fuera;
- los pasos de escala y sesgo y el ancho del acumulador;
- el ancho de los pesos (8 o 16 bits);
- `key_shift` y las cadenas de productos;
- la carga inicial de la memoria vectorial.

Las pruebas que lo reproducen están detrás de `LLM2CHIP_STORIES_GATES=1`:
- `a_real_model_s_token_on_the_die_matches_the_exact_machine` en `tiny_model.rs`, solo el die, en segundos;
- la comparación con puertas de `a_real_model_writes_the_same_text_through_the_whole_chip`.

Hasta resolverlo, **un chip para un modelo real no está verificado en puertas**.

## Reglas para la placa

- **Reloj único.** El puerto del anfitrión, el de la imagen y los enlaces entre chips son síncronos: válido y listo se leen en el mismo flanco del reloj del chip. Quien esté al otro lado debe usar ese reloj, o una copia en fase, por ejemplo un puente FPGA o un microcontrolador alimentado por la salida de reloj de la placa. Un dispositivo con reloj propio necesita un FIFO de doble reloj fuera del chip.
- **Reset.** El pin puede soltarse en cualquier momento: pasa por dos biestables antes de usarse y después el chip se queda 15 ciclos más en reset (`RESET_RELEASE` = 17 ciclos desde el pin). Antes de enviar bytes al anfitrión hay que esperar a que `booting` baje.
- **JTAG.** TCK, TMS y TDI pasan por dos biestables cada uno. El reloj del chip debe ir al menos 4 veces más rápido que TCK, con cada semiperiodo de TCK de 4 ciclos o más.
- **Pin `follower`.** Es estático: se fija en la placa y no cambia con el chip en marcha.

## Lo que falta para fabricar

- **Un PDK y sus bibliotecas**: celdas estándar en lugar de nuestras puertas genéricas, y las SRAM del compilador de memorias de la fundición, o generadores abiertos como OpenRAM.
- **Colocación, reloj y rutado**, el análisis de tiempos en todas las esquinas de proceso, la potencia y la caída de tensión, y las comprobaciones DRC y LVS hasta el GDSII. Con un PDK abierto (SKY130, GF180 o IHP SG13G2), OpenROAD lo hace.
- **Anillo de pads** con protección contra descargas, y encapsulado.
- **Pruebas de fábrica**: hay autotest de memorias y JTAG, pero no cadenas de scan ni patrones ATPG.
- **El fallo abierto de arriba.**
- **Silicio**: nada se ha probado en una FPGA física ni en un chip.
