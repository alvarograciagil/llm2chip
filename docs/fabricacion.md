# Hacia la fabricación

Qué está verificado del chip, qué reglas debe cumplir la placa y qué falta para fabricarlo.

## Lo que está verificado

| Qué | Cómo | Prueba |
|---|---|---|
| El chip entero (CPU, firmware, cargador, die, muestreador) escribe lo mismo, en los mismos ciclos, a nivel de instrucciones y de puertas | Modelos de juguete, de texto y con imagen | `crates/sim/tests/chip.rs`, `crates/sim/tests/vision_chip.rs` |
| Un simulador ajeno, Icarus Verilog, da lo mismo que el nuestro con el netlist exportado | El chip entero con su firmware, puerta a puerta; con `LLM2CHIP_ICARUS=1` | `text_goes_through_the_whole_chip_in_icarus_verilog` (unos 40 min), `the_whole_chip_reads_a_picture_in_icarus_verilog` (horas) |
| Un modelo real, stories260K, por el chip entero, a nivel de instrucciones y de puertas | El firmware tokeniza «Once upon a time» y escribe «, there was», lo mismo que la máquina del die dirigida desde el anfitrión y que la coma flotante; las puertas, en los mismos 1 909 169 ciclos | `a_real_model_writes_the_same_text_through_the_whole_chip` (unos 4 min), y solo el die en `a_real_model_s_token_on_the_die_matches_the_exact_machine` |
| Bloques, die y celdas en Verilog frente al simulador de puertas | Icarus Verilog, ciclo a ciclo | `crates/netlist/tests` |
| Síntesis y rutado para FPGA | Yosys y nextpnr (ECP5), Gowin y Cyclone IV por síntesis | `docs/fpga.md` |

### Un fallo que solo salía con un modelo real

stories260K fue el primer modelo cuyo microprograma pasa de 2^13 palabras (9 813), y en puertas su die calculaba ceros.

La causa estaba en cómo viaja un inmediato:
- **Dónde va:** un valor fijo de la instrucción viaja en el campo del destino de salto, y el die extiende su signo desde el último bit de ese campo.
- **Por qué se rompía:** el microprograma se ensamblaba para un die de prueba con destinos de 13 bits y se cargaba en el die final, de 14. Un −15 guardado en 13 bits (0x1FF1) se leía en 14 como +8 177.
- **Qué lo destapó:** la RoPE de cada cabeza resta 15 al exponente, así que el de las claves salía 8 192 más alto y la caché las guardaba como ceros.

Ahora el inmediato se guarda con todo su signo y cada die toma los bits que tiene su campo (`a_negative_immediate_keeps_its_sign_on_a_die_with_wider_jump_targets`).

Por el camino salió otro fallo, este del simulador de instrucciones: con la tabla de embedding en la imagen, leía siempre la fila del token 0.

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
- **Silicio**: nada se ha probado en una FPGA física ni en un chip.
