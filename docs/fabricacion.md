# Hacia la fabricación

Qué está verificado del chip, qué reglas debe cumplir la placa y qué falta para fabricarlo.

## Lo que está verificado

| Qué | Cómo | Prueba |
|---|---|---|
| El chip entero (CPU, firmware, cargador, die, muestreador) escribe lo mismo, en los mismos ciclos, a nivel de instrucciones y de puertas | Modelos de juguete, de texto y con imagen | `crates/sim/tests/chip.rs`, `crates/sim/tests/vision_chip.rs` |
| Un simulador ajeno, Icarus Verilog, da lo mismo que el nuestro con el netlist exportado | El chip entero con su firmware, puerta a puerta; con `LLM2CHIP_ICARUS=1` | `text_goes_through_the_whole_chip_in_icarus_verilog` (226 632 ciclos, unos 40 min), `the_whole_chip_reads_a_picture_in_icarus_verilog` (1 296 777 ciclos, unas 4 h) |
| Un modelo real, stories260K, por el chip entero, a nivel de instrucciones y de puertas | El firmware tokeniza «Once upon a time» y escribe «, there was», lo mismo que la máquina del die dirigida desde el anfitrión y que la coma flotante; las puertas, en los mismos 1 909 169 ciclos | `a_real_model_writes_the_same_text_through_the_whole_chip` (unos 4 min), y solo el die en `a_real_model_s_token_on_the_die_matches_the_exact_machine` |
| Bloques, die y celdas en Verilog frente al simulador de puertas | Icarus Verilog, ciclo a ciclo | `crates/netlist/tests` |
| Síntesis y rutado para FPGA | Yosys y nextpnr (ECP5), Gowin y Cyclone IV por síntesis | `docs/fpga.md` |
| Cadena de scan y patrones de prueba de fábrica | Los patrones del ATPG, aplicados en el simulador de puertas como lo haría un probador, dan lo previsto; con un fallo inyectado, fallan si el ATPG dijo que lo encontraban y no fallan si dijo que era imposible. El Verilog con scan da lo mismo en Icarus | `crates/netlist/tests/scan.rs` |

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

## Pruebas de fábrica: scan y ATPG

Cada chip que sale de la oblea hay que probarlo: un defecto deja un cable clavado a 0 o a 1. Para eso hacen falta dos piezas:
- **La cadena de scan** (`netlist::scan::insert_scan`): une todos los biestables en un registro de desplazamiento, de `scan_in` a `scan_out`.
  - Con `scan_enable`, cada ciclo desplaza la cadena un puesto: el probador carga cualquier estado y lee lo que el chip guarda.
  - Sin él, un ciclo de reloj captura lo que calcula la lógica.
  - `test_mode` pone cada memoria detrás de una sombra: la lógica lee biestables de la cadena en lugar de la memoria, y esos biestables capturan una mezcla de la dirección, los datos y la escritura que entran en ella. El contenido de las memorias lo prueba el autotest de marcha.
  - Con las tres señales a 0, el circuito funciona igual que sin cadena (comprobado ciclo a ciclo), con un multiplexor más delante de cada biestable.
- **Los patrones** (`netlist::atpg::generate`): con la cadena, probar el circuito secuencial se reduce a probar la lógica entre biestables.
  1. Primero, patrones al azar, 64 a la vez, simulando todos los fallos.
  2. Para lo que queda, PODEM busca un patrón fallo a fallo. Elige los objetivos con medidas SCOAP y descarta pronto los caminos por los que la diferencia ya no puede salir.
  3. Si la búsqueda se agota, el fallo es imposible de detectar: la lógica en ese punto es redundante.

  Los valores tienen tres estados (0, 1 y desconocido) para lo que sale de las memorias, que la prueba no controla.

La cuenta de fallos es la clásica: cada señal clavada en su origen y cada entrada de puerta donde la señal se reparte. Los fallos del camino de desplazamiento los detecta la prueba de la cadena, que pasa un `0011…` por ella.

`cargo run --release -p netlist --example atpg` da esta tabla:

| Bloque | Biestables en la cadena | Fallos | Detectados | Por la cadena | Imposibles | Abandonados | Patrones | Cobertura de fallos | Cobertura de prueba | Segundos |
|---|---|---|---|---|---|---|---|---|---|---|
| celda | 132 | 7134 | 5436 | 405 | 1293 | 0 | 113 + 7 | 81.88 % | 100.00 % | 0.0 |
| cpu | 236 | 24596 | 21874 | 711 | 1866 | 145 | 272 + 361 | 91.82 % | 99.36 % | 21.7 |
| muestreo | 363 | 48226 | 32872 | 1090 | 13459 | 805 | 418 + 866 | 70.42 % | 97.68 % | 147.0 |

Los bloques son la celda de pesos de 4 bits y activaciones de 16 con 32 pesos, la CPU RV32 y el muestreador de logits de 16 bits con 8 candidatos. Un die pequeño entero (tres celdas, el secuenciador, la unidad vectorial y las tablas) tarda mucho más con la búsqueda actual, que vuelve a simular todo el circuito en cada decisión: `cargo run --release -p netlist --example atpg -- die`.

- **Cobertura de prueba** es la medida habitual: los fallos encontrados entre los que algún patrón puede encontrar.
- **Cobertura de fallos** cuenta también los imposibles, que aquí son muchos. No son defectos de la prueba, sino lógica sobrante del netlist, que no se optimiza:
  - la primera fila del multiplicador suma un cero;
  - el acarreo de salida de muchos sumadores no se usa;
  - las comparaciones del muestreador usan un sumador entero y solo leen su acarreo.

  La síntesis con Yosys aplana el circuito y quita esa lógica. Medido con las memorias como cajas negras (`synth -flatten`, `abc -g AND,OR,XOR,MUX`):

  | Bloque | Celdas nuestras | Tras Yosys | Diferencia | Fallos imposibles |
  |---|---|---|---|---|
  | celda | 1 380 | 972 | −30 % | 22 % |
  | cpu | 4 151 | 3 459 | −17 % | 7 % |
  | muestreo | 9 121 | 5 698 | −38 % | 25 % |

  La reducción va con la proporción de fallos imposibles. Parte de ella es otra cosa: Yosys mete el multiplexor de habilitación dentro del biestable. Falta medir la cobertura sobre el netlist sintetizado, para lo que hay que leer el Verilog que escribe Yosys.

Una prueba de scan escribe en las memorias al desplazar, igual que el autotest de marcha, así que se hace antes de cargar la imagen.

**El die entero** no se prueba aplanado en una sola lista de fallos: sus celdas son idénticas y se prueban a la vez (`scan::insert_scan_parallel`).
- **Celda comprobable:** cada celda pasa a una versión cuyas entradas, en modo de prueba, salen de un envoltorio: un registro por unidad vectorial, compartido por sus celdas, con tantos biestables como bits de entrada tiene la celda.
- **Cadenas paralelas:** salen todas de `scan_cells_in`, tras el envoltorio, así que el mismo desplazamiento carga todas las celdas igual.
  - `scan_cells_out` da la cadena de la primera celda.
  - `scan_cells_differ` sube si la de alguna otra difiere de ella.
- **La salida de la celda** ya es un registro (`total_reg`) dentro de su cadena: no necesita envoltorio de salida.
- **Los patrones** se generan sobre una porción: el envoltorio y la celda comprobable, encadenados como en el die.

En el die de prueba (tres celdas):

| | |
|---|---|
| Cadena principal (el resto del die) | 1 304 biestables |
| Cadena de cada celda | 367 biestables, 51 de ellos del envoltorio |
| Fallos de la celda | 101 486, cobertura de prueba del 98,0 % con 571 patrones |

Comprobado en el simulador de puertas, con los patrones de la celda aplicados a las tres a la vez:
- sin fallos, la primera celda da lo previsto y ninguna difiere;
- con un fallo en la tercera celda, sube la diferencia;
- con un fallo en la primera, además, su salida no es la prevista.

Con las cadenas apagadas, el die hace lo mismo ciclo a ciclo.

**El coste**:
- **Por celda:** un multiplexor por bit de entrada (51) y uno por biestable de la cadena. No hay biestables nuevos en la celda, salvo las sombras de sus memorias.
- **Por unidad vectorial:** 51 biestables del envoltorio.
- **El tiempo de prueba de cada patrón** es lo que tarda en desplazarse la cadena más larga: unos 1 300 ciclos aquí, haya las celdas que haya. Con una sola cadena en serie, un die de 100 000 celdas de unos 300 biestables cada una necesitaría unos 30 millones de ciclos por patrón.

Las señales de prueba son ahora pines propios. La alternativa que ahorra pines es manejar la cadena desde el TAP JTAG con instrucciones de scan interno.

## Lo que falta para fabricar

- **Un PDK y sus bibliotecas**: celdas estándar en lugar de nuestras puertas genéricas, y las SRAM del compilador de memorias de la fundición, o generadores abiertos como OpenRAM.
- **Colocación, reloj y rutado**, el análisis de tiempos en todas las esquinas de proceso, la potencia y la caída de tensión, y las comprobaciones DRC y LVS hasta el GDSII. Con un PDK abierto (SKY130, GF180 o IHP SG13G2), OpenROAD lo hace.
- **Anillo de pads** con protección contra descargas, y encapsulado.
- **Pruebas de fábrica**:
  - hay autotest de memorias, JTAG, cadena de scan y patrones ATPG por bloque;
  - falta meter la cadena en el chip por defecto, con sus pines o detrás del TAP;
  - falta el formato de los probadores, STIL o WGL;
  - faltan los fallos de retardo (transición) a velocidad de reloj.
- **Silicio**: nada se ha probado en una FPGA física ni en un chip.
