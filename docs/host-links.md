# The host port's physical links

A chip speaks to its host in messages (see [the host port: messages](full-chip.md#the-host-port-messages)). This page covers the wires those messages cross. Every chip has a **byte port**. A design can add any of three serial links: a **serial port (UART)**, a **quad SPI target** and an **I2C target**. They are chosen with `host_uart`, `uart_baud`, `host_qspi`, `host_i2c` and `i2c_khz` in the application, and with `netlist::soc::HostLinks` in the generator. All of them carry the same messages, framed the same way.

| Link | Pins | Rate | Flow control | Cost |
|---|---|---|---|---|
| Byte port | `host_rx_valid`, `host_rx_ready`, `host_rx_data` (8), `host_tx_valid`, `host_tx_ready`, `host_tx_data` (8) | A byte a cycle each way | Valid and ready, both ways | Always there |
| Serial port | `uart_rx`, `uart_tx`, `uart_cts_n` | A byte in ten bits at the chosen rate | The chip's CTS line | 242 gates, 49 flip-flops |
| Quad SPI target | `qspi_sck`, `qspi_cs_n`, `qspi_in` (4), `qspi_out` (4), `qspi_oe`, `qspi_irq` | A byte in two clocks, the clock up to an eighth of the chip's | A status command | 227 gates, 42 flip-flops |
| I2C target | `i2c_scl`, `i2c_sda`, `i2c_sda_low`, `i2c_irq` | A byte in nine clocks, the clock up to 1 MHz and a twentieth of the chip's | Room and waiting at the start of each read; a byte that is not taken is not acknowledged | 275 gates, 36 flip-flops |

What each carries, before the framing and the link's own overhead:

| Link | At a chip clock of 1 GHz | At an FPGA's 35 MHz |
|---|---|---|
| Byte port | 1 GB/s | 35 MB/s |
| Serial port at 3 Mbaud | 300 KB/s | 300 KB/s |
| Quad SPI | 62 MB/s | 2.2 MB/s |
| I2C at 1 MHz | 111 KB/s | 111 KB/s (the chip needs 20 MHz or more) |

For scale, text takes a few bytes per token, listening to speech about 64 KB/s and speaking about 192 KB/s. The estimate warns when the fastest serial link a design has is slower than its speech needs.

## What all the links share

**Messages.** A message is its kind (a byte), its bytes and an end byte, `0xC0`. Inside a message `0xC0` goes as `0xDB 0xDC`, and `0xDB` as `0xDB 0xDD`. The port's hardware takes the escapes out on the way in and puts them in on the way out. A host reader starts a message at the first byte that is neither `0xC0` nor `0xDB`, and skips any `0xC0` between messages. `netlist::port` in Rust and `tools/host/chip_host.py` in Python do both.

**Queues.** The port has a receive queue and a send queue of `2^fifo_bits` entries each: 64 in every chip the generator makes today. An entry is one byte with its place in its message: a start (the kind), a byte inside, or the end. An escaped byte takes one entry but two bytes on the wire. So on the way in, `room` entries always hold at least `room` bytes. On the way out, `waiting` entries give at least `waiting` bytes.

**Which link answers.** The chip answers on the link the last message came from. Before any message, it speaks on the first serial link it has, in this order: quad SPI, I2C, serial port. Without any, it uses the byte port. Whatever waits in the send queue goes out on the link the next message names, so a host that speaks first also gets the boot report.

**One link at a time.** A message that starts on one link holds the port until its end. Meanwhile, each of the others behaves this way:

| Link | While another link holds the port |
|---|---|
| Byte port | Waits: its ready is low |
| Serial port | Its bytes are dropped and counted; its CTS line tells the host to wait |
| Quad SPI | Its bytes are dropped and counted |
| I2C | Its bytes are not acknowledged, and the host sends them again |

The same happens when the receive queue is full. `debug_select` = 6 (`DEBUG_HOST_DROPPED`) reads the count of dropped bytes. Two hosts should not both speak to one chip. If two have to, the second should speak only once the first has its answer's end.

**Reading past what waits.** With nothing to send, the quad SPI and I2C targets send `0xC0`, which a reader skips between messages. In the middle of a message, though, that same byte would end it early on the host's side. A host should therefore read no more bytes than the `waiting` count the link gives. Each waiting entry gives at least one byte, so a read of that length never runs dry. If the read stops in the middle of an escape, the rest of it comes in the next one.

**Clocks.** No serial link shares a clock with the chip. Each input pin goes through two flip-flops (I2C's data line through three) before the chip's logic sees it. That is why each link's clock has to be several times slower than the chip's, as set out below.

## The byte port

A plain synchronous port on the chip's own clock, for logic beside the chip: a board's FPGA fabric, a test bench, the browser's simulation.

| Pin | Direction | Meaning |
|---|---|---|
| `host_rx_data` (8) | In | A byte to the chip |
| `host_rx_valid` | In | `host_rx_data` holds a byte |
| `host_rx_ready` | Out | The chip takes it at this clock edge |
| `host_tx_data` (8) | Out | A byte from the chip |
| `host_tx_valid` | Out | `host_tx_data` holds a byte |
| `host_tx_ready` | In | The host takes it at this clock edge |

A byte moves on a rising edge of the chip's clock when its valid and ready are both high, as on an AXI stream. The host may hold valid high with the next byte as long as it likes. `host_rx_ready` is low while the receive queue is full, or while another link holds the port.

## The serial port (UART)

8N1: a start bit (low), eight data bits with the least significant first, a stop bit (high). Both lines idle high. The rate is set in the design as a whole number of the chip's clocks a bit, the divider: `HostLinks::uart`, or `--uart CLOCKS_A_BIT` in the FPGA example.

| Pin | Direction | Meaning |
|---|---|---|
| `uart_rx` | In | The host's transmit line |
| `uart_tx` | Out | The chip's transmit line, high while idle and in reset |
| `uart_cts_n` | Out | Clear to send, low when the host may send: connect it to the host's CTS input |

**Receiving.** The chip sees `uart_rx` two or three cycles late, through its two flip-flops. A fall starts a byte. The start bit is checked half a bit later: if the line is back up, it was noise and the chip goes back to waiting. Then each data bit is sampled in its middle, every divider cycles, and the stop bit too. A byte with a low stop bit is a framing error and is dropped. The receiver is free again from the middle of the stop bit, so the host can send bytes back to back.

**Sending.** Each byte is a start bit, the eight bits and a full stop bit, a divider's worth of cycles each, back to back while the send queue has bytes for this link.

**The rate.** The divider is the chip's clock over the baud rate, rounded. The chip's rate and the host's must agree within what the middle sampling allows. The last bit sampled, the stop bit, sits 9.5 bits after the start bit's fall. It has to land within half a bit, less the two or three cycles the flip-flops take:

`mismatch < (0.5 − 3 / divider) / 9.5`

That is 3.3 % at 16 clocks a bit, 4 % at 25 and 1.3 % at 8. The mismatch adds the rounding of the divider and the error of the host's own rate. For example, 48 MHz / 3 Mbaud is exactly 16, and 50 MHz / 2 Mbaud exactly 25. 50 MHz / 3 Mbaud gives 16.7, which rounds to 17: 2 % slow, within the limit but with little to spare. Use at least 8 clocks a bit, better 16 or more.

**Flow control.** A serial port cannot hold the host back by itself. `uart_cts_n` goes high once three quarters of the receive queue are in use (48 entries of 64, leaving 16 free), and while another link holds the port. A USB serial adapter with hardware flow control (FTDI, CP210x, CH340) stops within a few bytes, so the quarter left over takes them. Without it, the host must not send more than the queue holds while the chip is busy. One case comes up easily: a long question sent while the chip still writes its last answer stays in the queue until the answer ends, and what does not fit is lost.

**A host.** Open the port at the rate, 8N1, with RTS/CTS flow control. Write a message, then read until its answer's end:

```bash
python3 tools/host/chip_host.py /dev/ttyUSB0 --baud 3000000 ask "Once upon a time"
```

## The quad SPI target

SPI mode 0, with four data lines and the most significant nibble first. The host drives the clock and the select.

| Pin | Direction | Meaning |
|---|---|---|
| `qspi_sck` | In | The host's clock, low while idle |
| `qspi_cs_n` | In | Select, low during a transaction |
| `qspi_in` (4) | In | The host's nibbles |
| `qspi_out` (4) | Out | The chip's nibbles |
| `qspi_oe` | Out | The chip drives `qspi_out`: high from the end of a read's dummy byte until the select rises. With shared data lines, it enables the pads' output |
| `qspi_irq` | Out | High while the chip has something to send on this link |

**A transaction.** The select falls. A byte is two clock pulses: the high nibble at the first rising edge, the low nibble at the second. The first byte is a command:

| Command | After it |
|---|---|
| `0x01` write | The host's bytes, into the receive queue, for as long as the select stays low |
| `0x02` read | A dummy byte while the lines turn round. Then the chip's bytes: the send queue's, or `0xC0` with nothing to send |
| `0x05` status | A dummy byte. Then two bytes: the free entries in the receive queue (`room`) and the entries waiting to be sent (`waiting`). Further bytes give `0xC0` |

```
qspi_cs_n  ‾‾\______________________________________________/‾‾
qspi_sck   ____/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_ … ______
qspi_in        [ 0 ][ 2 ] (host lets go)
qspi_out                  (dummy)    [ hi ][ lo ][ hi ][ lo ] …
qspi_oe    _______________________/‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\__
                command 0x02 | dummy  |  byte 1  |  byte 2
```

**Timing.** All of it in the chip's clock cycles:

| What | Requirement |
|---|---|
| The clock | High and low at least four cycles each: a period of at least 8, so at most an eighth of the chip's clock |
| The host's nibble | Stable from a cycle before each rising edge to a cycle after it, as in mode 0 when the host changes it on the falling edge |
| The chip's nibble | Changes 3 or 4 cycles after each rising edge, and holds until the next one: the host reads it at the next rising edge |
| The select | Low at least 3 cycles before the first rising edge, high at least 3 cycles between transactions |
| A read's bytes | The next byte is fetched 3 cycles after the last rising edge of the one before, ready for the next edge. A byte leaves the send queue only once its last nibble has been clocked out, so a read cut short sends it again |

The select's rise ends the transaction: a half-written byte is lost, and the command is forgotten.

**A host.** Poll the status, or wait for the interrupt line:

1. Status: `0x05`, a dummy byte, then `room` and `waiting`.
2. Write: `0x01` and no more than `room` bytes.
3. Read: `0x02`, a dummy byte, then no more than `waiting` bytes.

## The I2C target

Standard I2C, Fast-mode Plus at most, as a target at a 7-bit address set in the design (`HostLinks::i2c`, or `--i2c 42` in hexadecimal in the FPGA example). It shares its bus with other targets.

| Pin | Direction | Meaning |
|---|---|---|
| `i2c_scl` | In | The bus clock |
| `i2c_sda` | In | The bus data line |
| `i2c_sda_low` | Out | Pulls the data line down. The pad must be open drain: low while this is high, and released otherwise. The board's pull-up raises the line |
| `i2c_irq` | Out | High while the chip has something to send on this link. Push-pull, and not part of the bus |

The chip never holds the clock low (no clock stretching), answers no general call and has no 10-bit address.

**A write.** A start, the address with the write bit (0), then the host's bytes, then a stop:

```
S | address·0 A | byte A | byte A | … | byte A | P
```

The chip acknowledges its address. It acknowledges each byte it has put in the receive queue. A byte it cannot take, with another link holding the port or the queue full, is not acknowledged. That byte is lost, and the host sends it again. A host that writes no more than `room` bytes is not refused for a full queue.

**A read.** A start, the address with the read bit (1), then the chip's bytes. The host acknowledges each one but the last:

```
S | address·1 A | room A | waiting A | byte A | … | byte N | P
```

Every read starts with two bytes: the free entries in the receive queue and the entries waiting to be sent. Then come the send queue's bytes, or `0xC0` with nothing to send. A host that only wants the status reads two bytes and does not acknowledge the second. A byte from the queue leaves it only once its last bit has been read, so a read cut short by a stop sends it again next time. A repeated start begins a new transaction.

**Timing.** In the chip's clock cycles:

| What | Requirement |
|---|---|
| The clock | High and low at least 5 cycles each. The bus's own limits (0.26 µs high and 0.5 µs low at 1 MHz) need a chip clock of at least 20 times the bus clock |
| The data line, from the host | The chip reads it about a cycle before the clock's rise as it sees it, so it needs a cycle of setup |
| The data line, from the chip | Changes 3 or 4 cycles after the clock falls, which gives the hold the bus asks for. At 20 MHz that is 150–200 ns, well within the 450 ns Fast-mode Plus allows |
| Start and stop | The data line falls or rises while the clock is high, at least 2 cycles from either clock edge |

The data line passes through one more flip-flop than the clock. So a data line that changes just as the clock falls, which the bus allows, is not taken for a start or a stop. There is no filter for spikes shorter than a cycle: on a noisy bus, put one outside, as Fast-mode asks of 50 ns spikes.

**A host.**

1. Read two bytes: `room` and `waiting`.
2. If `room` > 0, write no more than `room` bytes in one transaction.
3. If `waiting` > 0, read `2 + waiting` bytes and drop the first two.

From Linux, for example a Raspberry Pi with the chip on bus 1 at 0x42:

```bash
python3 tools/host/chip_host.py --i2c 1:0x42 ask "Once upon a time"
```

From a microcontroller, see [`tools/host/chip_i2c/`](../tools/host/chip_i2c).

## Tests

`each_link_is_answered_on_its_own` (in `crates/netlist/tests/links.rs`) runs the control system in gates with the real firmware, all four links at once, and a model of each host on its own clock. It checks:

- The boot report comes out on the quad SPI link.
- A status asked on the serial port is answered there, with its flow control.
- An option asked on the byte port is answered there.
- An option written over quad SPI, and another over I2C, are each read back on their own link, escapes and all.
- I2C refuses another address, and refuses a byte while the byte port holds the port.
- A serial byte sent while another link holds the port is dropped and counted.
