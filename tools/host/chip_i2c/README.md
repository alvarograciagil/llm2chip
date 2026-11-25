# A microcontroller host over I2C

`chip_i2c.ino` is an Arduino sketch that speaks to an llm2chip chip over its I2C target. It works on any board with `Wire`: AVR, ESP32, RP2040 and others. Type a question in the serial monitor at 115,200 baud, and the answer comes back as the chip writes it. Lines that start with `/` are commands:

| Line | Message |
|---|---|
| `/status` | Where the conversation stands: positions in use, capacity, percentage |
| `/options` | The generation's options and their values |
| `/set NUMBER VALUE` | Sets an option (5 most tokens, 6 greedy, 7 top-k, 8 temperature × 256, 9 seed, 20 reasoning budget) |
| `/stop` | Ends the answer being written |
| `/clear` | Clears the conversation |

**Wiring.** Connect SDA and SCL to the chip's `i2c_sda` and `i2c_scl`, with its `i2c_sda_low` driving an open-drain pad. Put pull-ups on both lines, and join the grounds. Wiring the chip's `i2c_irq` to a pin and setting `IRQ_PIN` saves polling.

**Settings** at the top of the sketch:

- `ADDRESS`: the chip's address.
- `BUS_HZ`: the bus clock. The chip's clock must be at least 20 times faster.
- `CHUNK`: what one `Wire` transaction moves, 32 bytes on AVR.

**What it does on the bus**, as [docs/host-links.md](../../../docs/host-links.md#the-i2c-target) sets out:

- Every read starts with two bytes, the chip's free room and what waits to be sent.
- The sketch writes no more than the room.
- It reads no more than what waits, because past that the chip sends `0xC0`, which would cut a message still being written.

`chip_link.h` holds the framing and the reader. It has no Arduino dependency: `c++ -std=c++11 check_link.cpp -o check && ./check` checks it on a computer.
