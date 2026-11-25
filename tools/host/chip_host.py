#!/usr/bin/env python3
"""A host for an llm2chip chip on a board, over its serial port.

The chip speaks in messages (see `netlist::port` and docs/full-chip.md): a kind byte, the
bytes, and an end byte 0xC0, with 0xC0 and 0xDB inside a message sent as 0xDB 0xDC and
0xDB 0xDD. This script frames what you ask, reads what the chip answers and prints it.

    pip install pyserial
    python3 chip_host.py PORT [--baud N] ask "Once upon a time"
    python3 chip_host.py PORT tokens 1 2 3
    python3 chip_host.py PORT status | options | clear | stop
    python3 chip_host.py PORT set temperature_x256 192

Without a port (`-` instead), it only checks its own framing.
"""

import struct
import sys

END, ESCAPE, ESCAPED_END, ESCAPED_ESCAPE = 0xC0, 0xDB, 0xDC, 0xDD
TEXT, TOKENS, PICTURE, STREAM = 0x01, 0x02, 0x03, 0x04
STATUS, OPTIONS, SET, STOP, CLEAR = 0x10, 0x11, 0x12, 0x13, 0x14
BOOT, READY, ANSWER_END, ERROR = 0x20, 0x21, 0x22, 0x2F
OPTION_NUMBERS = {"max_new_tokens": 5, "greedy": 6, "top_k": 7, "temperature_x256": 8, "seed": 9, "think_budget": 20}
ENDINGS = ["end token", "limit", "loop", "full context", "stopped", "cleared"]
ERRORS = {1: "no plan for this many chips", 2: "unknown message", 3: "not taken now", 4: "no such option", 5: "text too long"}


def frame(kind, payload=b""):
    out = bytearray([kind])
    for byte in payload:
        if byte == END:
            out += bytes([ESCAPE, ESCAPED_END])
        elif byte == ESCAPE:
            out += bytes([ESCAPE, ESCAPED_ESCAPE])
        else:
            out.append(byte)
    out.append(END)
    return bytes(out)


class Reader:
    """Reads the chip's bytes as they come, a whole message at a time."""

    def __init__(self):
        self.kind, self.payload, self.escaping = None, bytearray(), False

    def read(self, data):
        done = []
        for byte in data:
            if self.kind is None:
                if byte not in (END, ESCAPE):
                    self.kind, self.payload = byte, bytearray()
            elif byte == END:
                done.append((self.kind, bytes(self.payload)))
                self.kind, self.escaping = None, False
            elif self.escaping:
                self.payload.append({ESCAPED_END: END, ESCAPED_ESCAPE: ESCAPE}.get(byte, byte))
                self.escaping = False
            elif byte == ESCAPE:
                self.escaping = True
            else:
                self.payload.append(byte)
        return done


def word(payload, at):
    return struct.unpack_from("<I", payload, at)[0] if len(payload) >= at + 4 else 0


def describe(kind, payload):
    """A message from the chip, as a line, and whether it ends what was asked."""
    if kind == TEXT:
        return payload.decode("utf-8", "replace"), False
    if kind == TOKENS:
        return " ".join(str(word(payload, i)) for i in range(0, len(payload), 4)) + " ", False
    if kind == BOOT:
        return f"[boot: {len(payload)} segments, {sum(b != ord('.') for b in payload)} failed]\n", False
    if kind == READY:
        return f"[ready: room for {word(payload, 0)} positions]\n", False
    if kind == ANSWER_END:
        ending = ENDINGS[payload[0]] if payload and payload[0] < len(ENDINGS) else "?"
        position, capacity = word(payload, 1), word(payload, 5)
        share = f", {100 * position / capacity:.1f} % of the context" if capacity else ""
        return f"\n[end: {ending}; {position} positions{share}]\n", True
    if kind == STATUS:
        position, capacity = word(payload, 1), word(payload, 5)
        share = f" ({100 * position / capacity:.1f} %)" if capacity else ""
        state = "answering" if payload[:1] == bytes([1]) else "waiting"
        return f"[{state}; {position} of {capacity} positions{share}]\n", True
    if kind == OPTIONS:
        names = {number: name for name, number in OPTION_NUMBERS.items()}
        pairs = [(payload[i], word(payload, i + 1)) for i in range(0, len(payload) - 4, 5)]
        return "".join(f"{names.get(n, n)} = {v}\n" for n, v in pairs), True
    if kind == ERROR:
        return f"[error: {ERRORS.get(payload[0] if payload else 0, '?')} ({word(payload, 1)})]\n", True
    return f"[message {kind:#x}: {payload.hex()}]\n", False


def request(args):
    what = args[0]
    if what == "ask":
        return frame(TEXT, " ".join(args[1:]).encode())
    if what == "tokens":
        return frame(TOKENS, b"".join(struct.pack("<I", int(t)) for t in args[1:]))
    if what == "set":
        number = OPTION_NUMBERS.get(args[1])
        if number is None:
            sys.exit(f"options: {', '.join(OPTION_NUMBERS)}")
        return frame(SET, bytes([number]) + struct.pack("<I", int(args[2])))
    kinds = {"status": STATUS, "options": OPTIONS, "stop": STOP, "clear": CLEAR}
    if what not in kinds:
        sys.exit(__doc__)
    return frame(kinds[what])


def self_check():
    payload = bytes(range(256)) + bytes([END, ESCAPE])
    wire = frame(STREAM, payload) + frame(STATUS)
    assert END not in wire[1:len(payload) + 3]
    assert Reader().read(wire) == [(STREAM, payload), (STATUS, b"")]
    print("framing: ok")


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    if args[0] == "-":
        return self_check()
    port, args = args[0], args[1:]
    baud = 3_000_000
    if args[:1] == ["--baud"]:
        baud, args = int(args[1]), args[2:]
    message = request(args)
    import serial

    with serial.Serial(port, baud, timeout=0.1) as link:
        link.write(message)
        reader, idle = Reader(), 0
        while idle < 600:
            data = link.read(4096)
            idle = 0 if data else idle + 1
            for kind, payload in reader.read(data):
                line, last = describe(kind, payload)
                print(line, end="", flush=True)
                if last:
                    return


if __name__ == "__main__":
    main()
