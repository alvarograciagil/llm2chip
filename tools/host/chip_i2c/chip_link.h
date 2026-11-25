// The messages of an llm2chip chip's host port, for a microcontroller: framing what to send
// and reading what comes back, a byte at a time. No Arduino dependency, so it builds and is
// checked on a computer too (`check_link.cpp`). See docs/host-links.md.
#pragma once

#include <stddef.h>
#include <stdint.h>

namespace chip {

const uint8_t END = 0xC0, ESCAPE = 0xDB, ESCAPED_END = 0xDC, ESCAPED_ESCAPE = 0xDD;

// From the host.
const uint8_t TEXT = 0x01, TOKENS = 0x02, STATUS = 0x10, OPTIONS = 0x11, SET = 0x12, STOP = 0x13, CLEAR = 0x14, LOAD = 0x15;
// From the chip (it answers TEXT, TOKENS, STATUS and OPTIONS in their own kinds).
const uint8_t BOOT = 0x20, READY = 0x21, ANSWER_END = 0x22, ERROR = 0x2F;

// A message framed into `out`: its length, or 0 if it does not fit in `capacity`.
inline size_t frame(uint8_t kind, const uint8_t* payload, size_t length, uint8_t* out, size_t capacity) {
    size_t at = 0;
    if (capacity < 2) return 0;
    out[at++] = kind;
    for (size_t i = 0; i < length; i++) {
        uint8_t byte = payload[i];
        bool special = byte == END || byte == ESCAPE;
        if (at + (special ? 2 : 1) + 1 > capacity) return 0;
        if (special) {
            out[at++] = ESCAPE;
            out[at++] = byte == END ? ESCAPED_END : ESCAPED_ESCAPE;
        } else {
            out[at++] = byte;
        }
    }
    out[at++] = END;
    return at;
}

// A little-endian 32-bit word of a payload.
inline uint32_t word(const uint8_t* payload, size_t at) {
    return (uint32_t)payload[at] | (uint32_t)payload[at + 1] << 8 | (uint32_t)payload[at + 2] << 16 | (uint32_t)payload[at + 3] << 24;
}

// Reads the chip's bytes as they come. `feed` says when a message is complete; its kind and
// bytes are then in `kind`, `payload` and `length` until the next byte. A message longer
// than the buffer keeps its first bytes and says so in `cut`.
template <size_t CAPACITY>
class Reader {
  public:
    uint8_t kind = 0;
    uint8_t payload[CAPACITY];
    size_t length = 0;
    bool cut = false;

    bool feed(uint8_t byte) {
        if (!inside) {
            // Between messages: ends are padding, and the first other byte is a kind.
            if (byte == END || byte == ESCAPE) return false;
            kind = byte;
            length = 0;
            cut = false;
            escaping = false;
            inside = true;
            return false;
        }
        if (byte == END) {
            inside = false;
            return true;
        }
        if (escaping) {
            escaping = false;
            byte = byte == ESCAPED_END ? END : byte == ESCAPED_ESCAPE ? ESCAPE : byte;
        } else if (byte == ESCAPE) {
            escaping = true;
            return false;
        }
        if (length < CAPACITY) {
            payload[length++] = byte;
        } else {
            cut = true;
        }
        return false;
    }

  private:
    bool inside = false, escaping = false;
};

}  // namespace chip
