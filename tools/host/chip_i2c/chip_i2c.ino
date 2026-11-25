// A host for an llm2chip chip over I2C, on any Arduino-compatible board (AVR, ESP32,
// RP2040...). Type a question in the serial monitor and the chip's answer comes back as it
// is written. Lines starting with / are commands: /status, /options, /clear, /stop,
// /set NUMBER VALUE. See docs/host-links.md for the link and its rules.
//
// Wiring: SDA and SCL to the chip's i2c_sda and i2c_scl (its i2c_sda_low on an open-drain
// pad), pull-ups on both lines, common ground. Optionally, the chip's i2c_irq to IRQ_PIN.

#include <Wire.h>

#include "chip_link.h"

const uint8_t ADDRESS = 0x42;     // the chip's address, as its design sets it
const uint32_t BUS_HZ = 400000;   // 100 kHz, 400 kHz or 1 MHz, if the board and the chip allow
const int IRQ_PIN = -1;           // a pin wired to i2c_irq, or -1 to poll
// What one Wire transaction moves: 32 bytes on AVR, more on most other boards.
const size_t CHUNK = 32;

chip::Reader<64> reader;
bool waiting_answer = false;

void receive();

// Two bytes at the start of every read: the chip's room and what waits to be sent.
bool status(uint8_t& room, uint8_t& waiting) {
    if (Wire.requestFrom(ADDRESS, (uint8_t)2) != 2) return false;
    room = Wire.read();
    waiting = Wire.read();
    return true;
}

// Sends a framed message, never more than the chip has room for.
bool send(const uint8_t* bytes, size_t length) {
    while (length > 0) {
        uint8_t room, waiting;
        if (!status(room, waiting)) return false;
        if (room == 0) {
            receive();  // the chip may be waiting for us to read
            continue;
        }
        size_t part = length < room ? length : room;
        if (part > CHUNK) part = CHUNK;
        Wire.beginTransmission(ADDRESS);
        Wire.write(bytes, part);
        // 3: a byte was refused, another link holds the port. How many went is unknown.
        if (Wire.endTransmission() != 0) return false;
        bytes += part;
        length -= part;
    }
    return true;
}

void message(uint8_t kind, const uint8_t* payload, size_t length) {
    static uint8_t framed[2 * 128 + 2];
    size_t size = chip::frame(kind, payload, length, framed, sizeof framed);
    if (size == 0) {
        Serial.println(F("[too long]"));
    } else if (!send(framed, size)) {
        Serial.println(F("[the chip did not take it]"));
    } else {
        waiting_answer = true;
    }
}

// Prints what a message from the chip says.
void show() {
    const uint8_t* p = reader.payload;
    switch (reader.kind) {
        case chip::TEXT:
            Serial.write(p, reader.length);
            break;
        case chip::BOOT:
            Serial.print(F("[boot: "));
            Serial.print(reader.length);
            Serial.println(F(" segments]"));
            break;
        case chip::READY:
            Serial.print(F("[ready: "));
            Serial.print(chip::word(p, 0));
            Serial.println(F(" positions]"));
            break;
        case chip::ANSWER_END:
        case chip::STATUS: {
            uint32_t position = chip::word(p, 1), capacity = chip::word(p, 5);
            Serial.print(reader.kind == chip::STATUS ? (p[0] ? F("\n[answering; ") : F("\n[waiting; ")) : F("\n[end; "));
            Serial.print(position);
            Serial.print(F(" of "));
            Serial.print(capacity);
            if (capacity) {
                Serial.print(F(" positions, "));
                Serial.print(100.0 * position / capacity, 1);
                Serial.print(F(" %"));
            }
            Serial.println(F("]"));
            waiting_answer = false;
            break;
        }
        case chip::OPTIONS:
            for (size_t i = 0; i + 5 <= reader.length; i += 5) {
                Serial.print(p[i]);
                Serial.print(F(" = "));
                Serial.println(chip::word(p, i + 1));
            }
            waiting_answer = false;
            break;
        case chip::ERROR:
            Serial.print(F("\n[error "));
            Serial.print(p[0]);
            Serial.print(F(", "));
            Serial.print(chip::word(p, 1));
            Serial.println(F("]"));
            waiting_answer = false;
            break;
    }
}

// Reads what waits, never more: past it the chip sends ends, which would cut a message
// that is still being written.
void receive() {
    uint8_t room, waiting;
    if (!status(room, waiting) || waiting == 0) return;
    size_t count = 2 + (size_t)waiting;
    if (count > CHUNK) count = CHUNK;
    Wire.requestFrom(ADDRESS, (uint8_t)count);
    Wire.read();
    Wire.read();
    while (Wire.available()) {
        if (reader.feed(Wire.read())) show();
    }
}

void setup() {
    Serial.begin(115200);
    Wire.begin();
    Wire.setClock(BUS_HZ);
    if (IRQ_PIN >= 0) pinMode(IRQ_PIN, INPUT);
    Serial.println(F("llm2chip over I2C: type a question, or /status, /options, /clear, /stop, /set N V"));
}

void loop() {
    static char line[128];
    static size_t length = 0;
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\r') continue;
        if (c != '\n') {
            if (length < sizeof line - 1) line[length++] = c;
            continue;
        }
        line[length] = 0;
        String text(line);
        length = 0;
        if (text == "/status") {
            message(chip::STATUS, nullptr, 0);
        } else if (text == "/options") {
            message(chip::OPTIONS, nullptr, 0);
        } else if (text == "/clear") {
            message(chip::CLEAR, nullptr, 0);
        } else if (text == "/stop") {
            message(chip::STOP, nullptr, 0);
        } else if (text.startsWith("/set ")) {
            int space = text.indexOf(' ', 5);
            uint8_t payload[5];
            uint32_t value = (uint32_t)text.substring(space + 1).toInt();
            payload[0] = (uint8_t)text.substring(5, space).toInt();
            for (int i = 0; i < 4; i++) payload[1 + i] = value >> (8 * i);
            message(chip::SET, payload, 5);
        } else if (text.length() > 0) {
            message(chip::TEXT, (const uint8_t*)line, text.length());
        }
    }
    // With the interrupt line, read only when it says there is something; without it, poll.
    if (IRQ_PIN < 0 || digitalRead(IRQ_PIN) == HIGH) {
        receive();
    }
    if (IRQ_PIN < 0) delay(waiting_answer ? 1 : 20);
}
