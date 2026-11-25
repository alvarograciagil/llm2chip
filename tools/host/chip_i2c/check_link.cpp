// Checks chip_link.h on a computer: c++ -std=c++11 check_link.cpp -o /tmp/check && /tmp/check
#include <assert.h>
#include <stdio.h>

#include "chip_link.h"

int main() {
    uint8_t payload[258];
    for (int i = 0; i < 256; i++) payload[i] = (uint8_t)i;
    payload[256] = chip::END;
    payload[257] = chip::ESCAPE;
    uint8_t wire[600];
    size_t length = chip::frame(chip::SET, payload, sizeof payload, wire, sizeof wire);
    assert(length == 1 + 258 + 4 + 1);
    for (size_t i = 1; i + 1 < length; i++) assert(wire[i] != chip::END);
    assert(chip::frame(chip::SET, payload, sizeof payload, wire, 100) == 0);

    // Padding ends before it, and a second message after it.
    chip::Reader<300> reader;
    int done = 0;
    uint8_t padding[] = {chip::END, chip::END};
    for (uint8_t b : padding) done += reader.feed(b);
    for (size_t i = 0; i < length; i++) {
        if (reader.feed(wire[i])) {
            done++;
            assert(reader.kind == chip::SET && reader.length == sizeof payload && !reader.cut);
            for (size_t j = 0; j < sizeof payload; j++) assert(reader.payload[j] == payload[j]);
        }
    }
    uint8_t status[] = {chip::STATUS, 1, 7, 0, 0, 0, 0, 1, 0, 0, chip::END};
    for (uint8_t b : status) {
        if (reader.feed(b)) {
            done++;
            assert(reader.kind == chip::STATUS && chip::word(reader.payload, 1) == 7 && chip::word(reader.payload, 5) == 256);
        }
    }
    assert(done == 2);

    // A message longer than the buffer keeps its start.
    chip::Reader<4> small;
    for (size_t i = 0; i < length; i++) small.feed(wire[i]);
    assert(small.cut && small.length == 4 && small.payload[3] == 3);
    printf("chip_link: ok\n");
}
