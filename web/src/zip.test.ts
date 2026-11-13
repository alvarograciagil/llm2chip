import { describe, expect, it } from "vitest";
import { crc32, makeZip } from "./zip";

describe("zip", () => {
  it("computes the standard CRC-32", () => {
    expect(crc32(new TextEncoder().encode("123456789"))).toBe(0xcbf43926);
  });

  it("records every file in the end of central directory", () => {
    const archive = makeZip({ "rtl/cell.v": "module cell; endmodule\n", "léeme.txt": "ñandú" });
    expect(archive.slice(0, 4)).toEqual(new Uint8Array([0x50, 0x4b, 0x03, 0x04]));
    const end = archive.length - 22;
    expect(archive[end]).toBe(0x50);
    expect(archive[end + 10]).toBe(2);
  });
});
