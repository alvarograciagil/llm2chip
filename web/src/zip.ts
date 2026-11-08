/** Writes an uncompressed zip archive. */
const encoder = new TextEncoder();

const crcTable = Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});

export function crc32(data: Uint8Array): number {
  let c = 0xffffffff;
  for (const byte of data) c = crcTable[(c ^ byte) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function writer(bytes: number[]) {
  return {
    u16: (v: number) => bytes.push(v & 0xff, (v >>> 8) & 0xff),
    u32: (v: number) => bytes.push(v & 0xff, (v >>> 8) & 0xff, (v >>> 16) & 0xff, (v >>> 24) & 0xff),
  };
}

export function makeZip(files: Record<string, string>): Uint8Array {
  const body: number[] = [];
  const central: number[] = [];
  const entries = Object.entries(files);
  const out = writer(body);
  const dir = writer(central);
  const UTF8 = 0x0800;
  for (const [name, text] of entries) {
    const nameBytes = encoder.encode(name);
    const data = encoder.encode(text);
    const crc = crc32(data);
    const offset = body.length;
    out.u32(0x04034b50);
    out.u16(20);
    out.u16(UTF8);
    out.u16(0);
    out.u32(0);
    out.u32(crc);
    out.u32(data.length);
    out.u32(data.length);
    out.u16(nameBytes.length);
    out.u16(0);
    body.push(...nameBytes);
    for (const byte of data) body.push(byte);
    dir.u32(0x02014b50);
    dir.u16(20);
    dir.u16(20);
    dir.u16(UTF8);
    dir.u16(0);
    dir.u32(0);
    dir.u32(crc);
    dir.u32(data.length);
    dir.u32(data.length);
    dir.u16(nameBytes.length);
    dir.u16(0);
    dir.u16(0);
    dir.u16(0);
    dir.u16(0);
    dir.u32(0);
    dir.u32(offset);
    central.push(...nameBytes);
  }
  const start = body.length;
  const end: number[] = [];
  const tail = writer(end);
  tail.u32(0x06054b50);
  tail.u16(0);
  tail.u16(0);
  tail.u16(entries.length);
  tail.u16(entries.length);
  tail.u32(central.length);
  tail.u32(start);
  tail.u16(0);
  const archive = new Uint8Array(start + central.length + end.length);
  archive.set(body, 0);
  archive.set(central, start);
  archive.set(end, start + central.length);
  return archive;
}
