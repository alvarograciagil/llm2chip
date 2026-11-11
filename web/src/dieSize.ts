/** The largest die a process takes: the scanner's reticle, or less where the open shuttles
 * of that process offer less room. */
export function dieLimit(nodeMax: number | undefined): number {
  return Math.min(858, nodeMax && nodeMax > 0 ? nodeMax : 858);
}

/** A die of `width` × `height` mm shrunk, keeping its shape, until it fits `limit` mm². */
export function fitDie(width: number, height: number, limit: number): { die_width_mm: number; die_height_mm: number; die_max_mm2: number } {
  const scale = Math.min(1, Math.sqrt(limit / (width * height)));
  const floor = (v: number) => Math.max(0.1, Math.floor(v * scale * 10) / 10);
  const [w, h] = scale < 1 ? [floor(width), floor(height)] : [width, height];
  return { die_width_mm: w, die_height_mm: h, die_max_mm2: Math.round(w * h * 100) / 100 };
}
