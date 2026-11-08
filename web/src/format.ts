const decimals = (value: number) => {
  const size = Math.abs(value);
  if (size >= 100 || Number.isInteger(value)) return 0;
  if (size >= 10) return 1;
  if (size >= 0.1) return 2;
  return 3;
};

export function formatNumber(value: number): string {
  const digits = decimals(value);
  return value.toLocaleString("es-ES", { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

/** Maps a slider position in [0, 1] onto a logarithmic range. */
export function fromLog(position: number, min: number, max: number): number {
  return min * Math.pow(max / min, position);
}

export function toLog(value: number, min: number, max: number): number {
  return Math.log(value / min) / Math.log(max / min);
}
