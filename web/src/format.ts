const decimals = (value: number) => {
  const size = Math.abs(value);
  if (size >= 100 || Number.isInteger(value)) return 0;
  if (size >= 10) return 1;
  if (size >= 0.1) return 2;
  return 3;
};

/** The number format of the interface's language: the page says which, as `writeLanguage` sets it. */
export function locale(): string {
  return typeof document !== "undefined" && document.documentElement.lang === "es" ? "es-ES" : "en-US";
}

export function formatNumber(value: number): string {
  const digits = decimals(value);
  return value.toLocaleString(locale(), { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

/** The time a token takes and the speed it gives, as "1.71 ms/token · 585 tok/s": in
 * milliseconds below a tenth of a second, so short times do not round to a misleading value. */
export function formatTokenTime(seconds: number): string {
  const time = seconds < 0.1 ? `${formatNumber(seconds * 1000)} ms/token` : `${formatNumber(seconds)} s/token`;
  return `${time} · ${formatNumber(1 / seconds)} tok/s`;
}

/** Maps a slider position in [0, 1] onto a logarithmic range. */
export function fromLog(position: number, min: number, max: number): number {
  return min * Math.pow(max / min, position);
}

export function toLog(value: number, min: number, max: number): number {
  return Math.log(value / min) / Math.log(max / min);
}
