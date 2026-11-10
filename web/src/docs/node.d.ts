// The app carries no Node types; the documentation test only reads and writes one file.
declare module "node:fs" {
  export function readFileSync(path: URL, encoding: "utf8"): string;
  export function writeFileSync(path: URL, text: string): void;
}

declare const process: { env: Record<string, string | undefined> };
