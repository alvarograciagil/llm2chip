// The app carries no Node types; the tests only read and write a few files.
declare module "node:fs" {
  export function readFileSync(path: URL | string, encoding: "utf8"): string;
  export function writeFileSync(path: URL | string, text: string): void;
  export function appendFileSync(path: string, text: string): void;
  export function readdirSync(path: string): string[];
  export function statSync(path: string): { isDirectory(): boolean };
}

declare module "node:path" {
  export function join(...parts: string[]): string;
  export function relative(from: string, to: string): string;
}

declare const process: { env: Record<string, string | undefined> };
