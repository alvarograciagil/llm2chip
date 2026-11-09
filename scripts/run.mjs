// Runs a project script with the shell of this system: `node run.mjs build-wasm`.
import { spawnSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const name = process.argv[2];
const [command, args] = process.platform === "win32" ? ["powershell", ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", join(here, `${name}.ps1`)]] : ["bash", [join(here, `${name}.sh`)]];
const result = spawnSync(command, args, { stdio: "inherit" });
process.exit(result.status ?? 1);
