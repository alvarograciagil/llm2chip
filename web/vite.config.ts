import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/** The engine writes a new token on every start, so it is read per request. */
function sessionToken(): string {
  try {
    return readFileSync(join(homedir(), ".llm2chip/session-token"), "utf8");
  } catch {
    return "";
  }
}

export default defineConfig({
  plugins: [react()],
  worker: { format: "es" },
  server: {
    port: 5173,
    strictPort: true,
    host: "127.0.0.1",
    proxy: {
      "/api": {
        target: "http://127.0.0.1:7878",
        ws: true,
        configure(proxy) {
          proxy.on("proxyReq", (request) => request.setHeader("x-llm2chip-token", sessionToken()));
          proxy.on("proxyReqWs", (request) => request.setHeader("x-llm2chip-token", sessionToken()));
        },
      },
    },
  },
});
