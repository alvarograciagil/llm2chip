import { randomBytes, timingSafeEqual } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import type { IncomingMessage, ServerResponse } from "node:http";
import { homedir, networkInterfaces } from "node:os";
import { join } from "node:path";
import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

const STATE = join(homedir(), ".llm2chip");
const LAN = process.env.LLM2CHIP_LAN === "1";
const PORT = Number(process.env.LLM2CHIP_WEB_PORT) || 5173;
const COOKIE = "llm2chip_lan";

/** The engine writes a new token on every start, so it is read per request. */
function sessionToken(): string {
  try {
    return readFileSync(join(STATE, "session-token"), "utf8");
  } catch {
    return "";
  }
}

/** The key other devices need. Set it yourself, or one is generated and kept. */
function lanKey(): string {
  if (process.env.LLM2CHIP_LAN_KEY) return process.env.LLM2CHIP_LAN_KEY;
  const path = join(STATE, "lan-key");
  if (existsSync(path)) return readFileSync(path, "utf8").trim();
  mkdirSync(STATE, { recursive: true });
  const key = randomBytes(16).toString("hex");
  writeFileSync(path, key, { mode: 0o600 });
  return key;
}

function sameKey(given: string | null | undefined, key: string): boolean {
  if (!given) return false;
  const [a, b] = [Buffer.from(given), Buffer.from(key)];
  return a.length === b.length && timingSafeEqual(a, b);
}

function isLocal(request: IncomingMessage): boolean {
  return ["127.0.0.1", "::1", "::ffff:127.0.0.1"].includes(request.socket.remoteAddress ?? "");
}

function cookieKey(request: IncomingMessage): string | undefined {
  return request.headers.cookie?.split("; ").find((part) => part.startsWith(`${COOKIE}=`))?.slice(COOKIE.length + 1);
}

/** Other devices must present the key once; a cookie then remembers it. */
function lanAccess(): Plugin {
  const key = lanKey();
  return {
    name: "llm2chip-lan-access",
    configureServer(server) {
      server.middlewares.use((request: IncomingMessage, response: ServerResponse, next: () => void) => {
        if (isLocal(request) || sameKey(cookieKey(request), key)) return next();
        const url = new URL(request.url ?? "/", "http://lan");
        if (sameKey(url.searchParams.get("key"), key)) {
          url.searchParams.delete("key");
          response.statusCode = 302;
          response.setHeader("set-cookie", `${COOKIE}=${key}; HttpOnly; SameSite=Strict; Path=/`);
          response.setHeader("location", url.pathname + url.search);
          return response.end();
        }
        response.statusCode = 401;
        response.setHeader("content-type", "text/plain; charset=utf-8");
        response.end("An access key is needed: open the full address shown by the machine running llm2chip.\n");
      });
      // WebSockets do not pass through the middlewares above.
      server.httpServer?.prependListener("upgrade", (request: IncomingMessage, socket) => {
        if (!isLocal(request) && !sameKey(cookieKey(request), key)) socket.destroy();
      });
      server.httpServer?.once("listening", () => {
        const address = Object.values(networkInterfaces()).flat().find((a) => a?.family === "IPv4" && !a.internal)?.address ?? "localhost";
        const link = `http://${address}:${PORT}/?key=${key}`;
        writeFileSync(join(STATE, "lan-url"), `${link}\n`, { mode: 0o600 });
        console.log(`\n  Red local: ${link}\n`);
      });
    },
  };
}

/** The engine only trusts local pages, so a page served here is vouched for by the proxy. */
function vouch(outgoing: { setHeader(name: string, value: string): void }, incoming: IncomingMessage) {
  outgoing.setHeader("x-llm2chip-token", sessionToken());
  const origin = incoming.headers.origin;
  if (LAN && origin && new URL(origin).host === incoming.headers.host) outgoing.setHeader("origin", `http://127.0.0.1:${PORT}`);
}

export default defineConfig({
  plugins: [react(), ...(LAN ? [lanAccess()] : [])],
  worker: { format: "es" },
  server: {
    port: PORT,
    strictPort: true,
    host: LAN ? "0.0.0.0" : "127.0.0.1",
    allowedHosts: LAN ? [".local"] : undefined,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:7878",
        ws: true,
        configure(proxy) {
          proxy.on("proxyReq", (outgoing, incoming) => vouch(outgoing, incoming));
          proxy.on("proxyReqWs", (outgoing, incoming) => vouch(outgoing, incoming));
        },
      },
    },
  },
});
