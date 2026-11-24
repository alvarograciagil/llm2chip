// What a link to the app asks for in its address: a view (`#simulator`), and for the
// simulator a model (`#simulator/stories260K`, `#simulator/owner/name`). Read once, at
// start, because the address then carries the design (`#d=…`).
import type { View } from "./store";

const VIEWS: View[] = ["floorplan", "logic", "schematic", "simulator", "images", "listen", "speak", "explorer", "optimize", "models", "numeric", "parameters", "viewer"];

export interface Link {
  view: View;
  /** The simulator's model: a name the demo offers, or a repository on Hugging Face. */
  model: string | null;
}

export function readLink(hash: string): Link | null {
  const [asked, ...rest] = decodeURIComponent(hash.replace(/^#/, "")).split("/");
  // The voice view that listened and spoke at once is now two: an old link listens.
  const view = asked === "voice" ? "listen" : asked;
  if (!VIEWS.includes(view as View)) return null;
  const model = rest.join("/").trim();
  return { view: view as View, model: view === "simulator" && model ? model : null };
}

export const LINK = typeof location === "undefined" ? null : readLink(location.hash);
