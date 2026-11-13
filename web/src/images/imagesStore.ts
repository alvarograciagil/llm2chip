import { create } from "zustand";
import type { ServerModel } from "../simulator/simulatorStore";

/** Side, in pixels, of the square a picture is cropped to before it goes to the engine. */
export const PICTURE_SIDE = 512;

/** A square picture, RGB row by row, and how it shows. */
export interface SquarePicture {
  side: number;
  rgb: number[];
  preview: string;
}

/** A rectangle on the picture: left, top, right, bottom, in its pixels. */
export type Region = [number, number, number, number];

interface Result {
  data: string;
  side: number;
  seconds: number;
  seed: number;
  /** Bits of the die's vectors when the chip's arithmetic made it. */
  chip_bits?: number;
  /** MAGVIT-v2 ran on the chip too. */
  magvit_on_chip?: boolean;
}

interface ImagesState {
  online: boolean | null;
  models: ServerModel[];
  path: string;
  prompt: string;
  steps: number;
  guidance: number;
  /** Empty draws a new one each time. */
  seed: string;
  /** Run the transformer in the die's arithmetic, much slower. */
  onChip: boolean;
  /** MAGVIT-v2 on the chip too: minutes more, to encode and to decode. */
  magvitOnChip: boolean;
  picture: SquarePicture | null;
  region: Region | null;
  running: boolean;
  phase: string;
  step: number;
  total: number;
  result: Result | null;
  error: string | null;
  set: (partial: Partial<ImagesState>) => void;
  connect: () => Promise<void>;
  start: () => Promise<void>;
  cancel: () => Promise<void>;
}

/** The largest centred square of `file`, scaled to `PICTURE_SIDE`. */
export async function readSquare(file: File): Promise<SquarePicture> {
  const bitmap = await createImageBitmap(file);
  const crop = Math.min(bitmap.width, bitmap.height);
  const side = Math.min(PICTURE_SIDE, crop);
  const canvas = document.createElement("canvas");
  canvas.width = side;
  canvas.height = side;
  const context = canvas.getContext("2d")!;
  context.drawImage(bitmap, (bitmap.width - crop) / 2, (bitmap.height - crop) / 2, crop, crop, 0, 0, side, side);
  const rgba = context.getImageData(0, 0, side, side).data;
  const rgb: number[] = new Array(side * side * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    rgb[j] = rgba[i];
    rgb[j + 1] = rgba[i + 1];
    rgb[j + 2] = rgba[i + 2];
  }
  return { side, rgb, preview: canvas.toDataURL("image/png") };
}

/** The rectangle between two points, inside a square of `side`, with the corners in order. */
export function regionBetween(a: [number, number], b: [number, number], side: number): Region {
  const clamp = (v: number) => Math.max(0, Math.min(side, Math.round(v)));
  return [clamp(Math.min(a[0], b[0])), clamp(Math.min(a[1], b[1])), clamp(Math.max(a[0], b[0])), clamp(Math.max(a[1], b[1]))];
}

/** A region worth sending: one that covers at least a code of the model, 16 × 16 pixels of 256. */
export function regionIsUsable(region: Region | null, side: number): region is Region {
  return region !== null && (region[2] - region[0]) * 16 >= side && (region[3] - region[1]) * 16 >= side;
}

let timer: ReturnType<typeof setInterval> | null = null;

export const useImages = create<ImagesState>((set, get) => ({
  online: null,
  models: [],
  path: "",
  prompt: "",
  steps: 16,
  guidance: 1.75,
  seed: "",
  onChip: false,
  magvitOnChip: false,
  picture: null,
  region: null,
  running: false,
  phase: "",
  step: 0,
  total: 0,
  result: null,
  error: null,
  set: (partial) => set(partial),

  connect: async () => {
    try {
      const response = await fetch("/api/models");
      const models = ((await response.json()) as ServerModel[]).filter((m) => m.images.edits);
      const path = models.some((m) => m.path === get().path) ? get().path : (models[0]?.path ?? "");
      set({ online: true, models, path });
      // A job left running, for instance before the page was reloaded.
      const status = await (await fetch("/api/retouch")).json();
      if (status.state === "running" && !get().running) {
        set({ running: true });
        poll(set);
      }
    } catch {
      set({ online: false });
    }
  },

  start: async () => {
    const { path, prompt, steps, guidance, seed, picture, region, onChip, magvitOnChip } = get();
    const body = {
      path,
      prompt,
      steps,
      guidance,
      seed: seed.trim() ? Number(seed) : null,
      chip_bits: onChip ? 16 : null,
      magvit_on_chip: onChip && magvitOnChip,
      picture: picture && regionIsUsable(region, picture.side) ? { width: picture.side, height: picture.side, rgb: picture.rgb, region } : null,
    };
    set({ running: true, error: null, phase: "Starting", step: 0, total: steps });
    const response = await fetch("/api/retouch", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    if (!response.ok) {
      set({ running: false, error: await response.text() });
      return;
    }
    poll(set);
  },

  cancel: async () => {
    await fetch("/api/retouch/cancel", { method: "POST" });
  },
}));

function poll(set: (partial: Partial<ImagesState>) => void) {
  if (timer) clearInterval(timer);
  timer = setInterval(async () => {
    try {
      const status = await (await fetch("/api/retouch")).json();
      if (status.state === "running") {
        set({ phase: status.phase, step: status.step, total: status.steps });
        return;
      }
      if (timer) clearInterval(timer);
      timer = null;
      if (status.state === "done") set({ running: false, result: status.result });
      else if (status.state === "error") set({ running: false, error: status.error === "cancelled" ? null : status.error });
      else set({ running: false });
    } catch {
      set({ online: false });
    }
  }, 1000);
}
