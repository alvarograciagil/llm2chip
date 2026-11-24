import { locale } from "../format";
import type { BlockArea, DiePlan, Floorplan, Part, Rect, Report, SideDiePlan } from "../engine";
import type { Overlay, Selection } from "../store";

export interface Camera {
  x: number;
  y: number;
  k: number;
}

export type Hit =
  | { kind: "die"; die: number; rect: Rect; label: string }
  | { kind: "array"; die: number; rect: Rect; label: string }
  | { kind: "block"; die: number; id: string; rect: Rect; label: string }
  | { kind: "cell"; die: number; row: number; col: number; rect: Rect; label: string }
  | { kind: "memory"; rect: Rect; label: string }
  | { kind: "side"; die: number; rect: Rect; label: string };

export interface Theme {
  canvas: string;
  package: string;
  die: string;
  hairline: string;
  label: string;
  secondary: string;
  accent: string;
  neutral: string;
  categories: Record<BlockArea["category"], string>;
  parts: Record<Part, string>;
  heatLow: string;
  heatHigh: string;
}

export interface Scene {
  report: Report;
  overlay: Overlay;
  selection: Selection;
  hover: Hit | null;
  theme: Theme;
  width: number;
  height: number;
  camera: Camera;
  /** The interface's language for the words drawn; English if not given. */
  t?: (text: string) => string;
}

const ARRAY_IDS = ["cell_weights", "cell_macs", "scale_units", "activation_noc"];
const FONT = '-apple-system, "SF Pro Text", "Helvetica Neue", "Segoe UI", Roboto, Cantarell, "Noto Sans", sans-serif';
/** Below this cell size in pixels, the array is one texture; above, cells are real. */
const CELL_MIN_PX = 8;
const CELL_DETAIL_PX = 64;

/** The camera that shows `rect`, or null until the view has a usable size. */
export function fitCamera(rect: Rect, width: number, height: number, padding: number): Camera | null {
  const k = Math.min((width - 2 * padding) / rect.w, (height - 2 * padding) / rect.h);
  if (!(k > 0) || !Number.isFinite(k)) return null;
  return { k, x: rect.x - (width / k - rect.w) / 2, y: rect.y - (height / k - rect.h) / 2 };
}

export function readTheme(el: Element): Theme {
  const css = getComputedStyle(el);
  const get = (name: string) => css.getPropertyValue(name).trim();
  return {
    canvas: get("--canvas"),
    package: get("--package"),
    die: get("--die"),
    hairline: get("--separator"),
    label: get("--label"),
    secondary: get("--label-2"),
    accent: get("--accent"),
    neutral: get("--neutral-fill"),
    categories: {
      memory: get("--cat-memory"),
      compute: get("--cat-compute"),
      interconnect: get("--cat-interconnect"),
      vector: get("--cat-vector"),
      state: get("--cat-state"),
      system: get("--cat-system"),
    },
    parts: { text: get("--part-text"), vision: get("--part-vision"), listen: get("--part-listen"), speak: get("--part-speak"), draw: get("--part-draw") },
    heatLow: get("--heat-low"),
    heatHigh: get("--heat-high"),
  };
}

export const PART_LABEL: Record<Part, string> = { text: "Text", vision: "Vision", listen: "Listening", speak: "Speaking", draw: "Image drawing" };
const SPEAKING = ["speech writer", "sound decoder"];

/** What a die is for, in words: "Text + Vision", or "Listening + Speaking (sound decoder)"
 * when a voice die holds only one of speaking's two stages. */
export function partsLabel(parts: Part[], stages: string[], t: (text: string) => string = String): string {
  return parts
    .map((part) => {
      if (part !== "speak") return t(PART_LABEL[part]);
      const said = stages.filter((s) => SPEAKING.includes(s));
      return said.length === SPEAKING.length || said.length === 0 ? t("Speaking") : `${t("Speaking")} (${said.map((s) => t(s)).join(", ")})`;
    })
    .join(" + ");
}

/** A die beside the language model's in a line: its number, parts, size and shape. */
export function sideLabel(v: SideDiePlan, t: (text: string) => string = String): string {
  const mm2 = v.parts_mm2.reduce((a, b) => a + b, 0);
  const shape = `${v.units} × ${v.lanes} ${t("units")}, ${v.cells.toLocaleString(locale())} ${t("cells")}, ${Math.round(v.clock_mhz)} MHz`;
  const note = v.parts.includes("draw") ? "reference die, measured with Show-o" : "reference dies: plan them with your models";
  const reference = v.reference ? ` · ${t(note)}` : "";
  return `${t("Die")} ${v.index} · ${partsLabel(v.parts, v.stages, t)} · ${Math.round(mm2).toLocaleString(locale())} mm² · ${shape}${reference}`;
}

/** The share of a die's weights each segment of the image encoder takes, and where it starts. */
function visionBand(die: DiePlan): { from: number; share: number } | null {
  let from = 0;
  for (const seg of die.cells.segments) {
    if (seg.part === "vision") return { from, share: seg.fraction };
    from += seg.fraction;
  }
  return null;
}

function parse(hex: string): [number, number, number] {
  const v = hex.replace("#", "");
  return [parseInt(v.slice(0, 2), 16), parseInt(v.slice(2, 4), 16), parseInt(v.slice(4, 6), 16)];
}

export function mix(a: string, b: string, t: number): string {
  const [ar, ag, ab] = parse(a);
  const [br, bg, bb] = parse(b);
  const f = (x: number, y: number) => Math.round(x + (y - x) * t);
  return `rgb(${f(ar, br)},${f(ag, bg)},${f(ab, bb)})`;
}

function inkFor(fill: string): string {
  const m = fill.match(/\d+/g);
  const [r, g, b] = m && fill.startsWith("rgb") ? m.map(Number) : parse(fill);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b > 150 ? "rgba(0,0,0,0.85)" : "rgba(255,255,255,0.95)";
}

interface Measure {
  activity: number;
  energy: number;
  bytes: number;
  area: number;
}

function arrayMeasure(report: Report): Measure {
  const parts = report.blocks.filter((b) => ARRAY_IDS.includes(b.id));
  return {
    activity: Math.max(0, ...parts.map((b) => b.activity)),
    energy: parts.reduce((s, b) => s + b.energy_mj, 0),
    bytes: parts.reduce((s, b) => s + b.bytes_token, 0),
    area: parts.reduce((s, b) => s + b.per_die_mm2, 0),
  };
}

const blockMeasure = (b: BlockArea): Measure => ({ activity: b.activity, energy: b.energy_mj, bytes: b.bytes_token, area: b.per_die_mm2 });

function overlayValue(m: Measure, overlay: Overlay): number {
  switch (overlay) {
    case "activity":
      return m.activity;
    case "energy":
      return m.energy;
    case "data":
      return m.bytes;
    case "density":
      return m.energy / Math.max(m.area, 1e-9);
    default:
      return 0;
  }
}

export function overlayRange(report: Report, overlay: Overlay): number {
  const all = [arrayMeasure(report), ...report.blocks.filter((b) => !ARRAY_IDS.includes(b.id)).map(blockMeasure)];
  return Math.max(1e-12, ...all.map((m) => overlayValue(m, overlay)));
}

function dieLayerColor(die: DiePlan, plan: Floorplan, theme: Theme): string {
  const middle = ((die.first_layer ?? 0) + (die.last_layer ?? 0)) / 2;
  return mix(theme.heatLow, theme.heatHigh, plan.num_layers > 1 ? middle / (plan.num_layers - 1) : 0.5);
}

function fillFor(scene: Scene, die: DiePlan, measure: Measure, category: BlockArea["category"], isArray: boolean): string {
  const { overlay, theme, report } = scene;
  if (overlay === "function") return theme.categories[category];
  // The die's own blocks serve the language model: the text colour, lighter than its cells.
  if (overlay === "part") return isArray ? theme.parts.text : mix(theme.parts.text, theme.die, 0.45);
  if (overlay === "layer") return isArray ? dieLayerColor(die, report.floorplan, theme) : theme.neutral;
  const value = overlayValue(measure, overlay);
  if (value <= 0) return theme.neutral;
  return mix(theme.heatLow, theme.heatHigh, value / overlayRange(report, overlay));
}

const toScreen = (cam: Camera, r: Rect): Rect => ({ x: (r.x - cam.x) * cam.k, y: (r.y - cam.y) * cam.k, w: r.w * cam.k, h: r.h * cam.k });
const visible = (s: Rect, w: number, h: number) => s.x + s.w >= 0 && s.y + s.h >= 0 && s.x <= w && s.y <= h;

function roundRect(ctx: CanvasRenderingContext2D, r: Rect, radius: number) {
  ctx.beginPath();
  ctx.roundRect(r.x, r.y, r.w, r.h, Math.min(radius, r.w / 2, r.h / 2));
}

function text(ctx: CanvasRenderingContext2D, value: string, x: number, y: number, maxWidth: number) {
  let shown = value;
  if (ctx.measureText(shown).width > maxWidth) {
    while (shown.length > 1 && ctx.measureText(`${shown}…`).width > maxWidth) shown = shown.slice(0, -1);
    shown += "…";
  }
  ctx.fillText(shown, x, y);
}

const dieOrigin = (die: DiePlan, r: Rect): Rect => ({ x: die.rect.x + r.x, y: die.rect.y + r.y, w: r.w, h: r.h });

function drawBlockLabel(ctx: CanvasRenderingContext2D, screen: Rect, label: string, fill: string) {
  if (screen.w < 54 || screen.h < 20) return;
  ctx.fillStyle = inkFor(fill);
  ctx.font = `500 11px ${FONT}`;
  ctx.textBaseline = "top";
  text(ctx, label, screen.x + 6, screen.y + 5, screen.w - 12);
}

function drawCell(ctx: CanvasRenderingContext2D, scene: Scene, die: DiePlan, s: Rect) {
  const { theme, overlay } = scene;
  const cells = die.cells;
  const inset = Math.min(1, s.w * 0.06);
  const body = { x: s.x + inset, y: s.y + inset, w: s.w - 2 * inset, h: s.h - 2 * inset };
  const sramH = body.h * cells.sram_fraction;
  const parts: [number, string][] = [
    [cells.mac_fraction, theme.categories.compute],
    [cells.scale_fraction, theme.categories.compute],
    [cells.noc_fraction, theme.categories.interconnect],
  ];

  if (overlay === "part") {
    let y = body.y;
    die.cells.segments.forEach((seg, i) => {
      const h = body.h * seg.fraction;
      ctx.fillStyle = mix(theme.parts[seg.part], theme.die, (i % 2) * 0.18);
      ctx.fillRect(body.x, y, body.w, h);
      if (h > 14 && s.w > 120) {
        ctx.fillStyle = inkFor(theme.parts[seg.part]);
        ctx.font = `500 10px ${FONT}`;
        ctx.textBaseline = "middle";
        text(ctx, (scene.t ?? String)(seg.label), body.x + 5, y + h / 2, body.w - 10);
      }
      y += h;
    });
    return;
  }

  if (overlay !== "function") {
    ctx.fillStyle = fillFor(scene, die, arrayMeasure(scene.report), "memory", true);
    ctx.fillRect(body.x, body.y, body.w, body.h);
    return;
  }

  if (s.w >= CELL_DETAIL_PX) {
    let y = body.y;
    die.cells.segments.forEach((seg, i) => {
      const h = sramH * seg.fraction;
      ctx.fillStyle = mix(theme.categories.memory, theme.die, (i % 2) * 0.22);
      ctx.fillRect(body.x, y, body.w, h);
      if (h > 14 && s.w > 120) {
        ctx.fillStyle = inkFor(theme.categories.memory);
        ctx.font = `500 10px ${FONT}`;
        ctx.textBaseline = "middle";
        text(ctx, seg.label, body.x + 5, y + h / 2, body.w - 10);
      }
      y += h;
    });
  } else {
    ctx.fillStyle = theme.categories.memory;
    ctx.fillRect(body.x, body.y, body.w, sramH);
  }

  const rest = body.h - sramH;
  const total = parts.reduce((sum, [f]) => sum + f, 0) || 1;
  let x = body.x;
  for (const [fraction, color] of parts) {
    const w = (body.w * fraction) / total;
    ctx.fillStyle = color;
    ctx.fillRect(x, body.y + sramH, w, rest);
    x += w;
  }
}

function drawArray(ctx: CanvasRenderingContext2D, scene: Scene, die: DiePlan) {
  const { camera, width, height, theme } = scene;
  const cells = die.cells;
  const world = dieOrigin(die, cells.rect);
  const s = toScreen(camera, world);
  if (!visible(s, width, height)) return;

  const cellW = world.w / cells.cols;
  const cellH = world.h / cells.rows;
  const px = Math.min(cellW, cellH) * camera.k;
  const fill = fillFor(scene, die, arrayMeasure(scene.report), "memory", true);

  if (px < CELL_MIN_PX) {
    ctx.fillStyle = fill;
    ctx.fillRect(s.x, s.y, s.w, s.h);
    // Faint divisions hint at the repeated structure before cells are resolved.
    ctx.strokeStyle = inkFor(fill) === "rgba(0,0,0,0.85)" ? "rgba(0,0,0,0.10)" : "rgba(255,255,255,0.10)";
    ctx.lineWidth = 1;
    const step = Math.max(1, Math.ceil(10 / px));
    const coarse = Math.pow(2, Math.ceil(Math.log2(step)));
    const every = coarse * Math.max(1, Math.ceil(Math.max(cells.cols, cells.rows) / coarse / 16));
    ctx.beginPath();
    for (let c = every; c < cells.cols; c += every) {
      const x = s.x + (c / cells.cols) * s.w;
      ctx.moveTo(x, s.y);
      ctx.lineTo(x, s.y + s.h);
    }
    for (let r = every; r < cells.rows; r += every) {
      const y = s.y + (r / cells.rows) * s.h;
      ctx.moveTo(s.x, y);
      ctx.lineTo(s.x + s.w, y);
    }
    ctx.stroke();
    const band = scene.overlay === "part" ? visionBand(die) : null;
    if (band) {
      // The image encoder's share of the die's cells, where it sits after the embedding.
      const b = { x: s.x, y: s.y + band.from * s.h, w: s.w, h: Math.max(1, band.share * s.h) };
      ctx.fillStyle = theme.parts.vision;
      ctx.fillRect(b.x, b.y, b.w, b.h);
      drawBlockLabel(ctx, b, (scene.t ?? String)("Vision encoder"), theme.parts.vision);
      if (b.y > s.y + 22) drawBlockLabel(ctx, s, (scene.t ?? String)(`Cell array · ${cells.count.toLocaleString(locale())}`), fill);
      return;
    }
    drawBlockLabel(ctx, s, (scene.t ?? String)(`Cell array · ${cells.count.toLocaleString(locale())}`), fill);
    return;
  }

  ctx.fillStyle = theme.die;
  ctx.fillRect(s.x, s.y, s.w, s.h);
  const firstCol = Math.max(0, Math.floor((camera.x - world.x) / cellW));
  const lastCol = Math.min(cells.cols - 1, Math.floor((camera.x + width / camera.k - world.x) / cellW));
  const firstRow = Math.max(0, Math.floor((camera.y - world.y) / cellH));
  const lastRow = Math.min(cells.rows - 1, Math.floor((camera.y + height / camera.k - world.y) / cellH));
  for (let r = firstRow; r <= lastRow; r++) {
    for (let c = firstCol; c <= lastCol; c++) {
      if (r * cells.cols + c >= cells.count) continue;
      drawCell(ctx, scene, die, toScreen(camera, { x: world.x + c * cellW, y: world.y + r * cellH, w: cellW, h: cellH }));
    }
  }
}

function isSelected(sel: Selection, die: number, id: string): boolean {
  if (!sel || sel.die !== die) return false;
  if (sel.kind === "array") return id === "array";
  return sel.kind === "block" && (sel.id === id || (id === "array" && ARRAY_IDS.includes(sel.id)));
}

function drawDie(ctx: CanvasRenderingContext2D, scene: Scene, die: DiePlan) {
  const { camera, width, height, theme, report } = scene;
  const s = toScreen(camera, die.rect);
  if (!visible(s, width, height)) return;

  ctx.save();
  roundRect(ctx, s, 4);
  ctx.fillStyle = theme.die;
  ctx.fill();
  ctx.clip();

  drawArray(ctx, scene, die);
  const gap = s.w > 200 ? 2 : s.w > 60 ? 1 : 0;

  for (const placed of die.blocks) {
    const block = report.blocks.find((b) => b.id === placed.id);
    if (!block) continue;
    const r = toScreen(camera, dieOrigin(die, placed.rect));
    if (!visible(r, width, height)) continue;
    const fill = fillFor(scene, die, blockMeasure(block), block.category, false);
    ctx.fillStyle = fill;
    ctx.fillRect(r.x + gap, r.y + gap, Math.max(0.5, r.w - gap), Math.max(0.5, r.h - gap));
    drawBlockLabel(ctx, r, block.label, fill);
  }
  ctx.restore();

  if (scene.selection && scene.selection.die === die.index) {
    ctx.strokeStyle = theme.accent;
    ctx.lineWidth = 2;
    if (scene.selection.kind === "die") {
      roundRect(ctx, s, 4);
      ctx.stroke();
    }
    if (isSelected(scene.selection, die.index, "array")) {
      const a = toScreen(camera, dieOrigin(die, die.cells.rect));
      ctx.strokeRect(a.x + 1, a.y + 1, a.w - 2, a.h - 2);
    }
    for (const placed of die.blocks) {
      if (isSelected(scene.selection, die.index, placed.id)) {
        const r = toScreen(camera, dieOrigin(die, placed.rect));
        ctx.strokeRect(r.x + 1, r.y + 1, r.w - 2, r.h - 2);
      }
    }
    if (scene.selection.kind === "cell") {
      const c = die.cells;
      const world = dieOrigin(die, c.rect);
      const w = world.w / c.cols;
      const h = world.h / c.rows;
      const r = toScreen(camera, { x: world.x + scene.selection.col * w, y: world.y + scene.selection.row * h, w, h });
      ctx.strokeRect(r.x, r.y, r.w, r.h);
    }
  }

  ctx.fillStyle = theme.secondary;
  ctx.font = `500 11px ${FONT}`;
  ctx.textBaseline = "bottom";
  const t = scene.t ?? String;
  const range = die.first_layer === null ? "" : die.first_layer === die.last_layer ? ` · ${t(`layer ${die.first_layer}`)}` : ` · ${t(`layers ${die.first_layer}–${die.last_layer}`)}`;
  if (s.w > 36) text(ctx, s.w > 110 ? `${t("Die")} ${die.index} · ${partsLabel(die.parts, [], t)}${range}` : `${die.index}`, s.x, s.y - 4, s.w);
}

const VOICE_PARTS: [number, BlockArea["category"], string][] = [
  [0, "compute", "Logic"],
  [1, "memory", "Cell memory"],
  [2, "vector", "Vector memory"],
  [3, "system", "Microcode"],
];

/** A die beside the language model's: its parts of silicon in bands, or its parts of the
 * chip side by side; dashed while it is a reference die. */
function drawSideDie(ctx: CanvasRenderingContext2D, scene: Scene, v: SideDiePlan, room: number) {
  const { camera, width, height, theme, overlay } = scene;
  const t = scene.t ?? String;
  const s = toScreen(camera, v.rect);
  if (!visible(s, width, height)) return;
  ctx.save();
  roundRect(ctx, s, 4);
  ctx.fillStyle = theme.die;
  ctx.fill();
  ctx.clip();
  if (overlay === "part") {
    v.parts.forEach((part, i) => {
      const w = s.w / v.parts.length;
      ctx.fillStyle = theme.parts[part];
      ctx.fillRect(s.x + i * w, s.y, w, s.h);
      drawBlockLabel(ctx, { x: s.x + i * w, y: s.y, w, h: s.h }, partsLabel([part], v.stages, t), theme.parts[part]);
    });
  } else if (overlay === "function") {
    const total = v.parts_mm2.reduce((a, b) => a + b, 0) || 1;
    let y = s.y;
    for (const [i, category, label] of VOICE_PARTS) {
      const h = (s.h * v.parts_mm2[i]) / total;
      ctx.fillStyle = theme.categories[category];
      ctx.fillRect(s.x, y, s.w, h);
      drawBlockLabel(ctx, { x: s.x, y, w: s.w, h }, t(label), theme.categories[category]);
      y += h;
    }
  } else {
    ctx.fillStyle = theme.neutral;
    ctx.fillRect(s.x, s.y, s.w, s.h);
    drawBlockLabel(ctx, s, partsLabel(v.parts, v.stages, t), theme.neutral);
  }
  ctx.restore();
  if (v.reference) {
    ctx.save();
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = theme.secondary;
    ctx.lineWidth = 1.5;
    roundRect(ctx, s, 4);
    ctx.stroke();
    ctx.restore();
  }
  ctx.fillStyle = theme.secondary;
  ctx.font = `500 11px ${FONT}`;
  ctx.textBaseline = "bottom";
  // The caption may run past a small die, up to the next one: what a die is for always shows.
  const caption = `${t("Die")} ${v.index} · ${partsLabel(v.parts, v.stages, t)}`;
  if (room * camera.k > 24) text(ctx, caption, s.x, s.y - 4, room * camera.k - 6);
}

function drawLinks(ctx: CanvasRenderingContext2D, scene: Scene) {
  const { camera, report, theme } = scene;
  const dies = report.floorplan.dies;
  ctx.strokeStyle = theme.accent;
  ctx.globalAlpha = 0.65;
  ctx.lineWidth = Math.max(1.5, Math.min(4, camera.k * 0.4));
  ctx.lineCap = "round";
  ctx.beginPath();
  dies.forEach((die, i) => {
    if (i === 0) return;
    const a = toScreen(camera, dies[i - 1].rect);
    const b = toScreen(camera, die.rect);
    const [ax, ay, bx, by] =
      Math.abs(a.y - b.y) < 1
        ? a.x < b.x
          ? [a.x + a.w, a.y + a.h / 2, b.x, b.y + b.h / 2]
          : [a.x, a.y + a.h / 2, b.x + b.w, b.y + b.h / 2]
        : [a.x + a.w * 0.9, a.y + a.h, b.x + b.w * 0.9, b.y];
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
  });
  ctx.stroke();
  ctx.globalAlpha = 1;
}

export function draw(ctx: CanvasRenderingContext2D, scene: Scene) {
  const { width, height, camera, theme, report } = scene;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = theme.canvas;
  ctx.fillRect(0, 0, width, height);

  const plan = report.floorplan;
  const pkg = toScreen(camera, { x: 0, y: 0, w: plan.width, h: plan.height });
  roundRect(ctx, pkg, 10);
  ctx.fillStyle = theme.package;
  ctx.fill();
  ctx.strokeStyle = theme.hairline;
  ctx.lineWidth = 1;
  ctx.stroke();

  drawLinks(ctx, scene);
  for (const die of plan.dies) drawDie(ctx, scene, die);
  const side = plan.side ?? [];
  side.forEach((v, i) => drawSideDie(ctx, scene, v, (side[i + 1]?.rect.x ?? plan.width) - v.rect.x));

  for (const memory of plan.memory) {
    const s = toScreen(camera, memory.rect);
    if (!visible(s, width, height)) continue;
    roundRect(ctx, s, 4);
    ctx.fillStyle = theme.neutral;
    ctx.fill();
    drawBlockLabel(ctx, s, (scene.t ?? String)(memory.label), theme.neutral);
  }

  if (scene.hover && scene.hover.kind !== "die") {
    const r = toScreen(camera, scene.hover.rect);
    ctx.fillStyle = "rgba(255,255,255,0.14)";
    ctx.fillRect(r.x, r.y, r.w, r.h);
  }
}

export function hitTest(report: Report, wx: number, wy: number, cameraK: number, t: (text: string) => string = String): Hit | null {
  const inside = (r: Rect) => wx >= r.x && wy >= r.y && wx < r.x + r.w && wy < r.y + r.h;
  const plan = report.floorplan;
  for (const memory of plan.memory) {
    if (inside(memory.rect)) return { kind: "memory", rect: memory.rect, label: memory.label };
  }
  for (const v of plan.side ?? []) {
    if (inside(v.rect)) return { kind: "side", die: v.index, rect: v.rect, label: sideLabel(v, t) };
  }
  for (const die of plan.dies) {
    if (!inside(die.rect)) continue;
    const array = dieOrigin(die, die.cells.rect);
    if (inside(array)) {
      const cells = die.cells;
      const w = array.w / cells.cols;
      const h = array.h / cells.rows;
      if (Math.min(w, h) * cameraK >= CELL_MIN_PX) {
        const col = Math.floor((wx - array.x) / w);
        const row = Math.floor((wy - array.y) / h);
        if (row * cells.cols + col < cells.count) {
          return { kind: "cell", die: die.index, row, col, rect: { x: array.x + col * w, y: array.y + row * h, w, h }, label: `Cell ${col}, ${row}` };
        }
      }
      return { kind: "array", die: die.index, rect: array, label: "Cell array" };
    }
    for (const placed of die.blocks) {
      const rect = dieOrigin(die, placed.rect);
      if (inside(rect)) {
        const block = report.blocks.find((b) => b.id === placed.id);
        return { kind: "block", die: die.index, id: placed.id, rect, label: block?.label ?? placed.id };
      }
    }
    return { kind: "die", die: die.index, rect: die.rect, label: `Die ${die.index}` };
  }
  return null;
}
