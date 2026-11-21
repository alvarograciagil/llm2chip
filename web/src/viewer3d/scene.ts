// The 3D scene of a trace: dies round their ring, lanes lit by what the die runs, links
// carrying words, the KV cache filling, all drawn for one cycle at a time.

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { cacheAt, cacheHolders, cacheOps, dieAt, divergedAt, linkWordsIn, type Trace, type WordClass } from "./trace";
import { layoutScene, linkPoint, type DieLayout, type SceneLayout, type Vec3 } from "./layout3d";

export interface Palette {
  canvas: string;
  die: string;
  label: string;
  separator: string;
  classes: Record<WordClass, string>;
  idle: string;
  stall: string;
  cache: string;
  cacheEmpty: string;
  link: string;
}

/** The scene's colours from the app's own tokens, so it follows light and dark. */
export function readPalette(el: Element): Palette {
  const style = getComputedStyle(el);
  const canvas = style.getPropertyValue("--canvas").trim() || "#f4f4f6";
  const v = (name: string, fallback: string) => opaque(style.getPropertyValue(name).trim() || fallback, canvas);
  return {
    canvas: v("--canvas", "#f4f4f6"),
    die: v("--die", "#d9d9de"),
    label: v("--label", "#1d1d1f"),
    separator: v("--separator", "#c6c6c8"),
    classes: {
      cells: v("--cat-memory", "#3478f6"),
      vector: v("--cat-compute", "#30a46c"),
      scalar: v("--green", "#34c759"),
      send: v("--cat-interconnect", "#f5a524"),
      take: v("--cat-interconnect", "#f5a524"),
      halt: v("--neutral-fill", "#8e8e93"),
    },
    idle: v("--neutral-fill", "#8e8e93"),
    stall: v("--red", "#ff3b30"),
    cache: v("--cat-vector", "#e5484d"),
    cacheEmpty: v("--neutral-fill", "#8e8e93"),
    link: v("--separator", "#c6c6c8"),
  };
}

/** Red, green, blue and alpha of a CSS colour written as #rgb, #rrggbb, rgb() or rgba(). */
function channels(color: string): [number, number, number, number] | null {
  const hex = /^#([\da-f]{3}|[\da-f]{6})$/i.exec(color);
  if (hex) {
    const h = hex[1].length === 3 ? [...hex[1]].map((c) => c + c).join("") : hex[1];
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16), 1];
  }
  const m = /^rgba?\(\s*([\d.]+)[\s,]+([\d.]+)[\s,]+([\d.]+)(?:[\s,/]+([\d.]+%?))?\s*\)$/.exec(color);
  if (!m) return null;
  const alpha = m[4] === undefined ? 1 : m[4].endsWith("%") ? parseFloat(m[4]) / 100 : parseFloat(m[4]);
  return [parseFloat(m[1]), parseFloat(m[2]), parseFloat(m[3]), alpha];
}

/** A colour with transparency, as some of the app's tokens are, laid over `background`:
 * the scene's materials take only solid colours. */
export function opaque(color: string, background: string): string {
  const c = channels(color);
  if (!c || c[3] >= 1) return color;
  const under = channels(background) ?? [255, 255, 255, 1];
  const mix = (k: number) => Math.round(c[3] * c[k] + (1 - c[3]) * under[k]);
  return `rgb(${mix(0)}, ${mix(1)}, ${mix(2)})`;
}

interface DieMeshes {
  slab: THREE.Mesh;
  lanes: THREE.InstancedMesh;
  cells: THREE.InstancedMesh;
  cache: THREE.InstancedMesh | null;
  /** The cache operations whose columns this die shows. */
  held: number[];
  outline: THREE.LineSegments;
}

interface LinkMeshes {
  tube: THREE.Mesh;
  packet: THREE.Mesh;
  curve: THREE.Curve<THREE.Vector3>;
}

/** Where a link leaves die `from` and reaches die `to`: on each slab's edge facing the other. */
function edges(from: DieLayout, to: DieLayout): [Vec3, Vec3] {
  const dx = to.center.x - from.center.x;
  const dz = to.center.z - from.center.z;
  const length = Math.hypot(dx, dz) || 1;
  const at = (d: DieLayout, sign: number): Vec3 => {
    // How far along the direction the slab's rectangle reaches.
    const reach = Math.min(Math.abs(d.width / 2 / (dx / length || 1e-9)), Math.abs(d.depth / 2 / (dz / length || 1e-9)));
    return { x: d.center.x + (sign * dx * reach) / length, y: 0, z: d.center.z + (sign * dz * reach) / length };
  };
  return [at(from, 1), at(to, -1)];
}

const SLOT_GEOMETRY = new THREE.BoxGeometry(0.3, 0.16, 0.3);

export class ChipScene {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private dies: DieMeshes[] = [];
  private links: LinkMeshes[] = [];
  private trace: Trace | null = null;
  private layout: SceneLayout | null = null;
  private palette: Palette;
  private raycaster = new THREE.Raycaster();
  private color = new THREE.Color();
  private frame = 0;
  private resize: ResizeObserver;
  /** Called with the die under the pointer when the scene is clicked. */
  onPick: (die: number | null) => void = () => {};
  /** Called on every frame after drawing, with the screen position of each die's top. */
  onFrame: (positions: { x: number; y: number; visible: boolean }[]) => void = () => {};
  private tick: () => void = () => {};

  constructor(private host: HTMLElement) {
    this.palette = readPalette(host);
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    host.appendChild(this.renderer.domElement);
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 500);
    this.camera.position.set(0, 14, 18);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x666677, 2.2));
    const sun = new THREE.DirectionalLight(0xffffff, 1.4);
    sun.position.set(6, 12, 8);
    this.scene.add(sun);
    this.resize = new ResizeObserver(() => this.fit());
    this.resize.observe(host);
    this.renderer.domElement.addEventListener("pointerup", this.pick);
    this.fit();
    const loop = () => {
      this.frame = requestAnimationFrame(loop);
      this.tick();
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
      this.onFrame(this.screenPositions());
    };
    loop();
  }

  /** Called before every frame: where the view moves time on. */
  setTick(tick: () => void) {
    this.tick = tick;
  }

  dispose() {
    cancelAnimationFrame(this.frame);
    this.resize.disconnect();
    this.renderer.domElement.removeEventListener("pointerup", this.pick);
    this.clear();
    this.controls.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }

  setPalette(palette: Palette) {
    this.palette = palette;
    if (this.trace) this.setTrace(this.trace);
  }

  private fit() {
    const { clientWidth: w, clientHeight: h } = this.host;
    if (w === 0 || h === 0) return;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  private clear() {
    for (const d of this.dies) {
      for (const m of [d.slab, d.lanes, d.cells, d.cache, d.outline]) {
        if (!m) continue;
        this.scene.remove(m);
        if (m.geometry !== SLOT_GEOMETRY) m.geometry.dispose();
        (m.material as THREE.Material).dispose();
      }
    }
    for (const l of this.links) {
      for (const m of [l.tube, l.packet]) {
        this.scene.remove(m);
        m.geometry.dispose();
        (m.material as THREE.Material).dispose();
      }
    }
    this.dies = [];
    this.links = [];
  }

  setTrace(trace: Trace) {
    this.clear();
    this.trace = trace;
    const layout = layoutScene(trace);
    this.layout = layout;
    this.scene.background = new THREE.Color(this.palette.canvas);
    const caches = cacheOps(trace);
    const holders = cacheHolders(trace);
    layout.dies.forEach((d, i) => {
      const slab = new THREE.Mesh(new THREE.BoxGeometry(d.width, 0.12, d.depth), new THREE.MeshStandardMaterial({ color: this.palette.die, roughness: 0.8 }));
      slab.position.set(d.center.x, 0, d.center.z);
      slab.userData.die = i;
      const outline = new THREE.LineSegments(new THREE.EdgesGeometry(slab.geometry), new THREE.LineBasicMaterial({ color: this.palette.separator }));
      outline.position.copy(slab.position);
      const lanes = new THREE.InstancedMesh(new THREE.BoxGeometry(layout.lane, 0.22, layout.lane), new THREE.MeshStandardMaterial({ roughness: 0.5 }), d.lanes.length);
      const m = new THREE.Matrix4();
      d.lanes.forEach((p, k) => lanes.setMatrixAt(k, m.makeTranslation(p.x, p.y, p.z)));
      lanes.userData.die = i;
      const cells = new THREE.InstancedMesh(new THREE.BoxGeometry(0.22, 0.12, 0.22), new THREE.MeshStandardMaterial({ roughness: 0.6 }), Math.max(d.cells.length, 1));
      d.cells.forEach((p, k) => cells.setMatrixAt(k, m.makeTranslation(p.x, p.y, p.z)));
      cells.count = d.cells.length;
      // Only the dies that keep a layer's cache get its column.
      const held = caches.filter((op) => holders.get(op)?.has(i));
      let cache: THREE.InstancedMesh | null = null;
      if (held.length > 0) {
        cache = new THREE.InstancedMesh(SLOT_GEOMETRY, new THREE.MeshStandardMaterial({ roughness: 0.4 }), held.length * trace.context);
        held.forEach((_, column) => d.cache[column].forEach((p, slot) => cache!.setMatrixAt(column * trace.context + slot, m.makeTranslation(p.x, p.y, p.z))));
      }
      for (const mesh of [slab, outline, lanes, cells, cache]) if (mesh) this.scene.add(mesh);
      this.dies.push({ slab, lanes, cells, cache, outline, held });
    });
    // Lanes of one link sit side by side; loops back bow towards the ring's centre.
    const lanesOf = new Map<string, number>();
    for (const link of trace.links) {
      const pair = `${link.from}-${link.to}-${link.port.replace(/\d+$/, "")}`;
      const lane = lanesOf.get(pair) ?? 0;
      lanesOf.set(pair, lane + 1);
      // From the edge of one die to the edge of the other, not over their units.
      const [a, b] = edges(layout.dies[link.from], layout.dies[link.to]);
      const inward = link.port.startsWith("loop");
      const points = Array.from({ length: 24 }, (_, k) => {
        const p = linkPoint(a, b, k / 23, lane, inward);
        return new THREE.Vector3(p.x, p.y, p.z);
      });
      const curve = new THREE.CatmullRomCurve3(points);
      const tube = new THREE.Mesh(new THREE.TubeGeometry(curve, 48, 0.022, 6), new THREE.MeshBasicMaterial({ color: this.palette.link }));
      const packet = new THREE.Mesh(new THREE.SphereGeometry(0.11, 12, 8), new THREE.MeshBasicMaterial({ color: this.palette.classes.send }));
      packet.visible = false;
      this.scene.add(tube, packet);
      this.links.push({ tube, packet, curve });
    }
    const span = Math.max(layout.radius * 2 + 6, 8);
    this.camera.position.set(0, span * 0.75, span * 0.95);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }

  /** Draws the chips as they are at `cycle`; `selected` is the die picked, if any, and
   * `window` how many cycles back a link still shows the words that crossed it. */
  show(cycle: number, selected: number | null, window = 16) {
    const trace = this.trace;
    const layout = this.layout;
    if (!trace || !layout) return;
    const p = this.palette;
    const filled = cacheAt(trace, cycle);
    this.dies.forEach((d, i) => {
      const state = dieAt(trace, i, cycle);
      const lit = state.halted ? p.idle : state.stalled ? p.stall : p.classes[state.word?.class ?? "halt"];
      this.color.set(lit);
      for (let k = 0; k < d.lanes.count; k++) d.lanes.setColorAt(k, this.color);
      if (d.lanes.instanceColor) d.lanes.instanceColor.needsUpdate = true;
      const material = d.lanes.material as THREE.MeshStandardMaterial;
      material.emissive.set(lit);
      material.emissiveIntensity = state.halted ? 0 : state.stalled ? 0.5 : 0.25;
      this.color.set(state.word?.class === "cells" && !state.stalled ? p.classes.cells : p.idle);
      for (let k = 0; k < d.cells.count; k++) d.cells.setColorAt(k, this.color);
      if (d.cells.instanceColor) d.cells.instanceColor.needsUpdate = true;
      if (d.cache) {
        const byOp = filled.get(i);
        d.held.forEach((op, column) => {
          const positions = byOp?.get(op);
          for (let slot = 0; slot < trace.context; slot++) {
            this.color.set(positions?.has(slot) ? p.cache : p.cacheEmpty);
            d.cache!.setColorAt(column * trace.context + slot, this.color);
          }
        });
        if (d.cache.instanceColor) d.cache.instanceColor.needsUpdate = true;
      }
      // A die that has parted from the exact machine keeps a red outline from then on.
      const diverged = divergedAt(trace, i, cycle) !== null;
      (d.outline.material as THREE.LineBasicMaterial).color.set(diverged ? p.stall : selected === i ? p.label : p.separator);
      d.slab.position.y = selected === i ? 0.04 : 0;
    });
    const from = Math.max(0, Math.floor(cycle - window));
    trace.links.forEach((link, k) => {
      const l = this.links[k];
      const { crossed, waited } = linkWordsIn(link, from, Math.floor(cycle));
      const state = waited > 0 ? 2 : crossed > 0 ? 1 : 0;
      (l.tube.material as THREE.MeshBasicMaterial).color.set(state === 2 ? p.stall : state === 1 ? p.classes.send : p.link);
      l.packet.visible = state !== 0;
      if (state !== 0) {
        // Words run along the link while any crossed lately; one held back waits at its end.
        const t = state === 2 ? 0.92 : (performance.now() / 600) % 1;
        l.packet.position.copy(l.curve.getPointAt(t));
        l.packet.scale.setScalar(1 + Math.min(crossed, 32) / 16);
        (l.packet.material as THREE.MeshBasicMaterial).color.set(state === 2 ? p.stall : p.classes.send);
      }
    });
  }

  private screenPositions() {
    if (!this.layout) return [];
    const { clientWidth: w, clientHeight: h } = this.host;
    return this.layout.dies.map((d) => {
      const v = new THREE.Vector3(d.center.x, 0.9, d.center.z - d.depth / 2).project(this.camera);
      return { x: ((v.x + 1) / 2) * w, y: ((1 - v.y) / 2) * h, visible: v.z < 1 };
    });
  }

  private pick = (event: PointerEvent) => {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const pointer = new THREE.Vector2(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
    this.raycaster.setFromCamera(pointer, this.camera);
    const hit = this.raycaster.intersectObjects(this.dies.flatMap((d) => [d.slab, d.lanes]))[0];
    this.onPick(hit ? (hit.object.userData.die as number) : null);
  };
}
