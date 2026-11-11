import { download } from "../download";
import { floorplanDef, floorplanSvg } from "../floorplanExport";
import { useCallback, useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import type { Rect, Report } from "../engine";
import { formatNumber } from "../format";
import { useDesign, type Overlay, type Selection } from "../store";
import { CATEGORY_LABEL } from "./Inspector";
import { useT } from "../i18n";
import { IconButton, Segmented } from "./controls";
import { draw, fitCamera, hitTest, overlayRange, readTheme, type Camera, type Hit, type Theme } from "./floorplanRender";

const OVERLAYS: [Overlay, string][] = [
  ["funcion", "Función"],
  ["capa", "Capa"],
  ["actividad", "Actividad"],
  ["energia", "Energía"],
  ["ancho", "Datos"],
  ["densidad", "Densidad"],
];

const OVERLAY_UNIT: Partial<Record<Overlay, [string, (v: number) => string]>> = {
  actividad: ["del token", (v) => `${formatNumber(v * 100)} %`],
  energia: ["por token", (v) => `${formatNumber(v)} mJ`],
  ancho: ["por token", (v) => `${formatNumber(v / 1e9)} GB`],
  densidad: ["mJ/mm² por token", (v) => formatNumber(v)],
};

const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const MIN_K_MARGIN = 0.5;
const MAX_CELL_PX = 2400;


function Legend({ report, overlay }: { report: Report; overlay: Overlay }) {
  const t = useT();
  if (overlay === "funcion") {
    const used = [...new Set(report.blocks.map((b) => b.category))];
    return (
      <div className="legend">
        {used.map((c) => (
          <span key={c}>
            <i className={`swatch cat-${c}`} aria-hidden="true" />
            {t(CATEGORY_LABEL[c])}
          </span>
        ))}
      </div>
    );
  }
  if (overlay === "capa") {
    return (
      <div className="legend">
        <span>Capa 0</span>
        <i className="ramp" aria-hidden="true" />
        <span>Capa {report.floorplan.num_layers - 1}</span>
      </div>
    );
  }
  const [unit, show] = OVERLAY_UNIT[overlay]!;
  return (
    <div className="legend">
      <span>0</span>
      <i className="ramp" aria-hidden="true" />
      <span>
        {show(overlayRange(report, overlay))} {unit}
      </span>
    </div>
  );
}

function Breadcrumb({ selection, onSystem, onDie }: { selection: Selection; onSystem: () => void; onDie: (die: number) => void }) {
  const t = useT();
  return (
    <nav className="breadcrumb" aria-label="Posición en el chip">
      <button onClick={onSystem}>{t("Sistema")}</button>
      {selection && (
        <>
          <span aria-hidden="true">›</span>
          <button onClick={() => onDie(selection.die)}>Die {selection.die}</button>
        </>
      )}
      {selection && (selection.kind === "array" || selection.kind === "cell") && (
        <>
          <span aria-hidden="true">›</span>
          <span>Matriz de celdas</span>
        </>
      )}
      {selection?.kind === "cell" && (
        <>
          <span aria-hidden="true">›</span>
          <span>
            Celda {selection.col}, {selection.row}
          </span>
        </>
      )}
    </nav>
  );
}

export function FloorplanView() {
  const t = useT();
  const designed = useDesign((s) => s.report);
  // The canvas draws each block's name in the interface's language.
  const report = useMemo(() => designed && { ...designed, blocks: designed.blocks.map((b) => ({ ...b, label: t(b.label) })) }, [designed, t]);
  const overlay = useDesign((s) => s.overlay);
  const setOverlay = useDesign((s) => s.setOverlay);
  const selection = useDesign((s) => s.selection);
  const setSelection = useDesign((s) => s.setSelection);
  const appearance = useDesign((s) => s.appearance);

  const host = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const camera = useRef<Camera>({ x: 0, y: 0, k: 1 });
  const target = useRef<Camera | null>(null);
  const size = useRef({ width: 1, height: 1 });
  const atFit = useRef(true);
  const placed = useRef(false);
  const frame = useRef(0);
  const hover = useRef<Hit | null>(null);
  const theme = useRef<Theme | null>(null);
  const latest = useRef({ report, overlay, selection });
  latest.current = { report, overlay, selection };
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null);

  const paint = useCallback(() => {
    const el = canvas.current;
    const { report: r, overlay: o, selection: s } = latest.current;
    if (!el || !r || !theme.current) return;
    const ctx = el.getContext("2d")!;
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw(ctx, { report: r, overlay: o, selection: s, hover: hover.current, theme: theme.current, ...size.current, camera: camera.current });
  }, []);

  const schedule = useCallback(() => {
    cancelAnimationFrame(frame.current);
    frame.current = requestAnimationFrame(() => {
      const goal = target.current;
      if (goal) {
        const cam = camera.current;
        const t = 0.26;
        const next = { x: cam.x + (goal.x - cam.x) * t, y: cam.y + (goal.y - cam.y) * t, k: cam.k * Math.pow(goal.k / cam.k, t) };
        const done = Math.abs(next.k / goal.k - 1) < 0.002 && Math.abs(next.x - goal.x) * next.k < 0.5 && Math.abs(next.y - goal.y) * next.k < 0.5;
        camera.current = done ? goal : next;
        if (done) target.current = null;
        else schedule();
      }
      paint();
    });
  }, [paint]);

  const moveTo = useCallback(
    (goal: Camera) => {
      if (reducedMotion()) {
        camera.current = goal;
        target.current = null;
      } else {
        target.current = goal;
      }
      schedule();
    },
    [schedule],
  );

  const fitAll = useCallback(() => {
    const r = latest.current.report;
    if (!r) return;
    atFit.current = true;
    const { width, height } = size.current;
    const goal = fitCamera({ x: 0, y: 0, w: r.floorplan.width, h: r.floorplan.height }, width, height, 56);
    if (!goal) return;
    if (placed.current) {
      moveTo(goal);
    } else {
      placed.current = true;
      camera.current = goal;
      schedule();
    }
  }, [moveTo, schedule]);

  const focus = useCallback(
    (rect: Rect) => {
      atFit.current = false;
      const { width, height } = size.current;
      const goal = fitCamera(rect, width, height, 48);
      if (goal) moveTo(goal);
    },
    [moveTo],
  );

  useEffect(() => {
    const el = host.current!;
    const updateTheme = () => {
      theme.current = readTheme(el);
      schedule();
    };
    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      const dpr = window.devicePixelRatio || 1;
      size.current = { width, height };
      const c = canvas.current!;
      c.width = Math.round(width * dpr);
      c.height = Math.round(height * dpr);
      if (atFit.current || !placed.current) fitAll();
      else schedule();
    });
    observer.observe(el);
    updateTheme();
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    media.addEventListener("change", updateTheme);
    return () => {
      observer.disconnect();
      media.removeEventListener("change", updateTheme);
    };
  }, [fitAll, schedule]);

  useEffect(() => {
    if (atFit.current) fitAll();
    else schedule();
  }, [report, fitAll, schedule]);

  useEffect(() => schedule(), [overlay, selection, schedule]);

  useEffect(() => {
    if (host.current) theme.current = readTheme(host.current);
    schedule();
  }, [appearance, schedule]);

  const zoomAt = useCallback(
    (factor: number, px: number, py: number) => {
      const r = latest.current.report;
      if (!r) return;
      const { width, height } = size.current;
      const fit = fitCamera({ x: 0, y: 0, w: r.floorplan.width, h: r.floorplan.height }, width, height, 56);
      if (!fit) return;
      atFit.current = false;
      target.current = null;
      const cam = camera.current;
      const min = fit.k * MIN_K_MARGIN;
      const cells = r.floorplan.dies[0].cells;
      const max = MAX_CELL_PX / Math.min(cells.rect.w / cells.cols, cells.rect.h / cells.rows);
      const k = Math.min(max, Math.max(min, cam.k * factor));
      const wx = cam.x + px / cam.k;
      const wy = cam.y + py / cam.k;
      camera.current = { k, x: wx - px / k, y: wy - py / k };
      schedule();
    },
    [schedule],
  );

  const pointer = (e: { clientX: number; clientY: number }) => {
    const box = canvas.current!.getBoundingClientRect();
    const px = e.clientX - box.left;
    const py = e.clientY - box.top;
    const cam = camera.current;
    return { px, py, wx: cam.x + px / cam.k, wy: cam.y + py / cam.k };
  };

  useEffect(() => {
    const el = canvas.current!;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const box = el.getBoundingClientRect();
      const px = e.clientX - box.left;
      const py = e.clientY - box.top;
      const isMouseWheel = e.deltaMode !== 0 || (e.deltaX === 0 && Number.isInteger(e.deltaY) && Math.abs(e.deltaY) >= 100);
      if (e.ctrlKey || isMouseWheel) {
        zoomAt(Math.exp(-e.deltaY * (e.ctrlKey ? 0.012 : 0.0025)), px, py);
      } else {
        atFit.current = false;
        target.current = null;
        const cam = camera.current;
        camera.current = { ...cam, x: cam.x + e.deltaX / cam.k, y: cam.y + e.deltaY / cam.k };
        schedule();
      }
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [zoomAt, schedule]);

  const drag = useRef<{ x: number; y: number; moved: boolean } | null>(null);

  const select = (hit: Hit | null) => {
    if (!hit || hit.kind === "memory") return setSelection(null);
    if (hit.kind === "die") return setSelection({ kind: "die", die: hit.die });
    if (hit.kind === "array") return setSelection({ kind: "array", die: hit.die });
    if (hit.kind === "block") return setSelection({ kind: "block", die: hit.die, id: hit.id });
    setSelection({ kind: "cell", die: hit.die, row: hit.row, col: hit.col });
  };

  const onPointerDown = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    canvas.current!.setPointerCapture(e.pointerId);
    drag.current = { x: e.clientX, y: e.clientY, moved: false };
  };

  const onPointerMove = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    const d = drag.current;
    if (d) {
      const dx = e.clientX - d.x;
      const dy = e.clientY - d.y;
      if (d.moved || Math.hypot(dx, dy) > 3) {
        d.moved = true;
        d.x = e.clientX;
        d.y = e.clientY;
        atFit.current = false;
        target.current = null;
        const cam = camera.current;
        camera.current = { ...cam, x: cam.x - dx / cam.k, y: cam.y - dy / cam.k };
        setTip(null);
        schedule();
      }
      return;
    }
    const r = latest.current.report;
    if (!r) return;
    const { px, py, wx, wy } = pointer(e);
    const hit = hitTest(r, wx, wy, camera.current.k);
    if (hit?.label !== hover.current?.label || hit?.kind !== hover.current?.kind) {
      hover.current = hit;
      schedule();
    }
    setTip(hit ? { x: px, y: py, text: hit.label } : null);
  };

  const onPointerUp = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    const moved = drag.current?.moved;
    drag.current = null;
    const r = latest.current.report;
    if (moved || !r) return;
    const { wx, wy } = pointer(e);
    select(hitTest(r, wx, wy, camera.current.k));
  };

  const onDoubleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const r = latest.current.report;
    if (!r) return;
    const { wx, wy } = pointer(e);
    const hit = hitTest(r, wx, wy, camera.current.k);
    if (hit) {
      select(hit);
      focus(hit.rect);
    }
  };

  const goToDie = (index: number) => {
    const die = latest.current.report?.floorplan.dies[index];
    if (die) {
      setSelection({ kind: "die", die: index });
      focus(die.rect);
    }
  };

  if (!report) return <section className="stage" ref={host} />;

  const { width, height } = size.current;

  return (
    <section className="stage" ref={host}>
      <canvas
        ref={canvas}
        role="img"
        aria-label={`Plano del chip: ${report.dies} dies de ${formatNumber(report.die_area_mm2)} milímetros cuadrados. La lista de bloques está en el inspector.`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={() => {
          hover.current = null;
          setTip(null);
          schedule();
        }}
        onDoubleClick={onDoubleClick}
      />
      <div className="hud top">
        <Breadcrumb
          selection={selection}
          onSystem={() => {
            setSelection(null);
            fitAll();
          }}
          onDie={goToDie}
        />
        <Segmented label={t("Capa de color")} value={overlay} options={OVERLAYS.map(([id, label]) => [id, t(label)])} onChange={setOverlay} />
      </div>
      <div className="hud bottom">
        <Legend report={report} overlay={overlay} />
        <div className="zoom">
          <IconButton label="Exportar plano en SVG" onClick={() => download(floorplanSvg(report), "llm2chip_plano.svg", "image/svg+xml")}>
            <path d="M6 3h6l3 3v11H6zM12 3v3h3M8 11l2 2 2-2M10 8v5" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
          </IconButton>
          <IconButton label="Exportar colocación en DEF" onClick={() => download(floorplanDef(report.floorplan, "llm2chip"), "llm2chip_plano.def")}>
            <path d="M4 4h12v12H4zM4 10h12M10 4v12" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" />
          </IconButton>
          <IconButton label="Alejar" onClick={() => zoomAt(0.6, width / 2, height / 2)}>
            <path d="M5 10h10" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
          </IconButton>
          <IconButton label="Encajar" onClick={fitAll}>
            <path d="M4 8V4h4M16 8V4h-4M4 12v4h4M16 12v4h-4" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
          </IconButton>
          <IconButton label="Acercar" onClick={() => zoomAt(1.6, width / 2, height / 2)}>
            <path d="M5 10h10M10 5v10" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
          </IconButton>
        </div>
      </div>
      {tip && (
        <div className="tooltip" style={{ left: tip.x + 14, top: tip.y + 14 }}>
          {tip.text}
        </div>
      )}
    </section>
  );
}
