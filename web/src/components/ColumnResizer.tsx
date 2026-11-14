import { useEffect, useState, type PointerEvent } from "react";
import { useT } from "../i18n";

const LIMITS = { left: [220, 520], right: [240, 560] } as const;
const DEFAULTS = { left: 292, right: 312 };
const KEY = "llm2chip.columns";

type Side = keyof typeof DEFAULTS;

function readWidths(): Record<Side, number> {
  try {
    return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(KEY) ?? "{}") };
  } catch {
    return DEFAULTS;
  }
}

function apply(widths: Record<Side, number>) {
  document.documentElement.style.setProperty("--left-column", `${widths.left}px`);
  document.documentElement.style.setProperty("--right-column", `${widths.right}px`);
}

/** The edges of the side columns, which can be dragged to make them wider or narrower.
 *  Each width is remembered in this browser; a double click restores it. */
export function ColumnResizers() {
  const t = useT();
  const [widths, setWidths] = useState(readWidths);
  useEffect(() => {
    apply(widths);
    try {
      localStorage.setItem(KEY, JSON.stringify(widths));
    } catch {
      // The width still applies until the page reloads.
    }
  }, [widths]);

  const drag = (side: Side) => (start: PointerEvent<HTMLDivElement>) => {
    start.preventDefault();
    const handle = start.currentTarget;
    handle.setPointerCapture(start.pointerId);
    const [from, origin] = [widths[side], start.clientX];
    const move = (e: globalThis.PointerEvent) => {
      const moved = side === "left" ? e.clientX - origin : origin - e.clientX;
      const [low, high] = LIMITS[side];
      setWidths((w) => ({ ...w, [side]: Math.round(Math.min(high, Math.max(low, from + moved))) }));
    };
    const end = () => {
      handle.removeEventListener("pointermove", move);
      handle.removeEventListener("pointerup", end);
      document.body.classList.remove("resizing");
    };
    document.body.classList.add("resizing");
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", end);
  };

  return (
    <>
      {(["left", "right"] as Side[]).map((side) => (
        <div
          key={side}
          className={`column-resizer ${side}`}
          role="separator"
          aria-orientation="vertical"
          aria-label={side === "left" ? t("Width of the parameter column") : t("Width of the right column")}
          title={t("Drag to change the width; double-click to reset it")}
          onPointerDown={drag(side)}
          onDoubleClick={() => setWidths((w) => ({ ...w, [side]: DEFAULTS[side] }))}
        />
      ))}
    </>
  );
}
