/** Where a die's label sits on screen: the point it hangs from, its depth (smaller is
 * nearer) and the size it has in full and with only the die's name. */
export interface LabelSpot {
  x: number;
  y: number;
  depth: number;
  visible: boolean;
  full: { w: number; h: number };
  short: { w: number; h: number };
}

export type LabelForm = "full" | "short" | "hidden";

/** Room left between two labels. */
const GAP = 4;

interface Box {
  left: number;
  right: number;
  top: number;
  bottom: number;
}

/** A label hangs above its point, centred on it. */
function boxOf(spot: LabelSpot, size: { w: number; h: number }): Box {
  return { left: spot.x - size.w / 2, right: spot.x + size.w / 2, top: spot.y - size.h, bottom: spot.y };
}

const overlaps = (a: Box, b: Box) => a.left < b.right + GAP && b.left < a.right + GAP && a.top < b.bottom + GAP && b.top < a.bottom + GAP;

/** Which labels show in full: the chosen die's always, then the nearest first, as long
 * as they cover no label already placed. One that would shows only the die's name. */
export function placeLabels(spots: LabelSpot[], selected: number | null): LabelForm[] {
  const forms: LabelForm[] = spots.map((s) => (s.visible ? "short" : "hidden"));
  const order = spots
    .map((_, i) => i)
    .filter((i) => spots[i].visible)
    .sort((a, b) => Number(b === selected) - Number(a === selected) || spots[a].depth - spots[b].depth);
  const placed: Box[] = [];
  for (const i of order) {
    const full = boxOf(spots[i], spots[i].full);
    if (i === selected || !placed.some((b) => overlaps(b, full))) {
      forms[i] = "full";
      placed.push(full);
    } else {
      placed.push(boxOf(spots[i], spots[i].short));
    }
  }
  return forms;
}
