import type { ReactNode } from "react";

const line = { fill: "none", stroke: "currentColor", strokeWidth: 1.5, strokeLinecap: "round", strokeLinejoin: "round" } as const;

/** One icon of 20×20, drawn with lines in the text colour. */
export function Icon({ name, size = 16 }: { name: IconName; size?: number }) {
  return (
    <svg className="icon" width={size} height={size} viewBox="0 0 20 20" aria-hidden="true">
      {ICONS[name]}
    </svg>
  );
}

export type IconName = keyof typeof ICONS;

const ICONS = {
  // A die seen from above: the chip and its blocks.
  floorplan: (
    <>
      <rect x="3" y="3" width="14" height="14" rx="2" {...line} />
      <path d="M3 9h8M11 3v14M11 12h6" {...line} />
    </>
  ),
  // A token going through the steps.
  logic: (
    <>
      <circle cx="5" cy="5" r="2" {...line} />
      <circle cx="15" cy="10" r="2" {...line} />
      <circle cx="5" cy="15" r="2" {...line} />
      <path d="M7 5.8 13 9M13 11 7 14.2" {...line} />
    </>
  ),
  // A logic gate.
  schematic: <path d="M4 5h5a5 5 0 0 1 0 10H4zM1.5 8H4M1.5 12H4M14 10h4.5" {...line} />,
  simulator: <path d="M6 4.5v11l9-5.5z" {...line} />,
  // A framed picture: a mountain under the sun.
  images: (
    <>
      <rect x="3" y="4" width="14" height="12" rx="2" {...line} />
      <circle cx="13" cy="8" r="1.4" {...line} />
      <path d="M3.5 14 8 9.5l3.5 3.5 2-2 3 3" {...line} />
    </>
  ),
  // A curve over axes, what the sweeps draw.
  explorer: <path d="M3 3v14h14M6 13l3.5-4 3 2.5L17 6" {...line} />,
  // A histogram: how the numbers spread.
  numeric: <path d="M4 16V11M8 16V6M12 16V9M16 16V13M3 16.5h14" {...line} />,
  // Stacked layers of a model.
  models: <path d="M10 3 17 6.5 10 10 3 6.5zM3 10l7 3.5 7-3.5M3 13.5 10 17l7-3.5" {...line} />,
  parameters: <path d="M4 5h12M4 10h12M4 15h12M7 3.5v3M13 8.5v3M9 13.5v3" {...line} />,
  info: (
    <>
      <circle cx="10" cy="10" r="7" {...line} />
      <path d="M10 9v4.5M10 6.5v.2" {...line} />
    </>
  ),
  warning: <path d="M10 3.5 17.5 16.5h-15zM10 8.5v3.5M10 14v.2" {...line} />,
  check: <path d="M4.5 10.5 8.5 14.5 15.5 6" {...line} />,
  chip: (
    <>
      <rect x="5" y="5" width="10" height="10" rx="1.5" {...line} />
      <path d="M8 2.5v2.5M12 2.5v2.5M8 15v2.5M12 15v2.5M2.5 8h2.5M2.5 12h2.5M15 8h2.5M15 12h2.5" {...line} />
    </>
  ),
  board: (
    <>
      <rect x="2.5" y="4.5" width="15" height="11" rx="1.5" {...line} />
      <rect x="7" y="7.5" width="5" height="5" rx="0.5" {...line} />
      <path d="M4.5 15.5v2M15.5 15.5v2" {...line} />
    </>
  ),
  image: (
    <>
      <rect x="3" y="4" width="14" height="12" rx="2" {...line} />
      <circle cx="7.5" cy="8.5" r="1.3" {...line} />
      <path d="m3.5 14.5 4.5-4 3 2.5 2.5-2 3 3" {...line} />
    </>
  ),
  send: <path d="M3.5 10 16.5 4 12 16.5 9.5 11zM9.5 11 16.5 4" {...line} />,
  stop: <rect x="5.5" y="5.5" width="9" height="9" rx="1.5" {...line} />,
  reload: <path d="M15.5 8.5A6 6 0 1 0 16 12M16 4v4.5h-4.5" {...line} />,
  download: <path d="M10 3.5v9M6 9l4 4 4-4M4 16.5h12" {...line} />,
  link: <path d="M11.5 3.5h5v5M16.5 3.5 9 11M14 11.5v4a1 1 0 0 1-1 1H4.5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h4" {...line} />,
  pencil: <path d="M13.5 3.5 16.5 6.5 7 16H4v-3zM11.5 5.5l3 3" {...line} />,
} satisfies Record<string, ReactNode>;
