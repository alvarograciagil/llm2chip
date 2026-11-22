import { DEMO } from "../demo";
import type { Report } from "../engine";
import { formatNumber } from "../format";
import { useT } from "../i18n";
import { useDesign } from "../store";
import { useLayout, type LayoutRequest } from "./layoutStore";

/** The processes OpenROAD can lay out, with an open PDK. */
const OPEN_NODES: Record<string, string> = { sky130: "SkyWater 130", gf180: "GF180", asap7: "ASAP7" };

/** The compute cell the design's circuit chose, as the layout flow takes it. */
function cellRequest(report: Report, node: string): LayoutRequest | null {
  const decision = report.decisions[0];
  if (!decision) return null;
  return {
    node,
    cell: decision.cell,
    cell_key: decision.cell_key,
    // Aim at the clock the estimate gives the cell, no faster than the process allows.
    period_ns: Math.max(1, Math.round((1 / Math.max(decision.fmax_ghz, 0.05)) * 10) / 10),
  };
}

/** How the estimator names a cell on a process, to know a measure belongs to it. */
const cellKey = (request: LayoutRequest) => `${request.node}:${request.cell_key}`;

/** The clock target follows the estimate, so it moves once the measured clock is applied. */
const sameCell = (a: LayoutRequest, b: LayoutRequest) => cellKey(a) === cellKey(b);

/** A real layout of the design's compute cell with OpenROAD, measured against the estimate. */
export function LayoutPanel({ report, allowed }: { report: Report; allowed: boolean | null }) {
  const t = useT();
  const params = useDesign((s) => s.params);
  const setParams = useDesign((s) => s.setParams);
  const layout = useLayout();
  const node = String(params.node);
  if (node.startsWith("fpga")) return null;
  const open = OPEN_NODES[node];
  const request = open ? cellRequest(report, node) : null;
  const decision = report.decisions[0];
  const stale = layout.request !== null && request !== null && !sameCell(layout.request, request);
  const result = layout.result;
  const key = layout.request ? cellKey(layout.request) : "";
  const applied = key !== "" && params.measured_cell === key && Number(params.measured_cell_um2) > 0;
  const usable = result !== null && !stale && (result.cell_area_um2 ?? 0) > 0;

  return (
    <>
      <h3>{t("Real layout of the cell")}</h3>
      <div className="inset padded form">
        {!open ? (
          <p className="muted-text">{t("This process has no open PDK. Choose SkyWater 130, GF180 or ASAP7 as the target to lay it out with OpenROAD.")}</p>
        ) : (
          <>
            <p className="muted-text">{t("The cell the die repeats thousands of times, with the circuit chosen for this design, placed and routed with OpenROAD on")} {open}.</p>
            {allowed === false && <p className="note-warning">{t("The design cannot be built on this process as it stands: adjust it under “Feasibility” before laying it out.")}</p>}
            {DEMO ? <p className="muted-text">{t("Laying it out needs the app on your computer, with Docker.")}</p> : <button className="pill primary-pill" disabled={layout.state === "running" || !request || allowed !== true} onClick={() => request && void layout.start(request)}>
              {layout.state === "running" ? `${t(layout.phase)}… ${layout.seconds} s` : result ? t("Lay it out again") : t("Lay it out")}
            </button>}
            {layout.state === "error" && <p className="note-warning">{layout.error}</p>}
            {stale && result && <p className="muted-text">{t("The circuit changed since this layout.")}</p>}
          </>
        )}
        {result && (
          <>
            {result.image && <img className="layout-image" src={result.image} alt={t("Layout of the cell")} />}
            <dl className="metrics">
              <div>
                <dt>{t("Cell area")}</dt>
                <dd>
                  {formatNumber(result.cell_area_um2 ?? 0)} µm²{decision && !applied ? ` · ${t("estimated")} ${formatNumber(decision.area_um2)}` : ""}
                </dd>
              </div>
              <div>
                <dt>{t("Maximum frequency")}</dt>
                <dd>
                  {formatNumber(result.fmax_mhz ?? 0)} MHz{decision && !applied ? ` · ${t("estimated")} ${formatNumber(decision.fmax_ghz * 1000)}` : ""}
                </dd>
              </div>
              <div>
                <dt>{t("Power")}</dt>
                <dd>{formatNumber((result.power_w ?? 0) * 1000)} mW</dd>
              </div>
              <div>
                <dt>{t("Standard cells")}</dt>
                <dd>{formatNumber(result.cells ?? 0)}</dd>
              </div>
              <div>
                <dt>{t("Rule violations")}</dt>
                <dd>{result.drc_errors}</dd>
              </div>
            </dl>
            <p className="muted-text">{t("The layout carries a few words of weights in flip-flops, just enough to keep its error correction; on the chip, the weight memory is a separate SRAM macro.")}</p>
            {applied ? (
              <>
                <p className="note-ok">{t("The estimator uses this cell's measured area and frequency.")}</p>
                <button className="pill" onClick={() => setParams({ measured_cell: "", measured_cell_um2: 0, measured_cell_fmax_ghz: 0 })}>
                  {t("Back to the estimate")}
                </button>
              </>
            ) : (
              usable && (
                <button
                  className="pill primary-pill"
                  onClick={() => setParams({ measured_cell: key, measured_cell_um2: result.cell_area_um2 ?? 0, measured_cell_fmax_ghz: (result.fmax_mhz ?? 0) / 1000 })}
                >
                  {t("Use the measured figures")}
                </button>
              )
            )}
            <p className="muted-text layout-path">GDS: {result.gds}</p>
          </>
        )}
      </div>
    </>
  );
}
