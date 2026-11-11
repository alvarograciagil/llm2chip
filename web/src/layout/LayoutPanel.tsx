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
      <h3>{t("Layout real de la celda")}</h3>
      <div className="inset padded form">
        {!open ? (
          <p className="muted-text">{t("Este proceso no tiene un PDK abierto. Elige SkyWater 130, GF180 o ASAP7 en el destino para hacer el layout con OpenROAD.")}</p>
        ) : (
          <>
            <p className="muted-text">{t("La celda que el die repite miles de veces, con el circuito elegido para este diseño, colocada y rutada con OpenROAD en")} {open}.</p>
            {allowed === false && <p className="note-warning">{t("El diseño no se puede fabricar en este proceso tal como está: ajústalo con «Viabilidad» antes de hacer el layout.")}</p>}
            <button className="pill primary-pill" disabled={layout.state === "running" || !request || allowed !== true} onClick={() => request && void layout.start(request)}>
              {layout.state === "running" ? `${t(layout.phase)}… ${layout.seconds} s` : result ? t("Repetir el layout") : t("Hacer el layout")}
            </button>
            {layout.state === "error" && <p className="note-warning">{layout.error}</p>}
            {stale && result && <p className="muted-text">{t("El circuito cambió desde este layout.")}</p>}
          </>
        )}
        {result && (
          <>
            {result.image && <img className="layout-image" src={result.image} alt={t("Layout de la celda")} />}
            <dl className="metrics">
              <div>
                <dt>{t("Área de la celda")}</dt>
                <dd>
                  {formatNumber(result.cell_area_um2 ?? 0)} µm²{decision && !applied ? ` · ${t("estimada")} ${formatNumber(decision.area_um2)}` : ""}
                </dd>
              </div>
              <div>
                <dt>{t("Frecuencia máxima")}</dt>
                <dd>
                  {formatNumber(result.fmax_mhz ?? 0)} MHz{decision && !applied ? ` · ${t("estimada")} ${formatNumber(decision.fmax_ghz * 1000)}` : ""}
                </dd>
              </div>
              <div>
                <dt>{t("Potencia")}</dt>
                <dd>{formatNumber((result.power_w ?? 0) * 1000)} mW</dd>
              </div>
              <div>
                <dt>{t("Celdas estándar")}</dt>
                <dd>{formatNumber(result.cells ?? 0)}</dd>
              </div>
              <div>
                <dt>{t("Violaciones de reglas")}</dt>
                <dd>{result.drc_errors}</dd>
              </div>
            </dl>
            <p className="muted-text">{t("El layout lleva unas pocas palabras de pesos en biestables, las justas para conservar su corrección de errores; en el chip, la memoria de pesos es una macro de SRAM aparte.")}</p>
            {applied ? (
              <>
                <p className="note-ok">{t("El estimador usa el área y la frecuencia medidas de esta celda.")}</p>
                <button className="pill" onClick={() => setParams({ measured_cell: "", measured_cell_um2: 0, measured_cell_fmax_ghz: 0 })}>
                  {t("Volver a la estimación")}
                </button>
              </>
            ) : (
              usable && (
                <button
                  className="pill primary-pill"
                  onClick={() => setParams({ measured_cell: key, measured_cell_um2: result.cell_area_um2 ?? 0, measured_cell_fmax_ghz: (result.fmax_mhz ?? 0) / 1000 })}
                >
                  {t("Usar las cifras medidas")}
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
