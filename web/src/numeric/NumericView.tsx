import { useEffect } from "react";
import { create } from "zustand";
import { Segmented } from "../components/controls";
import { formatNumber } from "../format";
import { useSimulator } from "../simulator/simulatorStore";

interface LabResult {
  layers: { layer: number; error: number }[];
  logits_error: number[];
  state_error: number[];
  prompt_tokens: number;
  agreements: number;
  compared: number;
  first_divergence: number | null;
  has_state: boolean;
}

type Source = "layers" | "tokens" | "state";

interface LabState {
  path: string;
  prompt: string;
  newTokens: number;
  activationBits: number;
  stateBits: number;
  tableBits: number;
  view: Source;
  running: boolean;
  error: string | null;
  result: LabResult | null;
  set: (patch: Partial<LabState>) => void;
  run: () => Promise<void>;
}

const useLab = create<LabState>((set, get) => ({
  path: "",
  prompt: "La capital de Francia es",
  newTokens: 16,
  activationBits: 16,
  stateBits: 16,
  tableBits: 12,
  view: "layers",
  running: false,
  error: null,
  result: null,
  set: (patch) => set(patch),
  async run() {
    const { path, prompt, newTokens, activationBits, stateBits, tableBits } = get();
    set({ running: true, error: null });
    try {
      const response = await fetch("/api/lab", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ path, prompt, new_tokens: newTokens, activation_bits: activationBits, state_bits: stateBits, table_bits: tableBits }) });
      if (!response.ok) throw new Error((await response.text()) || `Error ${response.status}`);
      set({ result: await response.json(), running: false });
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e), running: false });
    }
  },
}));

const percent = (value: number) => `${(value * 100).toLocaleString("es-ES", { maximumSignificantDigits: 3 })} %`;

/** One relative error per row, on a shared scale so that rows can be compared. */
function ErrorBars({ rows }: { rows: { label: string; value: number }[] }) {
  const largest = Math.max(...rows.map((r) => r.value), 1e-12);
  return (
    <div className="lab-bars" role="table" aria-label="Error relativo">
      {rows.map((row) => (
        <div className="lab-row" role="row" key={row.label} title={`${row.label}: ${percent(row.value)}`}>
          <span role="cell" className="lab-label">{row.label}</span>
          <span role="cell" className="lab-track"><i style={{ width: `${(100 * row.value) / largest}%` }} /></span>
          <span role="cell" className="lab-value">{percent(row.value)}</span>
        </div>
      ))}
    </div>
  );
}

export function NumericView() {
  const { result, view, set, running, error } = useLab();
  const rows = !result
    ? []
    : view === "layers"
      ? result.layers.map((l) => ({ label: `Capa ${l.layer}`, value: l.error }))
      : view === "tokens"
        ? result.logits_error.map((value, i) => ({ label: `Token ${i + 1}`, value }))
        : result.state_error.map((value, i) => ({ label: `Token ${i + 1}`, value }));
  const options: [Source, string][] = [["layers", "Por capa"], ["tokens", "Logits por token"]];
  if (result?.has_state) options.push(["state", "Estado recurrente"]);

  return (
    <section className="stage models-stage">
      <div className="models-column">
        {error && (
          <div className="issue error" role="alert">
            <span className="issue-icon" aria-hidden="true">!</span>
            <div><p>{error}</p></div>
          </div>
        )}
        {!result && !error && <p className="placeholder models-empty">{running ? "Ejecutando el modelo en coma flotante y en la aritmética del chip…" : "Elige un modelo y una aritmética para ver cuánto se aleja el chip de la referencia."}</p>}
        {result && (
          <>
            <Segmented label="Vista" value={view} options={options} onChange={(value) => set({ view: value })} />
            <ErrorBars rows={rows} />
          </>
        )}
      </div>
    </section>
  );
}

export function NumericPanel() {
  const lab = useLab();
  const models = useSimulator((s) => s.models);
  const connect = useSimulator((s) => s.connect);
  useEffect(() => {
    void connect();
  }, [connect]);
  const runnable = models.filter((m) => m.supported);
  const path = lab.path || runnable[0]?.path || "";
  const { result } = lab;
  const number = (key: "newTokens" | "activationBits" | "stateBits" | "tableBits", label: string, min: number, max: number) => (
    <label>
      {label}
      <input type="number" min={min} max={max} value={lab[key]} onChange={(e) => lab.set({ [key]: Number(e.target.value) })} />
    </label>
  );

  return (
    <aside className="inspector essential">
      <h2 className="detail-title">Laboratorio numérico</h2>
      <p className="detail-sub">Compara la aritmética del chip con la referencia en coma flotante.</p>
      <div className="inset padded form">
        <label>
          Modelo
          <select value={path} onChange={(e) => lab.set({ path: e.target.value })}>
            {runnable.map((m) => <option key={m.path} value={m.path}>{m.name}</option>)}
          </select>
        </label>
        <label>
          Texto
          <textarea rows={3} value={lab.prompt} onChange={(e) => lab.set({ prompt: e.target.value })} />
        </label>
        {number("newTokens", "Tokens generados", 1, 256)}
        {number("activationBits", "Bits de activación", 4, 24)}
        {number("stateBits", "Bits de estado", 4, 24)}
        {number("tableBits", "Bits de las tablas", 4, 16)}
        <button className="pill primary-pill" disabled={!path || lab.running} onClick={() => { lab.set({ path }); void lab.run(); }}>{lab.running ? "Ejecutando…" : "Comparar"}</button>
      </div>
      {result && (
        <>
          <h3>Resultado</h3>
          <dl className="inset padded lab-summary">
            <div><dt>Token elegido igual</dt><dd>{formatNumber(result.agreements)} de {formatNumber(result.compared)}</dd></div>
            <div><dt>Primera diferencia</dt><dd>{result.first_divergence === null ? "Ninguna" : `Token ${result.first_divergence + 1}`}</dd></div>
          </dl>
        </>
      )}
    </aside>
  );
}
