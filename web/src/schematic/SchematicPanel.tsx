import { Segmented } from "../components/controls";
import { useDesign } from "../store";
import { download } from "../download";
import { makeZip } from "../zip";
import { formatNumber } from "../format";
import { currentModule, toSigned, useCellSim, validateInputs, type DieDemo } from "./simStore";

const COVERAGE: [string, boolean][] = [
  ["Celda: memoria, multiplicador y acumulador", true],
  ["Escala y sesgo por grupo", true],
  ["Red de activaciones", true],
  ["Unidad vectorial y tabla de funciones", true],
  ["Banco de vectores", true],
  ["Secuenciador y microprograma", true],
  ["Exponente por vector", true],
  ["Atención con softmax, como microprograma", true],
  ["Regla delta de DeltaNet, como microprograma", true],
  ["Normas RMS y L2, y tablas de funciones", true],
  ["Enlaces entre dies", true],
  ["Un modelo Qwen3.5 completo, compilado y ejecutado en puertas", true],
  ["Varios carriles vectoriales en paralelo", false],
  ["Mezcla de expertos y cabeza MTP en puertas", true],
  ["Microprograma fijo, con bucles y parámetros", true],
  ["CPU, carga de la imagen y muestreo en el chip", true],
  ["ECC, BIST y reparación", false],
];

const GATE_NAMES: Record<string, string> = { and: "AND", or: "OR", xor: "XOR", not: "NOT", mux: "Selectores", dff: "Biestables", sram: "Memorias" };

function Selected() {
  const design = useCellSim((s) => s.design);
  const path = useCellSim((s) => s.path);
  const selected = useCellSim((s) => s.selected);
  const enter = useCellSim((s) => s.enter);
  const module = currentModule(design, path);
  const inst = module?.instances.find((i) => i.name === selected);
  if (!inst || !design) return null;
  const child = inst.kind.kind === "module" ? design.modules[inst.kind.of] : null;
  const primNames: Record<string, string> = { and2: "Puerta AND", or2: "Puerta OR", xor2: "Puerta XOR", not: "Inversor", mux2: "Selector", dff: "Biestable D", sram: "Memoria SRAM síncrona" };
  return (
    <>
      <h3>Selección</h3>
      <div className="inset padded">
        <strong>{child ? child.title : primNames[(inst.kind as { of: { type: string } }).of.type]}</strong>
        <p className="muted-text">{child ? child.description : `Instancia ${inst.name}, en ${inst.group}.`}</p>
        {child && (
          <button className="link" onClick={() => enter(inst.name)}>
            Entrar en este bloque
          </button>
        )}
      </div>
    </>
  );
}

const DEMO_TEXT: Record<DieDemo, string> = {
  mlp: "Tres celdas ejecutan una capa MLP: dos matrices, activación, producto, otra matriz y residual.",
  attention: "Una consulta contra tres tokens en caché: producto escalar, softmax con tablas y suma ponderada.",
  delta: "Una cabeza de Gated DeltaNet: decaimiento, memoria, error, actualización del estado y salida.",
  boot: "El chip completo arranca desde su imagen: el controlador copia el firmware y este carga cada bloque en el die, comprobando su CRC.",
};

function DiePanel() {
  const die = useCellSim((s) => s.die);
  const counts = useCellSim((s) => s.dieCounts);
  const cycle = useCellSim((s) => s.cycle);
  const done = useCellSim((s) => s.dieDone);
  const results = useCellSim((s) => s.dieResults);
  const step = useCellSim((s) => s.step);
  const runAll = useCellSim((s) => s.runAll);
  const restart = useCellSim((s) => s.restart);
  const demo = useCellSim((s) => s.demo);
  const setDemo = useCellSim((s) => s.setDemo);
  if (!die || !counts) return null;
  const within = (v: number, e: number) => Math.abs(v - e) <= results!.tolerance * Math.max(Math.abs(e), 0.05);
  const ok = results && results.exact && results.got.every((v, i) => within(v, results.expected[i]));
  return (
    <>
      <div className="scope">
        <Segmented label="Programa" value={demo} options={[["mlp", "MLP"], ["attention", "Atención"], ["delta", "DeltaNet"], ["boot", "Arranque"]]} onChange={setDemo} />
      </div>
      <p className="detail-sub">{DEMO_TEXT[demo]}</p>
      <div className="sim-actions">
        <button className="pill primary-pill" disabled={done} onClick={step}>Paso</button>
        <button className="pill" disabled={done} onClick={runAll}>Ejecutar todo</button>
        <button className="pill" onClick={restart}>Reiniciar</button>
      </div>
      <dl className="inset metrics">
        <div><dt>Estado</dt><dd>{done ? "Terminado" : cycle > 0 ? "Ejecutando" : "Cargado"}</dd></div>
        <div><dt>Ciclo de reloj</dt><dd>{formatNumber(cycle)}</dd></div>
      </dl>
      {results?.text !== undefined && (
        <>
          <h3>Salida por el puerto de host</h3>
          <div className="inset padded"><code>{results.text}</code></div>
          <div className={`issue ${results.exact ? "ok" : "error"}`} role="status">
            <span className="issue-icon" aria-hidden="true">{results.exact ? "✓" : "!"}</span>
            <div><strong>{results.exact ? "Todos los bloques han pasado la comprobación de CRC." : "Algún bloque no ha pasado la comprobación de CRC."}</strong></div>
          </div>
        </>
      )}
      {results && results.text === undefined && (
        <>
          <h3>Resultado de la capa</h3>
          <dl className="inset metrics">
            {results.got.map((v, i) => (
              <div key={i}><dt>Salida {i} (esperada {formatNumber(results.expected[i])})</dt><dd>{formatNumber(v)}</dd></div>
            ))}
          </dl>
          <div className={`issue ${ok ? "ok" : "error"}`} role="status">
            <span className="issue-icon" aria-hidden="true">{ok ? "✓" : "!"}</span>
            <div><strong>{ok ? (results.tolerance > 0 ? `Coincide bit a bit con el modelo entero y, dentro del ${results.tolerance * 100} %, con la coma flotante.` : "El die reproduce la capa bit a bit.") : "El die no coincide con la referencia."}</strong></div>
          </div>
        </>
      )}
      <h3>{demo === "boot" ? "Circuito del chip" : "Circuito del die"}</h3>
      <dl className="inset metrics">
        {Object.entries(counts).map(([name, count]) => (
          <div key={name}><dt>{GATE_NAMES[name] ?? name}</dt><dd>{formatNumber(count)}</dd></div>
        ))}
      </dl>
      <div className="sim-actions">
        <button className="pill" onClick={() => download(die.verilog(), demo === "boot" ? "llm2chip_chip.v" : "llm2chip_die.v")}>Exportar a Verilog</button>
        <button className="pill" onClick={() => download(die.vhdl(), demo === "boot" ? "llm2chip_chip.vhd" : "llm2chip_die.vhd")}>Exportar a VHDL</button>
      </div>
    </>
  );
}

function Coverage() {
  return (
    <>
      <h3>Qué hay ya en el esquemático</h3>
      <div className="inset">
        {COVERAGE.map(([name, done]) => (
          <div className="coverage" key={name}>
            <span className={done ? "cov-mark done" : "cov-mark"} aria-hidden="true">{done ? "✓" : "○"}</span>
            <span>{name}</span>
            <small>{done ? "Funcional" : "Pendiente"}</small>
          </div>
        ))}
      </div>
    </>
  );
}

export function SchematicPanel() {
  const scope = useCellSim((s) => s.scope);
  const setScope = useCellSim((s) => s.setScope);
  const selector = (
    <div className="scope">
      <Segmented label="Alcance" value={scope} options={[["cell", "Celda"], ["die", "Die reducido"]]} onChange={setScope} />
    </div>
  );
  if (scope === "die") {
    return (
      <aside className="inspector">
        {selector}
        <h2 className="detail-title">Die</h2>
        <DiePanel />
        <Selected />
        <Coverage />
      </aside>
    );
  }
  return <CellPanel selector={selector} />;
}

function CellPanel({ selector }: { selector: React.ReactNode }) {
  const frequency = useDesign((s) => s.report?.metrics.find((m) => m.id === "freq_ghz")?.value ?? 1);
  const info = useCellSim((s) => s.info);
  const handle = useCellSim((s) => s.handle);
  const script = useCellSim((s) => s.script);
  const index = useCellSim((s) => s.index);
  const cycle = useCellSim((s) => s.cycle);
  const weightsText = useCellSim((s) => s.weightsText);
  const activationsText = useCellSim((s) => s.activationsText);
  const inputError = useCellSim((s) => s.inputError);
  const setWeights = useCellSim((s) => s.setWeights);
  const setActivations = useCellSim((s) => s.setActivations);
  const step = useCellSim((s) => s.step);
  const runAll = useCellSim((s) => s.runAll);
  const restart = useCellSim((s) => s.restart);
  const path = useCellSim((s) => s.path);
  const goTo = useCellSim((s) => s.goTo);
  const topTotal = useCellSim((s) => (s.path.length === 0 ? s.snapshot.total : undefined));

  if (!info || !handle) return <aside className="inspector" />;

  const parsed = validateInputs(info, weightsText, activationsText);
  const expected = "error" in parsed ? null : BigInt(handle.expected(parsed.weights.map(String).join(","), parsed.activations.map(String).join(",")));
  const finished = index >= script.length && script.length > 0;
  const total = topTotal !== undefined ? toSigned(topTotal) : null;
  const status = script[index]?.label ?? (finished ? "Terminado" : "Sin programa");
  const counts = Object.entries(info.counts);

  return (
    <aside className="inspector">
      {selector}
      <h2 className="detail-title">Simulación</h2>
      <p className="detail-sub">Puerta a puerta, sobre el circuito que ves.</p>

      <div className="inset padded form">
        <label>
          Códigos de peso ({info.config.weight_bits} bits)
          <textarea rows={2} spellCheck={false} value={weightsText} onChange={(e) => setWeights(e.target.value)} />
        </label>
        <label>
          Activaciones ({info.config.activation_bits} bits con signo)
          <textarea rows={2} spellCheck={false} value={activationsText} onChange={(e) => setActivations(e.target.value)} />
        </label>
      </div>
      {inputError && (
        <div className="issue error" role="alert">
          <span className="issue-icon" aria-hidden="true">!</span>
          <div>
            <strong>{inputError}</strong>
          </div>
        </div>
      )}

      <div className="sim-actions">
        <button className="pill primary-pill" disabled={finished || script.length === 0} onClick={step}>
          Paso
        </button>
        <button className="pill" disabled={finished || script.length === 0} onClick={runAll}>
          Ejecutar todo
        </button>
        <button className="pill" onClick={restart}>
          Reiniciar
        </button>
      </div>

      <dl className="inset metrics">
        <div>
          <dt>Fase</dt>
          <dd>{status}</dd>
        </div>
        <div>
          <dt>Ciclo de reloj</dt>
          <dd>{formatNumber(cycle)}</dd>
        </div>
        <div>
          <dt>Resultado en el circuito</dt>
          <dd>{total === null ? (
            <button className="link inline" onClick={() => goTo(0)}>Ver en la celda</button>
          ) : total.toString()}</dd>
        </div>
        <div>
          <dt>Esperado (Σ peso × activación)</dt>
          <dd>{expected?.toString() ?? "—"}</dd>
        </div>
      </dl>
      {finished && total !== null && expected !== null && (
        <div className={`issue ${total === expected ? "ok" : "error"}`} role="status">
          <span className="issue-icon" aria-hidden="true">{total === expected ? "✓" : "!"}</span>
          <div>
            <strong>{total === expected ? "El circuito da el resultado correcto." : "El circuito no coincide con la referencia."}</strong>
          </div>
        </div>
      )}

      <Selected />

      <h3>Circuito de la celda</h3>
      <dl className="inset metrics">
        {counts.map(([name, count]) => (
          <div key={name}>
            <dt>{GATE_NAMES[name] ?? name}</dt>
            <dd>{formatNumber(count)}</dd>
          </div>
        ))}
        <div>
          <dt>Latencia</dt>
          <dd>{info.latency} ciclos</dd>
        </div>
        <div>
          <dt>Memoria de pesos</dt>
          <dd>{formatNumber(2 ** info.config.addr_bits)} × {info.config.weight_bits} bits</dd>
        </div>
      </dl>
      <button className="plain" onClick={() => download(makeZip(JSON.parse(handle.project(1 / Math.max(frequency, 0.001)))) as BlobPart, "llm2chip_celda.zip", "application/zip")}>
        Exportar proyecto completo
      </button>
      <button className="plain" onClick={() => download(handle.verilog(), "llm2chip_celda.v")}>
        Exportar a Verilog
      </button>

      <Coverage />
      {path.length > 0 && <p className="legend-note">Pulsa Esc para subir un nivel.</p>}
    </aside>
  );
}
