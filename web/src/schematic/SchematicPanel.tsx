import { Segmented } from "../components/controls";
import { useDesign } from "../store";
import { download } from "../download";
import { makeZip } from "../zip";
import { formatNumber } from "../format";
import { currentModule, toSigned, useCellSim, validateInputs, type DieDemo } from "./simStore";

const COVERAGE: [string, boolean][] = [
  ["Cell: memory, multiplier and accumulator", true],
  ["Scale and bias per group", true],
  ["Activation network", true],
  ["Vector unit and function table", true],
  ["Vector bank", true],
  ["Sequencer and microcode", true],
  ["Exponent per vector", true],
  ["Attention with softmax, as a microprogram", true],
  ["DeltaNet delta rule, as a microprogram", true],
  ["RMS and L2 norms, and function tables", true],
  ["Die-to-die links", true],
  ["A whole Qwen3.5 model, compiled and run in gates", true],
  ["Several vector lanes in parallel", false],
  ["Mixture of experts and MTP head in gates", true],
  ["Fixed microprogram, with loops and parameters", true],
  ["CPU, image loading and sampling on the chip", true],
  ["ECC, BIST and repair", false],
];

const GATE_NAMES: Record<string, string> = { and: "AND", or: "OR", xor: "XOR", not: "NOT", mux: "Selectors", dff: "Flip-flops", sram: "Memories", rom: "ROM" };

function Selected() {
  const design = useCellSim((s) => s.design);
  const path = useCellSim((s) => s.path);
  const selected = useCellSim((s) => s.selected);
  const enter = useCellSim((s) => s.enter);
  const module = currentModule(design, path);
  const inst = module?.instances.find((i) => i.name === selected);
  if (!inst || !design) return null;
  const child = inst.kind.kind === "module" ? design.modules[inst.kind.of] : null;
  const primNames: Record<string, string> = { and2: "AND gate", or2: "OR gate", xor2: "XOR gate", not: "Inverter", mux2: "Selector", dff: "D flip-flop", sram: "Synchronous SRAM", rom: "ROM, written at manufacturing" };
  return (
    <>
      <h3>Selection</h3>
      <div className="inset padded">
        <strong>{child ? child.title : primNames[(inst.kind as { of: { type: string } }).of.type]}</strong>
        <p className="muted-text">{child ? child.description : `Instance ${inst.name}, in ${inst.group}.`}</p>
        {child && (
          <button className="link" onClick={() => enter(inst.name)}>
            Enter this block
          </button>
        )}
      </div>
    </>
  );
}

const DEMO_TEXT: Record<DieDemo, string> = {
  mlp: "Three cells run an MLP layer: two matrices, activation, product, another matrix and residual.",
  attention: "A query against three cached tokens: dot product, softmax with tables and weighted sum.",
  delta: "A Gated DeltaNet head: decay, memory, error, state update and output.",
  boot: "The whole chip boots from its image: the controller copies the firmware, and the firmware loads each block into the die, checking its CRC.",
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
  const activity = useCellSim((s) => s.dieActivity);
  if (!die || !counts) return null;
  const within = (v: number, e: number) => Math.abs(v - e) <= results!.tolerance * Math.max(Math.abs(e), 0.05);
  const ok = results && results.exact && results.got.every((v, i) => within(v, results.expected[i]));
  return (
    <>
      <div className="scope">
        <Segmented label="Program" value={demo} options={[["mlp", "MLP"], ["attention", "Attention"], ["delta", "DeltaNet"], ["boot", "Start-up"]]} onChange={setDemo} />
      </div>
      <p className="detail-sub">{DEMO_TEXT[demo]}</p>
      <div className="sim-actions">
        <button className="pill primary-pill" disabled={done} onClick={step}>Step</button>
        <button className="pill" disabled={done} onClick={runAll}>Run all</button>
        <button className="pill" onClick={restart}>Restart</button>
      </div>
      {activity && demo !== "boot" && (
        <div className="inset padded sim-explain" aria-live="polite">
          <strong>{done ? "Program finished" : `Word ${activity.pc} of the microprogram`}</strong>
          <p>{done ? "The sequencer has stopped: the result is in the vector bank." : activity.text}</p>
        </div>
      )}
      <dl className="inset metrics">
        <div><dt>State</dt><dd>{done ? "Finished" : cycle > 0 ? "Running" : "Loaded"}</dd></div>
        <div><dt>Clock cycle</dt><dd>{formatNumber(cycle)}</dd></div>
      </dl>
      {results?.text !== undefined && (
        <>
          <h3>Output on the host port</h3>
          <div className="inset padded"><code>{results.text}</code></div>
          <div className={`issue ${results.exact ? "ok" : "error"}`} role="status">
            <span className="issue-icon" aria-hidden="true">{results.exact ? "✓" : "!"}</span>
            <div><strong>{results.exact ? "Every block has passed the CRC check." : "Some block has not passed the CRC check."}</strong></div>
          </div>
        </>
      )}
      {results && results.text === undefined && (
        <>
          <h3>Layer result</h3>
          <dl className="inset metrics">
            {results.got.map((v, i) => (
              <div key={i}><dt>Output {i} (expected {formatNumber(results.expected[i])})</dt><dd>{formatNumber(v)}</dd></div>
            ))}
          </dl>
          <div className={`issue ${ok ? "ok" : "error"}`} role="status">
            <span className="issue-icon" aria-hidden="true">{ok ? "✓" : "!"}</span>
            <div><strong>{ok ? (results.tolerance > 0 ? `It matches the integer model bit for bit and, within ${results.tolerance * 100} %, floating point.` : "The die reproduces the layer bit for bit.") : "The die does not match the reference."}</strong></div>
          </div>
        </>
      )}
      <h3>{demo === "boot" ? "Chip circuit" : "Die circuit"}</h3>
      <dl className="inset metrics">
        {Object.entries(counts).map(([name, count]) => (
          <div key={name}><dt>{GATE_NAMES[name] ?? name}</dt><dd>{formatNumber(count)}</dd></div>
        ))}
      </dl>
      <div className="sim-actions">
        <button className="pill" onClick={() => download(die.verilog(), demo === "boot" ? "llm2chip_chip.v" : "llm2chip_die.v")}>Export to Verilog</button>
        <button className="pill" onClick={() => download(die.vhdl(), demo === "boot" ? "llm2chip_chip.vhd" : "llm2chip_die.vhd")}>Export to VHDL</button>
      </div>
    </>
  );
}

function Coverage() {
  return (
    <>
      <h3>What the schematic already has</h3>
      <div className="inset">
        {COVERAGE.map(([name, done]) => (
          <div className="coverage" key={name}>
            <span className={done ? "cov-mark done" : "cov-mark"} aria-hidden="true">{done ? "✓" : "○"}</span>
            <span>{name}</span>
            <small>{done ? "Working" : "Pending"}</small>
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
      <Segmented label="Scope" value={scope} options={[["cell", "Cell"], ["die", "Reduced die"]]} onChange={setScope} />
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
  const done = index > 0 ? script[index - 1] : null;
  const next = script[index]?.label ?? (finished ? "Nothing: the sum is in the accumulator" : "—");
  const counts = Object.entries(info.counts);

  return (
    <aside className="inspector">
      {selector}
      <h2 className="detail-title">Simulation</h2>
      <p className="detail-sub">Gate by gate, on the circuit you see.</p>

      <div className="inset padded form">
        <label>
          Weight codes ({info.config.weight_bits} bits)
          <textarea rows={2} spellCheck={false} value={weightsText} onChange={(e) => setWeights(e.target.value)} />
        </label>
        <label>
          Activations ({info.config.activation_bits} signed bits)
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
          Step
        </button>
        <button className="pill" disabled={finished || script.length === 0} onClick={runAll}>
          Run all
        </button>
        <button className="pill" onClick={restart}>
          Restart
        </button>
      </div>

      <div className="inset padded sim-explain" aria-live="polite">
        <strong>{done ? done.label : "Before starting"}</strong>
        <p>{done ? done.explain : "Press Step to advance one clock cycle: each step sets the cell's inputs, the circuit computes and the wires with a value light up in the schematic."}</p>
      </div>
      <dl className="inset metrics">
        <div>
          <dt>Next step</dt>
          <dd>{next}</dd>
        </div>
        <div>
          <dt>Clock cycle</dt>
          <dd>{formatNumber(cycle)}</dd>
        </div>
        <div>
          <dt>Result in the circuit</dt>
          <dd>{total === null ? (
            <button className="link inline" onClick={() => goTo(0)}>See in the cell</button>
          ) : total.toString()}</dd>
        </div>
        <div>
          <dt>Expected (Σ weight × activation)</dt>
          <dd>{expected?.toString() ?? "—"}</dd>
        </div>
      </dl>
      {finished && total !== null && expected !== null && (
        <div className={`issue ${total === expected ? "ok" : "error"}`} role="status">
          <span className="issue-icon" aria-hidden="true">{total === expected ? "✓" : "!"}</span>
          <div>
            <strong>{total === expected ? "The circuit gives the correct result." : "The circuit does not match the reference."}</strong>
          </div>
        </div>
      )}

      <Selected />

      <h3>Cell circuit</h3>
      <dl className="inset metrics">
        {counts.map(([name, count]) => (
          <div key={name}>
            <dt>{GATE_NAMES[name] ?? name}</dt>
            <dd>{formatNumber(count)}</dd>
          </div>
        ))}
        <div>
          <dt>Latency</dt>
          <dd>{info.latency} cycles</dd>
        </div>
        <div>
          <dt>Weight memory</dt>
          <dd>{formatNumber(2 ** info.config.addr_bits)} × {info.config.weight_bits} bits</dd>
        </div>
      </dl>
      <button className="plain" onClick={() => download(makeZip(JSON.parse(handle.project(1 / Math.max(frequency, 0.001)))) as BlobPart, "llm2chip_cell.zip", "application/zip")}>
        Export the whole project
      </button>
      <button className="plain" onClick={() => download(handle.verilog(), "llm2chip_cell.v")}>
        Export to Verilog
      </button>

      <Coverage />
      {path.length > 0 && <p className="legend-note">Press Esc to go up a level.</p>}
    </aside>
  );
}
