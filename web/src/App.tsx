import { useEffect } from "react";
import { FloorplanView } from "./components/FloorplanView";
import { Inspector } from "./components/Inspector";
import { Sidebar } from "./components/Sidebar";
import { Toolbar } from "./components/Toolbar";
import { CalibrationPanel, ExplorerView } from "./explorer/ExplorerView";
import { SimulatorPanel, SimulatorView } from "./simulator/SimulatorView";
import { SchematicPanel } from "./schematic/SchematicPanel";
import { SchematicView } from "./schematic/SchematicView";
import { useCellSim } from "./schematic/simStore";
import { useDesign } from "./store";

export function App() {
  const undo = useDesign((s) => s.undo);
  const redo = useDesign((s) => s.redo);
  const view = useDesign((s) => s.view);
  const model = useDesign((s) => s.model);
  const params = useDesign((s) => s.params);
  const cellKb = useDesign((s) => s.report?.metrics.find((m) => m.id === "cell_sram_kb")?.value);
  const rebuild = useCellSim((s) => s.rebuild);
  const cellKey = [model?.id, params.weight_bits, params.use_model_mix, params.activation_bits, params.accumulator_bits, params.mac_stages, params.adder_stages, cellKb].join("|");

  useEffect(() => {
    if (view === "esquematico" && model) rebuild(model.config, JSON.stringify(params));
    // The netlist only depends on the values in cellKey.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view, cellKey, rebuild]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const editing = (e.target as HTMLElement).matches("input[type=text], input[type=search], textarea");
      if (editing || !(e.metaKey || e.ctrlKey) || e.key.toLowerCase() !== "z") return;
      e.preventDefault();
      if (e.shiftKey) redo();
      else undo();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [undo, redo]);

  return (
    <div className="window">
      <Toolbar />
      <main className="split">
        <Sidebar />
        {view === "plano" && <FloorplanView />}
        {view === "plano" && <Inspector />}
        {view === "esquematico" && <SchematicView />}
        {view === "esquematico" && <SchematicPanel />}
        {view === "simulador" && <SimulatorView />}
        {view === "simulador" && <SimulatorPanel />}
        {view === "explorador" && <ExplorerView />}
        {view === "explorador" && <CalibrationPanel />}
      </main>
    </div>
  );
}
