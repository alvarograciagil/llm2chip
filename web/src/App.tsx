import { LogicPanel, LogicView } from "./logic/LogicView";
import { lazy, Suspense, useEffect } from "react";
import { FloorplanView } from "./components/FloorplanView";
import { Inspector } from "./components/Inspector";
import { Sidebar } from "./components/Sidebar";
import { Toolbar } from "./components/Toolbar";
import { ColumnResizers } from "./components/ColumnResizer";
import { CalibrationPanel, ExplorerView } from "./explorer/ExplorerView";
import { SimulatorPanel, SimulatorView } from "./simulator/SimulatorView";
import { ImagesPanel, ImagesView } from "./images/ImagesView";
import { VoicePanel, VoiceView } from "./voice/VoiceView";
import { SchematicPanel } from "./schematic/SchematicPanel";
import { SchematicView } from "./schematic/SchematicView";
import { ModelsPanel, ModelsView } from "./models/ModelsView";
import { NumericPanel, NumericView } from "./numeric/NumericView";
import { ParamDocsView } from "./docs/ParamDocsView";
import { OptimizePanel, OptimizeView } from "./optimize/OptimizeView";
import { useCellSim } from "./schematic/simStore";
import { useDesign } from "./store";
import { DEMO } from "./demo";
import { DemoStage, ENGINE_VIEWS } from "./components/EngineMissing";

// The 3D viewer brings Three.js, which only downloads when the view is opened.
const ChipViewer = lazy(() => import("./viewer3d/ChipViewer"));

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
    if (view === "schematic" && model) rebuild(model.config, JSON.stringify(params));
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
        <ColumnResizers />
        <Sidebar />
        {DEMO && ENGINE_VIEWS.has(view) ? <DemoStage view={view} /> : <Views />}
      </main>
    </div>
  );
}

function Views() {
  const view = useDesign((s) => s.view);
  return (
    <>
        {view === "floorplan" && <FloorplanView />}
        {view === "floorplan" && <Inspector />}
        {view === "logic" && <LogicView />}
        {view === "logic" && <LogicPanel />}
        {view === "schematic" && <SchematicView />}
        {view === "schematic" && <SchematicPanel />}
        {view === "simulator" && <SimulatorView />}
        {view === "simulator" && <SimulatorPanel />}
        {view === "images" && <ImagesView />}
        {view === "images" && <ImagesPanel />}
        {view === "voice" && <VoiceView />}
        {view === "voice" && <VoicePanel />}
        {view === "explorer" && <ExplorerView />}
        {view === "explorer" && <CalibrationPanel />}
        {view === "optimize" && <OptimizeView />}
        {view === "optimize" && <OptimizePanel />}
        {view === "numeric" && <NumericView />}
        {view === "numeric" && <NumericPanel />}
        {view === "models" && <ModelsView />}
        {view === "models" && <ModelsPanel />}
        {view === "parameters" && <ParamDocsView />}
        {view === "viewer" && (
          <Suspense fallback={<section className="stage" />}>
            <ChipViewer />
          </Suspense>
        )}
    </>
  );
}
