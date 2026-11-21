import { NetlistStage } from "./NetlistStage";
import { useCellSim } from "./simStore";

/** The schematic of the cell or the reduced die the simulator runs. */
export function SchematicView() {
  const design = useCellSim((s) => s.design);
  const path = useCellSim((s) => s.path);
  const area = useCellSim((s) => s.area);
  const snapshot = useCellSim((s) => s.snapshot);
  const selected = useCellSim((s) => s.selected);
  const { select, enter, setArea, goTo } = useCellSim.getState();
  return <NetlistStage design={design} path={path} area={area} snapshot={snapshot} selected={selected} onSelect={select} onEnter={enter} onArea={setArea} onGoTo={goTo} />;
}
