import { useEffect, useMemo } from "react";
import { useT } from "../i18n";
import { NetlistStage } from "../schematic/NetlistStage";
import { currentModule } from "../schematic/simStore";
import { frameAt, scaledActivity, useInside, enginePath } from "./inside";
import { useViewer } from "./viewerStore";
import type { Trace } from "./trace";

/** A die from the inside, down to its gates, at the cycle on show: its netlist as a
 * schematic with the value of every wire, and each block shaded by how much of it changed
 * since the cycle before. */
export function InsideDie({ trace }: { trace: Trace }) {
  const t = useT();
  const die = useInside((s) => s.die);
  const path = useInside((s) => s.path);
  const area = useInside((s) => s.area);
  const selected = useInside((s) => s.selected);
  const designs = useInside((s) => s.designs);
  const frames = useInside((s) => s.frames);
  const loading = useInside((s) => s.loading);
  const error = useInside((s) => s.error);
  const { open, goTo, setArea, select, leave, fetch } = useInside.getState();
  const cycle = useViewer((s) => Math.floor(s.cycle));
  const design = die !== null ? (designs[die] ?? null) : null;

  useEffect(() => {
    if (die !== null) void fetch(trace, cycle);
  }, [die, path, cycle, trace, fetch]);

  const shown = frames && frames.die === die && frames.path === enginePath(path) ? frames.list : undefined;
  const frame = frameAt(shown, cycle);
  const module = currentModule(design, path);
  const activity = useMemo(() => scaledActivity(module, frame), [module, frame]);
  if (die === null) return null;
  return (
    <div className="viewer-inside">
      {design ? (
        <NetlistStage
          design={design}
          path={path}
          area={area}
          snapshot={frame?.nets ?? {}}
          selected={selected}
          onSelect={select}
          onEnter={open}
          onArea={setArea}
          onGoTo={goTo}
          activity={activity}
          label={t(`Inside die ${die}`)}
          extra={
            <div className="viewer-inside-bar">
              <button className="pill" onClick={leave}>{t("Back to the chips")}</button>
              <span className="muted-text">{loading ? t("Replaying the gates…") : frame ? t(`Cycle ${frame.cycle}`) : ""}</span>
            </div>
          }
        />
      ) : (
        <div className="viewer-empty">
          <p>{loading ? t("Replaying the gates…") : (error ? t(error) : "")}</p>
          <button className="pill" onClick={leave}>{t("Back to the chips")}</button>
        </div>
      )}
      {design && error && <div className="viewer-inside-error">{t(error)}</div>}
    </div>
  );
}
