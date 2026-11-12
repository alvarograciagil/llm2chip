import { describe, expect, it } from "vitest";
import { addActivity, busiestWork, cpuCycles, describeStep, groupCycles, GROUPS, type StepActivity } from "./activity";

const step: StepActivity = { cpu: { order: [400, 90], wait_die: [11000, 3000], sample: [120, 30], send: [60, 15] }, routines: { token: [1, 11000] }, samples: [1, 600] };

describe("chip activity", () => {
  it("puts every task in one group", () => {
    const tasks = GROUPS.flatMap((g) => g.tasks);
    expect(new Set(tasks).size).toBe(tasks.length);
    expect(tasks.length).toBe(11);
  });

  it("adds steps and keeps the cycles of the CPU", () => {
    const both = addActivity(step, step);
    expect(cpuCycles(both)).toBe(2 * cpuCycles(step));
    expect(both.routines.token).toEqual([2, 22000]);
    expect(groupCycles(both).reduce((s, g) => s + g.cycles, 0)).toBe(cpuCycles(both));
  });

  it("tells the CPU's work from its waiting", () => {
    expect(busiestWork(step)).toBe("die");
    expect(busiestWork({ cpu: { wait_die: [5, 1] }, routines: {}, samples: [0, 0] })).toBeNull();
  });

  it("says what the chip did in the order it does it", () => {
    const lines = describeStep({ kind: "token", index: 3, text: " París", cycles: 0, activity: step }, String, true);
    expect(lines[0]).toMatch(/registros de parámetros/);
    expect(lines.some((l) => l.includes("«pasar un token por todas las capas»: 11000 ciclos"))).toBe(true);
    expect(lines.at(-1)).toMatch(/« París»/);
  });
});
