# 3D chip viewer (plan)

A view of the chip at work, in three dimensions: the dies and their units, tokens going
through the layers, the KV cache filling up, the links between chips and where the work
stalls. It is meant both to show the project and to debug it, so everything it draws comes
from a simulation of the circuit, never from an animation made up beside it.

Work happens on the `chip-viewer` branch, one commit per phase, and reaches `main` only
once it is agreed.

## Optional by construction

The viewer is an add-on: nothing in the compiler, the simulators or the existing views
depends on it.

- **Rust:** the trace recorder is its own workspace crate, `crates/trace`, on top of `sim`
  and `netlist`. No `cfg(feature)` inside the core crates, so no feature matrix to test;
  `scripts/check.sh` covers it like any other crate.
- **Engine:** one more job route, `/api/trace`, in the pattern of `/api/optimize`
  (start, status, cancel).
- **Web:** the view lives in `web/src/viewer3d/` and is loaded with `React.lazy`, so
  `three` goes into its own chunk and is only downloaded when the view is opened.

The only changes to the core are two small, behaviour-free additions:

1. **Operation marks in the microcode.** A `Microprogram` has words, entries and
   parameters but no record of which words belong to which operation of the model.
   Lowering records, for each routine, the first word of every `program.ops[i]` it emits.
   No word changes; the existing tests prove it.
2. **The multi-die harness as a library.** The code that joins dies with their links and
   runs them cycle by cycle (`run_joined`, `loaded_die`) and the tiny checkpoints live in
   `crates/sim/tests`. They move to `crates/trace`, and the tests call that copy.

## What a trace holds

A trace is recorded once and played back in the browser at any speed, forwards and back.
It stores changes and runs, not frames:

- per die and cycle: the microcode address (`pc`), whether it has halted, and for every
  link lane `tx_valid`, `rx_ready` and `ack`. These come from top-level ports, read with
  `Simulator::get` (a `snapshot` per cycle would be far too slow);
- what each word does, decoded once per address with `decode_word`/`describe_word`:
  its operation, the units and lanes it works on, cell passes, link sends and takes;
- the operation of the model each word belongs to (from the marks above), and with it the
  layer, the token and the routine;
- the KV cache: which slots of which unit a word writes, from the registers `Layout` gives
  the keys and values;
- stalls: cycles where a die waits on a link (valid without ready, or a take without
  data), or has halted while others still run.

## Two sources

1. **Gates, tiny model (debugging).** A small Qwen3.5-shaped model on N dies of M units,
   joined by their links and simulated gate by gate. Exact, cycle by cycle; runs in the
   engine within seconds.
2. **The real design (showing).** Qwen3.8-27B on 16 dies in groups of 8 (or any design of
   the optimiser), at operation granularity from `ShareOut::cycles_by_op` and
   `sizing::timing`. Gates cannot simulate a design that size. It needs the checkpoint on
   disk, runs only in the engine, and its traces are cached under `~/.llm2chip/cache`;
   none is committed. Link latency is not modelled there, as in the sizing.

## Phases

1. **Recorder.** Operation marks; harness moved to `crates/trace`; trace format and
   recorder over the gates; an example that writes a trace to a file.
   Done when a test shows the trace's cycles per routine equal what the joined run takes,
   and the final registers match the exact machine.
2. **3D view.** `three` pinned to an exact version. The dies, units and lanes as a grid,
   the ring of links, the KV cache as columns that fill up; playback with speed (real time
   to one word at a time), steps by word, operation, layer and token, a timeline to scrub,
   and an inspector for a unit (current word explained, its registers, its cache slots).
   Stalls in red. A `/api/trace` job. First visible result: the tiny model on gates.
3. **Divergences and breakpoints.** Compare every operation's outputs with the exact
   machine and mark the first die, unit and register that differ; single-die word-level
   lockstep against the microcode VM. Breakpoints on an operation kind, a layer, the first
   stall or the first divergence.
   Done when a test that injects a known fault finds the right operation and unit.
4. **Real design.** The Qwen design at operation granularity: which dies work and which
   wait, layer by layer, for any chip count and sharing.
5. **Extras.** Accumulated use as a heat map and where the cycles go (critical path);
   packets on the ring with queue occupancy and the rounds of exponent agreement; two
   designs side by side (for instance 12 dies in groups of 4 against 16 in groups of 8,
   or by layers against mixed); guided camera tours with captions for presentations, and
   export to video; power by block from the activity, through the estimator.

Tests follow the two tiers: anything over ~20 s is `#[ignore = "slow: …"]`. Every new
text in the app gets its Spanish translation (the coverage test enforces it).
