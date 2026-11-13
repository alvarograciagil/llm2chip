#!/usr/bin/env python3
"""Compares the generator's gate count with what Yosys and ABC can optimise it to."""

import json
import pathlib
import re
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
AREA = {"and": 0.0875, "or": 0.0875, "xor": 0.1312, "not": 0.0437, "mux": 0.1312, "dff": 0.2916}
FORMATS = [("fixed", 4, 0), ("fixed", 8, 0), ("fixed", 16, 0), ("fixed", 32, 0), ("float", 8, 4), ("float", 16, 5), ("float", 32, 8), ("posit", 16, 1), ("log", 8, 4)]
CASES = [(*f, kind) for kind in ("array", "wallace", "booth") for f in FORMATS]
STORED_WORDS = 4


def area(counts):
    return sum(AREA[gate] * counts.get(gate, 0) for gate in AREA)


def optimise(verilog, folder):
    source = folder / "cell.v"
    stat = folder / "stat.txt"
    source.write_text(verilog)
    script = f"read_verilog {source}; synth -top compute_cell -flatten -noabc; abc -g AND,OR,XOR,MUX; opt_clean; tee -o {stat} stat"
    subprocess.run(["yosys", "-q", "-p", script], check=True, capture_output=True)
    names = {"$_AND_": "and", "$_OR_": "or", "$_XOR_": "xor", "$_NOT_": "not", "$_MUX_": "mux", "$_DFF_P_": "dff", "$_DFFE_PP_": "dffe"}
    counts = {}
    for count, name in re.findall(r"(\d+)\s+(\$_\w+_)", stat.read_text()):
        counts[names.get(name, name)] = counts.get(names.get(name, name), 0) + int(count)
    enabled = counts.pop("dffe", 0)
    counts["dff"] = counts.get("dff", 0) + enabled
    counts["mux"] = counts.get("mux", 0) + enabled
    return counts


def main():
    exporter = ROOT / "target/release/examples/export"
    print("| Multiplier | Format | Bits | Exp | Generator µm² | After ABC µm² | Ratio | Path (FA) |\n|---|---|---|---|---|---|---|---|")
    ratios = {}
    for coding, bits, exponent, kind in CASES:
        result = subprocess.run([exporter, coding, str(bits), str(exponent), kind], check=True, capture_output=True, text=True)
        report = json.loads(result.stderr)
        raw = {k: v for k, v in report["counts"].items() if k in AREA}
        with tempfile.TemporaryDirectory() as tmp:
            optimised = optimise(result.stdout, pathlib.Path(tmp))
        # The test cell's four stored words become flip-flops in Yosys; only logic counts.
        memory = {"dff": STORED_WORDS * bits, "mux": STORED_WORDS * bits}
        optimised = {k: v - memory.get(k, 0) for k, v in optimised.items()}
        before, after = area(raw), area(optimised)
        ratios.setdefault(kind, []).append(after / before)
        print(f"| {kind} | {coding} | {bits} | {exponent} | {before:.1f} | {after:.1f} | {after / before:.2f} | {report['path']:.1f} |")
    for kind, values in ratios.items():
        print(f"\nMedia {kind}: {sum(values) / len(values):.3f}")


if __name__ == "__main__":
    main()
