#!/usr/bin/env bash
# Lays out a tile on SkyWater 130: one compute cell whose weight memory is a real SRAM macro
# of the open PDK (OpenRAM's sky130_sram_1rw1r_128x256_8, 32 Kbit), not flip-flops. The
# cell keeps 4096 weights of 8 bits; each word of the macro holds 16 of them.
#
#   tools/asic/tile.sh [period_ns] [folder]
#
# Writes the GDS, the layout images and summary.json (area, fmax, power, DRC) in the folder.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
period=${1:-10}
out=${2:-$HOME/.llm2chip/asic/tesela-sky130hd}
macro=sky130_sram_1rw1r_128x256_8
ram=/OpenROAD-flow-scripts/flow/platforms/sky130ram/$macro
mkdir -p "$out"

cargo build --release -q -p netlist --example export --manifest-path "$root/Cargo.toml"
"$root/target/release/examples/export" --json '{"weight_bits":8,"activation_bits":16,"accumulator_bits":32,"mac_stages":3,"adder_stages":2,"addr_bits":12,"coding":"affine","exponent_bits":0,"multiplier":"wallace","fixed_weights":false,"ecc":false,"skip_zeros":false}' > "$out/cell.v" 2>/dev/null

# The cell's memory, 4096 × 8, becomes the macro: the high address bits pick the word, the
# low ones (registered, as the word arrives a cycle later) the weight in it, and a write
# reaches only that weight's byte through the macro's mask.
python3 - "$out/cell.v" "$out/design.v" "$macro" <<'PY'
import re, sys
source, target, macro = sys.argv[1:]
text = open(source).read()
start = text.index("module llm2chip_sram")
end = text.index("endmodule", start) + len("endmodule")
wrapper = f"""module llm2chip_sram #(parameter ADDR_BITS = 12, parameter WIDTH = 8) (
  input clk, input [ADDR_BITS-1:0] addr, input we,
  input [WIDTH-1:0] wdata, output [WIDTH-1:0] rdata
);
  reg [3:0] lane;
  always @(posedge clk) lane <= addr[3:0];
  wire [127:0] word;
  {macro} memory (
    .clk0(clk), .csb0(1'b0), .web0(~we), .wmask0(16'b1 << addr[3:0]), .addr0(addr[11:4]),
    .din0({{16{{wdata}}}}), .dout0(word),
    .clk1(clk), .csb1(1'b1), .addr1(8'b0), .dout1()
  );
  assign rdata = word[lane*8 +: 8];
endmodule"""
assert "ADDR_BITS(12), .WIDTH(8)" in text, "the tile expects a 4096 × 8 memory"
open(target, "w").write(text[:start] + wrapper + text[end:])
PY

cat > "$out/config.mk" <<CONFIG
export DESIGN_NAME = compute_cell
export DESIGN_NICKNAME = tesela
export PLATFORM = sky130hd
export VERILOG_FILES = /work/design.v $ram/$macro.v
export VERILOG_FILES_BLACKBOX = $ram/$macro.v
export SDC_FILE = /work/constraint.sdc
export ADDITIONAL_LEFS = $ram/$macro.lef
export ADDITIONAL_LIBS = $ram/${macro}_TT_1p8V_25C.lib
export ADDITIONAL_GDS = $ram/$macro.gds
export CORE_UTILIZATION = 40
export PLACE_DENSITY = 0.3
export MACRO_PLACE_HALO = 30 30
export PDN_TCL = /work/pdn.tcl
export TAPCELL_TCL = /work/tapcell.tcl
# OpenRAM's abstract of this macro has supply pads of vdd and gnd that overlap on met3
# (two columns of them), which the power grid analysis reports as shorts and stops at; the
# analysis is left out, and the routing's own checks stay.
export PWR_NETS_VOLTAGES =
export GND_NETS_VOLTAGES =
CONFIG
# The rows stop 10 µm short of the macro, so the cells' grid and its vias keep clear of it.
cat > "$out/tapcell.tcl" <<'TAP'
cut_rows -halo_width_x 10 -halo_width_y 10
tapcell -distance 14 -tapcell_master "$::env(TAP_CELL_NAME)" -halo_width_x 10 -halo_width_y 10
TAP
# OpenRAM names its supply pins vdd and gnd; the platform's grid expects VDD and VSS. The
# macro's supply is thousands of small met3 pads under its met1-met3 obstruction: its own
# grid lays met4 stripes over it, with vias down to the pads and up to met5, while the
# cells' grid keeps 10 µm away.
cat > "$out/pdn.tcl" <<'PDN'
add_global_connection -net {VDD} -inst_pattern {.*} -pin_pattern {^vdd$} -power
add_global_connection -net {VSS} -inst_pattern {.*} -pin_pattern {^gnd$} -ground
add_global_connection -net {VDD} -inst_pattern {.*} -pin_pattern {VPWR} -power
add_global_connection -net {VDD} -inst_pattern {.*} -pin_pattern {VPB}
add_global_connection -net {VSS} -inst_pattern {.*} -pin_pattern {VGND} -ground
add_global_connection -net {VSS} -inst_pattern {.*} -pin_pattern {VNB}
global_connect
set_voltage_domain -name {CORE} -power {VDD} -ground {VSS}
define_pdn_grid -name {grid} -voltage_domains {CORE} -pins {met5}
add_pdn_stripe -grid {grid} -layer {met1} -width {0.48} -pitch {5.44} -offset {0} -followpins
add_pdn_stripe -grid {grid} -layer {met4} -width {1.600} -pitch {27.140} -offset {13.570}
add_pdn_stripe -grid {grid} -layer {met5} -width {1.600} -pitch {27.200} -offset {13.600}
add_pdn_connect -grid {grid} -layers {met1 met4}
add_pdn_connect -grid {grid} -layers {met4 met5}
define_pdn_grid -name {sram} -voltage_domains {CORE} -macro -orient {R0 R180 MX MY R90 R270 MXR90 MYR90} -halo {0.0 0.0 0.0 0.0} -default
add_pdn_stripe -grid {sram} -layer {met4} -width {1.600} -pitch {20.000} -offset {5.000}
add_pdn_connect -grid {sram} -layers {met3 met4}
add_pdn_connect -grid {sram} -layers {met4 met5}
PDN
cat > "$out/constraint.sdc" <<SDC
current_design compute_cell
set clk_period $period
create_clock -name clk -period \$clk_period [get_ports clk]
set non_clock_inputs [lsearch -inline -all -not -exact [all_inputs] [get_ports clk]]
set_input_delay [expr \$clk_period * 0.2] -clock clk \$non_clock_inputs
set_output_delay [expr \$clk_period * 0.2] -clock clk [all_outputs]
SDC
docker run --rm --platform linux/amd64 -v "$out":/work openroad/orfs:latest bash -c \
  "source /OpenROAD-flow-scripts/env.sh >/dev/null && cd /OpenROAD-flow-scripts/flow && make DESIGN_CONFIG=/work/config.mk WORK_HOME=/work/out" > "$out/flow.log" 2>&1
base="$out/out/logs/sky130hd/tesela/base"
reports="$out/out/reports/sky130hd/tesela/base"
python3 - "$base/6_report.json" "$reports/6_finish.rpt" "$out/summary.json" <<'PY'
import json, re, sys
metrics = json.load(open(sys.argv[1]))
finish = open(sys.argv[2]).read()
fmax = re.search(r"fmax = ([0-9.]+)", finish)
summary = {
    "area_cells_um2": metrics.get("finish__design__instance__area__stdcell"),
    "area_macros_um2": metrics.get("finish__design__instance__area__macros"),
    "area_die_um2": metrics.get("finish__design__die__area"),
    "utilization": metrics.get("finish__design__instance__utilization"),
    "cells": metrics.get("finish__design__instance__count__stdcell"),
    "power_w": metrics.get("finish__power__total"),
    "fmax_mhz": float(fmax.group(1)) if fmax else None,
    "drc_violations": metrics.get("detailedroute__route__drc_errors"),
}
json.dump(summary, open(sys.argv[3], "w"), indent=2)
print(json.dumps(summary, ensure_ascii=False))
PY
for image in final_all final_routing final_placement; do
  [ -f "$reports/$image.webp" ] && cp "$reports/$image.webp" "$out/$image.webp"
done
echo "GDS: $out/out/results/sky130hd/tesela/base/6_final.gds"
