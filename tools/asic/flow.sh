#!/usr/bin/env bash
# Lays out a generated Verilog with OpenROAD-flow-scripts (Docker image openroad/orfs):
# synthesis, floorplan, placement, clock tree, routing and GDS on an open PDK.
#
#   tools/asic/flow.sh diseño.v MODULO [plataforma] [periodo_ns] [carpeta]
#
# Platforms of the image: sky130hd, sky130hs, gf180, asap7, nangate45, ihp-sg13g2.
# Writes the GDS, the layout images and resumen.json (area, fmax, power, DRC) in the folder.
set -euo pipefail
verilog=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
top=$2
platform=${3:-sky130hd}
period=${4:-10}
# ASAP7's libraries count picoseconds; the others, nanoseconds.
clock=$period
[ "$platform" = asap7 ] && clock=$(python3 -c "print($period * 1000)")
out=${5:-$HOME/.llm2chip/asic/$top-$platform}
mkdir -p "$out"
cp "$verilog" "$out/design.v"
cat > "$out/config.mk" <<CONFIG
export DESIGN_NAME = $top
export DESIGN_NICKNAME = $top
export PLATFORM = $platform
export VERILOG_FILES = /work/design.v
export SDC_FILE = /work/constraint.sdc
export CORE_UTILIZATION = 40
export PLACE_DENSITY = 0.60
CONFIG
cat > "$out/constraint.sdc" <<SDC
current_design $top
set clk_period $clock
create_clock -name clk -period \$clk_period [get_ports clk]
set non_clock_inputs [lsearch -inline -all -not -exact [all_inputs] [get_ports clk]]
set_input_delay [expr \$clk_period * 0.2] -clock clk \$non_clock_inputs
set_output_delay [expr \$clk_period * 0.2] -clock clk [all_outputs]
SDC
docker run --rm --platform linux/amd64 -v "$out":/work openroad/orfs:latest bash -c \
  "source /OpenROAD-flow-scripts/env.sh >/dev/null && cd /OpenROAD-flow-scripts/flow && make DESIGN_CONFIG=/work/config.mk WORK_HOME=/work/out" > "$out/flujo.log" 2>&1
base="$out/out/logs/$platform/$top/base"
reports="$out/out/reports/$platform/$top/base"
python3 - "$base/6_report.json" "$reports/6_finish.rpt" "$out/resumen.json" <<'PY'
import json, re, sys
metrics = json.load(open(sys.argv[1]))
finish = open(sys.argv[2]).read()
fmax = re.search(r"fmax = ([0-9.]+)", finish)
summary = {
    "area_celdas_um2": metrics.get("finish__design__instance__area__stdcell"),
    "area_die_um2": metrics.get("finish__design__die__area"),
    "ocupacion": metrics.get("finish__design__instance__utilization"),
    "celdas": metrics.get("finish__design__instance__count__stdcell"),
    "potencia_w": metrics.get("finish__power__total"),
    "fmax_mhz": float(fmax.group(1)) if fmax else None,
    "violaciones_drc": metrics.get("detailedroute__route__drc_errors", metrics.get("finish__timing__drv__setup_violation_count")),
}
json.dump(summary, open(sys.argv[3], "w"), indent=2)
print(json.dumps(summary, ensure_ascii=False))
PY
for image in final_all final_routing final_placement; do
  [ -f "$reports/$image.webp" ] && cp "$reports/$image.webp" "$out/$image.webp"
done
echo "GDS: $out/out/results/$platform/$top/base/6_final.gds"
