#!/usr/bin/env bash
# Lays out every cell of tools/asic/cells.json on every platform, two at a time, and
# gathers what OpenROAD measured in tests/golden/asic.json, the reference the estimator's
# tests compare against.
#
#   tools/asic/calibrate.sh [plataforma...]
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cells="$root/tools/asic/cells.json"
work=$HOME/.llm2chip/asic/calibracion
mkdir -p "$work"
platforms=("$@")
[ ${#platforms[@]} -eq 0 ] && platforms=($(python3 -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['platforms']))" "$cells"))
cargo build --release -q -p netlist --example export
export_bin="$root/target/release/examples/export"

jobs=()
for name in $(python3 -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['cells']))" "$cells"); do
  config=$(python3 -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))['cells'][sys.argv[2]]))" "$cells" "$name")
  "$export_bin" --json "$config" > "$work/$name.v" 2>/dev/null
  for platform in "${platforms[@]}"; do
    jobs+=("$name $platform")
  done
done

run() {
  local name=$1 platform=$2
  local period
  period=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['platforms'][sys.argv[2]])" "$cells" "$platform")
  local out="$work/$name-$platform"
  if [ -f "$out/resumen.json" ]; then echo "ya medido: $name $platform"; return; fi
  "$root/tools/asic/flow.sh" "$work/$name.v" compute_cell "$platform" "$period" "$out" > /dev/null 2>&1 \
    && echo "medido: $name $platform $(cat "$out/resumen.json" | tr -d '\n ')" \
    || echo "falló: $name $platform (ver $out/flujo.log)"
}

export -f run
export cells work root
printf '%s\n' "${jobs[@]}" | xargs -P 2 -L 1 bash -c 'run "$0" "$1"'

python3 - "$cells" "$work" "$root/tests/golden/asic.json" <<'PY'
import json, os, sys
spec = json.load(open(sys.argv[1]))
measures = []
for name, cell in spec["cells"].items():
    for platform, period in spec["platforms"].items():
        path = os.path.join(sys.argv[2], f"{name}-{platform}", "resumen.json")
        if not os.path.exists(path):
            continue
        r = json.load(open(path))
        measures.append({"cell": name, "platform": platform, "period_ns": period, "config": cell,
                         "area_um2": r["area_celdas_um2"], "fmax_mhz": r["fmax_mhz"], "power_w": r["potencia_w"],
                         "cells": r["celdas"], "drc": r["violaciones_drc"]})
json.dump({"how": "tools/asic/calibrate.sh: OpenROAD-flow-scripts, CORE_UTILIZATION 40, PLACE_DENSITY 0.60, 4 words of weights in flip-flops", "measures": measures}, open(sys.argv[3], "w"), indent=1, ensure_ascii=False)
print(f"{len(measures)} medidas en {sys.argv[3]}")
PY
