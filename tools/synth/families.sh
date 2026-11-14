#!/usr/bin/env bash
# Synthesises the chip's Verilog for each FPGA family with Yosys, each run with a time
# limit, and prints what it takes: the figures behind `base_logic_k` in crates/arch/src/fpga.rs.
#
#   cargo run --release -p sim --example fpga -- chip.v --dsp --factor-steps 2
#   tools/synth/families.sh chip.v [OUT_DIR] [ecp5 xilinx gowin cyclone4 ...]
#
# Needs OSS CAD Suite, unpacked in ~/.llm2chip/tools/oss-cad-suite or where OSS_CAD_SUITE points.
# `ecp5-pnr` also places and routes on an ECP5-85F and reports the clock it reaches.
set -u
chip=${1:?"the chip's Verilog"}
out=${2:-synth-out}
shift $(( $# < 2 ? $# : 2 ))
families=${*:-ecp5 xilinx gowin cyclone4}
bin=${OSS_CAD_SUITE:-$HOME/.llm2chip/tools/oss-cad-suite}/bin
limit=${SYNTH_LIMIT:-3600}
mkdir -p "$out"
chip=$(cd "$(dirname "$chip")" && pwd)/$(basename "$chip")

# macOS has no `timeout`: perl's alarm ends the run after `limit` seconds.
limited() { perl -e 'alarm shift; exec @ARGV or die "$!"' "$limit" "$@"; }

run() {
    local name=$1 script=$2
    local start=$SECONDS
    if limited "$bin/yosys" -q -l "$out/$name.log" -p "read_verilog $chip; $script; tee -o $out/$name.txt stat" >/dev/null 2>&1; then
        echo "== $name ($((SECONDS - start)) s)"
        awk '/Number of cells|cells$/{p=1} p && /[0-9]+ +[A-Za-z$]/' "$out/$name.txt" | sort -u -k2 | grep -E "LUT|lcell|ALU|DFF|dff|FD|RAM|DP16|BRAM|DSP|MULT|mac|CCU2|RAMB|SB_" | awk '{print "  " $0}'
    else
        echo "== $name: did not finish within $limit s or failed, see $out/$name.log"
    fi
}

for family in $families; do
    case $family in
        ecp5) run ecp5 "synth_ecp5 -top llm2chip_top -json $out/ecp5.json" ;;
        ecp5-pnr)
            run ecp5 "synth_ecp5 -top llm2chip_top -json $out/ecp5.json"
            limited "$bin/nextpnr-ecp5" --85k --package CABGA756 --json "$out/ecp5.json" --freq 50 --timing-allow-fail --log "$out/ecp5-pnr.log" >/dev/null 2>&1
            grep -E "Max frequency for clock" "$out/ecp5-pnr.log" | tail -1 | sed 's/^/  /'
            ;;
        xilinx) run xilinx "synth_xilinx -family xc7 -flatten -top llm2chip_top" ;;
        gowin) run gowin "synth_gowin -top llm2chip_top" ;;
        # Yosys fails to map memories to the Cyclone IV's M9K blocks, so they stay black
        # boxes: the count is the logic alone, which is what this family runs short of.
        cyclone4) run cyclone4 "blackbox llm2chip_sram llm2chip_ram2p llm2chip_rom; synth_intel -family cycloneive -nobram -top llm2chip_top" ;;
        *) echo "unknown family: $family" ;;
    esac
done
