//! The logic each FPGA family needs for the smallest chip, synthesised with Yosys, against
//! the figures the board catalog uses to say whether a design fits.

#[path = "common/mod.rs"]
mod common;

use common::*;
use compiler::compile_qwen3_5;
use sim::chip::{fpga_chip, FpgaOptions};
use sim::Weights;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Cells of each kind in a Yosys `stat` report.
fn counts(report: &str) -> HashMap<String, u64> {
    report
        .lines()
        .filter_map(|line| {
            let mut words = line.split_whitespace();
            let count = words.next()?.parse().ok()?;
            let kind = words.next()?;
            words.next().is_none().then(|| (kind.to_string(), count))
        })
        .collect()
}

fn synthesise(yosys: &Path, chip: &Path, script: &str) -> HashMap<String, u64> {
    let report = chip.with_extension(format!("{}.txt", script.split_whitespace().next().unwrap()));
    let run = Command::new(yosys).args(["-q", "-p"]).arg(format!("read_verilog {}; {script}; tee -o {} stat", chip.display(), report.display())).output().unwrap();
    assert!(run.status.success(), "{script}: {}", String::from_utf8_lossy(&run.stderr));
    counts(&std::fs::read_to_string(report).unwrap())
}

#[test]
#[ignore = "lenta: sintetiza el chip con Yosys para cuatro familias, varios minutos"]
fn the_catalog_knows_how_much_logic_the_smallest_chip_takes() {
    let yosys = PathBuf::from(std::env::var("HOME").unwrap()).join(".llm2chip/tools/oss-cad-suite/bin/yosys");
    if !yosys.exists() {
        eprintln!("Yosys no está instalado: se omite");
        return;
    }
    let dir = std::env::temp_dir().join(format!("llm2chip-familias-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_checkpoint(&dir);
    // The chip `tools/synth/families.sh` measures: the test model, products in DSP blocks,
    // scale and bias in two cycles, for one board on its own.
    let chip = fpga_chip(&compile_qwen3_5(CONFIG).unwrap(), &Weights::open(&dir).unwrap(), fpga_die_config(), 8, FpgaOptions { factor_steps: 2, ..Default::default() }).unwrap();
    let verilog = dir.join("chip.v");
    std::fs::write(&verilog, netlist::to_verilog_for_fpga(&chip.design)).unwrap();

    let scripts = [
        ("ecp5", "synth_ecp5 -top llm2chip_top"),
        ("xilinx", "synth_xilinx -family xc7 -flatten -top llm2chip_top"),
        ("gowin", "synth_gowin -top llm2chip_top"),
        // Yosys does not map memories to the Cyclone IV's blocks: the logic alone.
        ("cyclone4", "blackbox llm2chip_sram llm2chip_ram2p llm2chip_rom; synth_intel -family cycloneive -nobram -top llm2chip_top"),
    ];
    let measured: HashMap<&str, HashMap<String, u64>> = std::thread::scope(|scope| {
        let runs: Vec<_> = scripts.iter().map(|(name, script)| (*name, scope.spawn(|| synthesise(&yosys, &verilog, script)))).collect();
        runs.into_iter().map(|(name, run)| (name, run.join().unwrap())).collect()
    });
    let sum = |family: &str, kinds: &[(&str, u64)]| -> f64 { kinds.iter().map(|(kind, weight)| measured[family].get(*kind).copied().unwrap_or(0) * weight).sum::<u64>() as f64 / 1000.0 };
    let luts = |family: &str, prefix: &str| -> f64 { measured[family].iter().filter(|(kind, _)| kind.starts_with(prefix)).map(|(_, n)| *n).sum::<u64>() as f64 / 1000.0 };
    // Distributed memory takes look-up tables of its own: four for each of these blocks.
    let xilinx = luts("xilinx", "LUT") + sum("xilinx", &[("RAM32M", 4), ("RAM64M", 4), ("RAM256X1S", 4), ("RAM128X1D", 4)]);
    let figures = [
        ("ulx3s-85f", luts("ecp5", "LUT4")),
        ("arty-a7-100t", xilinx),
        ("de10-nano", luts("gowin", "LUT") + sum("gowin", &[("ALU", 1)])),
        ("cyclone-iv-ep4ce10", sum("cyclone4", &[("cycloneive_lcell_comb", 1)])),
    ];
    for (board, thousands) in figures {
        let board = arch::fpga::boards().iter().find(|b| b.id == board).unwrap();
        let catalog = arch::fpga::base_logic_k(board);
        eprintln!("{}: medido {thousands:.1} K, catálogo {catalog:.1} K", board.fpga);
        assert!((thousands - catalog).abs() <= 0.1 * catalog, "{}: el catálogo dice {catalog:.1} K y la síntesis da {thousands:.1} K", board.fpga);
    }
    std::fs::remove_dir_all(&dir).ok();
}
