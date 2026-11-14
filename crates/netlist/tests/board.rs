//! The demo for a small board: a cell behind a serial port, simulated in iverilog.

use netlist::board::{cell_demo_files, OMDAZZ_EP4CE10};
use std::process::Command;

#[test]
fn the_cell_behind_the_serial_port_multiplies_a_matrix_exactly() {
    if Command::new("iverilog").arg("-V").output().is_err() {
        eprintln!("iverilog is not installed: skipped");
        return;
    }
    let files = cell_demo_files(&OMDAZZ_EP4CE10, 4, 16);
    let dir = std::env::temp_dir().join(format!("llm2chip-board-{}", std::process::id()));
    for (path, text) in &files {
        let path = dir.join(path);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }
    let built = Command::new("iverilog").current_dir(&dir).args(["-g2005", "-o", "tb.vvp", "rtl/compute_cell.v", "rtl/cell_demo.v", "tb/testbench.v"]).output().unwrap();
    assert!(built.status.success(), "{}", String::from_utf8_lossy(&built.stderr));
    let run = Command::new("vvp").current_dir(&dir).arg("tb.vvp").output().unwrap();
    let out = String::from_utf8_lossy(&run.stdout);
    assert!(out.contains("PASS"), "{out}");
    // Pins nobody has checked on the board's schematic say so where they are set.
    assert!(files["quartus/cell_demo.qsf"].contains("UNVERIFIED"));
    std::fs::remove_dir_all(&dir).ok();
}
