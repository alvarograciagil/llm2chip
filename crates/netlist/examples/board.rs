//! Writes the demo for a small board: a compute cell behind the serial port, with its
//! testbench, Quartus project and a host script.
//!
//! `cargo run -p netlist --example board -- DIR [rows] [cols]`

use netlist::board::{cell_demo_files, OMDAZZ_EP4CE10};
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(args.get(1).expect("carpeta de salida"));
    let rows = args.get(2).map_or(8, |n| n.parse().expect("filas"));
    let cols = args.get(3).map_or(64, |n| n.parse().expect("columnas"));
    for (path, text) in cell_demo_files(&OMDAZZ_EP4CE10, rows, cols) {
        let path = dir.join(path);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, text).unwrap();
    }
    eprintln!("proyecto en {}", dir.display());
}
