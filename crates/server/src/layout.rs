//! The real layout of the design's compute cell: its Verilog through OpenROAD-flow-scripts
//! (the `openroad/orfs` Docker image), on the open PDK of the chosen process. The cell is
//! what a die repeats thousands of times, so its measured area, clock and power are the
//! numbers that matter most for the estimate.

use netlist::{cell_design, to_verilog, CellConfig};
use serde::Deserialize;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;

/// The cell to lay out, as the design's circuit chose it, and where.
#[derive(Clone, Deserialize)]
pub struct LayoutRequest {
    pub node: String,
    /// The cell exactly as the estimator chose it (`CircuitDecision::cell`), and its name.
    pub cell: CellConfig,
    #[serde(default)]
    pub cell_key: String,
    /// The clock the layout aims for.
    pub period_ns: f64,
}

/// A layout under way: the step it is at, and in the end what it measured.
#[derive(Default)]
pub struct LayoutJob {
    pub phase: Mutex<String>,
    pub outcome: Mutex<Option<Result<serde_json::Value, String>>>,
}

/// The OpenROAD platform of a process the estimator knows, if it has an open PDK.
pub fn platform_for(node: &str) -> Option<&'static str> {
    match node {
        "sky130" => Some("sky130hd"),
        "gf180" => Some("gf180"),
        "asap7" => Some("asap7"),
        _ => None,
    }
}

/// The platform's unit of time in nanoseconds: ASAP7's libraries count picoseconds.
pub fn time_unit_ns(platform: &str) -> f64 {
    if platform == "asap7" { 0.001 } else { 1.0 }
}

/// What the flow's log says it is doing, in the person's words: it announces each stage
/// as `Running script.tcl, stage 3_3_place_gp`.
fn phase_of(line: &str) -> Option<&'static str> {
    let stage = line.strip_prefix("Running ")?.split("stage ").nth(1)?;
    [("1", "Sintetizando con Yosys"), ("2", "Haciendo el floorplan"), ("3", "Colocando las celdas"), ("4", "Construyendo el árbol de reloj"), ("5", "Rutando"), ("6", "Comprobando y escribiendo el GDS")]
        .into_iter()
        .find(|(step, _)| stage.starts_with(step))
        .map(|(_, phase)| phase)
}

pub fn run(request: &LayoutRequest, job: &LayoutJob, home: &Path) -> Result<serde_json::Value, String> {
    let set_phase = |phase: &str| *job.phase.lock().unwrap() = phase.to_string();
    let platform = platform_for(&request.node).ok_or("este proceso no tiene un PDK abierto: elige SkyWater 130, GF180 o ASAP7")?;
    let docker = Command::new("docker").arg("info").stdout(Stdio::null()).stderr(Stdio::null()).status();
    if !docker.is_ok_and(|s| s.success()) {
        return Err("Docker no está en marcha: ábrelo para hacer layouts con OpenROAD".into());
    }
    set_phase("Generando el Verilog de la celda");
    // A few words of memory: what is measured is the logic around it.
    let mut config = CellConfig { addr_bits: 2, mac_stages: request.cell.mac_stages.max(1), adder_stages: request.cell.adder_stages.max(1), ..request.cell };
    config.accumulator_bits = config.required_accumulator().max(config.accumulator_bits);
    let design = cell_design(config);
    let top = design.top.clone();
    let key: String = request.cell_key.chars().filter(|c| c.is_ascii_alphanumeric() || "-+_".contains(*c)).collect();
    let name = format!("{key}-{platform}-{}", request.period_ns).to_lowercase();
    let dir: PathBuf = home.join(".llm2chip/asic").join(&name);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("design.v"), to_verilog(&design)).map_err(|e| e.to_string())?;
    std::fs::write(
        dir.join("config.mk"),
        format!("export DESIGN_NAME = {top}\nexport DESIGN_NICKNAME = {top}\nexport PLATFORM = {platform}\nexport VERILOG_FILES = /work/design.v\nexport SDC_FILE = /work/constraint.sdc\nexport CORE_UTILIZATION = 40\nexport PLACE_DENSITY = 0.60\n"),
    )
    .map_err(|e| e.to_string())?;
    std::fs::write(
        dir.join("constraint.sdc"),
        format!("current_design {top}\nset clk_period {}\ncreate_clock -name clk -period $clk_period [get_ports clk]\nset non_clock_inputs [lsearch -inline -all -not -exact [all_inputs] [get_ports clk]]\nset_input_delay [expr $clk_period * 0.2] -clock clk $non_clock_inputs\nset_output_delay [expr $clk_period * 0.2] -clock clk [all_outputs]\n", request.period_ns / time_unit_ns(platform)),
    )
    .map_err(|e| e.to_string())?;

    set_phase("Sintetizando con Yosys");
    let mut child = Command::new("docker")
        .args(["run", "--rm", "--platform", "linux/amd64", "-v", &format!("{}:/work", dir.display()), "openroad/orfs:latest", "bash", "-c", "source /OpenROAD-flow-scripts/env.sh >/dev/null && cd /OpenROAD-flow-scripts/flow && make DESIGN_CONFIG=/work/config.mk WORK_HOME=/work/out 2>&1"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("no se pudo lanzar OpenROAD: {e}"))?;
    let mut tail: Vec<String> = Vec::new();
    for line in BufReader::new(child.stdout.take().expect("salida")).lines().map_while(Result::ok) {
        if let Some(phase) = phase_of(&line) {
            set_phase(phase);
        }
        tail.push(line);
        if tail.len() > 20 {
            tail.remove(0);
        }
    }
    let status = child.wait().map_err(|e| e.to_string())?;
    let logs = dir.join(format!("out/logs/{platform}/{top}/base"));
    let reports = dir.join(format!("out/reports/{platform}/{top}/base"));
    if !status.success() || !logs.join("6_report.json").exists() {
        return Err(format!("OpenROAD no terminó: {}", tail.join("\n")));
    }
    let metrics: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(logs.join("6_report.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let finish = std::fs::read_to_string(reports.join("6_finish.rpt")).unwrap_or_default();
    let fmax = finish.split("fmax = ").nth(1).and_then(|rest| rest.split_whitespace().next()).and_then(|v| v.parse::<f64>().ok());
    let image = std::fs::read(reports.join("final_all.webp")).ok().map(|bytes| format!("data:image/webp;base64,{}", crate::base64(&bytes)));
    let number = |key: &str| metrics.get(key).and_then(serde_json::Value::as_f64);
    Ok(serde_json::json!({
        "platform": platform,
        "cell_area_um2": number("finish__design__instance__area__stdcell"),
        "die_area_um2": number("finish__design__die__area"),
        "cells": number("finish__design__instance__count__stdcell"),
        "power_w": number("finish__power__total"),
        "fmax_mhz": fmax,
        "drc_errors": number("detailedroute__route__drc_errors").unwrap_or(0.0),
        "period_ns": request.period_ns,
        "image": image,
        "gds": dir.join(format!("out/results/{platform}/{top}/base/6_final.gds")),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_open_processes_can_be_laid_out() {
        assert_eq!(platform_for("sky130"), Some("sky130hd"));
        assert_eq!(platform_for("gf180"), Some("gf180"));
        assert_eq!(platform_for("n6"), None, "sin PDK abierto");
        assert_eq!(10.0 / time_unit_ns("asap7"), 10_000.0, "ASAP7 cuenta picosegundos");
        assert_eq!(time_unit_ns("sky130hd"), 1.0);
    }

    #[test]
    fn the_log_tells_the_step() {
        assert_eq!(phase_of("Running global_place.tcl, stage 3_3_place_gp"), Some("Colocando las celdas"));
        assert_eq!(phase_of("[INFO DPL-0303] Running algorithm for global swaps."), None);
    }
}
