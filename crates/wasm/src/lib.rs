//! Browser bindings. Values cross the boundary as JSON strings.

use arch::{catalog, estimate, nodes, ChipParams};
use model_ir::ModelGraph;
use netlist::block::Behavior;
use netlist::demo;
use netlist::export::{project_files, ProjectOptions};
use netlist::chip::{chip_design, ChipConfig};
use netlist::die::word_bits;
use netlist::image::{build_image, Segment, Target};
use netlist::sampler::{exponential_table, SamplerConfig};
use netlist::soc::{FirmwareImage, SocConfig};
use netlist::{cell_design, die_design, to_verilog, to_vhdl, CellConfig, ControlWord, Design, DieConfig, RomWord, Simulator, TestStep};
use wasm_bindgen::prelude::*;

const BUILTIN_MODELS: [(&str, &str, &str); 6] = [
    ("qwen3.8-27b", "Qwen3.8-27B", include_str!("../../../models/qwen3.8-27b.json")),
    ("qwen3.6-35b-a3b", "Qwen3.6-35B-A3B (MoE)", include_str!("../../../models/qwen3.6-35b-a3b.json")),
    ("llama-3.1-8b", "Llama 3.1 8B", include_str!("../../../models/llama-3.1-8b.json")),
    ("stories110m", "stories110M", include_str!("../../../models/stories110m.json")),
    ("stories15m", "stories15M", include_str!("../../../models/stories15m.json")),
    ("stories260k", "stories260K", include_str!("../../../models/stories260k.json")),
];

const PRESETS: [&str; 4] = [
    include_str!("../../../presets/qwen-sram.json"),
    include_str!("../../../presets/llama-rom.json"),
    include_str!("../../../presets/fpga-u250.json"),
    include_str!("../../../presets/stories-180nm.json"),
];

fn to_js<T: serde::Serialize>(value: &T) -> Result<String, JsError> {
    serde_json::to_string(value).map_err(|e| JsError::new(&e.to_string()))
}

#[wasm_bindgen]
pub fn builtin_models() -> Result<String, JsError> {
    let list: Vec<_> = BUILTIN_MODELS
        .iter()
        .map(|(id, name, config)| serde_json::json!({ "id": id, "name": name, "config": config }))
        .collect();
    to_js(&list)
}

#[wasm_bindgen]
pub fn presets() -> Result<String, JsError> {
    let list: Vec<serde_json::Value> = PRESETS
        .iter()
        .map(|p| serde_json::from_str(p).map_err(|e| JsError::new(&e.to_string())))
        .collect::<Result<_, _>>()?;
    to_js(&list)
}

#[wasm_bindgen]
pub fn default_params() -> Result<String, JsError> {
    to_js(&ChipParams::default())
}

#[wasm_bindgen]
pub fn optimization_catalog() -> Result<String, JsError> {
    to_js(&catalog())
}

#[wasm_bindgen]
pub fn tech_nodes() -> Result<String, JsError> {
    to_js(&nodes())
}

#[wasm_bindgen]
pub fn estimate_design(name: &str, config: &str, params: &str) -> Result<String, JsError> {
    let model = ModelGraph::from_config_json(name, config).map_err(|e| JsError::new(&e.0))?;
    let params: ChipParams =
        serde_json::from_str(params).map_err(|e| JsError::new(&format!("parámetros inválidos: {e}")))?;
    let report = estimate(&model, &params).map_err(|e| JsError::new(&e))?;
    to_js(&report)
}

fn cell_config(config: &str, params: &str) -> Result<CellConfig, JsError> {
    let model = ModelGraph::from_config_json("modelo", config).map_err(|e| JsError::new(&e.0))?;
    let p: ChipParams = serde_json::from_str(params).map_err(|e| JsError::new(&e.to_string()))?;
    let report = estimate(&model, &p).map_err(|e| JsError::new(&e))?;
    let cell_bits = report.metric("cell_sram_kb").unwrap_or(1.0) * 8_000.0;
    let words = (cell_bits / p.cell_weight_bits() as f64).ceil().max(2.0);
    Ok(p.cell_config((words.log2().ceil() as u32).clamp(1, 20)))
}

#[wasm_bindgen]
pub fn schedule_design(name: &str, config: &str, params: &str) -> Result<String, JsError> {
    let model = ModelGraph::from_config_json(name, config).map_err(|e| JsError::new(&e.0))?;
    let p: ChipParams = serde_json::from_str(params).map_err(|e| JsError::new(&e.to_string()))?;
    let report = estimate(&model, &p).map_err(|e| JsError::new(&e))?;
    let factor = if p.has("aritmetica_distribuida") { 4.0 } else { 1.0 };
    let ghz = report.metric("freq_ghz").unwrap_or(1.0);
    to_js(&compiler::schedule(&model, &p, &report.floorplan, ghz, factor))
}

/// The operations one token goes through, as the chip will execute them.
#[wasm_bindgen]
pub fn program_design(config: &str) -> Result<String, JsError> {
    to_js(&compiler::compile_qwen3_5(config).map_err(|e| JsError::new(&e))?)
}

/// Evaluates one design per entry in the JSON array `variants`, each merged over `params`.
#[wasm_bindgen]
pub fn sweep_design(name: &str, config: &str, params: &str, variants: &str) -> Result<String, JsError> {
    let model = ModelGraph::from_config_json(name, config).map_err(|e| JsError::new(&e.0))?;
    let base: serde_json::Value = serde_json::from_str(params).map_err(|e| JsError::new(&e.to_string()))?;
    let variants: Vec<serde_json::Map<String, serde_json::Value>> = serde_json::from_str(variants).map_err(|e| JsError::new(&e.to_string()))?;
    let mut results = Vec::with_capacity(variants.len());
    for overrides in variants {
        let mut merged = base.clone();
        for (key, value) in overrides {
            merged[key] = value;
        }
        let p: ChipParams = serde_json::from_value(merged).map_err(|e| JsError::new(&e.to_string()))?;
        let report = estimate(&model, &p).map_err(|e| JsError::new(&e))?;
        let values: serde_json::Map<String, serde_json::Value> = report.metrics.iter().map(|m| (m.id.to_string(), serde_json::json!(m.value))).collect();
        let errors = report.violations.iter().filter(|v| v.severity == arch::Severity::Error).count();
        results.push(serde_json::json!({ "metrics": values, "errors": errors }));
    }
    to_js(&results)
}

/// A live gate-level simulation of the design's compute cell.
#[wasm_bindgen]
pub struct CellSimulation {
    design: Design,
    config: CellConfig,
    sim: Simulator,
}

#[wasm_bindgen]
impl CellSimulation {
    #[wasm_bindgen(constructor)]
    pub fn new(config: &str, params: &str) -> Result<CellSimulation, JsError> {
        let config = cell_config(config, params)?;
        let design = cell_design(config);
        let sim = Simulator::new(&design).map_err(|e| JsError::new(&e.0))?;
        Ok(CellSimulation { design, config, sim })
    }

    pub fn design(&self) -> Result<String, JsError> {
        to_js(&self.design)
    }

    pub fn config(&self) -> Result<String, JsError> {
        to_js(&serde_json::json!({
            "config": self.config,
            "latency": self.config.latency(),
            "counts": self.design.flat_counts(),
        }))
    }

    pub fn verilog(&self) -> String {
        to_verilog(&self.design)
    }

    pub fn vhdl(&self) -> String {
        to_vhdl(&self.design)
    }

    /// The files of an exportable project, as a JSON object of path to contents.
    pub fn project(&self, clock_ns: f64) -> Result<String, JsError> {
        let c = self.config;
        let step = |pairs: &[(&str, u128)]| TestStep { inputs: pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect() };
        let mut steps = vec![step(&[("clear", 1)])];
        let weights = 1u128 << c.addr_bits.min(4);
        let value = |address: u128| (address + 1) & ((1 << c.weight_bits) - 1);
        let mut rom = Vec::new();
        if c.fixed_weights {
            rom.extend((0..weights).map(|address| RomWord { path: "weight_rom".into(), address: address as usize, value: value(address) }));
            steps.push(step(&[("clear", 1)]));
        } else {
            for address in 0..weights {
                steps.push(step(&[("clear", 0), ("load_en", 1), ("load_addr", address), ("load_data", value(address))]));
            }
            steps.push(step(&[("load_en", 0), ("clear", 1)]));
        }
        for x in 0..(1u128 << c.addr_bits.min(4)) {
            steps.push(step(&[("clear", 0), ("run", 1), ("x", x * 100 + 1)]));
        }
        for _ in 0..c.latency() {
            steps.push(step(&[("run", 0)]));
        }
        let report = serde_json::json!({ "config": c, "latency": c.latency(), "counts": self.design.flat_counts() }).to_string();
        let options = ProjectOptions { name: "Celda de cálculo", description: "Una celda de cálculo del chip generado: memoria de pesos, decodificador, multiplicador y acumulador.", clock_ns, watch: "total", report, rom };
        to_js(&project_files(&self.design, &steps, &options))
    }

    /// Fixes one word of a ROM, as the mask does at manufacture.
    pub fn burn(&mut self, memory: &str, address: u32, value: &str) -> Result<(), JsError> {
        let parsed: u128 = value.parse().map_err(|_| JsError::new("valor inválido"))?;
        self.sim.poke_memory(&format!("/{memory}"), address as usize, parsed).map_err(|e| JsError::new(&e.0))
    }

    /// `value` is a decimal string so 64-bit inputs keep every bit.
    pub fn set(&mut self, port: &str, value: &str) -> Result<(), JsError> {
        let parsed: u64 = value.parse().map_err(|_| JsError::new("valor inválido"))?;
        self.sim.set(port, parsed).map_err(|e| JsError::new(&e.0))
    }

    pub fn tick(&mut self) {
        self.sim.tick();
    }

    pub fn settle(&mut self) {
        self.sim.settle();
    }

    pub fn cycle(&self) -> f64 {
        self.sim.cycle() as f64
    }

    pub fn snapshot(&self, path: &str) -> Result<String, JsError> {
        to_js(&self.sim.snapshot(path))
    }

    /// The accumulator value these stored codes and activations must produce, as a
    /// signed decimal string. Floating formats are read with an exponent base of zero.
    pub fn expected(&self, codes: &str, activations: &str) -> Result<String, JsError> {
        let c = self.config;
        let parse = |text: &str| text.split(',').map(|v| v.trim().parse::<i128>()).collect::<Result<Vec<i128>, _>>().map_err(|_| JsError::new("lista inválida"));
        let shift_bits = 32 - (c.accumulator_bits - 1).leading_zeros();
        let mut total = 0i128;
        for (code, x) in parse(codes)?.iter().zip(parse(activations)?) {
            let raw = *code as u64;
            let (significand, shift) = if c.coding == netlist::format::WeightCoding::Codebook {
                (((raw << (64 - c.weight_bits)) as i64) >> (64 - c.weight_bits), 0)
            } else {
                let (significand, exponent) = c.decode_code(raw);
                (significand, exponent.min((1u64 << shift_bits) - 1) as u32)
            };
            total = total.wrapping_add((significand as i128 * x) << shift);
        }
        let excess = 128 - c.accumulator_bits;
        Ok((total << excess >> excess).to_string())
    }
}

/// The reduced die running one worked example, gate by gate.
#[wasm_bindgen]
pub struct DieSimulation {
    design: Design,
    config: DieConfig,
    sim: Simulator,
    demo: Demo,
    started: bool,
}

enum Demo {
    Mlp(demo::Example),
    Block(demo::BlockDemo),
    Boot(BootDemo),
}

/// The complete chip starting up from its image, which the app plays the host of.
struct BootDemo {
    image: Vec<u32>,
    pending: Option<u32>,
    output: Vec<u8>,
    segments: usize,
}

impl BootDemo {
    fn new(config: DieConfig) -> Result<Self, String> {
        let firmware = FirmwareImage::from_elf(include_bytes!("../../../firmware/prebuilt/firmware.elf"))?;
        let beats = word_bits(&config).div_ceil(32);
        let segments = vec![
            Segment { target: Target::Microcode, unit: 0, address: 0, beats, words: (0..4).flat_map(|_| ControlWord::default().encode_words(&config)).collect() },
            Segment { target: Target::Vector, unit: 0, address: 0, beats: 1, words: vec![100, 200, 300, 400] },
            Segment { target: Target::Function, unit: 0, address: 0, beats: 1, words: (0..16).map(|i| i * 3).collect() },
            Segment { target: Target::SamplerTable, unit: 0, address: 0, beats: 1, words: exponential_table()[..16].to_vec() },
        ];
        let image = build_image(&firmware.program, &firmware.data, &[], &[], &segments)?;
        Ok(Self { image, pending: None, output: Vec::new(), segments: segments.len() })
    }

    /// One cycle of the host side, before the clock edge.
    fn serve(&mut self, sim: &mut Simulator) -> Result<(), JsError> {
        let fail = |e: netlist::SimError| JsError::new(&e.0);
        sim.set("image_resp_valid", self.pending.is_some() as u64).map_err(fail)?;
        sim.set("image_resp_data", self.pending.unwrap_or(0) as u64).map_err(fail)?;
        sim.set("host_tx_ready", 1).map_err(fail)?;
        sim.settle();
        self.pending = (sim.get("image_req_valid") == Some(1)).then(|| self.image[sim.get("image_req_addr").unwrap_or(0) as usize]);
        if sim.get("host_tx_valid") == Some(1) {
            self.output.push(sim.get("host_tx_data").unwrap_or(0) as u8);
        }
        Ok(())
    }
}

#[wasm_bindgen]
impl DieSimulation {
    /// `kind` is "mlp", "attention" or "delta".
    #[wasm_bindgen(constructor)]
    pub fn new(params: &str, kind: &str) -> Result<DieSimulation, JsError> {
        let p: ChipParams = serde_json::from_str(params).map_err(|e| JsError::new(&e.to_string()))?;
        if kind == "boot" {
            let config = demo::block_config();
            let chip = ChipConfig { soc: SocConfig { program_bits: 12, data_bits: 13, fifo_bits: 6 }, die: config, sampler: SamplerConfig { logit_bits: 16, index_bits: 12, entries: 4, lanes: 1 } };
            let design = chip_design(chip);
            let mut sim = Simulator::new(&design).map_err(|e| JsError::new(&e.0))?;
            sim.set("reset", 1).map_err(|e| JsError::new(&e.0))?;
            sim.tick();
            sim.set("reset", 0).map_err(|e| JsError::new(&e.0))?;
            // An empty prompt: the firmware loads everything and stops.
            let mut boot = BootDemo::new(config).map_err(|e| JsError::new(&e))?;
            for _ in 0..=netlist::chip::RESET_STRETCH {
                sim.tick();
            }
            sim.set("host_rx_valid", 1).map_err(|e| JsError::new(&e.0))?;
            sim.set("host_rx_data", 0).map_err(|e| JsError::new(&e.0))?;
            boot.serve(&mut sim)?;
            sim.tick();
            sim.set("host_rx_valid", 0).map_err(|e| JsError::new(&e.0))?;
            return Ok(DieSimulation { design, config, sim, demo: Demo::Boot(boot), started: true });
        }
        let config = if kind == "mlp" {
            DieConfig {
                cells: 3,
                cell: CellConfig { weight_bits: 8, activation_bits: 16, accumulator_bits: 32, mac_stages: p.mac_stages, adder_stages: p.adder_stages, addr_bits: 4, coding: Default::default(), exponent_bits: 0, multiplier: netlist::multiplier::MultiplierKind::Array, fixed_weights: p.weight_backend == arch::WeightBackend::Rom, ecc: false },
                scale_bits: 16,
                reg_bits: 3,
                index_bits: 4,
                microcode_bits: 9,
                broadcast_stages: p.noc_stages + 1,
                function_bits: 8,
                function_select_bits: 3,
            }
        } else {
            demo::block_config()
        };
        let design = die_design(config);
        let mut sim = Simulator::new(&design).map_err(|e| JsError::new(&e.0))?;
        let demo = match kind {
            "mlp" => {
                let example = demo::example();
                demo::load(&mut sim, config, &example);
                Demo::Mlp(example)
            }
            "attention" | "delta" => {
                let built = if kind == "attention" { demo::attention_demo(&config) } else { demo::delta_demo(&config) };
                for step in demo::block_host_steps(config, &built.assembly) {
                    for (name, value) in &step.inputs {
                        sim.set_wide(name, *value).map_err(|e| JsError::new(&e.0))?;
                    }
                    sim.tick();
                }
                Demo::Block(built)
            }
            other => return Err(JsError::new(&format!("demostración desconocida: {other}"))),
        };
        Ok(DieSimulation { design, config, sim, demo, started: false })
    }

    pub fn design(&self) -> Result<String, JsError> {
        to_js(&self.design)
    }

    pub fn counts(&self) -> Result<String, JsError> {
        to_js(&self.design.flat_counts())
    }

    pub fn verilog(&self) -> String {
        to_verilog(&self.design)
    }

    pub fn vhdl(&self) -> String {
        to_vhdl(&self.design)
    }

    /// One clock edge. Returns true once the program has halted.
    pub fn tick(&mut self) -> Result<bool, JsError> {
        if !self.started {
            self.sim.set("start", 0).map_err(|e| JsError::new(&e.0))?;
            self.started = true;
        }
        if let Demo::Boot(boot) = &mut self.demo {
            boot.serve(&mut self.sim)?;
        }
        self.sim.tick();
        Ok(self.sim.get("halted") == Some(1))
    }

    pub fn cycle(&self) -> f64 {
        self.sim.cycle() as f64
    }

    pub fn snapshot(&self, path: &str) -> Result<String, JsError> {
        to_js(&self.sim.snapshot(path))
    }

    pub fn results(&mut self) -> Result<String, JsError> {
        match &self.demo {
            Demo::Mlp(example) => {
                let expected = demo::reference(example);
                let got: Vec<i64> = (0..demo::RESULT_LEN).map(|i| demo::read_register(&mut self.sim, self.config, demo::RESULT_REG, i)).collect();
                let exact = got == expected;
                to_js(&serde_json::json!({ "got": got, "expected": expected, "exact": exact, "tolerance": 0.0 }))
            }
            Demo::Block(built) => {
                let mut model = Behavior::new(&self.config, &built.assembly.initial);
                model.run(&built.assembly.steps);
                let (mut got, mut exact) = (Vec::new(), true);
                for i in 0..built.output_len {
                    let (mantissa, value) = demo::read_block_value(&mut self.sim, self.config, built.output_reg, i);
                    exact &= mantissa == model.mantissa((built.output_reg, i));
                    got.push(value);
                }
                to_js(&serde_json::json!({ "got": got, "expected": built.expected, "exact": exact, "tolerance": built.tolerance }))
            }
            Demo::Boot(boot) => {
                let text = String::from_utf8_lossy(&boot.output).into_owned();
                let exact = text == ".".repeat(boot.segments);
                to_js(&serde_json::json!({ "got": [], "expected": [], "exact": exact, "tolerance": 0.0, "text": text }))
            }
        }
    }
}
