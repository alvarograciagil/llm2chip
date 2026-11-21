//! The small Qwen-shaped model that the gate-level tests run.
#![allow(dead_code)]

use netlist::{CellConfig, DieConfig};
use std::path::Path;

// The models, dies and loading the viewer's traces share with these tests.
#[allow(unused_imports)]
pub use chip_trace::board::{load_assembly, load_microprogram, load_register_table, mask};
#[allow(unused_imports)]
pub use chip_trace::tiny::*;
use std::sync::OnceLock;

/// Eight letters, one token each: enough for the chip tests' text in very little memory.
pub fn tiny_tokenizer() -> sim::tokenizer::TokenizerTables {
    let json = r#"{"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":{"type":"Fuse"},
      "model":{"type":"BPE","vocab":{"h":0,"y":1,"c":2,"d":3,"e":4,"f":5,"g":6,"o":7},"merges":[]}}"#;
    sim::tokenizer::TokenizerTables::from_json(json, 64, 256).unwrap()
}

/// The smallest die that runs the test model with eight cells, whose memories fit in
/// an FPGA's block RAM: 2^9 registers of 2^7 words and 2^13 words of microcode.
#[allow(dead_code)]
pub fn fpga_die_config() -> DieConfig {
    let base = cell_die_config();
    DieConfig { cells: 8, cell: CellConfig { addr_bits: 8, ..base.cell }, index_bits: 7, microcode_bits: 13, ..base }
}


/// The firmware, built once per test run: tests running at the same time used to start
/// their own builds, and one could find the binary half-written by another.
pub fn firmware() -> Option<netlist::soc::FirmwareImage> {
    static FIRMWARE: OnceLock<Option<netlist::soc::FirmwareImage>> = OnceLock::new();
    FIRMWARE
        .get_or_init(|| {
            let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../firmware");
            let built = std::process::Command::new("cargo").current_dir(&dir).env_remove("RUSTFLAGS").env_remove("CARGO_ENCODED_RUSTFLAGS").args(["build", "--release"]).output().ok()?;
            if !built.status.success() {
                eprintln!("firmware skipped: {}", String::from_utf8_lossy(&built.stderr));
                return None;
            }
            let elf = std::fs::read(dir.join("target/riscv32im-unknown-none-elf/release/firmware")).ok()?;
            Some(netlist::soc::FirmwareImage::from_elf(&elf).expect("the firmware's ELF"))
        })
        .clone()
}

/// The toy image encoder: hidden 8 in 2 heads of 4, 2 blocks, patches of 2×2 pixels in 2
/// frames, 2×2 patches to a token of 8 values, the toy language model's width.
#[allow(dead_code)]
pub const VISION: (usize, usize, usize, usize, usize, usize, usize) = (8, 2, 2, 16, 2, 2, 2);

/// Writes the toy encoder next to a checkpoint: its config, the preprocessor's size limits
/// (images of 4 to 64 tokens) and its weights as plain floats.
#[allow(dead_code)]
pub fn make_vision_checkpoint(dir: &Path) {
    let (hidden, heads, depth, intermediate, patch, temporal, merge) = VISION;
    let side = 3;
    let config = serde_json::json!({
        "model_type": "qwen3_5",
        "image_token_id": 7, "vision_start_token_id": 5, "vision_end_token_id": 6,
        "text_config": { "rope_parameters": { "mrope_section": [1, 1, 0] } },
        "vision_config": { "depth": depth, "hidden_size": hidden, "num_heads": heads, "intermediate_size": intermediate, "patch_size": patch, "temporal_patch_size": temporal, "spatial_merge_size": merge, "out_hidden_size": 8, "num_position_embeddings": side * side, "rope_parameters": { "rope_theta": 100.0 } }
    });
    std::fs::write(dir.join("config.json"), serde_json::to_string_pretty(&config).unwrap()).unwrap();
    let pixels = (patch * merge) * (patch * merge);
    std::fs::write(dir.join("preprocessor_config.json"), serde_json::json!({ "size": { "shortest_edge": 4 * pixels, "longest_edge": 64 * pixels } }).to_string()).unwrap();

    let mut rng = Rng(0x0BAD_5EED_1234_5678);
    let mut w = Writer::default();
    let mut put = |w: &mut Writer, name: &str, shape: &[usize], spread: f32, offset: f32| {
        let values: Vec<f32> = (0..shape.iter().product()).map(|_| offset + spread * rng.unit()).collect();
        let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        w.tensor(&format!("vision_tower.{name}"), "F32", shape, bytes);
    };
    let input = 3 * temporal * patch * patch;
    put(&mut w, "patch_embed.proj.weight", &[hidden, input], 0.1, 0.0);
    put(&mut w, "patch_embed.proj.bias", &[hidden], 0.1, 0.0);
    put(&mut w, "pos_embed.weight", &[side * side, hidden], 0.2, 0.0);
    for block in 0..depth {
        let b = |s: &str| format!("blocks.{block}.{s}");
        put(&mut w, &b("norm1.weight"), &[hidden], 0.2, 1.0);
        put(&mut w, &b("norm1.bias"), &[hidden], 0.1, 0.0);
        put(&mut w, &b("attn.qkv.weight"), &[3 * hidden, hidden], 0.3, 0.0);
        put(&mut w, &b("attn.qkv.bias"), &[3 * hidden], 0.1, 0.0);
        put(&mut w, &b("attn.proj.weight"), &[hidden, hidden], 0.3, 0.0);
        put(&mut w, &b("attn.proj.bias"), &[hidden], 0.1, 0.0);
        put(&mut w, &b("norm2.weight"), &[hidden], 0.2, 1.0);
        put(&mut w, &b("norm2.bias"), &[hidden], 0.1, 0.0);
        put(&mut w, &b("mlp.linear_fc1.weight"), &[intermediate, hidden], 0.3, 0.0);
        put(&mut w, &b("mlp.linear_fc1.bias"), &[intermediate], 0.1, 0.0);
        put(&mut w, &b("mlp.linear_fc2.weight"), &[hidden, intermediate], 0.2, 0.0);
        put(&mut w, &b("mlp.linear_fc2.bias"), &[hidden], 0.1, 0.0);
    }
    let merged = hidden * merge * merge;
    put(&mut w, "merger.norm.weight", &[hidden], 0.2, 1.0);
    put(&mut w, "merger.norm.bias", &[hidden], 0.1, 0.0);
    put(&mut w, "merger.linear_fc1.weight", &[merged, merged], 0.2, 0.0);
    put(&mut w, "merger.linear_fc1.bias", &[merged], 0.1, 0.0);
    put(&mut w, "merger.linear_fc2.weight", &[8, merged], 0.2, 0.0);
    put(&mut w, "merger.linear_fc2.bias", &[8], 0.1, 0.0);
    w.write(&dir.join("model-vision.safetensors"));
}

/// A small picture, `width` × `height` RGB, the same every time.
#[allow(dead_code)]
pub fn test_picture(width: usize, height: usize) -> Vec<u8> {
    (0..width * height * 3).map(|i| ((i * 37 + i / 3 * 11) % 251) as u8).collect()
}

/// Runs a whole chip in Icarus Verilog, from its exported netlist, as the gate tests run it
/// in this crate's simulator: its output bytes and cycles. `None` without Icarus Verilog.
#[allow(dead_code)]
pub fn chip_in_iverilog(design: &netlist::Design, image: &[u32], input: &[u8], limit: u64, tag: &str) -> Option<(Vec<u8>, u64)> {
    std::process::Command::new("iverilog").arg("-V").output().ok()?;
    let dir = std::env::temp_dir().join(format!("llm2chip-iverilog-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let hex = |values: Vec<String>| values.join("\n") + "\n";
    std::fs::write(dir.join("image.hex"), hex(image.iter().map(|w| format!("{w:08x}")).collect())).unwrap();
    std::fs::write(dir.join("input.hex"), hex(input.iter().map(|b| format!("{b:02x}")).collect())).unwrap();
    let source = dir.join("chip.v");
    std::fs::write(&source, format!("{}\n{}", netlist::to_verilog(design), netlist::chip_testbench(design, image.len(), input.len(), limit))).unwrap();
    let binary = dir.join("chip.vvp");
    let compile = std::process::Command::new("iverilog").args(["-g2005", "-s", "testbench", "-o"]).arg(&binary).arg(&source).output().unwrap();
    assert!(compile.status.success(), "{}", String::from_utf8_lossy(&compile.stderr));
    let run = std::process::Command::new("vvp").current_dir(&dir).arg(&binary).output().unwrap();
    let text = String::from_utf8_lossy(&run.stdout);
    let output = text.lines().filter_map(|l| l.strip_prefix("B ")).map(|b| b.trim().parse().unwrap()).collect();
    let cycles = text.lines().find_map(|l| l.strip_prefix("CYCLES ")).map(|c| c.trim().parse().unwrap());
    std::fs::remove_dir_all(&dir).ok();
    Some((output, cycles.unwrap_or_else(|| panic!("Icarus Verilog did not finish: {}", text.lines().last().unwrap_or_default()))))
}

// The die driven on its gates as its firmware would, for the tests that need it.

/// What the firmware does for each routine: set the parameters and start the die.
#[allow(dead_code)]
pub fn enter(sim: &mut netlist::Simulator, code: &netlist::Microprogram, routine: usize, token: u32, position: u32) -> u64 {
    sim.set("start", 0).unwrap();
    sim.set("param_write", 1).unwrap();
    for parameter in &code.parameters {
        sim.set("param_index", parameter.register as u64).unwrap();
        sim.set("param_data", parameter.value(token, position) as u64).unwrap();
        sim.tick();
    }
    sim.set("param_write", 0).unwrap();
    sim.set("entry", code.entries[routine] as u64).unwrap();
    sim.set("start", 1).unwrap();
    sim.tick();
    sim.set("start", 0).unwrap();
    let mut cycles = 0u64;
    while sim.get("halted").unwrap() == 0 {
        sim.tick();
        cycles += 1;
        assert!(cycles < 20_000_000, "the sequencer does not stop");
    }
    cycles
}

/// Runs the whole chip in gates, the host sending a byte whenever the chip can take it.
#[allow(dead_code)]
pub fn gates(design: &netlist::Design, image: &[u32], input: &[u8], limit: u64) -> (Vec<u8>, u64) {
    let mut sim = netlist::Simulator::new(design).unwrap();
    sim.set("reset", 1).unwrap();
    sim.set("cpu_hold", 1).unwrap();
    sim.tick();
    sim.set("reset", 0).unwrap();
    for _ in 0..=netlist::chip::RESET_RELEASE {
        sim.tick();
    }
    let mut pending = None;
    let mut sent = 0;
    let mut step = |sim: &mut netlist::Simulator, pending: &mut Option<u32>| {
        sim.set("image_resp_valid", pending.is_some() as u64).unwrap();
        sim.set("image_resp_data", pending.unwrap_or(0) as u64).unwrap();
        let offer = sent < input.len();
        sim.set("host_rx_valid", offer as u64).unwrap();
        sim.set("host_rx_data", input.get(sent).copied().unwrap_or(0) as u64).unwrap();
        sim.settle();
        if offer && sim.get("host_rx_ready").unwrap() == 1 {
            sent += 1;
        }
        *pending = (sim.get("image_req_valid").unwrap() == 1).then(|| image[sim.get("image_req_addr").unwrap() as usize]);
    };
    let mut boot = 0;
    loop {
        step(&mut sim, &mut pending);
        if sim.get("booting").unwrap() == 0 {
            break;
        }
        sim.tick();
        boot += 1;
        assert!(boot < 1_000_000, "boot does not finish");
    }
    sim.set("cpu_hold", 0).unwrap();
    sim.set("host_tx_ready", 1).unwrap();
    let mut output = Vec::new();
    for cycle in 0..limit {
        step(&mut sim, &mut pending);
        if sim.get("host_tx_valid").unwrap() == 1 {
            output.push(sim.get("host_tx_data").unwrap() as u8);
        }
        if sim.get("halted").unwrap() == 1 {
            return (output, cycle + 1);
        }
        sim.tick();
    }
    sim.set("debug_select", netlist::chip::DEBUG_CPU_PC).unwrap();
    sim.settle();
    panic!("the firmware did not finish within {limit} cycles: output {output:?}, CPU at {:#x}, {sent} of {} bytes sent", sim.get("debug_value").unwrap(), input.len());
}
