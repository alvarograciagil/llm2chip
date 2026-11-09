//! The firmware's tokenizer, on the instruction-level CPU, against the reference library.

use netlist::chip::{chip_info, ChipConfig, ChipModel, DieEngine, GenerationSetup};
use netlist::image::build_image;
use netlist::rv32::Rv32Iss;
use netlist::sampler::{exponential_table, SamplerConfig, SamplerModel};
use netlist::soc::{FirmwareImage, SocBus, SocConfig};
use netlist::{CellConfig, DieConfig, Microprogram, Parameter};
use sim::tokenizer::TokenizerTables;
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::rc::Rc;
use tokenizers::Tokenizer;

fn build_firmware() -> Option<FirmwareImage> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../firmware");
    let built = Command::new("cargo").current_dir(&dir).env_remove("RUSTFLAGS").env_remove("CARGO_ENCODED_RUSTFLAGS").args(["build", "--release"]).output().ok()?;
    built.status.success().then(|| FirmwareImage::from_elf(&std::fs::read(dir.join("target/riscv32im-unknown-none-elf/release/firmware")).unwrap()).unwrap())
}

fn checkpoint() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var_os("HOME")?).join(".mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed-FP16/tokenizer.json");
    path.exists().then_some(path)
}

/// Keeps what the firmware sends to the die, and does nothing else.
struct Recorder(Rc<RefCell<Vec<u32>>>);

impl DieEngine for Recorder {
    fn run(&mut self, routine: usize, token: u32, _position: u32) {
        if routine == 0 {
            self.0.borrow_mut().push(token);
        }
    }
    fn cycles(&self, _routine: usize, _token: u32, _position: u32) -> u64 {
        0
    }
    fn read(&mut self, _address: u32) -> i64 {
        0
    }
}

const TEXTS: &[&str] = &[
    "The capital of France is",
    "¿Cuál es la capital de Francia? Responde en una frase.",
    "<|im_start|>user\nHola, ¿qué tal?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "don't we'll they've I'M John's 'quoted' 're",
    "fn main() {\n    let x = vec![1, 2, 3];\n\tprintln!(\"{:?}\", x); // comentario\r\n}\n\n\n",
    "3.14159 + 2,5e-10 = 1234567890 ٣٤ Ⅷ ½",
    "日本語のテキストと中文混合 한국어 Русский текст العربية",
    "emojis 😀👍🏽 🇪🇸 family 👨‍👩‍👧‍👦 end",
    "   leading and trailing   ",
    "tabs\t\tand  double  spaces \n \n trailing newline\n",
    "a<|endoftext|>b<|im_end|><|im_start|>",
    "punct!!! ...??? --- (paréntesis) [corchetes] {llaves} «comillas» “curly” — dash…",
    "ñandú acción über naïve café",
    "x",
];

#[test]
#[cfg_attr(debug_assertions, ignore = "millones de instrucciones: solo con optimizaciones")]
fn the_firmware_tokenizes_like_the_reference() {
    let (Some(firmware), Some(path)) = (build_firmware(), checkpoint()) else {
        eprintln!("omitido: falta el firmware o el tokenizador real");
        return;
    };
    let reference = Tokenizer::from_file(&path).unwrap();
    let tables = TokenizerTables::from_file(&path).unwrap().to_words();
    eprintln!("tablas: {} palabras", tables.len());

    let cell = CellConfig { weight_bits: 8, activation_bits: 16, accumulator_bits: 32, mac_stages: 1, adder_stages: 1, addr_bits: 3, coding: Default::default(), exponent_bits: 0, multiplier: netlist::multiplier::MultiplierKind::Array, fixed_weights: false, ecc: false };
    let die = DieConfig { cells: 1, cell, scale_bits: 16, reg_bits: 12, index_bits: 18, microcode_bits: 2, broadcast_stages: 1, function_bits: 12, function_select_bits: 3 };
    let code = Microprogram { words: Vec::new(), entries: vec![0, 1], parameters: vec![Parameter { register: 1, token: 1, position: 0, constant: 0 }, Parameter { register: 2, token: 0, position: 1, constant: 0 }] };
    let setup = GenerationSetup { max_new: 0, greedy: true, limit: 1, temperature: 256, seed: 1, raw_tokens: false, stop: [u32::MAX, u32::MAX] };
    let info = chip_info(&die, &code, 1, 248_077, &setup);
    let image = build_image(&firmware.program, &firmware.data, &info, &tables, &[]).unwrap();
    let soc = SocConfig { program_bits: 14, data_bits: 22, fifo_bits: 6 };
    let _ = ChipConfig { soc, die, sampler: SamplerConfig { logit_bits: 16, index_bits: 18, entries: 4, lanes: 1 } };

    let mut failures = Vec::new();
    for text in TEXTS {
        let expected = reference.encode(*text, false).unwrap().get_ids().to_vec();
        let tokens = Rc::new(RefCell::new(Vec::new()));
        let mut input = text.as_bytes().to_vec();
        input.push(0);
        let sampler = SamplerModel { config: SamplerConfig { logit_bits: 16, index_bits: 18, entries: 4, lanes: 1 }, rng: 1, table: exponential_table() };
        let model = ChipModel::new(Recorder(tokens.clone()), code.clone(), sampler);
        let mut bus = SocBus::booted_with(soc, &image, &input, model).unwrap();
        let mut cpu = Rv32Iss::default();
        while !cpu.halted {
            cpu.step(&mut bus);
            assert!(cpu.cycle < 2_000_000_000, "el firmware no termina con {text:?}");
        }
        let got = tokens.borrow().clone();
        eprintln!("{:>9} ciclos, {:>3} tokens: {text:?}", cpu.cycle, got.len());
        if got != expected {
            failures.push(format!("{text:?}\n  firmware:   {got:?}\n  referencia: {expected:?}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
