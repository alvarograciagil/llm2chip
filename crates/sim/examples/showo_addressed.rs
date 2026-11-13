//! The real Show-o on a die addressed as the firmware addresses it: the embedding table in
//! the image, a row at a time, and the die's memories sized so that register and element
//! fit in 32 bits. Prints the die's sizes and, with `pasada`, runs one pass with the text
//! and compares each masked place's best code with floating point's.
//!
//! `cargo run --release -p sim --example showo_addressed -- <carpeta de Show-o> [celdas] [pasada]`

use netlist::format::{WeightCoding, WeightFormat};
use netlist::{CellConfig, DieConfig};
use sim::chip::VmEngine;
use sim::lower::Layout;
use sim::showo::ShowO;
use sim::showo_chip::{addressed_die, firmware_info, program, program_guided, weights, Addressed, FillDie};
use std::path::Path;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(&args[1]);
    let cells: u32 = args.get(2).map_or(1024, |s| s.parse().unwrap());
    let pass = args.get(3).is_some_and(|s| s == "pasada");
    let started = std::time::Instant::now();
    let format = WeightFormat { coding: WeightCoding::Affine, bits: 8, exponent_bits: 0, bias: true, group_size: 64, fine_scale: false, keep_source: false };
    let open = || -> Result<_, String> { weights(dir, sim::Weights::open(dir)?.with_format(Some(format))) };
    let (shape, config, w) = open()?;
    let program = program(&shape);
    let context = (config.max_text + 1 + shape.image_rows).next_power_of_two();
    let base = DieConfig {
        cells,
        cell: CellConfig { weight_bits: 8, activation_bits: 16, accumulator_bits: 48, mac_stages: 1, adder_stages: 1, addr_bits: 3, coding: WeightCoding::Affine, exponent_bits: 0, multiplier: netlist::multiplier::MultiplierKind::Array, fixed_weights: false, ecc: false, skip_zeros: false },
        scale_bits: 21,
        reg_bits: 12,
        index_bits: 12,
        microcode_bits: 18,
        broadcast_stages: 1,
        function_bits: 12,
        function_select_bits: 3,
        features: Default::default(),
    };
    let guided = program_guided(&shape, Some(1.75));
    match addressed_die(&guided, &w, base, context) {
        Ok(die) => eprintln!("con guía: {} + {} = {} bits de dirección", die.reg_bits, die.index_bits, die.reg_bits + die.index_bits),
        Err(e) => eprintln!("con guía: {e}"),
    }
    let die = addressed_die(&program, &w, base, context)?;
    eprintln!(
        "{cells} celdas: {} bits de registro + {} de elemento = {} bits de dirección; {} bits de profundidad de celda; {} de microcódigo ({:.0} s)",
        die.reg_bits,
        die.index_bits,
        die.reg_bits + die.index_bits,
        die.cell.addr_bits,
        die.microcode_bits,
        started.elapsed().as_secs_f64()
    );
    if !pass {
        return Ok(());
    }

    let model = ShowO::open(dir)?.ok_or("no es un Show-o")?;
    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("phi/tokenizer.json")).map_err(|e| e.to_string())?;
    let text = tokenizer.encode("a green apple on a wooden table", false).map_err(|e| e.to_string())?.get_ids().to_vec();
    let side = config.side();
    let rgb: Vec<u8> = (0..side * side)
        .flat_map(|i| {
            let (x, y) = ((i % side) as f32 - 128.0, (i / side) as f32 - 128.0);
            if x * x + y * y < 80.0 * 80.0 { [220u8, 30, 30] } else { [255, 255, 255] }
        })
        .collect();
    let region: Vec<bool> = (0..side * side).map(|i| (64..192).contains(&(i % side)) && (64..192).contains(&(i / side))).collect();
    let codes = model.masked_codes(&rgb, side, side, &region)?;
    let image: Vec<u32> = codes.iter().map(|c| c.map_or(config.mask_token, |c| config.first_code() + c)).collect();
    let float = model.scores(&text, &image);
    eprintln!("coma flotante en {:.0} s", started.elapsed().as_secs_f64());

    let layout = Layout::with_cells_table_outside(&program, &w, die, context)?;
    let info = firmware_info(&program, &layout, &die)?;
    let mut engine = VmEngine::new_table_outside(program.clone(), open()?.2, die, context)?;
    eprintln!("die preparado en {:.0} s", started.elapsed().as_secs_f64());
    let mut chip = Addressed { engine: &mut engine, layout: &layout, program: &program, info: &info, bits: die.cell.activation_bits };
    let (ids, first) = config.text_prefix(&text);
    let rows: Vec<u32> = std::iter::once(config.start_of_image()).chain(image.iter().copied()).chain([config.end_of_image()]).collect();
    let (n, count) = (ids.len() as u32, rows.len() as u32);
    for slot in 0..n + count {
        chip.rotate(slot, first[0] as u32 + slot);
    }
    for (slot, id) in ids.iter().enumerate() {
        chip.run(sim::lower::ENTRY_TOKEN, *id, slot as u32);
    }
    let layers = shape.layers;
    for layer in 0..layers {
        for row in 0..count {
            if layer == 0 {
                chip.run(sim::lower::ENTRY_VISION, rows[row as usize], 0);
            }
            chip.run(sim::lower::ENTRY_VISION + 1 + 2 * layer, row, n + row);
        }
        for row in 0..count {
            chip.run(sim::lower::ENTRY_VISION + 2 + 2 * layer, row, n + count - 1);
        }
    }
    let codebook = config.codebook as usize;
    let (mut agree, mut masked) = (0, 0);
    for row in 1..count - 1 {
        if codes[row as usize - 1].is_some() {
            continue;
        }
        chip.run(sim::lower::ENTRY_VISION + 1 + 2 * layers, row, 0);
        let (_, scores) = chip.scores();
        let best = |v: &mut dyn Iterator<Item = f32>| v.enumerate().fold((0, f32::NEG_INFINITY), |b, (i, s)| if s > b.1 { (i, s) } else { b }).0;
        let chip_best = best(&mut scores.iter().map(|s| *s as f32));
        let float_best = best(&mut float[(row as usize - 1) * codebook..][..codebook].iter().copied());
        agree += (chip_best == float_best) as usize;
        masked += 1;
    }
    eprintln!("die direccionado: mejor código igual que en coma flotante en {agree} de {masked} sitios tapados ({:.0} s)", started.elapsed().as_secs_f64());
    Ok(())
}
