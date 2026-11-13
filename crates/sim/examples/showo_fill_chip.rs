//! Show-o retouches a picture as the chip's firmware does it: MaskGIT's steps in integers
//! (`maskgit-rt`), the guidance in the die, every routine on the die's exact machine with the
//! weights whole in cells of 16 bits; MAGVIT-v2 on the host.
//!
//! `cargo run --release -p sim --example showo_fill_chip -- <carpeta de Show-o> salida.bmp [pasos] [semilla]`

use netlist::format::WeightCoding;
use sim::chip::VmEngine;
use sim::lower::Layout;
use sim::showo::Magvit;
use sim::showo_chip::{fill_on_die, firmware_info, program_guided, schedule, weights, Fill};
use std::path::Path;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(&args[1]);
    let steps: usize = args.get(3).map_or(4, |s| s.parse().unwrap());
    let seed: u32 = args.get(4).map_or(7, |s| s.parse().unwrap());
    let started = std::time::Instant::now();
    let open = || weights(dir, sim::Weights::open(dir).unwrap()).unwrap();
    let (shape, config, w) = open();
    let program = program_guided(&shape, Some(1.75));
    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("phi/tokenizer.json")).map_err(|e| e.to_string())?;
    let text = tokenizer.encode("a yellow lemon", false).map_err(|e| e.to_string())?.get_ids().to_vec();
    let (ids, first) = config.text_prefix(&text);
    let (plain, plain_first) = config.text_prefix(&[]);
    // A red disc on white, its middle masked.
    let side = config.side();
    let rgb: Vec<u8> = (0..side * side)
        .flat_map(|i| {
            let (x, y) = ((i % side) as f32 - 128.0, (i / side) as f32 - 128.0);
            if x * x + y * y < 80.0 * 80.0 { [220u8, 30, 30] } else { [255, 255, 255] }
        })
        .collect();
    let region: Vec<bool> = (0..side * side).map(|i| (48..208).contains(&(i % side)) && (48..208).contains(&(i / side))).collect();
    let magvit = Magvit::open(&dir.join("magvitv2"))?;
    let codes = config.masked_codes(&magvit, &rgb, side, side, &region)?;
    // The die: the exact machine's, with sixteen cells deep enough for every weight.
    let context = 512;
    let die = netlist::DieConfig {
        cells: 16,
        cell: netlist::CellConfig { weight_bits: 16, activation_bits: 16, accumulator_bits: 40, mac_stages: 1, adder_stages: 1, addr_bits: 27, coding: WeightCoding::Affine, exponent_bits: 0, multiplier: netlist::multiplier::MultiplierKind::Array, fixed_weights: false, ecc: false, skip_zeros: false },
        scale_bits: 16,
        reg_bits: 24,
        index_bits: 27,
        microcode_bits: 20,
        broadcast_stages: 1,
        function_bits: 12,
        function_select_bits: 3,
        features: netlist::DieFeatures::default().with_key_shift(9),
    };
    let layout = Layout::with_cells(&program, &w, die, context)?;
    let info = firmware_info(&program, &layout, &die)?;
    let mut engine = VmEngine::new(program.clone(), open().2, die, context)?;
    eprintln!("preparado en {:.0} s", started.elapsed().as_secs_f64());
    let plan = schedule(codes.len(), steps);
    let fill = Fill { text: (&ids, first[0] as u32), plain: Some((&plain, plain_first[0] as u32)), schedule: &plan, seed };
    let filled = fill_on_die(&mut engine, &layout, &program, &info, &config, die.cell.activation_bits, &codes, &fill);
    eprintln!("rellenado en {:.0} s", started.elapsed().as_secs_f64());
    let picture = magvit.decode(&filled)?;
    let row = (side * 3).next_multiple_of(4);
    let mut bmp = b"BM".to_vec();
    bmp.extend(((54 + row * side) as u32).to_le_bytes());
    bmp.extend([0; 4]);
    bmp.extend(54u32.to_le_bytes());
    bmp.extend(40u32.to_le_bytes());
    bmp.extend((side as i32).to_le_bytes());
    bmp.extend((-(side as i32)).to_le_bytes());
    bmp.extend(1u16.to_le_bytes());
    bmp.extend(24u16.to_le_bytes());
    bmp.extend([0; 24]);
    for y in 0..side {
        for x in 0..side {
            let p = &picture[(y * side + x) * 3..][..3];
            bmp.extend([p[2], p[1], p[0]]);
        }
        bmp.resize(bmp.len() + row - side * 3, 0);
    }
    std::fs::write(&args[2], bmp).map_err(|e| e.to_string())?;
    eprintln!("{}: {:.0} s", args[2], started.elapsed().as_secs_f64());
    Ok(())
}
