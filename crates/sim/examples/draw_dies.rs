//! The silicon of drawing: Show-o's transformer, guided in the die, on the die the firmware
//! addresses, and MAGVIT-v2's encoder and decoder, each on a few shapes, counted as the voice
//! planner counts its stages (`sizing::routines_die`, N6). Prints each one's logic, cell
//! memory, vector memory and microcode, its clock, and what stays outside the die.
//!
//! `arch::drawing`'s reference die comes from this: Show-o on 16 units × 64 lanes and 4,096
//! cells. MAGVIT-v2 keeps whole maps of the picture in the vector memory, which comes to
//! more than a reticle, so it stays on the host.
//!
//! `cargo run --release -p sim --example draw_dies -- <Show-o folder> [shapes]`
//!
//! Shapes are `units,lanes,cells` in log2, separated by `/`; by default `0,6,12/4,6,12/4,6,13`.

use netlist::format::{WeightCoding, WeightFormat};
use netlist::{CellConfig, DieConfig};
use sim::lower::Layout;
use sim::magvit_chip::MagvitChip;
use sim::showo_chip::{addressed_die, program_guided, weights};
use sim::sizing::{die_of, reference_die, routines_die, Costs, N6};
use std::path::Path;

fn show(name: &str, shape: (u32, u32, u32), parts: [f64; 4], clock: f64) {
    let [logic, cells, vectors, microcode] = parts;
    println!(
        "{name} on {} units × {} lanes, {} cells: logic {logic:.3}, cells {cells:.3}, vectors {vectors:.3}, microcode {microcode:.3} = {:.1} mm², {:.0} MHz",
        1u32 << shape.0,
        1u32 << shape.1,
        1u64 << shape.2,
        parts.iter().sum::<f64>(),
        clock / 1e6
    );
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(args.get(1).ok_or("the Show-o folder")?);
    let shapes: Vec<(u32, u32, u32)> = args
        .get(2)
        .map_or("0,6,12/4,6,12/4,6,13", String::as_str)
        .split('/')
        .map(|s| {
            let v: Vec<u32> = s.split(',').map(|n| n.parse().expect("a number")).collect();
            (v[0], v[1], v[2])
        })
        .collect();
    let started = std::time::Instant::now();
    let costs = Costs::measure(reference_die(8));

    // MAGVIT-v2: its weights as they come, in cells as wide as the vectors (16 bits).
    let magvit = dir.join("magvitv2");
    let wide = {
        let base = reference_die(8);
        DieConfig { cell: CellConfig { weight_bits: 16, accumulator_bits: 2 * 16 + 8, ..base.cell }, ..base }
    };
    for &shape in &shapes {
        for (name, chip) in [
            ("MAGVIT-v2 decoder", MagvitChip::decoder_on(&magvit, 16, die_of(wide, 1 << 17, shape), Some(256))),
            ("MAGVIT-v2 encoder", MagvitChip::encoder_on(&magvit, 256, die_of(wide, 1 << 17, shape), Some(256))),
        ] {
            match chip.and_then(|c| {
                let (program, layout, _) = c.parts();
                // Its routines are the rows' part: counted as one list, the same instructions.
                let ops = &program.rows().ok_or("no routines")?.ops;
                let code = layout.routines_microprogram(program, &[(ops.as_slice(), false)])?;
                Ok(routines_die(layout, &code, &costs, &N6, shape, 1e9))
            }) {
                Ok((parts, clock)) => show(name, shape, parts, clock),
                Err(e) => println!("{name} on {shape:?}: {e}"),
            }
        }
        eprintln!("({:.0} s)", started.elapsed().as_secs_f64());
    }

    // Show-o: 8-bit affine codes in groups of 64, guidance in the die, the embedding table in
    // the image and a row at a time brought in by the loader.
    let format = WeightFormat { coding: WeightCoding::Affine, bits: 8, exponent_bits: 0, bias: true, group_size: 64, fine_scale: false, keep_source: false };
    let (shape_o, config, w) = weights(dir, sim::Weights::open(dir)?.with_format(Some(format)))?;
    let program = program_guided(&shape_o, Some(1.75));
    let context = (config.max_text + 1 + shape_o.image_rows).next_power_of_two();
    let base = DieConfig { cell: CellConfig { coding: WeightCoding::Affine, accumulator_bits: 48, addr_bits: 3, ..reference_die(8).cell }, reg_bits: 12, index_bits: 12, microcode_bits: 18, ..reference_die(8) };
    for &shape in &shapes {
        let result = addressed_die(&program, &w, die_of(base, context as u32, shape), context).and_then(|die| {
            let layout = Layout::with_cells_table_outside(&program, &w, die, context)?;
            let code = layout.microprogram(&program)?;
            let outside = layout.outside_table().map_or(0, |t| t.mantissas.len());
            Ok((routines_die(&layout, &code, &costs, &N6, shape, 1e9), outside, die.cell.activation_bits))
        });
        match result {
            Ok(((parts, clock), outside, bits)) => {
                show("Show-o", shape, parts, clock);
                println!("  embedding table outside the die: {outside} elements, {:.0} MB at {bits} bits", outside as f64 * bits as f64 / 8e6);
            }
            Err(e) => println!("Show-o on {shape:?}: {e}"),
        }
        eprintln!("({:.0} s)", started.elapsed().as_secs_f64());
    }
    Ok(())
}
