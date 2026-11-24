//! A chip that listens, thinks and speaks, planned as a system: its four stages (Moonshine,
//! a language model, OuteTTS and DAC) measured on a few shapes of die from their own
//! microprograms, with the optimiser's counts in N6, and the planner choosing which share a
//! die and at which shape, for the least silicon that gives the first output in time.
//!
//!     cargo run --release -p sim --example voice_system -- MOONSHINE LLM OUTETTS DAC [FIRST_OUTPUT_MS] [SECONDS] [WORDS] [voice]
//!
//! A `-` in place of Moonshine plans a chip that only speaks (its question comes as text);
//! in place of OuteTTS and DAC, one that only listens (its answer goes as text).
//! `FIRST_OUTPUT_MS` is the longest wait from the end of the question to the first sound,
//! or to the answer's first token when the chip only listens (500); `SECONDS` the longest
//! question (3) and `WORDS` the tokens of the answer's first sentence (12).

use sim::voice_plan::{plan_voice, turn_energy, VoiceAsk, VoiceModels};
use std::path::PathBuf;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [listen, think, speak, sound, rest @ ..] = args.as_slice() else {
        return Err("usage: voice_system MOONSHINE LLM OUTETTS DAC [FIRST_SOUND_MS] [SECONDS] [WORDS]".into());
    };
    let given = |folder: &String| (folder != "-").then(|| PathBuf::from(folder));
    let models = VoiceModels { listen: given(listen), think: PathBuf::from(think), speak: given(speak), sound: given(sound) };
    let defaults = VoiceAsk::default();
    // `voice` among the arguments: the writer reads its speaker's profile before each sentence.
    let speaker = rest.iter().any(|r| r == "voice");
    let rest: Vec<&String> = rest.iter().filter(|r| *r != "voice").collect();
    let number = |i: usize| rest.get(i).and_then(|s| s.parse::<f64>().ok());
    let ask = VoiceAsk { first_output_seconds: number(0).map_or(defaults.first_output_seconds, |ms| ms / 1e3), seconds: number(1).unwrap_or(defaults.seconds), words: number(2).map_or(defaults.words, |w| w as u32), speaker, ..defaults };
    let (best, weighed, stages) = plan_voice(&models, &ask, &mut |what| eprintln!("measuring {what}"))?;
    let shape = |s: (u32, u32, u32)| format!("{} × {}, {} cells", 1u32 << s.0, 1u32 << s.1, 1u64 << s.2);
    println!("groupings weighed:");
    for w in &weighed {
        let groups: Vec<String> = w.groups.iter().map(|g| g.iter().map(|s| stages[*s].name.as_str()).collect::<Vec<_>>().join(" + ")).collect();
        match (&w.best, &w.reason) {
            (Some(p), _) => println!("  [{}]: {:.0} mm², first output {:.0} ms", groups.join("] ["), p.area_mm2, p.first_output_seconds * 1e3),
            (None, Some(why)) => println!("  [{}]: no, {why}", groups.join("] [")),
            _ => {}
        }
    }
    let Some(plan) = best else {
        println!("no system gives its first output in {:.0} ms", ask.first_output_seconds * 1e3);
        return Ok(());
    };
    println!("chosen: {} dies, {:.0} mm², first output {:.0} ms after the question ends", plan.dies.iter().map(|d| d.count).sum::<u32>(), plan.area_mm2, plan.first_output_seconds * 1e3);
    for die in &plan.dies {
        let names: Vec<&str> = die.stages.iter().map(|s| stages[*s].name.as_str()).collect();
        println!("  {} die(s) of {} at {:.0} MHz, {:.0} mm² (logic {:.0}, cells {:.0}, vectors {:.0}): {}", die.count, shape(die.shape), die.clock_hz / 1e6, die.mm2(), die.parts_mm2[0], die.parts_mm2[1], die.parts_mm2[2], names.join(" + "));
    }
    let energy = turn_energy(&plan, &models.roles()?, &ask);
    for (((stage, timing), variant), joules) in stages.iter().zip(&plan.stages).zip(&plan.variants).zip(&energy.stages_j) {
        println!("  {}: first output after {:.1} ms, then {:.0} a second{}; {:.1} mJ a turn", stage.name, timing.seconds_to(1) * 1e3, timing.outputs_per_second(), if variant.is_empty() { String::new() } else { format!(" ({variant})") }, joules * 1e3);
    }
    println!("a turn ({:.1} s): {:.2} J, {:.2} J of it leakage; {:.2} W on average", energy.seconds, energy.total_j, energy.leakage_j, energy.total_j / energy.seconds);
    Ok(())
}
