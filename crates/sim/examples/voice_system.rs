//! A chip that listens, thinks and speaks, planned as a system: its four stages (Moonshine,
//! a language model, OuteTTS and DAC) measured on a few shapes of die from their own
//! microprograms, with the optimiser's counts in N6, and the planner choosing which share a
//! die and at which shape, for the least silicon that gives the first sound in time.
//!
//!     cargo run --release -p sim --example voice_system -- MOONSHINE LLM OUTETTS DAC [FIRST_SOUND_MS] [SECONDS] [WORDS]
//!
//! `FIRST_SOUND_MS` is the longest wait from the end of the question to the first sound
//! (500), `SECONDS` the longest question (3) and `WORDS` the tokens of the answer's first
//! sentence (12).

use sim::voice_plan::{plan_voice, VoiceAsk, VoiceModels};
use std::path::PathBuf;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [listen, think, speak, sound, rest @ ..] = args.as_slice() else {
        return Err("usage: voice_system MOONSHINE LLM OUTETTS DAC [FIRST_SOUND_MS] [SECONDS] [WORDS]".into());
    };
    let models = VoiceModels { listen: PathBuf::from(listen), think: PathBuf::from(think), speak: PathBuf::from(speak), sound: PathBuf::from(sound) };
    let number = |i: usize| rest.get(i).and_then(|s| s.parse::<f64>().ok());
    let defaults = VoiceAsk::default();
    let ask = VoiceAsk { first_sound_seconds: number(0).map_or(defaults.first_sound_seconds, |ms| ms / 1e3), seconds: number(1).unwrap_or(defaults.seconds), words: number(2).map_or(defaults.words, |w| w as u32), ..defaults };
    let (best, weighed, stages) = plan_voice(&models, &ask, &mut |what| eprintln!("measuring {what}"))?;
    let shape = |s: (u32, u32, u32)| format!("{} × {}, {} cells", 1u32 << s.0, 1u32 << s.1, 1u64 << s.2);
    println!("groupings weighed:");
    for w in &weighed {
        let groups: Vec<String> = w.groups.iter().map(|g| g.iter().map(|s| stages[*s].name.as_str()).collect::<Vec<_>>().join(" + ")).collect();
        match (&w.best, &w.reason) {
            (Some(p), _) => println!("  [{}]: {:.0} mm², first sound {:.0} ms", groups.join("] ["), p.area_mm2, p.first_output_seconds * 1e3),
            (None, Some(why)) => println!("  [{}]: no, {why}", groups.join("] [")),
            _ => {}
        }
    }
    let Some(plan) = best else {
        println!("no system meets the first sound in {:.0} ms", ask.first_sound_seconds * 1e3);
        return Ok(());
    };
    println!("chosen: {} dies, {:.0} mm², first sound {:.0} ms after the question ends", plan.dies.iter().map(|d| d.count).sum::<u32>(), plan.area_mm2, plan.first_output_seconds * 1e3);
    for die in &plan.dies {
        let names: Vec<&str> = die.stages.iter().map(|s| stages[*s].name.as_str()).collect();
        println!("  {} die(s) of {} at {:.0} MHz, {:.0} mm² (logic {:.0}, cells {:.0}, vectors {:.0}): {}", die.count, shape(die.shape), die.clock_hz / 1e6, die.mm2(), die.parts_mm2[0], die.parts_mm2[1], die.parts_mm2[2], names.join(" + "));
    }
    for ((stage, timing), variant) in stages.iter().zip(&plan.stages).zip(&plan.variants) {
        println!("  {}: first output after {:.1} ms, then {:.0} a second{}", stage.name, timing.seconds_to(1) * 1e3, timing.outputs_per_second(), if variant.is_empty() { String::new() } else { format!(" ({variant})") });
    }
    Ok(())
}
