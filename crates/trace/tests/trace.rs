//! Traces tell what the gates did, without changing it.

use chip_trace::board::run_joined;
use chip_trace::record::WordClass;
use chip_trace::{tiny_board, wires, TinyBoard, TinyDies, TinyModel};
use netlist::DieProgram;
use sim::chip::Sharing;
use sim::lower::{Layout, ENTRY_LOGITS, ENTRY_TOKEN};
use sim::Weights;
use std::path::PathBuf;

fn checkpoint(model: TinyModel, tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("llm2chip-trace-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    model.write(&dir);
    dir
}

/// Marking where each operation starts lays the words out exactly as before, and every
/// operation of a token gets its mark, in order.
#[test]
fn marks_change_no_word_and_follow_the_operations() {
    let dir = checkpoint(TinyModel::Plain, "marks");
    let program = compiler::compile_qwen3_5(&TinyModel::Plain.config()).unwrap();
    let weights = Weights::open(&dir).unwrap();
    let board = tiny_board(TinyModel::Plain, &dir, 1, Sharing::Layers, TinyDies::default()).unwrap();
    let c = board.configs[0];

    let layout = Layout::with_cells(&program, &weights, c, chip_trace::TINY_CONTEXT).unwrap();
    let steps = layout.routine_steps(&program, &program.ops[..program.logits_from], true).unwrap();
    let marks: Vec<(usize, u32)> = (0..=steps.len()).map(|s| (s, s as u32)).collect();
    let (mut plain, mut marked) = (DieProgram::new(c), DieProgram::new(c));
    plain.assemble(&steps);
    marked.assemble_marked(&steps, &marks);
    let encode = |words: Vec<netlist::ControlWord>| words.iter().map(|w| w.encode_words(&c)).collect::<Vec<_>>();
    assert_eq!(encode(plain.finish()), encode(marked.finish()));

    let code = &board.dies[0].1;
    let token: Vec<u32> = code.marks.iter().filter(|(at, _)| *at < code.entries[ENTRY_LOGITS]).map(|(_, op)| *op).collect();
    assert_eq!(token, (0..program.logits_from as u32).collect::<Vec<_>>());
    assert!(code.marks.windows(2).all(|w| w[0].0 <= w[1].0), "marks in address order");
    assert!(code.marks.iter().all(|(at, _)| (*at as usize) < code.words.len()));
    std::fs::remove_dir_all(dir).ok();
}

fn fresh(dir: &std::path::Path, chips: usize, sharing: Sharing) -> TinyBoard {
    tiny_board(TinyModel::Plain, dir, chips, sharing, TinyDies::default()).unwrap()
}

/// Two groups of two chips run a token and its logits while the recorder watches: they
/// take the cycles they take unwatched and reach the logits of one die alone, and the
/// trace accounts for every cycle of every die.
#[test]
fn a_trace_records_the_dies_without_changing_them() {
    let dir = checkpoint(TinyModel::Plain, "board");
    let sharing = Sharing::Mixed(2);
    let tokens = [3u32];

    let mut watched = fresh(&dir, 4, sharing);
    let trace = watched.record("two groups of two", &tokens).unwrap();

    let mut unwatched = fresh(&dir, 4, sharing);
    let pairs = wires(4, sharing);
    let cycles = [run_joined(&mut unwatched.dies, &pairs, ENTRY_TOKEN, 3, 0), run_joined(&mut unwatched.dies, &pairs, ENTRY_LOGITS, 3, 0)];
    assert_eq!(trace.steps.iter().map(|s| s.cycles).collect::<Vec<_>>(), cycles);
    assert_eq!(watched.logits(), unwatched.logits());

    let mut alone = fresh(&dir, 1, Sharing::Layers);
    alone.record("alone", &tokens).unwrap();
    assert_eq!(watched.logits(), alone.logits(), "the logits of one die alone");

    let total: u64 = cycles.iter().sum();
    assert_eq!(trace.steps[1].start, cycles[0]);
    for die in &trace.dies {
        assert_eq!(die.pcs.len() as u64, total);
        assert!(die.pcs.iter().all(|pc| *pc < die.words.len() as i32));
        assert!(die.stalls.iter().all(|(start, length)| start + length <= total));
        assert!(die.words.iter().any(|w| w.class == WordClass::Cells));
    }
    // Every link carried something, and the groups' rings closed over their loops.
    assert_eq!(trace.links.len(), 6);
    assert!(trace.links.iter().all(|l| l.spans.iter().any(|(_, _, state)| *state == 1)), "{:?}", trace.links.iter().map(|l| l.spans.len()).collect::<Vec<_>>());
    // The attention layer writes its cache once per token on the group that holds it.
    let attention = trace.ops.iter().filter(|o| o.kind == "KvAppend").count();
    assert!(attention > 0);
    assert!(!trace.kv.is_empty() && trace.kv.iter().all(|k| trace.ops[k.op as usize].kind == "KvAppend" && k.position == 0));
    let json = serde_json::to_string(&trace).unwrap();
    assert_eq!(serde_json::from_str::<chip_trace::record::Trace>(&json).unwrap(), trace);
    std::fs::remove_dir_all(dir).ok();
}

/// Dies that compute what they should part from the exact machine nowhere, however they
/// share the model.
#[test]
#[ignore = "slow: four boards gate by gate, each checked against the exact machine"]
fn dies_that_work_do_not_diverge() {
    let dir = checkpoint(TinyModel::Plain, "agree");
    for (chips, sharing) in [(1, Sharing::Layers), (2, Sharing::Layers), (2, Sharing::Rows), (4, Sharing::Mixed(2))] {
        let mut board = fresh(&dir, chips, sharing);
        let trace = board.record_checked("checked", &[3], &dir).unwrap();
        assert_eq!(trace.divergences, vec![], "{chips} chips, {sharing:?}");
        // Checking reads the registers between routines without changing what comes after.
        if chips == 4 {
            let mut plain = fresh(&dir, chips, sharing);
            plain.record("plain", &[3]).unwrap();
            assert_eq!(board.logits(), plain.logits(), "{chips} chips, {sharing:?}");
        }
    }
    std::fs::remove_dir_all(dir).ok();
    // Chips that share the attention's heads each make only their heads' slice of some
    // buffers: that slice is what is checked.
    let dir = checkpoint(TinyModel::Grouped, "agree-heads");
    let mut board = tiny_board(TinyModel::Grouped, &dir, 4, Sharing::Mixed(2), TinyDies::default()).unwrap();
    let trace = board.record_checked("heads", &[3], &dir).unwrap();
    assert_eq!(trace.divergences, vec![]);
    std::fs::remove_dir_all(dir).ok();
}

/// A die that forgets to store one operation's result is caught at that operation,
/// on that die, in the first routine it runs it, from the cycle it started it; the die
/// before it agrees until what the broken one sends it reaches the logits.
#[test]
fn a_broken_operation_is_found_where_it_breaks() {
    let dir = checkpoint(TinyModel::Plain, "broken");
    let mut board = fresh(&dir, 2, Sharing::Layers);
    // The last operation of its part that the second die keeps the result of whole and
    // writes with words of its own.
    let program = board.program.clone();
    let op = chip_trace::check::checked(&board, 1, 0..program.logits_from)
        .into_iter()
        .map(|(op, _)| op as u32)
        .rev()
        .find(|op| {
            let (from, to) = chip_trace::op_words(&board.dies[1].1, *op);
            board.dies[1].1.words[from..to].iter().any(|w| w.write_en)
        })
        .expect("an operation to break");
    board.break_op(1, op);
    let trace = board.record_checked("broken", &[3, 5], &dir).unwrap();
    let first = trace.divergences.first().expect("the fault is found");
    assert_eq!((first.die, first.op, first.step), (1, op, 0), "{:?}", trace.divergences);
    let start = trace.dies[1].pcs[first.cycle as usize];
    assert_eq!(trace.dies[1].words[start as usize].op, Some(op));
    // The first die agrees on its own layers; only the logits it makes from the broken
    // die's hidden state differ.
    assert!(trace.divergences.iter().filter(|d| d.die == 0).all(|d| trace.steps[d.step].routine == "logits"), "{:?}", trace.divergences);
    std::fs::remove_dir_all(dir).ok();
}

/// A design traced operation by operation: the chips of a group start every operation
/// together and wait for the slowest, groups take turns, the first group makes the
/// logits, and a token comes to about what the sizing counts from the whole microprograms.
#[test]
fn a_design_traced_by_operations_follows_its_groups() {
    use chip_trace::design::{design_trace, Design};
    let dir = checkpoint(TinyModel::Plain, "design");
    let program = compiler::compile_qwen3_5(&TinyModel::Plain.config()).unwrap();
    let weights = Weights::open(&dir).unwrap();
    let design = Design { chips: 4, sharing: Sharing::Mixed(2), cluster_bits: 1, lane_bits: 2, cell_bits: 3, context: 16, factor_steps: 1 };
    let trace = design_trace("tiny", &program, &weights, design, &[0, 15]).unwrap();
    assert_eq!(trace.source, "timing");
    assert_eq!(trace.steps.iter().map(|s| s.routine.as_str()).collect::<Vec<_>>(), ["token", "logits", "token", "logits"]);
    for pair in [(0, 1), (2, 3)] {
        let starts = |d: usize| trace.dies[d].spans.iter().map(|s| (s.0, s.2)).collect::<Vec<_>>();
        assert_eq!(starts(pair.0), starts(pair.1), "the members of a group start each operation together");
    }
    // The second group starts once the first is done, and makes no logits.
    let first_end = trace.dies[0].spans.iter().filter(|s| s.0 < trace.steps[1].start).map(|s| s.0 + s.1).max().unwrap();
    assert!(trace.dies[2].spans[0].0 >= first_end);
    assert!(trace.dies[2].spans.iter().all(|s| (s.2 as usize) < program.logits_from));
    assert!(trace.dies[0].spans.iter().any(|s| (s.2 as usize) >= program.logits_from));
    // Against the sizing, which counts each chip's whole microprogram.
    let die = sim::sizing::die_of(sim::sizing::reference_die(1), 16, (1, 2, 3));
    let built = sim::sizing::built(&program, &weights, die, 16, 4, Sharing::Mixed(2)).unwrap();
    let sized = sim::sizing::timing(&built.programs, die, 16, 1, 1, Sharing::Mixed(2)).token_cycles;
    let traced = trace.steps[2].cycles + trace.steps[3].cycles;
    eprintln!("traced {traced}, sized {sized}");
    assert!(traced as f64 > 0.8 * sized as f64 && (traced as f64) < 1.3 * sized as f64, "traced {traced}, sized {sized}");
    std::fs::remove_dir_all(dir).ok();
}
