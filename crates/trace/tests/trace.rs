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
