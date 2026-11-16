//! The factory test in STIL (IEEE 1450), the format testers read: the scan chain, the timing
//! of a test cycle and every pattern `atpg::generate` made, for a design with one chain
//! (`scan::insert_scan`).
//!
//! Each test cycle drives the inputs at its start, strobes the outputs at 40 ns and pulses
//! `clk` from 50 to 80 ns, so outputs are read before the clock edge. The file has:
//! - **the chain test**: `0011…` shifted through the chain and read back, which finds the
//!   faults on the shift path;
//! - **each pattern**: its state shifted in while the previous pattern's result shifts out
//!   (`load_unload`), then one cycle that sets the input pins, reads the output pins and
//!   captures with the clock (`capture`);
//! - a last `load_unload` that shifts out the last result.
//!
//! Shift data goes in the order it is shifted: the first character goes into `scan_in` on
//! the first shift cycle, and is the one that ends farthest from it; the first character
//! expected on `scan_out` is the one already next to it.
//!
//! `read` takes back the subset of STIL this module writes and expands it into test cycles,
//! and `run` applies them on the gate simulator as a tester would.

use crate::atpg::AtpgReport;
use crate::scan::{SCAN_ENABLE, SCAN_IN, SCAN_OUT, TEST_MODE};
use crate::{Design, Dir, Simulator};
use std::collections::HashMap;
use std::fmt::Write;

/// The clock pin, as the exported Verilog names it.
pub const CLOCK: &str = "clk";
const TABLE: &str = "_default_WFT_";
const CHAIN_PINS: [&str; 4] = [SCAN_ENABLE, TEST_MODE, SCAN_IN, SCAN_OUT];

/// The signals of each port: one per bit, `name[i]` when it has more than one.
fn bit_names(name: &str, width: u32) -> Vec<String> {
    if width == 1 {
        vec![name.to_string()]
    } else {
        (0..width).map(|i| format!("{name}[{i}]")).collect()
    }
}

/// Pins as `(name, width)`.
type Pins = Vec<(String, u32)>;

/// The design's pins outside the chain, inputs and outputs.
fn pins(design: &Design) -> (Pins, Pins) {
    let top = &design.modules[&design.top];
    let of = |dir: Dir| top.ports.iter().filter(|p| p.dir == dir && !CHAIN_PINS.contains(&p.name.as_str())).map(|p| (p.name.clone(), top.nets[p.net].width)).collect();
    (of(Dir::Input), of(Dir::Output))
}

fn quoted(names: &[String]) -> String {
    names.iter().map(|n| format!("\"{n}\"")).collect::<Vec<_>>().join(" + ")
}

fn drive(bits: impl IntoIterator<Item = bool>) -> String {
    bits.into_iter().map(|b| if b { '1' } else { '0' }).collect()
}

fn expect(bits: impl IntoIterator<Item = Option<bool>>) -> String {
    bits.into_iter().map(|b| match b {
        Some(true) => 'H',
        Some(false) => 'L',
        None => 'X',
    }).collect()
}

/// The chain test's data: `0011…`, as many as the chain is long.
fn chain_test(length: usize) -> Vec<bool> {
    (0..length).map(|i| (i / 2) % 2 == 1).collect()
}

/// The STIL of a design with a scan chain and the patterns `atpg::generate` made for it.
pub fn write(design: &Design, report: &AtpgReport) -> String {
    let (inputs, outputs) = pins(design);
    let pi: Vec<String> = inputs.iter().flat_map(|(n, w)| bit_names(n, *w)).collect();
    let po: Vec<String> = outputs.iter().flat_map(|(n, w)| bit_names(n, *w)).collect();
    let length = report.chain_length;
    let mut s = String::new();
    let _ = writeln!(s, "STIL 1.0;\n");
    let _ = writeln!(s, "Header {{\n    Title \"llm2chip factory test: {}\";\n    Source \"{} patterns over a scan chain of {} flip-flops\";\n}}\n", design.top, report.patterns.len(), length);

    let _ = writeln!(s, "Signals {{");
    let _ = writeln!(s, "    \"{CLOCK}\" In;\n    \"{TEST_MODE}\" In;\n    \"{SCAN_ENABLE}\" In;\n    \"{SCAN_IN}\" In {{ ScanIn; }}\n    \"{SCAN_OUT}\" Out {{ ScanOut; }}");
    for name in &pi {
        let _ = writeln!(s, "    \"{name}\" In;");
    }
    for name in &po {
        let _ = writeln!(s, "    \"{name}\" Out;");
    }
    let _ = writeln!(s, "}}\n");

    let _ = writeln!(s, "SignalGroups {{");
    if !pi.is_empty() {
        let _ = writeln!(s, "    \"_pi\" = '{}';", quoted(&pi));
    }
    if !po.is_empty() {
        let _ = writeln!(s, "    \"_po\" = '{}';", quoted(&po));
    }
    let _ = writeln!(s, "}}\n");

    let _ = writeln!(s, "Timing {{\n    WaveformTable \"{TABLE}\" {{\n        Period '100ns';\n        Waveforms {{");
    let _ = writeln!(s, "            \"{CLOCK}\" {{ 0 {{ '0ns' D; }} P {{ '0ns' D; '50ns' U; '80ns' D; }} }}");
    let ins: Vec<String> = [TEST_MODE, SCAN_ENABLE, SCAN_IN].iter().map(|n| n.to_string()).chain((!pi.is_empty()).then(|| "_pi".to_string())).collect();
    for name in ins {
        let _ = writeln!(s, "            \"{name}\" {{ 01 {{ '0ns' D/U; }} }}");
    }
    let outs: Vec<String> = [SCAN_OUT.to_string()].into_iter().chain((!po.is_empty()).then(|| "_po".to_string())).collect();
    for name in outs {
        let _ = writeln!(s, "            \"{name}\" {{ LHX {{ '0ns' X; '40ns' L/H/X; }} }}");
    }
    let _ = writeln!(s, "        }}\n    }}\n}}\n");

    let _ = writeln!(s, "ScanStructures {{\n    ScanChain \"chain\" {{ ScanLength {length}; ScanIn \"{SCAN_IN}\"; ScanOut \"{SCAN_OUT}\"; }}\n}}\n");

    // Outputs are not compared while shifting, nor the chain's while capturing.
    let masked_po = if po.is_empty() { String::new() } else { format!(" \"_po\" = {};", "X".repeat(po.len())) };
    let capture_pi = if pi.is_empty() { String::new() } else { " \"_pi\" = #;".to_string() };
    let capture_po = if po.is_empty() { String::new() } else { " \"_po\" = #;".to_string() };
    let _ = writeln!(s, "Procedures {{");
    let _ = writeln!(s, "    \"load_unload\" {{\n        W \"{TABLE}\";\n        V {{ \"{CLOCK}\" = 0; \"{TEST_MODE}\" = 1; \"{SCAN_ENABLE}\" = 1;{masked_po} }}\n        Shift {{ V {{ \"{SCAN_IN}\" = #; \"{SCAN_OUT}\" = #; \"{CLOCK}\" = P; }} }}\n    }}");
    let _ = writeln!(s, "    \"capture\" {{\n        W \"{TABLE}\";\n        V {{ \"{SCAN_ENABLE}\" = 0; \"{SCAN_OUT}\" = X;{capture_pi}{capture_po} \"{CLOCK}\" = P; }}\n    }}");
    let _ = writeln!(s, "}}\n");

    let _ = writeln!(s, "PatternBurst \"_burst_\" {{\n    PatList {{ \"_pattern_\" {{ }} }}\n}}\n");
    let _ = writeln!(s, "PatternExec {{\n    PatternBurst \"_burst_\";\n}}\n");

    let _ = writeln!(s, "Pattern \"_pattern_\" {{\n    W \"{TABLE}\";");
    // The chain test: shifted in, then read back as it was.
    let flush = chain_test(length);
    let _ = writeln!(s, "    \"chain test\": Call \"load_unload\" {{ \"{SCAN_IN}\" = {}; \"{SCAN_OUT}\" = {}; }}", drive(flush.iter().copied()), "X".repeat(length));
    let mut previous: Option<String> = Some(expect(flush.iter().map(|b| Some(*b))));
    for (index, p) in report.patterns.iter().enumerate() {
        // The state goes in last flip-flop first, and the result comes out the same way.
        let load = drive(p.load.iter().rev().copied());
        let unload = previous.take().unwrap_or_else(|| "X".repeat(length));
        let _ = writeln!(s, "    \"pattern {index}\": Call \"load_unload\" {{ \"{SCAN_IN}\" = {load}; \"{SCAN_OUT}\" = {unload}; }}");
        let mut data = String::new();
        if !pi.is_empty() {
            let bits = inputs.iter().flat_map(|(name, width)| {
                let given = p.inputs.iter().find(|(n, _)| n == name).map(|(_, b)| b.clone()).unwrap_or_default();
                (0..*width as usize).map(move |i| given.get(i).copied().unwrap_or(false))
            });
            let _ = write!(data, " \"_pi\" = {};", drive(bits));
        }
        if !po.is_empty() {
            let bits = outputs.iter().flat_map(|(name, width)| {
                let given = p.outputs.iter().find(|(n, _)| n == name).map(|(_, b)| b.clone()).unwrap_or_default();
                (0..*width as usize).map(move |i| given.get(i).copied().flatten())
            });
            let _ = write!(data, " \"_po\" = {};", expect(bits));
        }
        let _ = writeln!(s, "    Call \"capture\" {{{data} }}");
        previous = Some(expect(p.unload.iter().rev().copied()));
    }
    let last = previous.unwrap_or_else(|| "X".repeat(length));
    let _ = writeln!(s, "    \"unload\": Call \"load_unload\" {{ \"{SCAN_IN}\" = {}; \"{SCAN_OUT}\" = {last}; }}", "0".repeat(length));
    let _ = writeln!(s, "}}");
    s
}

/// One test cycle: the inputs it drives, the outputs it expects before the clock, and
/// whether it pulses the clock.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Cycle {
    pub drive: Vec<(String, bool)>,
    pub expect: Vec<(String, bool)>,
    pub pulse: bool,
}

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Word(String),
    Quoted(String),
    Expression(String),
    Symbol(char),
}

fn tokens(text: &str) -> Result<Vec<Token>, String> {
    let mut out = Vec::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            c if c.is_whitespace() => {}
            '"' | '\'' => {
                let mut value = String::new();
                loop {
                    match chars.next() {
                        Some(end) if end == c => break,
                        Some(other) => value.push(other),
                        None => return Err("a quote is not closed".into()),
                    }
                }
                out.push(if c == '"' { Token::Quoted(value) } else { Token::Expression(value) });
            }
            '{' | '}' | ';' | '=' | ':' => out.push(Token::Symbol(c)),
            _ => {
                let mut word = c.to_string();
                while let Some(next) = chars.peek().copied().filter(|n| !n.is_whitespace() && !"{};=:\"'".contains(*n)) {
                    word.push(next);
                    chars.next();
                }
                out.push(Token::Word(word));
            }
        }
    }
    Ok(out)
}

/// A statement: its words up to `;`, or up to a block of statements.
#[derive(Clone, Debug)]
struct Node {
    head: Vec<Token>,
    body: Option<Vec<Node>>,
}

fn nodes(list: &[Token], at: &mut usize) -> Result<Vec<Node>, String> {
    let mut out = Vec::new();
    let mut head = Vec::new();
    while *at < list.len() {
        let token = list[*at].clone();
        *at += 1;
        match token {
            Token::Symbol(';') => out.push(Node { head: std::mem::take(&mut head), body: None }),
            Token::Symbol('{') => {
                let body = nodes(list, at)?;
                out.push(Node { head: std::mem::take(&mut head), body: Some(body) });
            }
            Token::Symbol('}') => {
                if !head.is_empty() {
                    return Err(format!("a statement does not end before a block closes: {head:?}"));
                }
                return Ok(out);
            }
            other => head.push(other),
        }
    }
    if !head.is_empty() {
        return Err(format!("a statement does not end: {head:?}"));
    }
    Ok(out)
}

fn name_of(token: &Token) -> Option<&str> {
    match token {
        Token::Quoted(s) | Token::Word(s) => Some(s),
        _ => None,
    }
}

/// The named block `keyword` at the top, if there is one.
fn block<'a>(top: &'a [Node], keyword: &str) -> Option<&'a [Node]> {
    top.iter().find(|n| n.head.first().and_then(name_of) == Some(keyword)).and_then(|n| n.body.as_deref())
}

/// `"signal" = data;` statements: the name and its characters.
fn assignments(list: &[Node]) -> Result<Vec<(String, String)>, String> {
    list.iter()
        .map(|n| match n.head.as_slice() {
            [name, Token::Symbol('='), value] => Ok((name_of(name).ok_or("a signal has no name")?.to_string(), name_of(value).ok_or("an assignment has no value")?.to_string())),
            other => Err(format!("not an assignment: {other:?}")),
        })
        .collect()
}

/// What `read` keeps of the file: the signals and their direction, the groups, and the
/// procedures' statements.
struct Program {
    outputs: HashMap<String, bool>,
    groups: HashMap<String, Vec<String>>,
    procedures: HashMap<String, Vec<Node>>,
}

impl Program {
    fn members(&self, name: &str) -> Vec<String> {
        self.groups.get(name).cloned().unwrap_or_else(|| vec![name.to_string()])
    }

    /// One `V`: assigns each signal of each name a character, `#` taking the next of the
    /// call's data for that name, and gives the cycle the state then makes.
    fn vector(&self, list: &[Node], state: &mut HashMap<String, char>, data: &mut HashMap<String, std::collections::VecDeque<char>>) -> Result<Cycle, String> {
        for (name, value) in assignments(list)? {
            let members = self.members(&name);
            let given: Vec<char> = if value == "#" {
                let queue = data.get_mut(&name).ok_or_else(|| format!("no data for {name}"))?;
                (0..members.len()).map(|_| queue.pop_front().ok_or_else(|| format!("the data for {name} ran out"))).collect::<Result<_, _>>()?
            } else if value.chars().count() == members.len() {
                value.chars().collect()
            } else if value.chars().count() == 1 {
                vec![value.chars().next().unwrap(); members.len()]
            } else {
                return Err(format!("{name} has {} signals and {value:?} does not fit them", members.len()));
            };
            for (member, c) in members.into_iter().zip(given) {
                state.insert(member, c);
            }
        }
        let mut cycle = Cycle::default();
        let mut sorted: Vec<(&String, &char)> = state.iter().collect();
        sorted.sort();
        for (signal, c) in sorted {
            match (self.outputs.get(signal), c) {
                (Some(false), '0' | '1') => cycle.drive.push((signal.clone(), *c == '1')),
                (Some(false), 'P') => cycle.pulse = true,
                (Some(true), 'L' | 'H') => cycle.expect.push((signal.clone(), *c == 'H')),
                (Some(true), 'X') | (Some(false), _) => {}
                (Some(true), other) => return Err(format!("{signal} is an output and {other} does not compare it")),
                (None, _) => return Err(format!("{signal} is not a signal")),
            }
        }
        Ok(cycle)
    }

    fn call(&self, procedure: &str, data: Vec<(String, String)>, state: &mut HashMap<String, char>, cycles: &mut Vec<Cycle>) -> Result<(), String> {
        let body = self.procedures.get(procedure).ok_or_else(|| format!("there is no procedure {procedure}"))?;
        let mut data: HashMap<String, std::collections::VecDeque<char>> = data.into_iter().map(|(name, value)| (name, value.chars().collect())).collect();
        for statement in body {
            match (statement.head.first().and_then(name_of), &statement.body) {
                (Some("W"), None) => {}
                (Some("V"), Some(list)) => cycles.push(self.vector(list, state, &mut data)?),
                (Some("Shift"), Some(list)) => {
                    // As many shifts as there is data left for the chain.
                    while data.values().any(|queue| !queue.is_empty()) {
                        for vector in list {
                            match (vector.head.first().and_then(name_of), &vector.body) {
                                (Some("V"), Some(inner)) => cycles.push(self.vector(inner, state, &mut data)?),
                                other => return Err(format!("a shift takes only vectors: {other:?}")),
                            }
                        }
                    }
                }
                other => return Err(format!("{procedure}: not understood: {other:?}")),
            }
        }
        Ok(())
    }
}

/// The test cycles of a STIL file of the kind `write` makes.
pub fn read(text: &str) -> Result<Vec<Cycle>, String> {
    let list = tokens(text)?;
    let mut at = 0;
    let top = nodes(&list, &mut at)?;
    let mut outputs = HashMap::new();
    for signal in block(&top, "Signals").ok_or("there is no Signals block")? {
        match signal.head.as_slice() {
            [name, Token::Word(dir)] => {
                outputs.insert(name_of(name).ok_or("a signal has no name")?.to_string(), dir == "Out");
            }
            other => return Err(format!("not a signal: {other:?}")),
        }
    }
    let mut groups = HashMap::new();
    for group in block(&top, "SignalGroups").unwrap_or(&[]) {
        match group.head.as_slice() {
            [name, Token::Symbol('='), Token::Expression(members)] => {
                let members = members.split('+').map(|m| m.trim().trim_matches('"').to_string()).collect();
                groups.insert(name_of(name).ok_or("a group has no name")?.to_string(), members);
            }
            other => return Err(format!("not a signal group: {other:?}")),
        }
    }
    let mut procedures = HashMap::new();
    for procedure in block(&top, "Procedures").unwrap_or(&[]) {
        let name = procedure.head.first().and_then(name_of).ok_or("a procedure has no name")?;
        procedures.insert(name.to_string(), procedure.body.clone().unwrap_or_default());
    }
    let program = Program { outputs, groups, procedures };
    let pattern = top.iter().find(|n| n.head.first().and_then(name_of) == Some("Pattern")).and_then(|n| n.body.as_deref()).ok_or("there is no Pattern")?;
    let (mut state, mut cycles) = (HashMap::new(), Vec::new());
    for statement in pattern {
        // A label goes before what it names.
        let head: &[Token] = match statement.head.as_slice() {
            [Token::Quoted(_), Token::Symbol(':'), rest @ ..] => rest,
            all => all,
        };
        match (head, &statement.body) {
            ([Token::Word(w), _], None) if w == "W" => {}
            ([Token::Word(call), name], Some(data)) if call == "Call" => program.call(name_of(name).ok_or("a call has no procedure")?, assignments(data)?, &mut state, &mut cycles)?,
            ([Token::Word(v)], Some(list)) if v == "V" => {
                let mut none = HashMap::new();
                cycles.push(program.vector(list, &mut state, &mut none)?);
            }
            other => return Err(format!("not understood in the pattern: {other:?}")),
        }
    }
    Ok(cycles)
}

/// The value of signal `name`: its port and bit.
fn port_bit(name: &str) -> (&str, usize) {
    match name.strip_suffix(']').and_then(|s| s.rsplit_once('[')) {
        Some((port, bit)) => (port, bit.parse().unwrap_or(0)),
        None => (name, 0),
    }
}

/// Applies test cycles on the gate simulator as a tester would: each drives its inputs,
/// compares its outputs once they settle, then pulses the clock if it says so. Gives the
/// cycles whose outputs did not match.
pub fn run(sim: &mut Simulator, cycles: &[Cycle]) -> Result<Vec<usize>, String> {
    let mut words: HashMap<String, Vec<u32>> = HashMap::new();
    let mut failed = Vec::new();
    for (index, cycle) in cycles.iter().enumerate() {
        for (signal, value) in &cycle.drive {
            if signal == CLOCK {
                continue;
            }
            let (port, bit) = port_bit(signal);
            let current = words.entry(port.to_string()).or_insert_with(|| sim.get_words(port).unwrap_or_default());
            if current.len() <= bit / 32 {
                current.resize(bit / 32 + 1, 0);
            }
            let mask = 1u32 << (bit % 32);
            current[bit / 32] = if *value { current[bit / 32] | mask } else { current[bit / 32] & !mask };
        }
        for (port, value) in &words {
            sim.set_words(port, value).map_err(|e| format!("{port}: {e:?}"))?;
        }
        sim.settle();
        let matched = cycle.expect.iter().all(|(signal, value)| {
            let (port, bit) = port_bit(signal);
            sim.get_words(port).is_some_and(|got| got.get(bit / 32).is_some_and(|w| (w >> (bit % 32) & 1 == 1) == *value))
        });
        if !matched {
            failed.push(index);
        }
        if cycle.pulse {
            sim.tick();
        }
    }
    Ok(failed)
}
