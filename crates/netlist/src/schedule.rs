//! A schedule of routines that the firmware runs by itself, for a die whose program is a set
//! of routines rather than a token's pass: a speech recogniser's encoder and decoder, say.
//! The host sends a few numbers and the input's vectors; the firmware works out the lengths,
//! orders every routine and writes the tokens the die picks. Nothing in here knows a model:
//! the compiler writes the records, and `expand` reads them as the firmware does, so a
//! schedule can be checked against the host's own order of routines.
//!
//! Each record is `RECORD_WORDS` words, its kind first:
//!
//! - `RECEIVE var`: a word from the host into a variable.
//! - `COMPUTE dst src mul pre div post max`: `dst = min((src · mul + pre) / div + post, max)`,
//!   in signed arithmetic.
//! - `VECTORS count add address exponent size routine`: `var[count] + add` vectors from the
//!   host, each its exponent then its elements, written to the die at `address` (element 0)
//!   and `exponent`; after each, `routine` runs with the vector's index as its token
//!   (`NO_ROUTINE` runs nothing).
//! - `REPEAT body count add`: the next `body` records, `var[count] + add` times, with the
//!   iteration's index in `INDEX`.
//! - `CALL routine ta tb tvar pa pb pvar`: runs `routine` with token `ta · i + tb + var[tvar]`
//!   and position `pa · i + pb + var[pvar]`, `i` being the innermost repeat's index.
//! - `HALVE count load second add`: adds `var[count]` rows pairwise until one is left: while
//!   more than one, the second half onto the first (`load i`, `second i + half`, `add i`),
//!   and an odd last row onto the first.
//! - `GENERATE body limit start`: writes up to `var[limit]` tokens, starting from token
//!   `start`: for each position, the next `body` records run with the token in `TOKEN` and
//!   the position in `POSITION`, then the die's logits are sampled; a stop token ends it, and
//!   any other is sent to the host and becomes the next one.
//! - `SEND address exponent`: sends the host the die's element at `address` and the exponent
//!   at `exponent`, each as a word.
//! - `STREAM body count`: the next `body` records for as long as the host sends a word other
//!   than zero before each time, the time's index in `INDEX`; how many times in `count`. For
//!   input that comes while it is being worked on, such as speech.
//! - `END`.

/// Words in a record.
pub const RECORD_WORDS: usize = 8;
pub const END: u32 = 0;
pub const RECEIVE: u32 = 1;
pub const COMPUTE: u32 = 2;
pub const VECTORS: u32 = 3;
pub const REPEAT: u32 = 4;
pub const CALL: u32 = 5;
pub const HALVE: u32 = 6;
pub const GENERATE: u32 = 7;
pub const SEND: u32 = 8;
pub const STREAM: u32 = 9;

/// Variables: 0 is always zero; the token and position of `GENERATE`; the repeat's index.
pub const VARIABLES: usize = 16;
pub const ZERO: u32 = 0;
pub const TOKEN: u32 = 1;
pub const POSITION: u32 = 2;
pub const INDEX: u32 = 3;
/// The first variable free for the schedule's own numbers.
pub const FIRST_FREE: u32 = 4;
pub const NO_ROUTINE: u32 = u32::MAX;

/// A record of the schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Record {
    Receive { var: u32 },
    Compute { dst: u32, src: u32, mul: i32, pre: i32, div: i32, post: i32, max: i32 },
    Vectors { count: u32, add: u32, address: u32, exponent: u32, size: u32, routine: u32 },
    Repeat { body: u32, count: u32, add: u32 },
    Call { routine: u32, token: (u32, u32, u32), position: (u32, u32, u32) },
    Halve { count: u32, load: u32, second: u32, add: u32 },
    Generate { body: u32, limit: u32, start: u32 },
    Send { address: u32, exponent: u32 },
    Stream { body: u32, count: u32 },
    End,
}

impl Record {
    pub fn words(&self) -> [u32; RECORD_WORDS] {
        let w = |kind: u32, rest: &[u32]| {
            let mut words = [0; RECORD_WORDS];
            words[0] = kind;
            words[1..=rest.len()].copy_from_slice(rest);
            words
        };
        match *self {
            Record::Receive { var } => w(RECEIVE, &[var]),
            Record::Compute { dst, src, mul, pre, div, post, max } => w(COMPUTE, &[dst, src, mul as u32, pre as u32, div as u32, post as u32, max as u32]),
            Record::Vectors { count, add, address, exponent, size, routine } => w(VECTORS, &[count, add, address, exponent, size, routine]),
            Record::Repeat { body, count, add } => w(REPEAT, &[body, count, add]),
            Record::Call { routine, token, position } => w(CALL, &[routine, token.0, token.1, token.2, position.0, position.1, position.2]),
            Record::Halve { count, load, second, add } => w(HALVE, &[count, load, second, add]),
            Record::Generate { body, limit, start } => w(GENERATE, &[body, limit, start]),
            Record::Send { address, exponent } => w(SEND, &[address, exponent]),
            Record::Stream { body, count } => w(STREAM, &[body, count]),
            Record::End => w(END, &[]),
        }
    }

    pub fn read(words: &[u32]) -> Result<Self, String> {
        let a = |i: usize| words.get(i).copied().unwrap_or(0);
        Ok(match a(0) {
            END => Record::End,
            RECEIVE => Record::Receive { var: a(1) },
            COMPUTE => Record::Compute { dst: a(1), src: a(2), mul: a(3) as i32, pre: a(4) as i32, div: a(5) as i32, post: a(6) as i32, max: a(7) as i32 },
            VECTORS => Record::Vectors { count: a(1), add: a(2), address: a(3), exponent: a(4), size: a(5), routine: a(6) },
            REPEAT => Record::Repeat { body: a(1), count: a(2), add: a(3) },
            CALL => Record::Call { routine: a(1), token: (a(2), a(3), a(4)), position: (a(5), a(6), a(7)) },
            HALVE => Record::Halve { count: a(1), load: a(2), second: a(3), add: a(4) },
            GENERATE => Record::Generate { body: a(1), limit: a(2), start: a(3) },
            SEND => Record::Send { address: a(1), exponent: a(2) },
            STREAM => Record::Stream { body: a(1), count: a(2) },
            kind => return Err(format!("no record of kind {kind}")),
        })
    }
}

/// The schedule's words, ended.
pub fn words(records: &[Record]) -> Vec<u32> {
    records.iter().chain(&[Record::End]).flat_map(|r| r.words()).collect()
}

/// What the firmware does with the outside world while it follows a schedule.
pub trait Driver {
    fn receive(&mut self) -> u32;
    /// Writes one element or exponent of the die's vector memory.
    fn write(&mut self, address: u32, value: u32);
    /// Reads one, as a signed number of the vectors' width.
    fn read(&mut self, address: u32) -> u32;
    fn run(&mut self, routine: u32, token: u32, position: u32);
    /// The token the die's logits give, after a routine of `GENERATE`'s body.
    fn sample(&mut self) -> u32;
    fn stops(&self, token: u32) -> bool;
    fn send(&mut self, token: u32);
}

type Vars = [u32; VARIABLES];
type VarReader = dyn Fn(&Vars, u32) -> Result<u32, String>;

/// Follows `words` as the firmware does.
pub fn expand(words: &[u32], driver: &mut impl Driver) -> Result<(), String> {
    let mut vars = [0u32; VARIABLES];
    let records = words.len() / RECORD_WORDS;
    let read = |at: usize| Record::read(&words[at * RECORD_WORDS..(at + 1) * RECORD_WORDS]);
    let var = |vars: &[u32; VARIABLES], v: u32| vars.get(v as usize).copied().ok_or(format!("no variable {v}"));
    // Runs the records from `at` to `end`, with `vars[INDEX]` as the repeat's index.
    fn block<D: Driver>(at: usize, end: usize, vars: &mut Vars, read: &dyn Fn(usize) -> Result<Record, String>, var: &VarReader, driver: &mut D) -> Result<bool, String> {
        let mut at = at;
        while at < end {
            let record = read(at)?;
            at += 1;
            match record {
                Record::End => return Ok(false),
                Record::Receive { var: v } => *vars.get_mut(v as usize).ok_or("no such variable")? = driver.receive(),
                Record::Compute { dst, src, mul, pre, div, post, max } => {
                    let value = (var(vars, src)? as i32).wrapping_mul(mul).wrapping_add(pre).checked_div(div).ok_or("division by zero")?.wrapping_add(post).min(max);
                    *vars.get_mut(dst as usize).ok_or("no such variable")? = value as u32;
                }
                Record::Vectors { count, add, address, exponent, size, routine } => {
                    for i in 0..var(vars, count)? + add {
                        let value = driver.receive();
                        driver.write(exponent, value);
                        for e in 0..size {
                            let value = driver.receive();
                            driver.write(address + e, value);
                        }
                        if routine != NO_ROUTINE {
                            driver.run(routine, i, 0);
                        }
                    }
                }
                Record::Repeat { body, count, add } => {
                    let outer = vars[INDEX as usize];
                    for i in 0..var(vars, count)? + add {
                        vars[INDEX as usize] = i;
                        if !block(at, at + body as usize, vars, read, var, driver)? {
                            return Ok(false);
                        }
                    }
                    vars[INDEX as usize] = outer;
                    at += body as usize;
                }
                Record::Call { routine, token, position } => {
                    let i = vars[INDEX as usize];
                    let t = token.0.wrapping_mul(i).wrapping_add(token.1).wrapping_add(var(vars, token.2)?);
                    let p = position.0.wrapping_mul(i).wrapping_add(position.1).wrapping_add(var(vars, position.2)?);
                    driver.run(routine, t, p);
                }
                Record::Halve { count, load, second, add } => {
                    let mut count = var(vars, count)?;
                    while count > 1 {
                        let half = count / 2;
                        for i in 0..half {
                            driver.run(load, i, 0);
                            driver.run(second, i + half, 0);
                            driver.run(add, i, 0);
                        }
                        if count % 2 == 1 {
                            driver.run(load, 0, 0);
                            driver.run(second, count - 1, 0);
                            driver.run(add, 0, 0);
                        }
                        count = half;
                    }
                }
                Record::Stream { body, count } => {
                    let outer = vars[INDEX as usize];
                    let mut times = 0;
                    while driver.receive() != 0 {
                        vars[INDEX as usize] = times;
                        if !block(at, at + body as usize, vars, read, var, driver)? {
                            return Ok(false);
                        }
                        times += 1;
                    }
                    vars[INDEX as usize] = outer;
                    *vars.get_mut(count as usize).ok_or("no such variable")? = times;
                    at += body as usize;
                }
                Record::Send { address, exponent } => {
                    for at in [address, exponent] {
                        let value = driver.read(at);
                        driver.send(value);
                    }
                }
                Record::Generate { body, limit, start } => {
                    vars[TOKEN as usize] = start;
                    for position in 0..var(vars, limit)? {
                        vars[POSITION as usize] = position;
                        if !block(at, at + body as usize, vars, read, var, driver)? {
                            return Ok(false);
                        }
                        let token = driver.sample();
                        if driver.stops(token) {
                            break;
                        }
                        driver.send(token);
                        vars[TOKEN as usize] = token;
                    }
                    at += body as usize;
                }
            }
        }
        Ok(true)
    }
    block(0, records, &mut vars, &read, &var, driver)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Log {
        input: std::collections::VecDeque<u32>,
        runs: Vec<(u32, u32, u32)>,
        writes: Vec<(u32, u32)>,
        sent: Vec<u32>,
    }

    impl Driver for Log {
        fn receive(&mut self) -> u32 {
            self.input.pop_front().unwrap()
        }
        fn write(&mut self, address: u32, value: u32) {
            self.writes.push((address, value));
        }
        fn read(&mut self, address: u32) -> u32 {
            address + 1
        }
        fn run(&mut self, routine: u32, token: u32, position: u32) {
            self.runs.push((routine, token, position));
        }
        fn sample(&mut self) -> u32 {
            self.runs.last().unwrap().2 + 10
        }
        fn stops(&self, token: u32) -> bool {
            token == 13
        }
        fn send(&mut self, token: u32) {
            self.sent.push(token);
        }
    }

    #[test]
    fn a_schedule_reads_back_as_written_and_runs_its_loops() {
        let records = [
            Record::Receive { var: 4 },
            Record::Compute { dst: 5, src: 4, mul: 1, pre: -1, div: 2, post: 1, max: 100 },
            Record::Vectors { count: ZERO, add: 1, address: 64, exponent: 9, size: 2, routine: 7 },
            Record::Repeat { body: 2, count: 5, add: 0 },
            Record::Call { routine: 1, token: (3, 0, ZERO), position: (0, 0, ZERO) },
            Record::Call { routine: 2, token: (1, 0, ZERO), position: (0, 1, 5) },
            Record::Halve { count: 4, load: 3, second: 4, add: 5 },
            Record::Generate { body: 1, limit: 4, start: 1 },
            Record::Call { routine: 6, token: (0, 0, TOKEN), position: (0, 0, POSITION) },
        ];
        let w = words(&records);
        for (i, r) in records.iter().enumerate() {
            assert_eq!(Record::read(&w[i * RECORD_WORDS..]).unwrap(), *r);
        }
        let mut log = Log { input: [5, 30, 31, 32].into(), ..Default::default() };
        expand(&w, &mut log).unwrap();
        assert_eq!(log.writes, [(9, 30), (64, 31), (65, 32)]);
        // 5 rows: (5 - 1) / 2 + 1 = 3 repeats; halving 5 rows: 2 pairs and the odd one, then 1 pair.
        let expected = [
            (7, 0, 0),
            (1, 0, 0), (2, 0, 4), (1, 3, 0), (2, 1, 4), (1, 6, 0), (2, 2, 4),
            (3, 0, 0), (4, 2, 0), (5, 0, 0), (3, 1, 0), (4, 3, 0), (5, 1, 0), (3, 0, 0), (4, 4, 0), (5, 0, 0),
            (3, 0, 0), (4, 1, 0), (5, 0, 0),
            (6, 1, 0), (6, 10, 1), (6, 11, 2), (6, 12, 3),
        ];
        assert_eq!(log.runs, expected);
        // Positions 0..3 sample 10, 11, 12, 13; 13 stops.
        assert_eq!(log.sent, [10, 11, 12]);
    }
}
