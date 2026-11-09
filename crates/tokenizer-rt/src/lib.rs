//! The tokenizer that runs on the chip, driven only by tables built from a model's
//! `tokenizer.json`. It follows the steps of the Hugging Face library: added tokens,
//! normalizer, pre-tokenizer, model and post-processor. It needs no heap, so the same
//! code runs in the firmware and in the tests that compare it with the library.
#![no_std]

pub const HEADER_WORDS: usize = 34;
pub const F_CLASSES: usize = 0;
pub const F_BLOB: usize = 1;
pub const F_NORMALIZER: usize = 2;
pub const F_PRE_TOKENIZER: usize = 3;
pub const F_WORD_CLASS: usize = 4;
pub const F_SPACE_CLASS: usize = 5;
pub const F_ADDED_RAW: usize = 6;
pub const F_ADDED_RAW_COUNT: usize = 7;
pub const F_ADDED_NORMALIZED: usize = 8;
pub const F_ADDED_NORMALIZED_COUNT: usize = 9;
pub const F_MODEL: usize = 10;
pub const F_VOCAB_SLOTS: usize = 11;
pub const F_VOCAB: usize = 12;
pub const F_SEED: usize = 13;
pub const F_MERGE_SLOTS: usize = 14;
pub const F_MERGES: usize = 15;
pub const F_BYTE_FALLBACK: usize = 16;
pub const F_FUSE_UNK: usize = 17;
pub const F_UNK: usize = 18;
pub const F_PREFIX: usize = 19;
pub const F_PREFIX_LEN: usize = 20;
pub const F_SUFFIX: usize = 21;
pub const F_SUFFIX_LEN: usize = 22;
pub const F_IGNORE_MERGES: usize = 23;
pub const F_POST_BEFORE: usize = 24;
pub const F_POST_AFTER: usize = 25;
pub const F_ADD_SPECIAL: usize = 26;
pub const F_OFFSETS: usize = 27;
pub const F_TOKENS: usize = 28;
pub const F_LEADING_SPACES: usize = 29;
pub const F_FIRST_TOKEN_SPACES: usize = 30;
pub const F_TEXT: usize = 31;
pub const F_TEXT_BYTES: usize = 32;
pub const F_ARENA_WORDS: usize = 33;

pub const MODEL_BPE: u32 = 0;
pub const MODEL_WORD_LEVEL: u32 = 1;

pub const ADDED_SINGLE_WORD: u32 = 1;
pub const ADDED_LSTRIP: u32 = 2;
pub const ADDED_RSTRIP: u32 = 4;
/// Where the first byte of an added token sits in its flags, to reject most positions cheaply.
pub const ADDED_FIRST_SHIFT: u32 = 8;

pub const R_MATCH: u32 = 0;
pub const R_CHAR: u32 = 1;
pub const R_CLASS: u32 = 2;
pub const R_SPLIT: u32 = 3;
pub const R_JUMP: u32 = 4;
pub const R_LOOK: u32 = 5;
pub const R_LINE_START: u32 = 6;
pub const R_LINE_END: u32 = 7;
pub const R_TEXT_START: u32 = 8;
pub const R_TEXT_END: u32 = 9;
pub const R_TEXT_END_NEWLINE: u32 = 10;

/// Every step takes four words: the operation and three arguments.
pub const STEP_WORDS: usize = 4;
pub const S_END: u32 = 0;
pub const S_NORMAL_FORM: u32 = 1;
pub const S_LOWERCASE: u32 = 2;
pub const S_STRIP: u32 = 3;
pub const S_REMOVE_CLASS: u32 = 4;
pub const S_REPLACE: u32 = 5;
pub const S_PREPEND: u32 = 6;
pub const S_BYTES: u32 = 7;
pub const S_SPLIT: u32 = 8;
pub const S_METASPACE: u32 = 9;

pub const SPLIT_REMOVED: u32 = 0;
pub const SPLIT_ISOLATED: u32 = 1;
pub const SPLIT_PREVIOUS: u32 = 2;
pub const SPLIT_NEXT: u32 = 3;
pub const SPLIT_CONTIGUOUS: u32 = 4;
/// Added to a split's mode when what is matched and what is not swap roles.
pub const SPLIT_INVERT: u32 = 8;

pub const META_NEVER: u32 = 0;
pub const META_FIRST: u32 = 1;
pub const META_ALWAYS: u32 = 2;

/// Words of one normal form's description, then the tables it points to.
pub const FORM_WORDS: usize = 7;

const ADDED_PIECE: u32 = u32::MAX;
const NO_RANK: u32 = u32::MAX;
const HANGUL_S: u32 = 0xAC00;
const HANGUL_L: u32 = 0x1100;
const HANGUL_V: u32 = 0x1161;
const HANGUL_T: u32 = 0x11A7;
const HANGUL_L_COUNT: u32 = 19;
const HANGUL_V_COUNT: u32 = 21;
const HANGUL_T_COUNT: u32 = 28;
const HANGUL_S_COUNT: u32 = HANGUL_L_COUNT * HANGUL_V_COUNT * HANGUL_T_COUNT;

/// The working memory given to `encode` was not enough for this text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutOfMemory;

type Outcome<T = ()> = Result<T, OutOfMemory>;

/// Two independent hashes of a string, kept together to tell its entry from any other.
pub fn hash_string(seed: u32, bytes: impl Iterator<Item = u8>) -> (u32, u32) {
    let (mut first, mut second) = (seed ^ 0x811C_9DC5, seed.wrapping_mul(0x9E37_79B1) ^ 0xDEAD_BEEF);
    for byte in bytes {
        first = (first ^ byte as u32).wrapping_mul(0x0100_0193);
        second = (second.rotate_left(5) ^ byte as u32).wrapping_mul(0x2F0B_4C87);
    }
    (first, second)
}

pub fn pair_slot(first: u32, second: u32) -> u32 {
    first.wrapping_mul(0x9E37_79B1) ^ second.wrapping_mul(0x85EB_CA77)
}

/// GPT-2's reversible map from a byte to a printable character.
pub fn byte_character(byte: u8) -> u32 {
    match byte {
        0..=0x20 => 256 + byte as u32,
        0x7F..=0xA0 => 256 + 33 + byte as u32 - 0x7F,
        0xAD => 256 + 67,
        _ => byte as u32,
    }
}

/// The UTF-8 character at `at` and the offset of the next one.
fn character(text: &[u8], at: usize) -> (u32, usize) {
    let first = text[at] as u32;
    let (length, initial) = match first {
        0..=0x7F => return (first, at + 1),
        0xC0..=0xDF => (2, first & 0x1F),
        0xE0..=0xEF => (3, first & 0x0F),
        _ => (4, first & 0x07),
    };
    let end = (at + length).min(text.len());
    let code = text[at + 1..end].iter().fold(initial, |code, byte| code << 6 | (*byte as u32 & 0x3F));
    (code, end)
}

/// The character that ends just before `at`, and where it starts.
fn character_before(text: &[u8], at: usize) -> (u32, usize) {
    let mut start = at - 1;
    while start > 0 && text[start] & 0xC0 == 0x80 {
        start -= 1;
    }
    (character(text, start).0, start)
}

struct Writer<'a> {
    buffer: &'a mut [u8],
    length: usize,
}

impl Writer<'_> {
    fn byte(&mut self, byte: u8) -> Outcome {
        *self.buffer.get_mut(self.length).ok_or(OutOfMemory)? = byte;
        self.length += 1;
        Ok(())
    }

    fn bytes(&mut self, bytes: &[u8]) -> Outcome {
        bytes.iter().try_for_each(|b| self.byte(*b))
    }

    fn character(&mut self, code: u32) -> Outcome {
        match code {
            0..=0x7F => self.byte(code as u8),
            0x80..=0x7FF => {
                self.byte(0xC0 | (code >> 6) as u8)?;
                self.byte(0x80 | (code & 0x3F) as u8)
            }
            0x800..=0xFFFF => {
                self.byte(0xE0 | (code >> 12) as u8)?;
                self.byte(0x80 | (code >> 6 & 0x3F) as u8)?;
                self.byte(0x80 | (code & 0x3F) as u8)
            }
            _ => {
                self.byte(0xF0 | (code >> 18) as u8)?;
                self.byte(0x80 | (code >> 12 & 0x3F) as u8)?;
                self.byte(0x80 | (code >> 6 & 0x3F) as u8)?;
                self.byte(0x80 | (code & 0x3F) as u8)
            }
        }
    }
}

/// The current list of pieces and the one the running step is writing.
struct Pieces<'a> {
    lists: [&'a mut [u32]; 2],
    live: usize,
    count: usize,
    written: usize,
}

impl Pieces<'_> {
    fn get(&self, index: usize) -> (u32, u32) {
        (self.lists[self.live][2 * index], self.lists[self.live][2 * index + 1])
    }

    fn begin(&mut self) {
        self.written = 0;
    }

    fn push(&mut self, first: u32, second: u32) -> Outcome {
        let target = &mut self.lists[1 - self.live];
        if 2 * self.written + 1 >= target.len() {
            return Err(OutOfMemory);
        }
        target[2 * self.written] = first;
        target[2 * self.written + 1] = second;
        self.written += 1;
        Ok(())
    }

    fn plain(&mut self, start: usize, length: usize, origin: bool) -> Outcome {
        if length == 0 {
            return Ok(());
        }
        self.push(start as u32, (length as u32) << 1 | origin as u32)
    }

    fn end(&mut self) {
        self.live = 1 - self.live;
        self.count = self.written;
    }
}

struct Work<'a> {
    bytes: &'a mut [u8],
    used: usize,
    pieces: Pieces<'a>,
    stack: &'a mut [u32],
}

impl<'a> Work<'a> {
    fn new(arena: &'a mut [u32]) -> Self {
        let quarter = arena.len() / 4;
        let (stack, rest) = arena.split_at_mut(quarter);
        let (first, rest) = rest.split_at_mut(quarter / 2);
        let (second, bytes) = rest.split_at_mut(quarter / 2);
        // SAFETY: every bit pattern is valid for both types, and u8 needs no alignment.
        let bytes = unsafe { core::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut u8, bytes.len() * 4) };
        Self { bytes, used: 0, pieces: Pieces { lists: [first, second], live: 0, count: 0, written: 0 }, stack }
    }
}

/// A stretch of the working text, and whether it begins where the text does.
#[derive(Clone, Copy)]
struct Piece {
    start: usize,
    length: usize,
    origin: bool,
}

/// How far the output of token decoding has got.
#[derive(Clone, Copy, Debug, Default)]
pub struct Decoding {
    leading_spaces: u32,
    first_token: bool,
}

#[derive(Clone, Copy)]
pub struct Tokenizer<'a> {
    t: &'a [u32],
}

impl<'a> Tokenizer<'a> {
    pub fn new(tables: &'a [u32]) -> Self {
        Self { t: tables }
    }

    fn field(&self, field: usize) -> usize {
        self.t[field] as usize
    }

    fn blob(&self, at: usize) -> u8 {
        (self.t[self.field(F_BLOB) + at / 4] >> (8 * (at % 4))) as u8
    }

    fn blob_bytes(&self, at: usize, length: usize) -> impl Iterator<Item = u8> + Clone + '_ {
        (at..at + length).map(move |i| self.blob(i))
    }

    fn in_class(&self, class: u32, code: u32) -> bool {
        let entry = self.field(F_CLASSES) + 2 * class as usize;
        let (table, count) = (self.t[entry] as usize, self.t[entry + 1] as usize);
        let (mut low, mut high) = (0, count);
        while low < high {
            let middle = (low + high) / 2;
            if code < self.t[table + 2 * middle] {
                high = middle;
            } else if code > self.t[table + 2 * middle + 1] {
                low = middle + 1;
            } else {
                return true;
            }
        }
        false
    }

    /// Backtracking in the order the alternatives are written, as Oniguruma does.
    fn run(&self, re: usize, start_pc: usize, start_at: usize, text: &[u8], stack: &mut [u32], base: usize) -> Outcome<Option<usize>> {
        let t = self.t;
        let (mut pc, mut at, mut sp) = (start_pc, start_at, base);
        loop {
            let moved = match t[re + pc] {
                R_MATCH => return Ok(Some(at)),
                R_CHAR | R_CLASS if at >= text.len() => false,
                R_CHAR => {
                    let (code, next) = character(text, at);
                    let same = code == t[re + pc + 1];
                    if same {
                        at = next;
                        pc += 2;
                    }
                    same
                }
                R_CLASS => {
                    let (code, next) = character(text, at);
                    let inside = self.in_class(t[re + pc + 1], code);
                    if inside {
                        at = next;
                        pc += 2;
                    }
                    inside
                }
                R_SPLIT => {
                    if sp + 2 > stack.len() {
                        return Err(OutOfMemory);
                    }
                    stack[sp] = t[re + pc + 2];
                    stack[sp + 1] = at as u32;
                    sp += 2;
                    pc = t[re + pc + 1] as usize;
                    true
                }
                R_JUMP => {
                    pc = t[re + pc + 1] as usize;
                    true
                }
                R_LOOK => {
                    let (negative, end) = (t[re + pc + 1] != 0, t[re + pc + 2] as usize);
                    let found = self.run(re, pc + 3, at, text, stack, sp)?.is_some();
                    let holds = found != negative;
                    if holds {
                        pc = end;
                    }
                    holds
                }
                assertion => {
                    let holds = match assertion {
                        R_LINE_START => at == 0 || text[at - 1] == b'\n',
                        R_LINE_END => at == text.len() || text[at] == b'\n',
                        R_TEXT_START => at == 0,
                        R_TEXT_END => at == text.len(),
                        _ => at == text.len() || (at + 1 == text.len() && text[at] == b'\n'),
                    };
                    if holds {
                        pc += 1;
                    }
                    holds
                }
            };
            if !moved {
                if sp == base {
                    return Ok(None);
                }
                sp -= 2;
                pc = stack[sp] as usize;
                at = stack[sp + 1] as usize;
            }
        }
    }

    /// Visits the text in order, as stretches that match the pattern and stretches that do not.
    fn each_segment(&self, re: usize, text: &[u8], stack: &mut [u32], visit: &mut impl FnMut(usize, usize, bool) -> Outcome) -> Outcome {
        if text.is_empty() {
            return Ok(());
        }
        let (mut last_end, mut last_match_end, mut previous) = (0, None, 0);
        'matches: while last_end <= text.len() {
            let mut start = last_end;
            let (found_start, found_end) = loop {
                if let Some(end) = self.run(re, 0, start, text, stack, 0)? {
                    break (start, end);
                }
                if start >= text.len() {
                    break 'matches;
                }
                start = character(text, start).1;
            };
            if found_start == found_end && Some(last_end) == last_match_end {
                if last_end >= text.len() {
                    break;
                }
                last_end = character(text, last_end).1;
                continue;
            }
            last_end = found_end;
            last_match_end = Some(found_end);
            if previous != found_start {
                visit(previous, found_start, false)?;
            }
            visit(found_start, found_end, true)?;
            previous = found_end;
        }
        if previous != text.len() {
            visit(previous, text.len(), false)?;
        }
        Ok(())
    }

    fn split(&self, re: usize, mode: u32, piece: Piece, work: &mut Work) -> Outcome {
        let Piece { start, length, origin } = piece;
        let Work { bytes, pieces, stack, .. } = work;
        let text = &bytes[start..start + length];
        let invert = mode & SPLIT_INVERT != 0;
        let behavior = mode & !SPLIT_INVERT;
        let mut emit = |from: usize, to: usize| pieces.plain(start + from, to - from, origin && from == 0);
        // One piece is held back, because the next stretch may join it.
        let mut pending: Option<(usize, usize, bool)> = None;
        let mut previous_match = false;
        self.each_segment(re, text, stack, &mut |from, to, matched| {
            let matched = matched != invert;
            match behavior {
                SPLIT_REMOVED => {
                    if !matched {
                        emit(from, to)?;
                    }
                }
                SPLIT_ISOLATED => emit(from, to)?,
                SPLIT_NEXT => match pending.take() {
                    Some((held, _, true)) if !matched => emit(held, to)?,
                    Some((held, end, _)) => {
                        emit(held, end)?;
                        pending = Some((from, to, matched));
                    }
                    None => pending = Some((from, to, matched)),
                },
                _ => {
                    let joins = if behavior == SPLIT_CONTIGUOUS { matched == previous_match } else { matched && !previous_match };
                    match (&mut pending, joins) {
                        (Some(held), true) => held.1 = to,
                        _ => {
                            if let Some((held, end, _)) = pending.replace((from, to, matched)) {
                                emit(held, end)?;
                            }
                        }
                    }
                }
            }
            previous_match = matched;
            Ok(())
        })?;
        if let Some((held, end, _)) = pending {
            emit(held, end)?;
        }
        Ok(())
    }

    fn table_lookup(&self, table: usize, count: usize, entry_words: usize, key: u32) -> Option<usize> {
        let (mut low, mut high) = (0, count);
        while low < high {
            let middle = (low + high) / 2;
            let at = table + entry_words * middle;
            match self.t[at].cmp(&key) {
                core::cmp::Ordering::Less => low = middle + 1,
                core::cmp::Ordering::Greater => high = middle,
                core::cmp::Ordering::Equal => return Some(at),
            }
        }
        None
    }

    fn combining_class(&self, form: usize, code: u32) -> u32 {
        if code < 0x300 {
            return 0;
        }
        let (table, count) = (self.t[form + 2] as usize, self.t[form + 3] as usize);
        let (mut low, mut high) = (0, count);
        while low < high {
            let middle = (low + high) / 2;
            let at = table + 3 * middle;
            if code < self.t[at] {
                high = middle;
            } else if code > self.t[at + 1] {
                low = middle + 1;
            } else {
                return self.t[at + 2];
            }
        }
        0
    }

    fn compose(&self, form: usize, first: u32, second: u32) -> Option<u32> {
        if (HANGUL_L..HANGUL_L + HANGUL_L_COUNT).contains(&first) && (HANGUL_V..HANGUL_V + HANGUL_V_COUNT).contains(&second) {
            return Some(HANGUL_S + ((first - HANGUL_L) * HANGUL_V_COUNT + second - HANGUL_V) * HANGUL_T_COUNT);
        }
        if (HANGUL_S..HANGUL_S + HANGUL_S_COUNT).contains(&first) && (first - HANGUL_S).is_multiple_of(HANGUL_T_COUNT) && (HANGUL_T + 1..HANGUL_T + HANGUL_T_COUNT).contains(&second) {
            return Some(first + second - HANGUL_T);
        }
        if second < 0x300 {
            return None;
        }
        let (table, count) = (self.t[form + 4] as usize, self.t[form + 5] as usize);
        let (mut low, mut high) = (0, count);
        while low < high {
            let middle = (low + high) / 2;
            let at = table + 3 * middle;
            match (self.t[at], self.t[at + 1]).cmp(&(first, second)) {
                core::cmp::Ordering::Less => low = middle + 1,
                core::cmp::Ordering::Greater => high = middle,
                core::cmp::Ordering::Equal => return Some(self.t[at + 2]),
            }
        }
        None
    }

    fn normal_form(&self, form: usize, text: &[u8], output: &mut Writer, scratch: &mut [u32]) -> Outcome {
        let t = self.t;
        let mut count = 0;
        let mut push = |code: u32| -> Outcome {
            *scratch.get_mut(count).ok_or(OutOfMemory)? = code;
            count += 1;
            Ok(())
        };
        let mut at = 0;
        while at < text.len() {
            let (code, next) = character(text, at);
            at = next;
            if (HANGUL_S..HANGUL_S + HANGUL_S_COUNT).contains(&code) {
                let index = code - HANGUL_S;
                push(HANGUL_L + index / (HANGUL_V_COUNT * HANGUL_T_COUNT))?;
                push(HANGUL_V + index % (HANGUL_V_COUNT * HANGUL_T_COUNT) / HANGUL_T_COUNT)?;
                if !index.is_multiple_of(HANGUL_T_COUNT) {
                    push(HANGUL_T + index % HANGUL_T_COUNT)?;
                }
            } else if let Some(entry) = (code >= 0xA0).then(|| self.table_lookup(t[form] as usize, t[form + 1] as usize, 3, code)).flatten() {
                let (from, length) = (t[entry + 1] as usize, t[entry + 2] as usize);
                for i in 0..length {
                    push(t[from + i])?;
                }
            } else {
                push(code)?;
            }
        }
        // Canonical ordering: marks sort by class, without crossing a starter.
        for i in 1..count {
            let (code, class) = (scratch[i], self.combining_class(form, scratch[i]));
            if class == 0 {
                continue;
            }
            let mut j = i;
            while j > 0 && self.combining_class(form, scratch[j - 1]) > class {
                scratch[j] = scratch[j - 1];
                j -= 1;
            }
            scratch[j] = code;
        }
        let mut kept = count;
        if t[form + 6] != 0 && count > 0 {
            let (mut write, mut starter, mut last_class) = (0, None, 0);
            for read in 0..count {
                let code = scratch[read];
                let class = self.combining_class(form, code);
                // Blocked when something of the same or a higher class, or a starter, lies between.
                let adjacent = starter.is_some_and(|place| write == place + 1);
                if let Some(place) = starter.filter(|_| adjacent || (last_class != 0 && last_class < class)) {
                    if let Some(composed) = self.compose(form, scratch[place], code) {
                        scratch[place] = composed;
                        continue;
                    }
                }
                if class == 0 {
                    starter = Some(write);
                }
                last_class = class;
                scratch[write] = code;
                write += 1;
            }
            kept = write;
        }
        scratch[..kept].iter().try_for_each(|code| output.character(*code))
    }

    fn transform(&self, operation: u32, [a, b, c]: [usize; 3], piece: Piece, work: &mut Work) -> Outcome {
        let Piece { start, length, origin } = piece;
        let Work { bytes, used, pieces, stack } = work;
        let (old, new) = bytes.split_at_mut(*used);
        let text = &old[start..start + length];
        let mut output = Writer { buffer: new, length: 0 };
        match operation {
            S_NORMAL_FORM => self.normal_form(a, text, &mut output, stack)?,
            S_LOWERCASE => {
                let mut at = 0;
                while at < text.len() {
                    let (code, next) = character(text, at);
                    at = next;
                    match self.table_lookup(a, b, 3, code) {
                        Some(entry) => {
                            output.character(self.t[entry + 1])?;
                            if self.t[entry + 2] != 0 {
                                output.character(self.t[entry + 2])?;
                            }
                        }
                        None => output.character(code)?,
                    }
                }
            }
            S_REMOVE_CLASS => {
                let mut at = 0;
                while at < text.len() {
                    let (code, next) = character(text, at);
                    if !self.in_class(a as u32, code) {
                        output.bytes(&text[at..next])?;
                    }
                    at = next;
                }
            }
            S_REPLACE => {
                let content = self.blob_bytes(b, c);
                self.each_segment(a, text, stack, &mut |from, to, matched| {
                    if matched {
                        content.clone().try_for_each(|byte| output.byte(byte))
                    } else {
                        output.bytes(&text[from..to])
                    }
                })?;
            }
            S_PREPEND => {
                self.blob_bytes(a, b).try_for_each(|byte| output.byte(byte))?;
                output.bytes(text)?;
            }
            S_BYTES => text.iter().try_for_each(|byte| output.character(byte_character(*byte)))?,
            _ => {
                // Metaspace puts its mark first, unless the text already starts with it.
                let mark = self.blob_bytes(a, b);
                let wanted = c as u32 == META_ALWAYS || (c as u32 == META_FIRST && origin);
                if wanted && !mark.clone().eq(text.iter().copied().take(b)) {
                    mark.clone().try_for_each(|byte| output.byte(byte))?;
                }
                output.bytes(text)?;
            }
        }
        let produced = output.length;
        pieces.plain(*used, produced, origin)?;
        *used += produced;
        Ok(())
    }

    fn strip(&self, left: bool, right: bool, class: u32, piece: Piece, work: &mut Work) -> Outcome {
        let Piece { start, length, origin } = piece;
        let text = &work.bytes[start..start + length];
        let (mut from, mut to) = (0, length);
        while left && from < to {
            let (code, next) = character(text, from);
            if !self.in_class(class, code) {
                break;
            }
            from = next;
        }
        while right && to > from {
            let (code, begin) = character_before(text, to);
            if !self.in_class(class, code) {
                break;
            }
            to = begin;
        }
        work.pieces.plain(start + from, to - from, origin && from == 0)
    }

    fn run_steps(&self, mut pc: usize, work: &mut Work) -> Outcome {
        loop {
            let operation = self.t[pc];
            if operation == S_END {
                return Ok(());
            }
            let (a, b, c) = (self.t[pc + 1] as usize, self.t[pc + 2] as usize, self.t[pc + 3] as usize);
            work.pieces.begin();
            for index in 0..work.pieces.count {
                let (first, second) = work.pieces.get(index);
                if second == ADDED_PIECE {
                    work.pieces.push(first, second)?;
                    continue;
                }
                let piece = Piece { start: first as usize, length: (second >> 1) as usize, origin: second & 1 != 0 };
                match operation {
                    S_SPLIT => self.split(a, b as u32, piece, work)?,
                    S_STRIP => self.strip(a != 0, b != 0, c as u32, piece, work)?,
                    S_PREPEND if piece.length == 0 => {}
                    _ => self.transform(operation, [a, b, c], piece, work)?,
                }
            }
            work.pieces.end();
            pc += STEP_WORDS;
        }
    }

    /// The added token that starts here, longest first: its table entry and length.
    fn added_at(&self, table: usize, count: usize, text: &[u8]) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize)> = None;
        for index in 0..count {
            let entry = table + 4 * index;
            if self.t[entry + 3] >> ADDED_FIRST_SHIFT != text[0] as u32 {
                continue;
            }
            let (offset, length) = (self.t[entry + 1] as usize, self.t[entry + 2] as usize);
            if length <= text.len() && best.is_none_or(|(_, longest)| length > longest) && self.blob_bytes(offset, length).eq(text[..length].iter().copied()) {
                best = Some((entry, length));
            }
        }
        best
    }

    fn split_added(&self, table: usize, count: usize, work: &mut Work) -> Outcome {
        let (word, space) = (self.t[F_WORD_CLASS], self.t[F_SPACE_CLASS]);
        work.pieces.begin();
        for index in 0..work.pieces.count {
            let (first, second) = work.pieces.get(index);
            if second == ADDED_PIECE {
                work.pieces.push(first, second)?;
                continue;
            }
            let (start, length, origin) = (first as usize, (second >> 1) as usize, second & 1 != 0);
            let Work { bytes, pieces, .. } = work;
            let text = &bytes[start..start + length];
            let (mut at, mut kept) = (0, 0);
            while at < length {
                let Some((entry, size)) = self.added_at(table, count, &text[at..]) else {
                    at += 1;
                    continue;
                };
                let flags = self.t[entry + 3];
                let (mut from, match_end) = (at, at + size);
                let mut to = match_end;
                at = match_end;
                if flags & ADDED_SINGLE_WORD != 0 {
                    let free_before = from == 0 || !self.in_class(word, character_before(text, from).0);
                    let free_after = to == length || !self.in_class(word, character(text, to).0);
                    if !free_before || !free_after {
                        continue;
                    }
                }
                if flags & ADDED_LSTRIP != 0 {
                    while from > kept {
                        let (code, begin) = character_before(text, from);
                        if !self.in_class(space, code) {
                            break;
                        }
                        from = begin;
                    }
                }
                if flags & ADDED_RSTRIP != 0 {
                    while to < length {
                        let (code, next) = character(text, to);
                        if !self.in_class(space, code) {
                            break;
                        }
                        to = next;
                    }
                }
                if kept < from {
                    pieces.plain(start + kept, from - kept, origin && kept == 0)?;
                }
                pieces.push(self.t[entry], ADDED_PIECE)?;
                kept = to;
            }
            if kept < length {
                pieces.plain(start + kept, length - kept, origin && kept == 0)?;
            }
        }
        work.pieces.end();
        Ok(())
    }

    fn vocabulary(&self, bytes: impl Iterator<Item = u8>) -> Option<u32> {
        let (first, second) = hash_string(self.t[F_SEED], bytes);
        let mask = self.field(F_VOCAB_SLOTS).checked_sub(1)?;
        let table = self.field(F_VOCAB);
        let mut slot = first as usize & mask;
        loop {
            let stored = self.t[table + 2 * slot + 1];
            if stored == 0 {
                return None;
            }
            if self.t[table + 2 * slot] == second {
                return Some(stored - 1);
            }
            slot = (slot + 1) & mask;
        }
    }

    fn merge(&self, first: u32, second: u32) -> (u32, u32) {
        let mask = self.t[F_MERGE_SLOTS] - 1;
        let table = self.t[F_MERGES];
        let mut slot = pair_slot(first, second);
        loop {
            slot &= mask;
            let at = (table + 4 * slot) as usize;
            let stored = self.t[at];
            if stored == 0 {
                return (NO_RANK, 0);
            }
            if stored == first + 1 && self.t[at + 1] == second {
                return (self.t[at + 2], self.t[at + 3]);
            }
            slot += 1;
        }
    }

    /// Always the lowest-ranked pair next, the leftmost if the same pair repeats.
    fn merge_symbols(&self, stack: &mut [u32], mut count: usize) -> usize {
        let total = count;
        let (symbols, rest) = stack.split_at_mut(total);
        let (ranks, results) = rest.split_at_mut(total);
        for i in 0..count.saturating_sub(1) {
            (ranks[i], results[i]) = self.merge(symbols[i], symbols[i + 1]);
        }
        while count > 1 {
            let mut best = 0;
            for i in 1..count - 1 {
                if ranks[i] < ranks[best] {
                    best = i;
                }
            }
            if ranks[best] == NO_RANK {
                break;
            }
            symbols[best] = results[best];
            symbols.copy_within(best + 2..count, best + 1);
            ranks.copy_within(best + 1..count - 1, best);
            results.copy_within(best + 1..count - 1, best);
            count -= 1;
            if best > 0 {
                (ranks[best - 1], results[best - 1]) = self.merge(symbols[best - 1], symbols[best]);
            }
            if best + 1 < count {
                (ranks[best], results[best]) = self.merge(symbols[best], symbols[best + 1]);
            }
        }
        count
    }

    fn encode_bpe(&self, text: &[u8], stack: &mut [u32], sink: &mut impl FnMut(u32)) -> Outcome {
        if self.t[F_IGNORE_MERGES] != 0 {
            if let Some(id) = self.vocabulary(text.iter().copied()) {
                sink(id);
                return Ok(());
            }
        }
        let characters = text.iter().filter(|b| **b & 0xC0 != 0x80).count();
        // Each character can become as many symbols as it has bytes with its affixes.
        let room = (text.len() + (self.field(F_PREFIX_LEN) + self.field(F_SUFFIX_LEN)) * characters) * 3;
        if room > stack.len() {
            return Err(OutOfMemory);
        }
        let (prefix, prefix_len, suffix, suffix_len) = (self.field(F_PREFIX), self.field(F_PREFIX_LEN), self.field(F_SUFFIX), self.field(F_SUFFIX_LEN));
        let fallback = self.field(F_BYTE_FALLBACK);
        let (unknown, fuse) = (self.t[F_UNK], self.t[F_FUSE_UNK] != 0);
        let mut count = 0;
        let mut held_unknown = false;
        let mut add = |stack: &mut [u32], id: u32| {
            stack[count] = id;
            count += 1;
        };
        let mut at = 0;
        while at < text.len() {
            let next = character(text, at).1;
            let piece = text[at..next].iter().copied();
            let head = self.blob_bytes(prefix, if at == 0 { 0 } else { prefix_len });
            let tail = self.blob_bytes(suffix, if next == text.len() { suffix_len } else { 0 });
            let whole = head.chain(piece).chain(tail);
            if let Some(id) = self.vocabulary(whole.clone()) {
                if held_unknown {
                    add(stack, unknown - 1);
                    held_unknown = false;
                }
                add(stack, id);
            } else if fallback != 0 && whole.clone().all(|byte| self.t[fallback + byte as usize] != 0) {
                whole.for_each(|byte| add(stack, self.t[fallback + byte as usize] - 1));
            } else if unknown != 0 {
                if held_unknown && !fuse {
                    add(stack, unknown - 1);
                }
                held_unknown = true;
            }
            at = next;
        }
        if held_unknown {
            add(stack, unknown - 1);
        }
        let kept = self.merge_symbols(stack, count);
        stack[..kept].iter().for_each(|id| sink(*id));
        Ok(())
    }

    fn post_tokens(&self, field: usize, sink: &mut impl FnMut(u32)) {
        if self.t[F_ADD_SPECIAL] != 0 {
            let table = self.field(field);
            for i in 0..self.t[table] as usize {
                sink(self.t[table + 1 + i]);
            }
        }
    }

    fn prepare<'w>(&self, text: &[u8], arena: &'w mut [u32], normalize_only: bool) -> Outcome<Work<'w>> {
        let work = Work::new(arena);
        if text.len() > work.bytes.len() {
            return Err(OutOfMemory);
        }
        self.fill(text, work, normalize_only)
    }

    fn fill<'w>(&self, text: &[u8], mut work: Work<'w>, normalize_only: bool) -> Outcome<Work<'w>> {
        work.bytes[..text.len()].copy_from_slice(text);
        work.used = text.len();
        work.pieces.begin();
        work.pieces.plain(0, text.len(), true)?;
        work.pieces.end();
        if !normalize_only {
            self.split_added(self.field(F_ADDED_RAW), self.field(F_ADDED_RAW_COUNT), &mut work)?;
        }
        self.run_steps(self.field(F_NORMALIZER), &mut work)?;
        if !normalize_only {
            self.split_added(self.field(F_ADDED_NORMALIZED), self.field(F_ADDED_NORMALIZED_COUNT), &mut work)?;
            self.run_steps(self.field(F_PRE_TOKENIZER), &mut work)?;
        }
        Ok(work)
    }

    /// Turns text into token numbers. Nothing reaches `sink` if the memory runs out.
    pub fn encode(&self, text: &[u8], arena: &mut [u32], sink: &mut impl FnMut(u32)) -> Outcome {
        let work = self.prepare(text, arena, false)?;
        self.post_tokens(F_POST_BEFORE, sink);
        for index in 0..work.pieces.count {
            let (first, second) = work.pieces.get(index);
            if second == ADDED_PIECE {
                sink(first);
                continue;
            }
            let (start, length) = (first as usize, (second >> 1) as usize);
            let piece = &work.bytes[start..start + length];
            if self.t[F_MODEL] == MODEL_WORD_LEVEL {
                sink(self.vocabulary(piece.iter().copied()).unwrap_or(self.t[F_UNK].wrapping_sub(1)));
            } else {
                self.encode_bpe(piece, work.stack, sink)?;
            }
        }
        self.post_tokens(F_POST_AFTER, sink);
        Ok(())
    }

    /// Runs only the normalizer, which is how normalized added tokens are prepared.
    pub fn normalize(&self, text: &[u8], arena: &mut [u32], output: &mut impl FnMut(u8)) -> Outcome {
        let work = self.prepare(text, arena, true)?;
        for index in 0..work.pieces.count {
            let (first, second) = work.pieces.get(index);
            let (start, length) = (first as usize, (second >> 1) as usize);
            work.bytes[start..start + length].iter().for_each(|b| output(*b));
        }
        Ok(())
    }

    pub fn decoding(&self) -> Decoding {
        Decoding { leading_spaces: self.t[F_LEADING_SPACES], first_token: true }
    }

    /// Writes one generated token's bytes. Special tokens write nothing.
    pub fn write_token(&self, token: u32, state: &mut Decoding, output: &mut impl FnMut(u8)) {
        if token >= self.t[F_TOKENS] {
            return;
        }
        let offsets = self.field(F_OFFSETS) + token as usize;
        let (from, to) = (self.t[offsets] as usize, self.t[offsets + 1] as usize);
        if from == to {
            return;
        }
        let drop_spaces = state.first_token && self.t[F_FIRST_TOKEN_SPACES] != 0;
        state.first_token = false;
        for byte in self.blob_bytes(from, to - from) {
            if byte == b' ' && (drop_spaces || state.leading_spaces > 0) {
                state.leading_spaces = state.leading_spaces.saturating_sub(1);
                continue;
            }
            state.leading_spaces = 0;
            output(byte);
        }
    }
}
