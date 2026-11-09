//! Firmware of the control CPU: loads the model, reads text, runs the die and writes text.
//! It talks to the outside world only through the host port.
#![no_std]
#![no_main]

use core::arch::global_asm;
use core::ptr::{read_volatile, write_volatile};

const HOST_RX_DATA: *const u32 = 0x2000_0000 as *const u32;
const HOST_RX_STATUS: *const u32 = 0x2000_0004 as *const u32;
const HOST_TX_DATA: *mut u32 = 0x2000_0008 as *mut u32;
const HOST_TX_STATUS: *const u32 = 0x2000_000C as *const u32;

const LOADER: *mut u32 = 0x3000_0000 as *mut u32;
const LOADER_START: usize = 6;
const LOADER_BUSY: usize = 7;
const LOADER_CRC: usize = 8;
const DATA_MEMORY: *const u32 = 0x1000_0000 as *const u32;
const DEFAULT_SEGMENT_TABLE: usize = 0x1000;
const PLAN_DIRECTORY: usize = 0x0FF0;
const SEGMENT_FIELDS: usize = 7;

const PERIPHERALS: *mut u32 = 0x4000_0000 as *mut u32;
const REG_ENTRY: usize = 32;
const REG_STATUS: usize = 33;
const REG_VECTOR_ADDRESS: usize = 34;
const REG_VECTOR_DATA: usize = 35;
const REG_SAMPLE_BASE: usize = 36;
const REG_SAMPLE_COUNT: usize = 37;
const REG_SAMPLE_INVERSE: usize = 38;
const REG_SAMPLE_LIMIT: usize = 39;
const REG_SAMPLE_GREEDY: usize = 40;
const REG_SAMPLE_SEED: usize = 41;
const REG_SAMPLE_START: usize = 42;
const REG_SAMPLE_DONE: usize = 43;
const REG_SAMPLE_TOKEN: usize = 44;
const REG_FOLLOWER: usize = 45;
const REG_CHAIN_DATA: usize = 46;
const REG_CHAIN_STATUS: usize = 47;

/// The description the image builder leaves in data memory.
const DEFAULT_INFO: usize = 0x0C00;
const INFO_ENTRY_TOKEN: usize = 0;
const INFO_LOGITS: usize = 2;
const INFO_LOGITS_EXPONENT: usize = 3;
const INFO_VOCABULARY: usize = 4;
const INFO_MAX_NEW: usize = 5;
const INFO_GREEDY: usize = 6;
const INFO_LIMIT: usize = 7;
const INFO_TEMPERATURE: usize = 8;
const INFO_SEED: usize = 9;
const INFO_ACTIVATION_BITS: usize = 10;
const INFO_PARAMETER_COUNT: usize = 11;
const INFO_RAW_TOKENS: usize = 12;
const INFO_STOP_FIRST: usize = 13;
const INFO_STOP_SECOND: usize = 14;
const INFO_PARAMETERS: usize = 15;

/// The tokenizer's tables, which the image builder leaves after the segment table.
const TOKENIZER: *const u32 = 0x1000_8000 as *const u32;
const TOK_MERGE_SLOTS: usize = 0;
const TOK_MERGES: usize = 1;
const TOK_LETTER_COUNT: usize = 2;
const TOK_LETTERS: usize = 3;
const TOK_NUMBER_COUNT: usize = 4;
const TOK_NUMBERS: usize = 5;
const TOK_BYTE_SYMBOLS: usize = 6;
const TOK_OFFSETS: usize = 7;
const TOK_BLOB: usize = 8;
const TOK_ADDED_COUNT: usize = 9;
const TOK_ADDED: usize = 10;
const TOK_TEXT: usize = 11;
const TOK_TEXT_CAPACITY: usize = 12;
const MAX_SYMBOLS: usize = 512;

const END_OF_TOKENS: u32 = u32::MAX;
const ROUTINE_TOKEN: u32 = 0;
const ROUTINE_LOGITS: u32 = 1;

global_asm!(
    r#"
    .section .text.entry
    .global _start
_start:
    la sp, __stack_top
    la a0, __bss_start
    la a1, __bss_end
1:  bgeu a0, a1, 2f
    sw zero, 0(a0)
    addi a0, a0, 4
    j 1b
2:  call main
    ebreak
"#
);

/// Words of data memory of this chip's description and segment table, and chip count.
static mut INFO_WORD: usize = DEFAULT_INFO;
static mut TABLE_WORD: usize = DEFAULT_SEGMENT_TABLE;
static mut CHIPS: u32 = 1;

fn data_word(word: usize) -> u32 {
    unsafe { read_volatile(DATA_MEMORY.add(word)) }
}

fn info(field: usize) -> u32 {
    data_word(unsafe { INFO_WORD } + field)
}

fn chips() -> u32 {
    unsafe { CHIPS }
}

fn peripheral_write(register: usize, value: u32) {
    unsafe { write_volatile(PERIPHERALS.add(register), value) }
}

fn peripheral_read(register: usize) -> u32 {
    unsafe { read_volatile(PERIPHERALS.add(register)) }
}

fn receive() -> u8 {
    unsafe {
        while read_volatile(HOST_RX_STATUS) == 0 {}
        read_volatile(HOST_RX_DATA) as u8
    }
}

fn send(byte: u8) {
    unsafe {
        while read_volatile(HOST_TX_STATUS) == 0 {}
        write_volatile(HOST_TX_DATA, byte as u32);
    }
}

/// Loads every segment of the image; each CRC mismatch is reported to the host.
fn load_segments() {
    unsafe {
        let table = TABLE_WORD;
        let count = data_word(table) as usize;
        for index in 0..count {
            let field = |i: usize| data_word(table + 1 + index * SEGMENT_FIELDS + i);
            for register in 0..6 {
                write_volatile(LOADER.add(register), field(register));
            }
            write_volatile(LOADER.add(LOADER_START), 1);
            while read_volatile(LOADER.add(LOADER_BUSY)) != 0 {}
            let good = read_volatile(LOADER.add(LOADER_CRC)) == field(6);
            send(if good { b'.' } else { b'E' });
        }
    }
}

const COUNT_WORD: u32 = 0xC000_0000;
const TOTAL_WORD: u32 = 0xD000_0000;
const WORD_VALUE: u32 = 0x0FFF_FFFF;
/// How long a chip waits for a neighbour's ready signal before concluding it is alone.
const LONELY_POLLS: u32 = 32;

fn chain_connected() -> bool {
    (0..LONELY_POLLS).any(|_| peripheral_read(REG_CHAIN_STATUS) & 2 != 0)
}

/// Finds out how many chips there are and which one this is, around the chain.
fn count_chips() -> (u32, u32) {
    if peripheral_read(REG_FOLLOWER) != 0 {
        let id = chain_receive() & WORD_VALUE;
        chain_send(COUNT_WORD | (id + 1));
        let total = chain_receive();
        chain_send(total);
        return (id, total & WORD_VALUE);
    }
    if !chain_connected() {
        return (0, 1);
    }
    chain_send(COUNT_WORD | 1);
    let total = chain_receive() & WORD_VALUE;
    chain_send(TOTAL_WORD | total);
    chain_receive();
    (0, total)
}

/// Takes this chip's part of the image, if the image has a plan for this many chips.
fn select_part(id: u32, total: u32) -> bool {
    let plans = data_word(PLAN_DIRECTORY) as usize;
    if plans == 0 {
        return total == 1;
    }
    let lists = data_word(PLAN_DIRECTORY + 1) as usize;
    for plan in 0..plans {
        if data_word(lists + 2 * plan) == total {
            let parts = data_word(lists + 2 * plan + 1) as usize;
            unsafe {
                INFO_WORD = data_word(parts + 2 * id as usize) as usize;
                TABLE_WORD = data_word(parts + 2 * id as usize + 1) as usize;
                CHIPS = total;
            }
            return true;
        }
    }
    false
}

fn chain_send(word: u32) {
    while peripheral_read(REG_CHAIN_STATUS) & 2 == 0 {}
    peripheral_write(REG_CHAIN_DATA, word);
}

fn chain_receive() -> u32 {
    while peripheral_read(REG_CHAIN_STATUS) & 1 == 0 {}
    peripheral_read(REG_CHAIN_DATA)
}

/// Starts one routine of this chip's die, without waiting for it.
fn start_die(routine: u32, token: u32, position: u32) {
    for p in 0..info(INFO_PARAMETER_COUNT) as usize {
        let at = INFO_PARAMETERS + 4 * p;
        let value = info(at + 1).wrapping_mul(token).wrapping_add(info(at + 2).wrapping_mul(position)).wrapping_add(info(at + 3));
        peripheral_write(info(at) as usize, value);
    }
    peripheral_write(REG_ENTRY, info(INFO_ENTRY_TOKEN + routine as usize));
}

fn wait_for_die() {
    while peripheral_read(REG_STATUS) == 0 {}
}

/// Orders a routine on every chip: the command goes down the chain before this die starts.
fn run_die(routine: u32, token: u32, position: u32) {
    if chips() > 1 {
        chain_send(routine);
        chain_send(token);
        chain_send(position);
    }
    start_die(routine, token, position);
    wait_for_die();
}

/// What every chip but the first does: obey the chain and pass the commands on.
fn serve_chain(id: u32) -> ! {
    let forward = id + 1 < chips();
    loop {
        let (routine, token, position) = (chain_receive(), chain_receive(), chain_receive());
        if forward {
            chain_send(routine);
            chain_send(token);
            chain_send(position);
        }
        start_die(routine, token, position);
        wait_for_die();
    }
}

/// Inverse temperature for the sampling unit: mantissa steps per table step.
fn sampling_inverse() -> u32 {
    peripheral_write(REG_VECTOR_ADDRESS, info(INFO_LOGITS_EXPONENT));
    let raw = peripheral_read(REG_VECTOR_DATA) as i32;
    let bits = info(INFO_ACTIVATION_BITS);
    let unused = 32 - bits;
    let exponent = (raw << unused) >> unused;
    let doubled = 17 + exponent - bits as i32;
    let scaled = if doubled >= 0 { (256u32 << doubled.min(16)) / info(INFO_TEMPERATURE).max(1) } else { (256u32 >> (-doubled).min(16)) / info(INFO_TEMPERATURE).max(1) };
    scaled.min(0xFFFF)
}

fn sample() -> u32 {
    peripheral_write(REG_SAMPLE_BASE, info(INFO_LOGITS));
    peripheral_write(REG_SAMPLE_COUNT, info(INFO_VOCABULARY));
    peripheral_write(REG_SAMPLE_INVERSE, sampling_inverse());
    peripheral_write(REG_SAMPLE_LIMIT, info(INFO_LIMIT));
    peripheral_write(REG_SAMPLE_GREEDY, info(INFO_GREEDY));
    peripheral_write(REG_SAMPLE_START, 1);
    while peripheral_read(REG_SAMPLE_DONE) == 0 {}
    peripheral_read(REG_SAMPLE_TOKEN)
}

fn tok(field: usize) -> u32 {
    unsafe { read_volatile(TOKENIZER.add(field)) }
}

fn tok_word(offset: u32) -> u32 {
    unsafe { read_volatile(TOKENIZER.add(offset as usize)) }
}

fn blob_byte(at: u32) -> u8 {
    unsafe { read_volatile((TOKENIZER.add(tok(TOK_BLOB) as usize) as *const u8).add(at as usize)) }
}

/// Whether `code` falls in one of a table's sorted, inclusive ranges.
fn in_ranges(code: u32, table: u32, count: u32) -> bool {
    let (mut low, mut high) = (0, count);
    while low < high {
        let middle = (low + high) / 2;
        if code < tok_word(table + 2 * middle) {
            high = middle;
        } else if code > tok_word(table + 2 * middle + 1) {
            low = middle + 1;
        } else {
            return true;
        }
    }
    false
}

fn is_letter(code: u32) -> bool {
    in_ranges(code, tok(TOK_LETTERS), tok(TOK_LETTER_COUNT))
}

fn is_number(code: u32) -> bool {
    in_ranges(code, tok(TOK_NUMBERS), tok(TOK_NUMBER_COUNT))
}

fn is_space(code: u32) -> bool {
    matches!(code, 9..=13 | 0x20 | 0x85 | 0xA0 | 0x1680 | 0x2000..=0x200A | 0x2028 | 0x2029 | 0x202F | 0x205F | 0x3000)
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

/// Where the piece that starts at `start` ends, following the model's splitting pattern.
fn piece_end(text: &[u8], start: usize) -> usize {
    let len = text.len();
    let (first, second) = character(text, start);
    if first == b'\'' as u32 {
        for suffix in [&b"s"[..], b"t", b"re", b"ve", b"m", b"ll", b"d"] {
            let end = second + suffix.len();
            if end <= len && text[second..end].eq_ignore_ascii_case(suffix) {
                return end;
            }
        }
    }
    let first_is_letter = is_letter(first);
    let first_is_number = is_number(first);
    let letters_from = if first_is_letter {
        Some(start)
    } else if !first_is_number && first != 13 && first != 10 && second < len && is_letter(character(text, second).0) {
        Some(second)
    } else {
        None
    };
    if let Some(mut at) = letters_from {
        while at < len {
            let (code, next) = character(text, at);
            if !is_letter(code) {
                break;
            }
            at = next;
        }
        return at;
    }
    if first_is_number {
        return second;
    }

    let symbols_from = if first == 32 { second } else { start };
    let mut at = symbols_from;
    while at < len {
        let (code, next) = character(text, at);
        if is_space(code) || is_letter(code) || is_number(code) {
            break;
        }
        at = next;
    }
    if at > symbols_from {
        while at < len && (text[at] == 13 || text[at] == 10) {
            at += 1;
        }
        return at;
    }

    // A run of white space: through its last line break, or up to the following text.
    let (mut end, mut break_end, mut previous) = (start, None, start);
    while end < len {
        let (code, next) = character(text, end);
        if !is_space(code) {
            break;
        }
        previous = end;
        end = next;
        if code == 13 || code == 10 {
            break_end = Some(end);
        }
    }
    if let Some(found) = break_end {
        return found;
    }
    if end < len && previous > start {
        return previous;
    }
    end.max(second)
}

/// The rank and result of merging two symbols, if the vocabulary has that pair.
fn merge(first: u32, second: u32) -> Option<(u32, u32)> {
    let mask = tok(TOK_MERGE_SLOTS) - 1;
    let mut slot = first.wrapping_mul(0x9E37_79B1) ^ second.wrapping_mul(0x85EB_CA77);
    let table = tok(TOK_MERGES);
    loop {
        slot &= mask;
        let at = table + 4 * slot;
        let stored = tok_word(at);
        if stored == 0 {
            return None;
        }
        if stored == first + 1 && tok_word(at + 1) == second {
            return Some((tok_word(at + 2), tok_word(at + 3)));
        }
        slot += 1;
    }
}

/// Byte-pair encoding of one piece; each token goes to `sink`.
fn encode_piece(piece: &[u8], sink: &mut impl FnMut(u32)) {
    let table = tok(TOK_BYTE_SYMBOLS);
    for chunk in piece.chunks(MAX_SYMBOLS) {
        let mut symbols = [0u32; MAX_SYMBOLS];
        for (slot, byte) in symbols.iter_mut().zip(chunk) {
            *slot = tok_word(table + *byte as u32);
        }
        let mut count = chunk.len();
        while count > 1 {
            let mut best: Option<(u32, u32, u32, u32)> = None;
            for pair in symbols[..count].windows(2) {
                if let Some((rank, id)) = merge(pair[0], pair[1]) {
                    if best.is_none_or(|(lowest, ..)| rank < lowest) {
                        best = Some((rank, pair[0], pair[1], id));
                    }
                }
            }
            let Some((_, first, second, id)) = best else { break };
            let (mut read, mut write) = (0, 0);
            while read < count {
                if read + 1 < count && symbols[read] == first && symbols[read + 1] == second {
                    symbols[write] = id;
                    read += 2;
                } else {
                    symbols[write] = symbols[read];
                    read += 1;
                }
                write += 1;
            }
            count = write;
        }
        for symbol in &symbols[..count] {
            sink(*symbol);
        }
    }
}

/// The special token at the start of `text`, longest first: its number and length.
fn added_token(text: &[u8]) -> Option<(u32, usize)> {
    let table = tok(TOK_ADDED);
    let mut best: Option<(u32, usize)> = None;
    for index in 0..tok(TOK_ADDED_COUNT) {
        let at = table + 3 * index;
        let (id, offset, length) = (tok_word(at), tok_word(at + 1), tok_word(at + 2) as usize);
        if length <= text.len() && best.is_none_or(|(_, longest)| length > longest) && (0..length).all(|i| blob_byte(offset + i as u32) == text[i]) {
            best = Some((id, length));
        }
    }
    best
}

fn encode_text(text: &[u8], sink: &mut impl FnMut(u32)) {
    let mut at = 0;
    while at < text.len() {
        let mut stop = at;
        while stop < text.len() && added_token(&text[stop..]).is_none() {
            stop += 1;
        }
        let mut piece = at;
        while piece < stop {
            let end = piece_end(&text[..stop], piece);
            encode_piece(&text[piece..end], sink);
            piece = end;
        }
        if let Some((id, length)) = (stop < text.len()).then(|| added_token(&text[stop..])).flatten() {
            sink(id);
            at = stop + length;
        } else {
            at = stop;
        }
    }
}

/// Reads the whole prompt from the host and hands over its tokens.
fn read_prompt(sink: &mut impl FnMut(u32)) {
    if info(INFO_RAW_TOKENS) != 0 {
        loop {
            let mut token = 0u32;
            for shift in 0..4 {
                token |= (receive() as u32) << (8 * shift);
            }
            if token == END_OF_TOKENS {
                return;
            }
            sink(token);
        }
    }
    let buffer = unsafe { core::slice::from_raw_parts_mut(TOKENIZER.add(tok(TOK_TEXT) as usize) as *mut u8, tok(TOK_TEXT_CAPACITY) as usize) };
    let mut length = 0;
    loop {
        let byte = receive();
        if byte == 0 {
            break;
        }
        if length < buffer.len() {
            buffer[length] = byte;
            length += 1;
        }
    }
    encode_text(&buffer[..length], sink);
}

fn emit(token: u32) {
    if info(INFO_RAW_TOKENS) != 0 {
        for shift in 0..4 {
            send((token >> (8 * shift)) as u8);
        }
        return;
    }
    let offsets = tok(TOK_OFFSETS);
    for at in tok_word(offsets + token)..tok_word(offsets + token + 1) {
        send(blob_byte(at));
    }
}

#[unsafe(no_mangle)]
extern "C" fn main() {
    let (id, total) = count_chips();
    if !select_part(id, total) {
        if id == 0 {
            send(b'N');
        }
        return;
    }
    load_segments();
    if id != 0 {
        serve_chain(id);
    }
    peripheral_write(REG_SAMPLE_SEED, info(INFO_SEED));
    let (entry_token, entry_logits) = (ROUTINE_TOKEN, ROUTINE_LOGITS);
    let (mut last, mut position) = (0, 0);
    read_prompt(&mut |token| {
        run_die(entry_token, token, position);
        last = token;
        position += 1;
    });
    if position == 0 {
        return;
    }
    let (first_stop, second_stop) = (info(INFO_STOP_FIRST), info(INFO_STOP_SECOND));
    for _ in 0..info(INFO_MAX_NEW) {
        run_die(entry_logits, last, position - 1);
        last = sample();
        if last == first_stop || last == second_stop {
            return;
        }
        emit(last);
        run_die(entry_token, last, position);
        position += 1;
    }
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {
        unsafe { core::arch::asm!("ebreak") };
    }
}
