//! Firmware of the control CPU: loads the model, reads text, runs the die and writes text.
//! It talks to the outside world only through the host port.
#![no_std]
#![no_main]

use core::arch::global_asm;
use core::ptr::{addr_of_mut, read_volatile, write_volatile};
use tokenizer_rt::{Decoding, LoopWatch, Tokenizer, F_ARENA_WORDS, F_TEXT, F_TEXT_BYTES};

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
const REG_QUEUE_ROOM: usize = 49;

/// The description the image builder leaves in data memory.
const DEFAULT_INFO: usize = 0x0C00;
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
const INFO_STOP_COUNT: usize = 13;
const INFO_STOP_TABLE: usize = 14;
const INFO_DRAFT_DEPTH: usize = 15;
const INFO_DRAFT_LOGITS: usize = 16;
const INFO_ENTRIES: usize = 17;
const INFO_THINK_OPEN: usize = 18;
const INFO_THINK_CLOSE: usize = 19;
const INFO_THINK_BUDGET: usize = 20;
const INFO_TABLE_SOURCE: usize = 21;
const INFO_TABLE_ADDRESS: usize = 22;
const INFO_TABLE_COLS: usize = 23;
const INFO_VISION: usize = 24;
const INFO_SHOWO: usize = 25;
const INFO_ROTATION_SOURCE: usize = 26;
const INFO_ROTATION_ADDRESS: usize = 27;
const INFO_ROTATION_COLS: usize = 28;
const INFO_SCHEDULE: usize = 29;
const INFO_QUEUE: usize = 30;
const INFO_SYSTEM: usize = 31;
/// Where the conversation table is in the description: zero answers one question and stops.
const INFO_CONVERSATION: usize = 32;
const INFO_PARAMETERS: usize = 33;
/// The load controller writes zeros instead of reading the image while this is set.
const LOADER_FILL: usize = 9;
/// The order along the chain to clear the conversation, with the positions it used.
const CLEAR_ROUTINE: u32 = u32::MAX;
/// The host port's status bit that says the clear pin was raised.
const CLEAR_REQUESTED: u32 = 2;
const TARGET_VECTOR: u32 = 3;

/// The tokenizer's tables, which the image builder leaves after the segment table.
const TOKENIZER: *const u32 = 0x1000_8000 as *const u32;

const END_OF_TOKENS: u32 = u32::MAX;
const ROUTINE_TOKEN: u32 = 0;
const ROUTINE_LOGITS: u32 = 1;
const ROUTINE_FEED_FROM_MAIN: u32 = 2;
const ROUTINE_FEED_FROM_DRAFT: u32 = 3;
const ROUTINE_DRAFT: u32 = 4;
const ROUTINE_DRAFT_LOGITS: u32 = 5;
const REG_DRAFT_RESULT: usize = 48;
const MAX_DRAFT_DEPTH: usize = 3;
const ROUTINE_TOKEN_FROM_IMAGE: u32 = 6;
const ROUTINE_VISION: u32 = 7;

/// The words of the image encoder's description, from `INFO_VISION`.
const V_PATCHES: usize = 0;
const V_IMAGE_TOKEN: usize = 1;
const V_MERGE: usize = 2;
const V_BLOCKS: usize = 3;
const V_PIXELS: usize = 4;
const V_POSITION: usize = 7;
const V_PATCH_ROPE: usize = 10;
const V_PATCH_ROPE_WIDTH: usize = 11;
const V_ROPE: usize = 12;
const V_ROPE_HALF: usize = 13;
const V_MROPE_ROWS: usize = 14;
const V_MROPE_COLS: usize = 15;

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
/// The clear pin was raised: what is being read or written is dropped.
static mut ABORT: bool = false;

fn aborted() -> bool {
    unsafe { ABORT }
}

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

/// The next byte from the host; 0 once the clear pin has been raised, and from then on.
#[inline(always)]
fn receive() -> u8 {
    unsafe {
        loop {
            // A byte waiting and no clear, the usual case, costs what it did before the pin.
            let status = read_volatile(HOST_RX_STATUS);
            if status == 1 {
                return read_volatile(HOST_RX_DATA) as u8;
            }
            if status & CLEAR_REQUESTED != 0 {
                ABORT = true;
                return 0;
            }
        }
    }
}

/// Has the load controller move `count` elements, or write zeros when `fill`, and waits.
fn transfer(fill: bool, [source, target, unit, address, count, beats]: [u32; 6]) {
    if count == 0 {
        return;
    }
    unsafe {
        write_volatile(LOADER.add(LOADER_FILL), fill as u32);
        for (register, value) in [source, target, unit, address, count, beats].iter().enumerate() {
            write_volatile(LOADER.add(register), *value);
        }
        write_volatile(LOADER.add(LOADER_START), 1);
        while read_volatile(LOADER.add(LOADER_BUSY)) != 0 {}
    }
}

/// Empties what a conversation left in the die: the vector memory's first contents again,
/// as the boot left them, and zeros over the cache and the recurrent state, as the
/// description's conversation table says. Its entries are a first address, how many
/// registers from it and the step between them, how far each reaches, and the elements of a
/// position's row when the cache's rows follow one another: then only the first `rows`.
fn clear_state(rows: u32) {
    let at = info(INFO_CONVERSATION) as usize;
    if at == 0 {
        return;
    }
    let table = unsafe { TABLE_WORD };
    for index in 0..data_word(table) as usize {
        let field = |i: usize| data_word(table + 1 + index * SEGMENT_FIELDS + i);
        if field(1) == TARGET_VECTOR {
            transfer(false, [field(0), field(1), field(2), field(3), field(4), field(5)]);
        }
    }
    for entry in 0..info(at + 1) as usize {
        let field = |i: usize| info(at + 2 + 5 * entry + i);
        let count = if field(4) != 0 { rows.wrapping_mul(field(4)).min(field(3)) } else { field(3) };
        for register in 0..field(1) {
            transfer(true, [0, TARGET_VECTOR, 0, field(0).wrapping_add(register.wrapping_mul(field(2))), count, 1]);
        }
    }
}

/// Clears what the conversation left, on this chip and down the chain, and forgets the
/// clear pin.
fn clear_conversation(rows: u32) {
    if chips() > 1 {
        chain_send(CLEAR_ROUTINE);
        chain_send(rows);
        chain_send(0);
    }
    clear_state(rows);
    unsafe {
        write_volatile(HOST_RX_STATUS as *mut u32, 0);
        ABORT = false;
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
    let table = unsafe { TABLE_WORD };
    for index in 0..data_word(table) as usize {
        let field = |i: usize| data_word(table + 1 + index * SEGMENT_FIELDS + i);
        transfer(false, [field(0), field(1), field(2), field(3), field(4), field(5)]);
        let good = unsafe { read_volatile(LOADER.add(LOADER_CRC)) } == field(6);
        send(if good { b'.' } else { b'E' });
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

/// Brings row `row` of a table that stays in the image, if there is one, into the register
/// the die reads it from: the table whose place, register and row length are in the
/// description fields `source`, `address` and `count`.
fn load_row(row: u32, [source, address, count]: [usize; 3]) {
    let cols = info(count);
    if cols == 0 {
        return;
    }
    transfer(false, [info(source).wrapping_add(row.wrapping_mul(cols)), TARGET_VECTOR, 0, info(address), cols, 1]);
}

/// `a · b`, without the multiplier's 33 cycles when `a` is 0 or 1, as most factors are.
fn times(a: u32, b: u32) -> u32 {
    match a {
        0 => 0,
        1 => b,
        _ => a.wrapping_mul(b),
    }
}

/// Starts one routine of this chip's die, without waiting for it.
fn start_die(routine: u32, token: u32, position: u32) {
    // Only these routines read a token's row of the table: with Show-o, the image's
    // embedding too, which reads the same table.
    if routine == ROUTINE_TOKEN || routine == ROUTINE_DRAFT || (routine == ROUTINE_VISION && info(INFO_SHOWO) != 0) {
        load_row(token, [INFO_TABLE_SOURCE, INFO_TABLE_ADDRESS, INFO_TABLE_COLS]);
    }
    // The routines that rotate by the token's position: a die that leaves its rotation
    // table in the image holds only that position's row.
    if routine == ROUTINE_TOKEN || routine == ROUTINE_DRAFT || routine == ROUTINE_TOKEN_FROM_IMAGE {
        load_row(position, [INFO_ROTATION_SOURCE, INFO_ROTATION_ADDRESS, INFO_ROTATION_COLS]);
    }
    for p in 0..info(INFO_PARAMETER_COUNT) as usize {
        let at = INFO_PARAMETERS + 7 * p;
        let window = info(at + 6);
        let windowed = if window > 0 { info(at + 4).wrapping_mul((position + 1).min(window)).wrapping_add(info(at + 5).wrapping_mul(position % window)) } else { 0 };
        let value = times(info(at + 1), token).wrapping_add(times(info(at + 2), position)).wrapping_add(info(at + 3)).wrapping_add(windowed);
        peripheral_write(info(at) as usize, value);
    }
    peripheral_write(REG_ENTRY, info(info(INFO_ENTRIES) as usize + routine as usize));
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
        if routine == CLEAR_ROUTINE {
            clear_state(token);
            continue;
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

/// The token the draft head finds most likely.
fn sample_draft() -> u32 {
    peripheral_write(REG_SAMPLE_BASE, info(INFO_DRAFT_LOGITS));
    peripheral_write(REG_SAMPLE_COUNT, info(INFO_VOCABULARY));
    peripheral_write(REG_SAMPLE_INVERSE, 0xFFFF);
    peripheral_write(REG_SAMPLE_LIMIT, 1);
    peripheral_write(REG_SAMPLE_GREEDY, 1);
    peripheral_write(REG_SAMPLE_START, 1);
    while peripheral_read(REG_SAMPLE_DONE) == 0 {}
    peripheral_read(REG_SAMPLE_TOKEN)
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

fn tokenizer() -> Tokenizer<'static> {
    Tokenizer::new(unsafe { core::slice::from_raw_parts(TOKENIZER, tok(F_TEXT) as usize) })
}

/// Tokenizes the text in its buffer. Nothing is produced if it does not fit.
fn encode_text(length: usize, sink: &mut impl FnMut(u32)) {
    let text_at = tok(F_TEXT) as usize;
    let text = unsafe { core::slice::from_raw_parts(TOKENIZER.add(text_at) as *const u8, length) };
    let arena_at = text_at + (tok(F_TEXT_BYTES) as usize).div_ceil(4);
    let arena = unsafe { core::slice::from_raw_parts_mut(TOKENIZER.add(arena_at) as *mut u32, tok(F_ARENA_WORDS) as usize) };
    let _ = tokenizer().encode(text, arena, sink);
}

/// Hands over the system prompt's tokens, which the chip carries, at the start of a
/// conversation, then reads the whole prompt from the host and hands over its tokens.
fn read_prompt(first: bool, sink: &mut impl FnMut(u32)) {
    let system = info(INFO_SYSTEM) as usize;
    if first && system != 0 {
        for at in system + 1..=system + info(system) as usize {
            sink(info(at));
        }
    }
    if info(INFO_RAW_TOKENS) != 0 {
        loop {
            let mut token = 0u32;
            for shift in 0..4 {
                token |= (receive() as u32) << (8 * shift);
            }
            if token == END_OF_TOKENS || aborted() {
                return;
            }
            sink(token);
        }
    }
    let buffer = unsafe { core::slice::from_raw_parts_mut(TOKENIZER.add(tok(F_TEXT) as usize) as *mut u8, tok(F_TEXT_BYTES) as usize) };
    let (mut length, mut fits) = (0, true);
    loop {
        let byte = receive();
        if byte == 0 {
            break;
        }
        match buffer.get_mut(length) {
            Some(slot) => {
                *slot = byte;
                length += 1;
            }
            None => fits = false,
        }
    }
    if fits && !aborted() {
        encode_text(length, sink);
    }
}

fn receive_word() -> u32 {
    let mut word = 0u32;
    for shift in 0..4 {
        word |= (receive() as u32) << (8 * shift);
    }
    word
}

fn vision(field: usize) -> u32 {
    info(info(INFO_VISION) as usize + field)
}

/// Writes one element, or an exponent, of the die's vector memory while the die is idle.
fn vector_write(address: u32, value: u32) {
    peripheral_write(REG_VECTOR_ADDRESS, address);
    peripheral_write(REG_VECTOR_DATA, value);
}

/// A vector from the host: its exponent, then its elements.
fn receive_vector(field: usize) {
    let (address, exponent, size) = (vision(field), vision(field + 1), vision(field + 2));
    vector_write(exponent, receive_word());
    for i in 0..size {
        vector_write(address + i, receive_word());
    }
}

/// Copies `count` elements of vector memory as the image left them, from `from` to `to`,
/// with the load controller: the image is never written, so it still holds every row of
/// the rotation table however the die's copy has been rewritten.
fn copy_from_image(from: u32, to: u32, count: u32) {
    unsafe {
        let table = TABLE_WORD;
        let segments = data_word(table) as usize;
        let mut done = 0;
        while done < count {
            let at = from + done;
            let mut found = None;
            for index in 0..segments {
                let field = |i: usize| data_word(table + 1 + index * SEGMENT_FIELDS + i);
                let (address, length) = (field(3), field(4));
                if field(1) == TARGET_VECTOR && field(5) == 1 && at >= address && at < address + length {
                    found = Some((field(0) + (at - address), (address + length - at).min(count - done)));
                    break;
                }
            }
            match found {
                Some((source, length)) => {
                    let fields = [source, TARGET_VECTOR, 0, to + done, length, 1];
                    for (register, value) in fields.iter().enumerate() {
                        write_volatile(LOADER.add(register), *value);
                    }
                    write_volatile(LOADER.add(LOADER_START), 1);
                    while read_volatile(LOADER.add(LOADER_BUSY)) != 0 {}
                    done += length;
                }
                // An element the image leaves out starts at zero.
                None => {
                    vector_write(to + done, 0);
                    done += 1;
                }
            }
        }
    }
}

/// An image the encoder has turned into tokens, waiting for its place in the prompt.
#[derive(Clone, Copy, Default)]
struct Picture {
    tokens: u32,
    /// Tokens per row of the image's grid of tokens, and rows.
    width: u32,
    height: u32,
}

/// Reads an image from the host, if it sends one, and runs the encoder over it: each patch's
/// pixels, learned position and rotation row, then each block twice over every patch.
fn read_picture() -> Option<Picture> {
    let count = receive_word();
    if count == 0 {
        return None;
    }
    let (rows, cols) = (receive_word(), receive_word());
    let width = vision(V_PATCH_ROPE_WIDTH);
    if count > vision(V_PATCHES) {
        // More than the encoder's stores hold: the image is read and left aside.
        for _ in 0..count * (vision(V_PIXELS + 2) + vision(V_POSITION + 2) + 2 + width) {
            receive_word();
        }
        return None;
    }
    for patch in 0..count {
        receive_vector(V_PIXELS);
        receive_vector(V_POSITION);
        for i in 0..width {
            vector_write(vision(V_PATCH_ROPE) + patch * width + i, receive_word());
        }
        run_die(ROUTINE_VISION, patch, patch);
    }
    for block in 0..vision(V_BLOCKS) {
        for patch in 0..count {
            run_die(ROUTINE_VISION + 1 + 2 * block, patch, patch);
        }
        // Every patch attends over all of them.
        for patch in 0..count {
            run_die(ROUTINE_VISION + 2 + 2 * block, patch, count - 1);
        }
    }
    let merge = vision(V_MERGE);
    Some(Picture { tokens: count / (merge * merge), width: cols / merge, height: rows / merge })
}

/// The words of Show-o's description, from `INFO_SHOWO`.
const S_ROWS: usize = 0;
const S_LAYERS: usize = 1;
const S_CODEBOOK: usize = 2;
const S_SCORES: usize = 3;
const S_SCORES_EXPONENT: usize = 4;
const S_ROPE: usize = 5;
const S_ROPE_HALF: usize = 6;
const S_GUIDED: usize = 7;
/// The most image rows, text tokens and steps a fill takes.
const MAX_ROWS: usize = 520;
const MAX_TEXT: usize = 160;
const MAX_STEPS: usize = 64;
/// A masked place.
const MASKED: u32 = u32::MAX;

fn showo(field: usize) -> u32 {
    info(info(INFO_SHOWO) as usize + field)
}

fn send_word(word: u32) {
    for byte in word.to_le_bytes() {
        send(byte);
    }
}

/// The rotation of the cache's position `slot` from the table's row for `from`, in a table
/// at `base` with `half` rotated pairs a row.
fn rotate_in(base: u32, half: u32, slot: u32, from: u32) {
    let width = 3 * half + 1;
    copy_from_image(base + from * width, base + slot * width, width);
}

/// One pass of Show-o over an image, as the host asks:
/// - the text, as its tokens after a count and the position of the first: each goes in the
///   cache's next place, rotated as its real position;
/// - the image's tokens (its start, codes or the mask, and its end);
/// - each layer in two passes over the image's rows, the first after the text in the cache;
/// - each code's score at every place of the image, sent back as each place's exponent and
///   then its mantissas.
fn showo_pass() {
    let steps = receive_word();
    if steps != 0 {
        showo_fill(steps as usize);
        return;
    }
    let (text, first) = (receive_word(), receive_word());
    let (rows, base, half) = (showo(S_ROWS), showo(S_ROPE), showo(S_ROPE_HALF));
    let rotate = |slot: u32, position: u32| {
        if position != slot {
            rotate_in(base, half, slot, position);
        }
    };
    for slot in 0..text {
        let token = receive_word();
        rotate(slot, first + slot);
        run_die(ROUTINE_TOKEN, token, slot);
    }
    // The image's tokens are kept to read their embeddings before the first layer.
    let mut image = [0u32; 512];
    for row in 0..rows as usize {
        image[row] = receive_word();
    }
    for row in 0..rows {
        rotate(text + row, first + text + row);
    }
    for layer in 0..showo(S_LAYERS) {
        for row in 0..rows {
            if layer == 0 {
                run_die(ROUTINE_VISION, image[row as usize], 0);
            }
            run_die(ROUTINE_VISION + 1 + 2 * layer, row, text + row);
        }
        for row in 0..rows {
            run_die(ROUTINE_VISION + 2 + 2 * layer, row, text + rows - 1);
        }
    }
    let score = ROUTINE_VISION + 1 + 2 * showo(S_LAYERS);
    for row in 1..rows - 1 {
        run_die(score, row, 0);
        peripheral_write(REG_VECTOR_ADDRESS, showo(S_SCORES_EXPONENT));
        send_word(peripheral_read(REG_VECTOR_DATA));
        for code in 0..showo(S_CODEBOOK) {
            peripheral_write(REG_VECTOR_ADDRESS, showo(S_SCORES) + code);
            send_word(peripheral_read(REG_VECTOR_DATA));
        }
    }
}

/// A mantissa or exponent of the die, as a signed number of the vectors' width.
fn die_value(address: u32) -> i32 {
    peripheral_write(REG_VECTOR_ADDRESS, address);
    let unused = 32 - info(INFO_ACTIVATION_BITS);
    ((peripheral_read(REG_VECTOR_DATA) as i32) << unused) >> unused
}

/// A text the host sent: its tokens and the position of the first.
struct Text {
    tokens: [u32; MAX_TEXT],
    count: u32,
    first: u32,
}

fn receive_text() -> Text {
    let (count, first) = (receive_word(), receive_word());
    let mut tokens = [0u32; MAX_TEXT];
    for token in tokens.iter_mut().take(count as usize) {
        *token = receive_word();
    }
    Text { tokens, count, first }
}

static mut CURRENT: [u32; MAX_ROWS] = [0; MAX_ROWS];
static mut SAMPLED: [u32; MAX_ROWS] = [0; MAX_ROWS];
static mut CONFIDENCE: [i64; MAX_ROWS] = [0; MAX_ROWS];

/// One pass over the image with `text`: its tokens, then each layer twice over the rows,
/// then each place's scores and `after` for it. The image's rows are its start, `current`
/// as codes (or the mask) and its end.
fn fill_pass(text: &Text, current: &[u32], markers: (u32, u32, u32, u32), after: &mut dyn FnMut(u32)) {
    let (start, end, mask, first_code) = markers;
    let (rows, base, half) = (showo(S_ROWS), showo(S_ROPE), showo(S_ROPE_HALF));
    for slot in 0..text.count + rows {
        if text.first + slot != slot {
            rotate_in(base, half, slot, text.first + slot);
        }
    }
    for slot in 0..text.count {
        run_die(ROUTINE_TOKEN, text.tokens[slot as usize], slot);
    }
    let token = |row: u32| -> u32 {
        if row == 0 {
            return start;
        }
        if row == rows - 1 {
            return end;
        }
        match current[row as usize - 1] {
            MASKED => mask,
            code => first_code + code,
        }
    };
    for layer in 0..showo(S_LAYERS) {
        for row in 0..rows {
            if layer == 0 {
                run_die(ROUTINE_VISION, token(row), 0);
            }
            run_die(ROUTINE_VISION + 1 + 2 * layer, row, text.count + row);
        }
        for row in 0..rows {
            run_die(ROUTINE_VISION + 2 + 2 * layer, row, text.count + rows - 1);
        }
    }
    let score = ROUTINE_VISION + 1 + 2 * showo(S_LAYERS);
    for row in 1..rows - 1 {
        run_die(score, row, 0);
        after(row);
    }
}

/// Show-o's MaskGIT fill, the steps in the chip: the host sends the image's markers (its
/// start, end, the mask and the first code), the text and, when the die guides, the empty
/// text, a seed, each step's masked count and noise temperature, and each place's code or
/// `MASKED`. Each step draws a code for every masked place and keeps the surest; at the
/// end the chip sends every place's code.
fn showo_fill(steps: usize) {
    let markers = (receive_word(), receive_word(), receive_word(), receive_word());
    let text = receive_text();
    let guided = showo(S_GUIDED) != 0;
    let plain = if guided { Some(receive_text()) } else { None };
    let mut rng = maskgit_rt::Rng(receive_word() | 1);
    let mut schedule = [(0i64, 0i64); MAX_STEPS];
    for slot in schedule.iter_mut().take(steps) {
        *slot = (receive_word() as i32 as i64, receive_word() as i32 as i64);
    }
    let places = showo(S_ROWS) as usize - 2;
    let bits = info(INFO_ACTIVATION_BITS);
    // SAFETY: the firmware runs on one core and nothing else touches these.
    let (current, sampled, confidence) = unsafe { (&mut (&mut *addr_of_mut!(CURRENT))[..places], &mut (&mut *addr_of_mut!(SAMPLED))[..places], &mut (&mut *addr_of_mut!(CONFIDENCE))[..places]) };
    for place in 0..places {
        current[place] = receive_word();
        sampled[place] = if current[place] == MASKED { 0 } else { current[place] };
    }
    let score = ROUTINE_VISION + 1 + 2 * showo(S_LAYERS);
    for &(remaining, temperature) in schedule.iter().take(steps) {
        confidence.fill(i64::MAX);
        {
            let now: &[u32] = current;
            let mut draw = |row: u32| {
                let place = row as usize - 1;
                if now[place] != MASKED {
                    sampled[place] = now[place];
                    return;
                }
                let exponent = die_value(showo(S_SCORES_EXPONENT));
                let base = showo(S_SCORES);
                let (code, sure) = maskgit_rt::draw(showo(S_CODEBOOK), &|c| die_value(base + c), exponent, bits, temperature, &mut rng);
                sampled[place] = code;
                confidence[place] = sure;
            };
            match &plain {
                None => fill_pass(&text, now, markers, &mut draw),
                Some(plain) => {
                    fill_pass(&text, now, markers, &mut |row| run_die(score + 1, row, 0));
                    fill_pass(plain, now, markers, &mut |row| {
                        run_die(score + 2, row, 0);
                        draw(row);
                    });
                }
            }
        }
        let unknown = current.iter().filter(|c| **c == MASKED).count();
        let cut = maskgit_rt::cut(confidence, maskgit_rt::masked(remaining, unknown));
        for place in 0..places {
            current[place] = if confidence[place] >= cut { sampled[place] } else { MASKED };
        }
    }
    for code in sampled.iter() {
        send_word(*code);
    }
}

/// The rotation of the cache's position `slot`: the table's own row for `from`.
fn rotate_as(slot: u32, from: u32) {
    let width = 3 * vision(V_ROPE_HALF) + 1;
    copy_from_image(vision(V_ROPE) + from * width, vision(V_ROPE) + slot * width, width);
}

/// The rotation of an image token at `slot`, at row `row` and column `column` of an image
/// that starts at `start`: under M-RoPE, frequency `i` follows the row if `i % 3 == 1` and
/// the column if `i % 3 == 2`, as far as their sections reach, and the start otherwise.
fn rotate_image_token(slot: u32, start: u32, row: u32, column: u32) {
    let half = vision(V_ROPE_HALF);
    let (base, width) = (vision(V_ROPE), 3 * half + 1);
    rotate_as(slot, start);
    for i in 0..half {
        let from = match i % 3 {
            1 if i < 3 * vision(V_MROPE_ROWS) => start + row,
            2 if i < 3 * vision(V_MROPE_COLS) => start + column,
            _ => continue,
        };
        for offset in [0, half, 2 * half] {
            copy_from_image(base + from * width + offset + i, base + slot * width + offset + i, 1);
        }
    }
}

/// Follows the model's reasoning, between its opening and closing tokens, and closes it
/// itself once it has taken as many tokens as the budget allows.
#[derive(Default)]
struct Reasoning {
    inside: bool,
    spent: u32,
}

impl Reasoning {
    fn see(&mut self, token: u32) {
        if token == info(INFO_THINK_OPEN) {
            (self.inside, self.spent) = (true, 0);
        } else if token == info(INFO_THINK_CLOSE) {
            self.inside = false;
        } else if self.inside {
            self.spent += 1;
        }
    }

    /// The token the chip goes on with in place of the one it sampled.
    fn steer(&mut self, sampled: u32) -> u32 {
        let token = if self.inside && sampled != info(INFO_THINK_CLOSE) && self.spent >= info(INFO_THINK_BUDGET) { info(INFO_THINK_CLOSE) } else { sampled };
        self.see(token);
        token
    }
}

fn emit(token: u32, decoding: &mut Decoding) {
    if info(INFO_RAW_TOKENS) != 0 {
        for shift in 0..4 {
            send((token >> (8 * shift)) as u8);
        }
        return;
    }
    tokenizer().write_token(token, decoding, &mut send);
}

/// A schedule's records: see `netlist::schedule`, which reads them the same way.
const RECORD_WORDS: usize = 8;
const S_INDEX: u32 = 3;
const S_TOKEN: u32 = 1;
const S_POSITION: u32 = 2;

fn field(at: usize, word: usize) -> u32 {
    data_word(info(INFO_SCHEDULE) as usize + at * RECORD_WORDS + word)
}

/// The most calls in a repeat that `repeat_calls` takes.
const FAST_CALLS: usize = 16;

/// A repeat whose body is only calls, run lean on a single chip: each call's entry and the
/// value of each parameter register at the first iteration are worked out once, with how
/// much each grows from one iteration to the next, so an iteration only writes and adds.
/// False, having run nothing, when the repeat is not of that kind; the general way then
/// runs it.
fn repeat_calls(first: usize, body: usize, count: u32, vars: &[u32; 16]) -> bool {
    let parameters = info(INFO_PARAMETER_COUNT) as usize;
    if body > FAST_CALLS || chips() > 1 || info(INFO_TABLE_COLS) != 0 || info(INFO_ROTATION_COLS) != 0 || parameters > 16 {
        return false;
    }
    for p in 0..parameters {
        if info(INFO_PARAMETERS + 7 * p + 6) != 0 {
            return false;
        }
    }
    let mut entries = [0u32; FAST_CALLS];
    let mut values = [[0u32; 16]; FAST_CALLS];
    let mut steps = [[0u32; 16]; FAST_CALLS];
    for k in 0..body {
        let f = |word| field(first + k, word);
        if f(0) != 5 {
            return false;
        }
        entries[k] = info(info(INFO_ENTRIES) as usize + f(1) as usize);
        let (token, position) = (f(3).wrapping_add(vars[(f(4) & 15) as usize]), f(6).wrapping_add(vars[(f(7) & 15) as usize]));
        for p in 0..parameters {
            let at = INFO_PARAMETERS + 7 * p;
            values[k][p] = times(info(at + 1), token).wrapping_add(times(info(at + 2), position)).wrapping_add(info(at + 3));
            steps[k][p] = times(info(at + 1), f(2)).wrapping_add(times(info(at + 2), f(5)));
        }
    }
    // Each parameter register's address, worked out once.
    let mut registers = [core::ptr::null_mut::<u32>(); 16];
    for (p, register) in registers.iter_mut().enumerate().take(parameters) {
        *register = unsafe { PERIPHERALS.add(info(INFO_PARAMETERS + 7 * p) as usize) };
    }
    let entry = unsafe { PERIPHERALS.add(REG_ENTRY) };
    let registers = &registers[..parameters];
    // Between two routines the CPU only writes the parameters and starts the die; the next
    // values are worked out while the die runs. With a command queue that has room for a
    // routine's orders, the CPU orders the next while the die runs this one.
    let queued = info(INFO_QUEUE) as usize > parameters;
    for _ in 0..count {
        for ((entry_word, values), steps) in entries.iter().zip(values.iter_mut()).zip(&steps).take(body) {
            if queued {
                while (peripheral_read(REG_QUEUE_ROOM) as usize) <= parameters {}
            } else {
                wait_for_die();
            }
            for (register, value) in registers.iter().zip(values.iter()) {
                unsafe { write_volatile(*register, *value) };
            }
            unsafe { write_volatile(entry, *entry_word) };
            for (value, step) in values.iter_mut().zip(steps.iter()).take(parameters) {
                *value = value.wrapping_add(*step);
            }
        }
    }
    wait_for_die();
    true
}

/// Follows the schedule's records from `at` to `end`; false once it meets its end.
fn follow(mut at: usize, end: usize, vars: &mut [u32; 16]) -> bool {
    while at < end {
        let here = at;
        let f = |word| field(here, word);
        let var = |vars: &[u32; 16], v: u32| vars[(v & 15) as usize];
        let set = |vars: &mut [u32; 16], v: u32, value: u32| {
            if v != 0 {
                vars[(v & 15) as usize] = value;
            }
        };
        let next = here + 1;
        at = next;
        match f(0) {
            1 => set(vars, f(1), receive_word()),
            2 => {
                let product = (var(vars, f(2)) as i32).wrapping_mul(f(3) as i32).wrapping_add(f(4) as i32);
                let divisor = f(5) as i32;
                let quotient = if divisor == 0 { 0 } else { product.wrapping_div(divisor) };
                set(vars, f(1), quotient.wrapping_add(f(6) as i32).min(f(7) as i32) as u32);
            }
            3 => {
                for i in 0..var(vars, f(1)) + f(2) {
                    vector_write(f(4), receive_word());
                    for e in 0..f(5) {
                        vector_write(f(3) + e, receive_word());
                    }
                    if f(6) != u32::MAX {
                        run_die(f(6), i, 0);
                    }
                }
            }
            4 => {
                let (body, outer) = (f(1) as usize, var(vars, S_INDEX));
                let count = var(vars, f(2)) + f(3);
                if repeat_calls(next, body, count, vars) {
                    at = next + body;
                    continue;
                }
                for i in 0..count {
                    set(vars, S_INDEX, i);
                    if !follow(next, next + body, vars) {
                        return false;
                    }
                }
                set(vars, S_INDEX, outer);
                at = next + body;
            }
            5 => {
                let i = var(vars, S_INDEX);
                let token = times(f(2), i).wrapping_add(f(3)).wrapping_add(var(vars, f(4)));
                let position = times(f(5), i).wrapping_add(f(6)).wrapping_add(var(vars, f(7)));
                run_die(f(1), token, position);
            }
            6 => {
                let mut count = var(vars, f(1));
                while count > 1 {
                    let half = count / 2;
                    for i in 0..half {
                        run_die(f(2), i, 0);
                        run_die(f(3), i + half, 0);
                        run_die(f(4), i, 0);
                    }
                    if count % 2 == 1 {
                        run_die(f(2), 0, 0);
                        run_die(f(3), count - 1, 0);
                        run_die(f(4), 0, 0);
                    }
                    count = half;
                }
            }
            7 => {
                let body = f(1) as usize;
                set(vars, S_TOKEN, f(3));
                let stops = info(INFO_STOP_TABLE)..info(INFO_STOP_TABLE) + info(INFO_STOP_COUNT);
                for position in 0..var(vars, f(2)) {
                    set(vars, S_POSITION, position);
                    if !follow(next, next + body, vars) {
                        return false;
                    }
                    let token = sample();
                    if stops.clone().any(|at| info(at as usize) == token) {
                        break;
                    }
                    send_word(token);
                    set(vars, S_TOKEN, token);
                }
                at = next + body;
            }
            8 => {
                send_word(die_value(f(1)) as u32);
                send_word(die_value(f(2)) as u32);
            }
            9 => {
                let (body, outer) = (f(1) as usize, var(vars, S_INDEX));
                let mut times = 0;
                while receive_word() != 0 {
                    set(vars, S_INDEX, times);
                    if !follow(next, next + body, vars) {
                        return false;
                    }
                    times += 1;
                }
                set(vars, S_INDEX, outer);
                set(vars, f(2), times);
                at = next + body;
            }
            _ => return false,
        }
    }
    true
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
    // A die run by routines follows its schedule: the host sends what it says.
    if info(INFO_SCHEDULE) != 0 {
        follow(0, usize::MAX, &mut [0; 16]);
        return;
    }
    // A chip that carries Show-o fills images instead of writing text.
    if info(INFO_SHOWO) != 0 {
        showo_pass();
        return;
    }
    let entry_token = ROUTINE_TOKEN;
    let depth = (info(INFO_DRAFT_DEPTH) as usize).min(MAX_DRAFT_DEPTH);
    let rotates = info(INFO_VISION) != 0 && vision(V_ROPE_HALF) != 0;
    let stops = info(INFO_STOP_TABLE) as usize..(info(INFO_STOP_TABLE) + info(INFO_STOP_COUNT)) as usize;
    // A chip that holds a conversation clears it at power-up, every place of its cache,
    // and again whenever the clear pin is raised, as far as the conversation went.
    let conversation = info(INFO_CONVERSATION) as usize;
    let capacity = if conversation != 0 { info(conversation) } else { u32::MAX };
    let mut rows = capacity;
    loop {
        if conversation != 0 {
            clear_conversation(rows);
        }
        let (mut last, mut position, mut draft_position) = (0, 0, 0u32);
        let mut reasoning = Reasoning::default();
        // After an image, the text's rotations run behind the cache by this much.
        let mut behind = 0u32;
        let mut first = true;
        loop {
            // A chip that carries an image encoder first hears whether an image comes.
            let mut picture = if info(INFO_VISION) != 0 { read_picture() } else { None };
            read_prompt(first, &mut |token| {
                // A full cache takes nothing more until the conversation is cleared.
                if position >= capacity {
                    return;
                }
                if let Some(image) = picture.filter(|_| token == vision(V_IMAGE_TOKEN)) {
                    // The image's tokens take the placeholder's place, one by one, and the text
                    // goes on after the longer side of the image.
                    let start = position;
                    for index in 0..image.tokens {
                        run_die(ROUTINE_VISION + 1 + 2 * vision(V_BLOCKS), index, index);
                        if rotates {
                            rotate_image_token(position, start, index / image.width, index % image.width);
                        }
                        run_die(ROUTINE_TOKEN_FROM_IMAGE, 0, position);
                        position += 1;
                    }
                    behind = image.tokens - image.width.max(image.height);
                    picture = None;
                    return;
                }
                reasoning.see(token);
                if rotates && behind > 0 {
                    rotate_as(position, position - behind);
                }
                // The draft head reads the state before each token, to predict the one after it.
                if depth > 0 && position > 0 {
                    run_die(ROUTINE_FEED_FROM_MAIN, 0, 0);
                    run_die(ROUTINE_DRAFT, token, draft_position);
                    draft_position += 1;
                }
                run_die(entry_token, token, position);
                last = token;
                position += 1;
            });
            first = false;
            if aborted() {
                break;
            }
            if position == 0 && conversation == 0 {
                return;
            }
            answer(&mut last, &mut position, &mut draft_position, &mut reasoning, (depth, rotates, behind, stops.clone(), capacity));
            if conversation == 0 {
                return;
            }
            // The answer's end, and the chip waits for the next turn of the conversation.
            if info(INFO_RAW_TOKENS) != 0 {
                emit(END_OF_TOKENS, &mut Decoding::default());
            } else {
                send(0);
            }
            if aborted() {
                break;
            }
        }
        rows = position;
    }
}

/// Writes the answer after the prompt, from `last` at `position`, until a stop token, a
/// loop, the most new tokens, a full cache or the clear pin.
fn answer(last: &mut u32, position: &mut u32, draft_position: &mut u32, reasoning: &mut Reasoning, (depth, rotates, behind, stops, capacity): (usize, bool, u32, core::ops::Range<usize>, u32)) {
    if *position == 0 {
        return;
    }
    let (entry_token, entry_logits) = (ROUTINE_TOKEN, ROUTINE_LOGITS);
    // Raw mode has no tokenizer in memory to ask.
    let mut decoding = if info(INFO_RAW_TOKENS) != 0 { Decoding::default() } else { tokenizer().decoding() };
    let mut pending = [(0u32, [0u32; MAX_DRAFT_DEPTH]); MAX_DRAFT_DEPTH + 1];
    let (mut waiting, mut produced, mut previous) = (0usize, 0u32, 0u32);
    // A model that never writes its end token stops once it repeats itself.
    let mut watch = LoopWatch::default();
    for _ in 0..info(INFO_MAX_NEW) {
        if unsafe { read_volatile(HOST_RX_STATUS) } & CLEAR_REQUESTED != 0 {
            unsafe { ABORT = true };
            return;
        }
        run_die(entry_logits, *last, *position - 1);
        *last = reasoning.steer(sample());
        if stops.clone().any(|at| info(at) == *last) {
            return;
        }
        if depth > 0 {
            if produced > 0 {
                // Earlier guesses are scored against what the model really produced.
                let mut kept = 0;
                for index in 0..waiting {
                    let (from, guesses) = pending[index];
                    let ahead = produced.wrapping_sub(from) as usize;
                    if (1..=depth).contains(&ahead) {
                        peripheral_write(REG_DRAFT_RESULT, ((ahead as u32 - 1) << 1) | (guesses[ahead - 1] == previous) as u32);
                    }
                    if ahead < depth {
                        pending[kept] = pending[index];
                        kept += 1;
                    }
                }
                waiting = kept;
                let mut guess = *last;
                let mut guesses = [0u32; MAX_DRAFT_DEPTH];
                run_die(ROUTINE_FEED_FROM_MAIN, 0, 0);
                for (step, slot) in guesses.iter_mut().enumerate().take(depth) {
                    run_die(ROUTINE_DRAFT_LOGITS, guess, *draft_position + step as u32);
                    guess = sample_draft();
                    *slot = guess;
                    run_die(ROUTINE_FEED_FROM_DRAFT, 0, 0);
                }
                if waiting < pending.len() {
                    pending[waiting] = (produced + 1, guesses);
                    waiting += 1;
                }
            }
            run_die(ROUTINE_FEED_FROM_MAIN, 0, 0);
            run_die(ROUTINE_DRAFT, *last, *draft_position);
            *draft_position += 1;
        }
        emit(*last, &mut decoding);
        if watch.push(*last) || *position >= capacity {
            return;
        }
        if rotates && behind > 0 {
            rotate_as(*position, *position - behind);
        }
        run_die(entry_token, *last, *position);
        *position += 1;
        produced += 1;
        previous = *last;
    }
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {
        unsafe { core::arch::asm!("ebreak") };
    }
}
