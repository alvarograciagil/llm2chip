//! The chip image: everything the chip needs at power-up, as a stream of 32-bit words.
//! Words 0 to 2 are the magic number and the sizes of the firmware's two memories, which
//! the load controller copies by itself; the firmware then loads every other segment.

use crate::soc::{CONFIG_WORD, PLAN_DIRECTORY_WORD, SEGMENT_TABLE_WORD, TOKENIZER_WORD};

pub const MAGIC: u32 = 0x4943_324C;
pub const HEADER_WORDS: usize = 3;
pub const SEGMENT_FIELDS: usize = 7;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Target {
    Program = 0,
    Data = 1,
    Microcode = 2,
    Vector = 3,
    Function = 4,
    CellWeight = 5,
    CellScale = 6,
    SamplerTable = 7,
    /// Where each register starts in a compact vector memory, and its loop step.
    RegisterBase = 8,
    /// A lookup table that stays in the image: the firmware loads one row of it into the
    /// register at `address` before each token, `unit` elements. It is not loaded at boot.
    RuntimeTable = 9,
    /// For each PC of a microcode kept as a dictionary, where its following word and its
    /// branch target are in the dictionary.
    MicrocodeIndex = 10,
}

/// A block of the image and where it goes on the chip.
#[derive(Clone, Debug)]
pub struct Segment {
    pub target: Target,
    pub unit: u32,
    pub address: u32,
    /// Image words per element written.
    pub beats: u32,
    pub words: Vec<u32>,
}

impl Segment {
    pub fn elements(&self) -> usize {
        self.words.len() / self.beats as usize
    }
}

/// CRC-32 of one more word, most significant polynomial bit first as in Ethernet, reflected.
pub fn crc32_word(crc: u32, word: u32) -> u32 {
    let mut value = crc ^ word;
    for _ in 0..32 {
        value = if value & 1 == 1 { (value >> 1) ^ 0xEDB8_8320 } else { value >> 1 };
    }
    value
}

pub fn crc32(words: &[u32]) -> u32 {
    words.iter().fold(u32::MAX, |crc, word| crc32_word(crc, *word))
}

/// Builds the image. The chip description and the segment table are placed inside
/// the firmware's data memory, where it finds them without any other help.
pub fn build_image(program: &[u32], data: &[u32], config: &[u32], tokenizer: &[u32], segments: &[Segment]) -> Result<Vec<u32>, String> {
    let (segments, tables) = runtime_tables(segments);
    let segments = segments.as_slice();
    let mut config = config.to_vec();
    if data.len() > CONFIG_WORD {
        return Err(format!("los datos del firmware ocupan {} palabras y el límite es {CONFIG_WORD}", data.len()));
    }
    if config.len() > PLAN_DIRECTORY_WORD - CONFIG_WORD {
        return Err(format!("la descripción del chip ocupa {} palabras y el límite es {}", config.len(), PLAN_DIRECTORY_WORD - CONFIG_WORD));
    }
    let table_words = 1 + segments.len() * SEGMENT_FIELDS;
    if table_words > TOKENIZER_WORD - SEGMENT_TABLE_WORD {
        return Err(format!("la tabla de segmentos ocupa {table_words} palabras y el límite es {}", TOKENIZER_WORD - SEGMENT_TABLE_WORD));
    }
    let data_words = if tokenizer.is_empty() { SEGMENT_TABLE_WORD + table_words } else { TOKENIZER_WORD + tokenizer.len() };
    let mut source = HEADER_WORDS + program.len() + data_words;

    let mut table = vec![segments.len() as u32];
    for segment in segments {
        table.extend([source as u32, segment.target as u32, segment.unit, segment.address, segment.elements() as u32, segment.beats, crc32(&segment.words)]);
        source += segment.words.len();
    }

    let mut tables_at = source;
    point_at_table(&mut config, tables.first(), &mut tables_at);
    let mut image = vec![MAGIC, program.len() as u32, data_words as u32];
    image.extend_from_slice(program);
    image.extend_from_slice(data);
    image.resize(HEADER_WORDS + program.len() + CONFIG_WORD, 0);
    image.extend_from_slice(&config);
    image.resize(HEADER_WORDS + program.len() + SEGMENT_TABLE_WORD, 0);
    image.extend(table);
    if !tokenizer.is_empty() {
        image.resize(HEADER_WORDS + program.len() + TOKENIZER_WORD, 0);
        image.extend_from_slice(tokenizer);
    }
    for segment in segments {
        image.extend_from_slice(&segment.words);
    }
    for table in tables.iter().take(1) {
        image.extend_from_slice(&table.words);
    }
    Ok(image)
}

/// Splits off the tables that stay in the image from the segments loaded at boot.
fn runtime_tables(segments: &[Segment]) -> (Vec<Segment>, Vec<Segment>) {
    let (tables, boot): (Vec<Segment>, Vec<Segment>) = segments.iter().cloned().partition(|s| s.target == Target::RuntimeTable);
    assert!(tables.len() <= 1, "la imagen solo admite una tabla que se lea en marcha");
    (boot, tables)
}

/// Tells the firmware where `table` will be in the image, which is `at` onwards, and moves
/// `at` past it.
fn point_at_table(info: &mut [u32], table: Option<&Segment>, at: &mut usize) {
    let Some(table) = table else { return };
    info[crate::chip::INFO_TABLE_SOURCE] = *at as u32;
    info[crate::chip::INFO_TABLE_ADDRESS] = table.address;
    info[crate::chip::INFO_TABLE_COLS] = table.unit;
    *at += table.words.len();
}

/// What one chip needs when the model is shared among a given number of them.
#[derive(Clone, Debug)]
pub struct ChipPart {
    pub info: Vec<u32>,
    pub segments: Vec<Segment>,
}

fn table_words(segments: &[Segment]) -> usize {
    1 + segments.len() * SEGMENT_FIELDS
}

fn segment_table(segments: &[Segment], source: &mut usize) -> Vec<u32> {
    let mut table = vec![segments.len() as u32];
    for segment in segments {
        table.extend([*source as u32, segment.target as u32, segment.unit, segment.address, segment.elements() as u32, segment.beats, crc32(&segment.words)]);
        *source += segment.words.len();
    }
    table
}

/// One image for every number of chips in `plans`, each given with one part per chip.
/// The firmware counts the chips at start-up and takes its own part.
pub fn build_plan_image(program: &[u32], data: &[u32], tokenizer: &[u32], plans: &[Vec<ChipPart>]) -> Result<Vec<u32>, String> {
    if data.len() > CONFIG_WORD {
        return Err(format!("los datos del firmware ocupan {} palabras y el límite es {CONFIG_WORD}", data.len()));
    }
    // The directory: for each plan its number of chips and where its parts are listed.
    let lists_at = TOKENIZER_WORD + tokenizer.len();
    let lists_len = 2 * plans.len() + plans.iter().map(|plan| 2 * plan.len()).sum::<usize>();
    let mut block = lists_at + lists_len;
    let mut lists = vec![0u32; 2 * plans.len()];
    for (index, plan) in plans.iter().enumerate() {
        lists[2 * index] = plan.len() as u32;
        lists[2 * index + 1] = (lists_at + lists.len()) as u32;
        for part in plan {
            lists.push(block as u32);
            lists.push((block + part.info.len()) as u32);
            block += part.info.len() + table_words(&runtime_tables(&part.segments).0);
        }
    }
    let data_words = block;
    let mut source = HEADER_WORDS + program.len() + data_words;
    let split: Vec<(Vec<Segment>, Vec<Segment>)> = plans.iter().flatten().map(|part| runtime_tables(&part.segments)).collect();
    let mut tables_at = source + split.iter().flat_map(|(boot, _)| boot).map(|s| s.words.len()).sum::<usize>();
    // Parts with the very same table share one copy of it.
    let mut placed: Vec<(&[u32], usize)> = Vec::new();
    let mut runtime_words: Vec<u32> = Vec::new();
    let mut blocks = Vec::new();
    for (part, (boot, tables)) in plans.iter().flatten().zip(&split) {
        let mut info = part.info.clone();
        if let Some(table) = tables.first() {
            match placed.iter().find(|(words, _)| *words == table.words.as_slice()) {
                Some((_, at)) => {
                    let mut at = *at;
                    point_at_table(&mut info, Some(table), &mut at);
                }
                None => {
                    placed.push((&table.words, tables_at));
                    runtime_words.extend_from_slice(&table.words);
                    point_at_table(&mut info, Some(table), &mut tables_at);
                }
            }
        }
        blocks.extend_from_slice(&info);
        blocks.extend(segment_table(boot, &mut source));
    }

    let mut image = vec![MAGIC, program.len() as u32, data_words as u32];
    image.extend_from_slice(program);
    image.extend_from_slice(data);
    let base = HEADER_WORDS + program.len();
    image.resize(base + PLAN_DIRECTORY_WORD, 0);
    image.extend([plans.len() as u32, lists_at as u32]);
    image.resize(base + TOKENIZER_WORD, 0);
    image.extend_from_slice(tokenizer);
    image.extend(lists);
    image.extend(blocks);
    for segment in split.iter().flat_map(|(boot, _)| boot) {
        image.extend_from_slice(&segment.words);
    }
    image.extend(runtime_words);
    Ok(image)
}
