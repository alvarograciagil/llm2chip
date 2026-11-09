//! The chip image: everything the chip needs at power-up, as a stream of 32-bit words.
//! Words 0 to 2 are the magic number and the sizes of the firmware's two memories, which
//! the load controller copies by itself; the firmware then loads every other segment.

use crate::soc::{CONFIG_WORD, SEGMENT_TABLE_WORD};

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
pub fn build_image(program: &[u32], data: &[u32], config: &[u32], segments: &[Segment]) -> Result<Vec<u32>, String> {
    if data.len() > CONFIG_WORD {
        return Err(format!("los datos del firmware ocupan {} palabras y el límite es {CONFIG_WORD}", data.len()));
    }
    if config.len() > SEGMENT_TABLE_WORD - CONFIG_WORD {
        return Err(format!("la descripción del chip ocupa {} palabras y el límite es {}", config.len(), SEGMENT_TABLE_WORD - CONFIG_WORD));
    }
    let table_words = 1 + segments.len() * SEGMENT_FIELDS;
    let data_words = SEGMENT_TABLE_WORD + table_words;
    let mut source = HEADER_WORDS + program.len() + data_words;

    let mut table = vec![segments.len() as u32];
    for segment in segments {
        table.extend([source as u32, segment.target as u32, segment.unit, segment.address, segment.elements() as u32, segment.beats, crc32(&segment.words)]);
        source += segment.words.len();
    }

    let mut image = vec![MAGIC, program.len() as u32, data_words as u32];
    image.extend_from_slice(program);
    image.extend_from_slice(data);
    image.resize(HEADER_WORDS + program.len() + CONFIG_WORD, 0);
    image.extend_from_slice(config);
    image.resize(HEADER_WORDS + program.len() + SEGMENT_TABLE_WORD, 0);
    image.extend(table);
    for segment in segments {
        image.extend_from_slice(&segment.words);
    }
    Ok(image)
}
