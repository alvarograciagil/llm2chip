//! The messages of the chip's host port, as a host writes and reads them.
//!
//! The port carries messages both ways, framed as SLIP frames are: a message is its kind (one
//! byte), its bytes and an end byte (`FRAME_END`), and either of those marks inside a message
//! goes as an escape and a code. The port's hardware undoes the framing as bytes arrive and
//! keeps each byte's place in its message beside it, so the firmware reads a message's bytes
//! at the cost of a plain byte, and a byte lost on the way spoils one message only.
//!
//! Every host of the chip uses this module: the tests, the engine's simulator, the gates in
//! the browser and the Verilog test bench.

use crate::soc::{FRAME_END, FRAME_ESCAPE, FRAME_ESCAPED_END, FRAME_ESCAPED_ESCAPE};

/// The kinds of message. The host sends the first group; the chip sends `TEXT`, `TOKENS`,
/// `STREAM`, `STATUS` and `OPTIONS` back as answers, and the rest of its own.
pub mod kind {
    /// A turn as text, for a chip that tokenizes; its answer comes back as text.
    pub const TEXT: u8 = 0x01;
    /// A turn as tokens, each 4 bytes little-endian; its answer comes back as tokens.
    pub const TOKENS: u8 = 0x02;
    /// A picture before the turn that places it, for a chip that reads images.
    pub const PICTURE: u8 = 0x03;
    /// Words for a chip run by a schedule or Show-o's passes; such a chip answers with words.
    pub const STREAM: u8 = 0x04;
    /// Asks where the conversation stands.
    pub const STATUS: u8 = 0x10;
    /// Asks the options and their values.
    pub const OPTIONS: u8 = 0x11;
    /// Sets an option: its number and its value.
    pub const SET: u8 = 0x12;
    /// Ends the answer being written.
    pub const STOP: u8 = 0x13;
    /// Clears the conversation, as the clear pin does.
    pub const CLEAR: u8 = 0x14;

    /// The chip has loaded its image: a byte a segment, `.` for a good checksum, `E` for one
    /// that failed.
    pub const BOOT: u8 = 0x20;
    /// The chip waits for a conversation's first turn: its capacity in positions, zero for a
    /// chip that answers once.
    pub const READY: u8 = 0x21;
    /// An answer has ended: why, the positions the conversation takes and its capacity.
    pub const END: u8 = 0x22;
    /// Something went wrong: a code and a detail.
    pub const ERROR: u8 = 0x2F;
}

/// Why an answer ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ending {
    /// The model's end token.
    EndToken,
    /// The most tokens an answer may take.
    Limit,
    /// The model was going round in circles.
    Loop,
    /// The conversation filled the chip's context.
    Full,
    /// The host asked to stop.
    Stopped,
    /// The conversation was cleared in the middle of it.
    Cleared,
    Unknown(u8),
}

impl Ending {
    pub fn from_code(code: u8) -> Self {
        match code {
            0 => Ending::EndToken,
            1 => Ending::Limit,
            2 => Ending::Loop,
            3 => Ending::Full,
            4 => Ending::Stopped,
            5 => Ending::Cleared,
            other => Ending::Unknown(other),
        }
    }
}

/// What an error message's code means.
pub mod error {
    /// The image has no plan for this many chips.
    pub const NO_PLAN: u8 = 1;
    /// A kind of message the chip does not know.
    pub const UNKNOWN: u8 = 2;
    /// A message the chip does not take now, or at all: text without a tokenizer, a picture
    /// without an encoder, a turn in the middle of a stream.
    pub const UNEXPECTED: u8 = 3;
    /// An option the chip does not have.
    pub const NO_OPTION: u8 = 4;
    /// A text longer than the tokenizer's working memory.
    pub const TOO_LONG: u8 = 5;
}

/// The options a host can read and set while the chip runs, by number: the fields of the
/// chip's description they are.
pub const OPTIONS: [(u8, &str); 6] = [(5, "max_new_tokens"), (6, "greedy"), (7, "top_k"), (8, "temperature_x256"), (9, "seed"), (20, "think_budget")];

/// The number of the option called `name`.
pub fn option(name: &str) -> Option<u8> {
    OPTIONS.iter().find(|(_, n)| *n == name).map(|(id, _)| *id)
}

/// A message framed for the wire.
pub fn frame(kind: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 2);
    out.push(kind);
    for &byte in payload {
        match byte {
            FRAME_END => out.extend([FRAME_ESCAPE, FRAME_ESCAPED_END]),
            FRAME_ESCAPE => out.extend([FRAME_ESCAPE, FRAME_ESCAPED_ESCAPE]),
            other => out.push(other),
        }
    }
    out.push(FRAME_END);
    out
}

fn le(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// A turn as text.
pub fn text(text: &str) -> Vec<u8> {
    frame(kind::TEXT, text.as_bytes())
}

/// A turn as tokens.
pub fn tokens(tokens: &[u32]) -> Vec<u8> {
    frame(kind::TOKENS, &le(tokens))
}

/// Words for a chip run by a schedule or Show-o's passes.
pub fn stream(words: &[u32]) -> Vec<u8> {
    frame(kind::STREAM, &le(words))
}

/// A picture's bytes, as `sim::vision_chip::picture_bytes` lays them out.
pub fn picture(bytes: &[u8]) -> Vec<u8> {
    frame(kind::PICTURE, bytes)
}

pub fn status() -> Vec<u8> {
    frame(kind::STATUS, &[])
}

pub fn options() -> Vec<u8> {
    frame(kind::OPTIONS, &[])
}

pub fn set(option: u8, value: u32) -> Vec<u8> {
    let mut payload = vec![option];
    payload.extend(value.to_le_bytes());
    frame(kind::SET, &payload)
}

pub fn stop() -> Vec<u8> {
    frame(kind::STOP, &[])
}

pub fn clear() -> Vec<u8> {
    frame(kind::CLEAR, &[])
}

/// A message as it arrived: its kind and its bytes, the framing undone.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub kind: u8,
    pub payload: Vec<u8>,
}

impl Message {
    /// The bytes as 32-bit words, little-endian.
    pub fn words(&self) -> Vec<u32> {
        self.payload.as_chunks::<4>().0.iter().map(|w| u32::from_le_bytes(*w)).collect()
    }

    fn word(&self, at: usize) -> u32 {
        self.payload.get(at..at + 4).map_or(0, |w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
    }

    /// What the message says.
    pub fn event(&self) -> Event {
        match self.kind {
            kind::BOOT => Event::Boot { segments: self.payload.len(), failed: self.payload.iter().filter(|b| **b != b'.').count() },
            kind::READY => Event::Ready { capacity: self.word(0) },
            kind::TEXT => Event::Text(self.payload.clone()),
            kind::TOKENS => Event::Tokens(self.words()),
            kind::STREAM => Event::Words(self.words()),
            kind::END => Event::End { ending: Ending::from_code(self.payload.first().copied().unwrap_or(0)), position: self.word(1), capacity: self.word(5) },
            kind::STATUS => Event::Status { answering: self.payload.first().is_some_and(|s| *s != 0), position: self.word(1), capacity: self.word(5) },
            kind::OPTIONS => Event::Options(self.payload.as_chunks::<5>().0.iter().map(|c| (c[0], u32::from_le_bytes([c[1], c[2], c[3], c[4]]))).collect()),
            kind::ERROR => Event::Error { code: self.payload.first().copied().unwrap_or(0), detail: self.word(1) },
            _ => Event::Other(self.clone()),
        }
    }
}

/// What a message from the chip says.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    /// The segments loaded, and how many of them failed their checksum.
    Boot { segments: usize, failed: usize },
    Ready { capacity: u32 },
    Text(Vec<u8>),
    Tokens(Vec<u32>),
    Words(Vec<u32>),
    End { ending: Ending, position: u32, capacity: u32 },
    Status { answering: bool, position: u32, capacity: u32 },
    Options(Vec<(u8, u32)>),
    Error { code: u8, detail: u32 },
    Other(Message),
}

/// Reads the chip's bytes as they come, a message at a time.
#[derive(Default)]
pub struct Reader {
    current: Option<Message>,
    escaping: bool,
}

impl Reader {
    /// The messages these bytes complete.
    pub fn read(&mut self, bytes: &[u8]) -> Vec<Message> {
        let mut done = Vec::new();
        for &byte in bytes {
            match &mut self.current {
                None => {
                    if byte != FRAME_END && byte != FRAME_ESCAPE {
                        self.current = Some(Message { kind: byte, payload: Vec::new() });
                    }
                }
                Some(message) => {
                    if byte == FRAME_END {
                        done.push(self.current.take().unwrap());
                        self.escaping = false;
                    } else if self.escaping {
                        message.payload.push(match byte {
                            FRAME_ESCAPED_END => FRAME_END,
                            FRAME_ESCAPED_ESCAPE => FRAME_ESCAPE,
                            other => other,
                        });
                        self.escaping = false;
                    } else if byte == FRAME_ESCAPE {
                        self.escaping = true;
                    } else {
                        message.payload.push(byte);
                    }
                }
            }
        }
        done
    }

    /// The bytes of the message still arriving, if one is.
    pub fn partial(&self) -> Option<&Message> {
        self.current.as_ref()
    }
}

/// The whole messages in these bytes.
pub fn messages(bytes: &[u8]) -> Vec<Message> {
    Reader::default().read(bytes)
}

/// What each whole message in these bytes says.
pub fn events(bytes: &[u8]) -> Vec<Event> {
    messages(bytes).iter().map(Message::event).collect()
}

/// The tokens of each answer the chip ended, in order.
pub fn answers(bytes: &[u8]) -> Vec<Vec<u32>> {
    let (mut all, mut current) = (Vec::new(), Vec::new());
    for event in events(bytes) {
        match event {
            Event::Tokens(tokens) => current.extend(tokens),
            Event::End { .. } => all.push(std::mem::take(&mut current)),
            _ => {}
        }
    }
    all
}

/// The text of each answer the chip ended, in order.
pub fn answer_texts(bytes: &[u8]) -> Vec<String> {
    let (mut all, mut current) = (Vec::new(), Vec::new());
    for event in events(bytes) {
        match event {
            Event::Text(text) => current.extend(text),
            Event::End { .. } => all.push(String::from_utf8_lossy(&std::mem::take(&mut current)).into_owned()),
            _ => {}
        }
    }
    all
}

/// Every word the chip sent in its stream messages, in order.
pub fn words(bytes: &[u8]) -> Vec<u32> {
    events(bytes).into_iter().flat_map(|e| if let Event::Words(w) = e { w } else { Vec::new() }).collect()
}

/// The text the chip sent, all its text messages joined, ended or not.
pub fn all_text(bytes: &[u8]) -> String {
    let joined: Vec<u8> = events(bytes).into_iter().flat_map(|e| if let Event::Text(t) = e { t } else { Vec::new() }).collect();
    String::from_utf8_lossy(&joined).into_owned()
}

/// The segments the chip loaded, and how many failed their checksum.
pub fn boot(bytes: &[u8]) -> Option<(usize, usize)> {
    events(bytes).into_iter().find_map(|e| if let Event::Boot { segments, failed } = e { Some((segments, failed)) } else { None })
}

/// The errors the chip reported, by code.
pub fn errors(bytes: &[u8]) -> Vec<u8> {
    events(bytes).into_iter().filter_map(|e| if let Event::Error { code, .. } = e { Some(code) } else { None }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_message_comes_back_as_it_went_whatever_its_bytes() {
        let payload: Vec<u8> = (0..=255).chain([FRAME_END, FRAME_ESCAPE, FRAME_END]).collect();
        let wire = frame(kind::STREAM, &payload);
        assert!(!wire[1..wire.len() - 1].contains(&FRAME_END), "no end inside a message");
        let got = messages(&[frame(kind::STATUS, &[]), wire.clone(), vec![FRAME_END, FRAME_END]].concat());
        assert_eq!(got, [Message { kind: kind::STATUS, payload: vec![] }, Message { kind: kind::STREAM, payload }]);
        // Byte by byte, as they arrive.
        let mut reader = Reader::default();
        let one_by_one: Vec<Message> = wire.iter().flat_map(|b| reader.read(&[*b])).collect();
        assert_eq!(one_by_one, messages(&wire));
    }

    #[test]
    fn a_lost_byte_spoils_one_message_only() {
        let mut wire = [tokens(&[1, 2, 3]), tokens(&[4])].concat();
        wire.remove(0);
        let got = messages(&wire);
        assert_eq!(got.last().unwrap().words(), [4]);
    }

    #[test]
    fn the_port_reads_framed_bytes_as_the_host_wrote_them() {
        let mut bus = crate::soc::SocBus::new(crate::soc::SocConfig { program_words: 16, data_bits: 4, fifo_bits: 4, links: Default::default() }, &[], &[], &[], crate::soc::NoPeripheral);
        bus.send(&[set(9, 0xC0DB_C0DB), stop()].concat());
        let entries: Vec<u16> = bus.rx.iter().copied().collect();
        let start = (crate::soc::MARK_START << 8) as u16;
        let end = (crate::soc::MARK_END << 8) as u16;
        assert_eq!(entries, [start | kind::SET as u16, 9, 0xDB, 0xC0, 0xDB, 0xC0, end, start | kind::STOP as u16, end]);
    }
}
