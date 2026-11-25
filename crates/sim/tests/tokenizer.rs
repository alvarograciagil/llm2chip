//! The firmware's tokenizer, on the instruction-level CPU, against the reference library.

use netlist::chip::{chip_info, ChipConfig, ChipModel, DieEngine, GenerationSetup};
use netlist::image::build_image;
use netlist::rv32::Rv32Iss;
use netlist::sampler::{exponential_table, SamplerConfig, SamplerModel};
use netlist::soc::{FirmwareImage, SocBus, SocConfig};
use netlist::{CellConfig, DieConfig, Microprogram, Parameter};
use sim::tokenizer::TokenizerTables;
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use tokenizers::Tokenizer;

mod common;

fn build_firmware() -> Option<FirmwareImage> {
    common::firmware()
}

/// The tokenizers of the models on this machine, one per distinct repository.
fn local_tokenizers() -> Vec<PathBuf> {
    let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else { return Vec::new() };
    let mut found = Vec::new();
    // Other tokenizers to compare, each file in its own folder under this one.
    let extra = std::env::var_os("LLM2CHIP_TOKENIZERS").map(PathBuf::from);
    for root in [home.join(".mtplx/models"), home.join(".llm2chip/models")].into_iter().chain(extra) {
        for entry in std::fs::read_dir(root).into_iter().flatten().flatten() {
            found.push(entry.path().join("tokenizer.json"));
        }
    }
    for repo in std::fs::read_dir(home.join(".cache/huggingface/hub")).into_iter().flatten().flatten() {
        for snapshot in std::fs::read_dir(repo.path().join("snapshots")).into_iter().flatten().flatten() {
            found.push(snapshot.path().join("tokenizer.json"));
        }
    }
    // Only the kinds the chip handles: a cached model of another family, such as a BERT with
    // WordPiece, is not one to compare.
    found.retain(|p| std::fs::read_to_string(p).ok().and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok()).is_some_and(|json| matches!(json["model"]["type"].as_str(), Some("BPE" | "WordLevel"))));
    found
}

/// Keeps what the firmware sends to the die, and does nothing else.
struct Recorder(Rc<RefCell<Vec<u32>>>);

impl DieEngine for Recorder {
    fn run(&mut self, routine: usize, token: u32, _position: u32) {
        if routine == 0 {
            self.0.borrow_mut().push(token);
        }
    }
    fn cycles(&self, _routine: usize, _token: u32, _position: u32) -> u64 {
        0
    }
    fn read(&mut self, _address: u32) -> i64 {
        0
    }
}

const TEXTS: &[&str] = &[
    "The capital of France is",
    "¿Cuál es la capital de Francia? Responde en una frase.",
    "<|im_start|>user\nHola, ¿qué tal?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "don't we'll they've I'M John's 'quoted' 're",
    "fn main() {\n    let x = vec![1, 2, 3];\n\tprintln!(\"{:?}\", x); // comentario\r\n}\n\n\n",
    "3.14159 + 2,5e-10 = 1234567890 ٣٤ Ⅷ ½",
    "日本語のテキストと中文混合 한국어 Русский текст العربية",
    "emojis 😀👍🏽 🇪🇸 family 👨‍👩‍👧‍👦 end",
    "   leading and trailing   ",
    "tabs\t\tand  double  spaces \n \n trailing newline\n",
    "a<|endoftext|>b<|im_end|><|im_start|>",
    "punct!!! ...??? --- (paréntesis) [corchetes] {llaves} «comillas» “curly” — dash…",
    "ñandú acción über naïve café",
    "cafe\u{301} n\u{303} s\u{307}\u{323} \u{1100}\u{1161}\u{11A8}",
    "x",
];

/// Each style of tokenizer that models publish, in miniature.
const STYLES: &[(&str, &str)] = &[
    ("gpt2", r#"{"added_tokens":[{"id":90,"content":"<|end|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":91,"content":"[MASK]","single_word":false,"lstrip":true,"rstrip":true,"normalized":false,"special":false},{"id":92,"content":"palabra","single_word":true,"lstrip":false,"rstrip":false,"normalized":false,"special":false},{"id":93,"content":"NORMAL","single_word":false,"lstrip":false,"rstrip":false,"normalized":true,"special":false}],"normalizer":{"type":"NFC"},"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":VOCAB_GPT2,"merges":MERGES_GPT2}}"#),
    ("llama3", r#"{"added_tokens":[{"id":90,"content":"<|begin|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],"normalizer":null,"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]},"post_processor":{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"<|begin|>","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[],"special_tokens":{"<|begin|>":{"id":"<|begin|>","ids":[90],"tokens":["<|begin|>"]}}},"decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":true,"vocab":VOCAB_GPT2,"merges":MERGES_GPT2}}"#),
    ("llama2", r#"{"added_tokens":[{"id":0,"content":"<unk>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":1,"content":"<s>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":2,"content":"</s>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],"normalizer":{"type":"Sequence","normalizers":[{"type":"Prepend","prepend":"▁"},{"type":"Replace","pattern":{"String":" "},"content":"▁"}]},"pre_tokenizer":null,"post_processor":{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"<s>","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[],"special_tokens":{"<s>":{"id":"<s>","ids":[1],"tokens":["<s>"]}}},"decoder":{"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"},{"type":"Strip","content":" ","start":1,"stop":0}]},"model":{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":true,"byte_fallback":true,"ignore_merges":false,"vocab":VOCAB_META,"merges":MERGES_META}}"#),
    ("metaspace", r#"{"added_tokens":[{"id":0,"content":"<unk>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":2,"content":"</s>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],"normalizer":null,"pre_tokenizer":{"type":"Metaspace","replacement":"▁","prepend_scheme":"first","split":false},"post_processor":null,"decoder":{"type":"Metaspace","replacement":"▁","prepend_scheme":"first","split":false},"model":{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":VOCAB_META,"merges":MERGES_META}}"#),
    ("metaspace-split", r#"{"added_tokens":[{"id":0,"content":"<unk>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],"normalizer":{"type":"Sequence","normalizers":[{"type":"NFKC"},{"type":"Lowercase"},{"type":"Strip","strip_left":true,"strip_right":true}]},"pre_tokenizer":{"type":"Metaspace","replacement":"▁","add_prefix_space":true},"post_processor":null,"decoder":{"type":"Metaspace","replacement":"▁","add_prefix_space":true},"model":{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":VOCAB_META,"merges":MERGES_META}}"#),
    ("deepseek", r#"{"added_tokens":[{"id":90,"content":"<|fin|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":true,"special":true}],"normalizer":{"type":"Sequence","normalizers":[]},"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"[!-/:-~\u2018-\u201F\u3000-\u3002]+"},"behavior":"Isolated","invert":false},{"type":"Split","pattern":{"Regex":"\\s+$"},"behavior":"Isolated","invert":false},{"type":"Split","pattern":{"Regex":"[\u4E00-\u9FA5\u0800-\u4E00\uAC00-\uD7FF]+"},"behavior":"Isolated","invert":false},{"type":"Digits","individual_digits":true},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]},"post_processor":null,"decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":VOCAB_GPT2,"merges":MERGES_GPT2}}"#),
    ("splits", r###"{"added_tokens":[],"normalizer":{"type":"Sequence","normalizers":[{"type":"NFD"},{"type":"StripAccents"},{"type":"Replace","pattern":{"Regex":"[0-9]+"},"content":"#"}]},"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"String":","},"behavior":"MergedWithNext","invert":false},{"type":"Split","pattern":{"Regex":"\\."},"behavior":"MergedWithPrevious","invert":false},{"type":"Split","pattern":{"Regex":"[aeiou]"},"behavior":"Contiguous","invert":true},{"type":"Punctuation","behavior":"Isolated"},{"type":"WhitespaceSplit"}]},"post_processor":null,"decoder":{"type":"Fuse"},"model":{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":"##","end_of_word_suffix":"</w>","fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":VOCAB_SUBWORD,"merges":MERGES_SUBWORD}}"###),
    ("words", r#"{"added_tokens":[],"normalizer":{"type":"Lowercase"},"pre_tokenizer":{"type":"Whitespace"},"post_processor":{"type":"RobertaProcessing","sep":["</s>",2],"cls":["<s>",1],"trim_offsets":true,"add_prefix_space":false},"decoder":null,"model":{"type":"WordLevel","unk_token":"<unk>","vocab":{"<unk>":0,"<s>":1,"</s>":2,"hola":3,"mundo":4,"a":5,"1":6}}}"#),
];

/// A vocabulary of every byte-as-character plus a few joined pieces.
fn gpt2_vocabulary() -> (String, String, usize) {
    let printable = |b: u32| (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
    let mut next = 0;
    let symbols: Vec<String> = (0..256u32).map(|b| if printable(b) { char::from_u32(b).unwrap().to_string() } else { next += 1; char::from_u32(255 + next).unwrap().to_string() }).collect();
    // Some bytes are left out on purpose: the library then simply drops them.
    let mut vocabulary: Vec<String> = symbols.iter().filter(|s| !["À", "ø"].contains(&s.as_str())).cloned().collect();
    let mut merges = Vec::new();
    for (first, second) in [("h", "o"), ("ho", "l"), ("hol", "a"), ("o", "l"), ("Ġ", "h"), ("Ġh", "ol"), ("t", "h"), ("th", "e"), ("Ġ", "the"), ("e", "n"), ("a", "b"), ("ab", "a"), ("1", "2")] {
        vocabulary.push(format!("{first}{second}"));
        merges.push(format!("[{:?},{:?}]", first, second));
    }
    vocabulary.sort();
    vocabulary.dedup();
    let entries: Vec<String> = vocabulary.iter().enumerate().map(|(i, s)| format!("{}:{}", serde_json::to_string(s).unwrap(), i)).collect();
    (format!("{{{}}}", entries.join(",")), format!("[{}]", merges.join(",")), vocabulary.len())
}

fn meta_vocabulary() -> (String, String) {
    let mut entries = vec![r#""<unk>":0,"<s>":1,"</s>":2"#.to_string()];
    entries.extend((0..256).map(|b| format!("\"<0x{b:02X}>\":{}", b + 3)));
    for (next, piece) in (259..).zip(["▁", "h", "o", "l", "a", "▁h", "▁ho", "ol", "▁hol", "e", "n", "t", "m", "u", "é", "日", "▁a", "1", "2"]) {
        entries.push(format!("{}:{next}", serde_json::to_string(piece).unwrap()));
    }
    let merges = r#"[["▁","h"],["▁h","o"],["o","l"],["▁ho","l"],["▁","a"]]"#;
    (format!("{{{}}}", entries.join(",")), merges.to_string())
}

fn subword_vocabulary() -> (String, String) {
    let mut entries = vec![r#""<unk>":0"#.to_string()];
    for (next, piece) in (1..).zip(["h", "##o", "##l", "##a", "##a</w>", "a", "a</w>", "e", "##e</w>", "##e", "#", "##n", "##n</w>", "ho", "hol", "n</w>", "e</w>", "##l</w>", "##o</w>", "o</w>"]) {
        entries.push(format!("{}:{next}", serde_json::to_string(piece).unwrap()));
    }
    (format!("{{{}}}", entries.join(",")), r###"[["h","##o"],["ho","##l"]]"###.to_string())
}

fn style_json(template: &str) -> String {
    let (gpt2, gpt2_merges, size) = gpt2_vocabulary();
    let (meta, meta_merges) = meta_vocabulary();
    let (subword, subword_merges) = subword_vocabulary();
    // The library expects added tokens to be numbered right after the vocabulary.
    let numbered = (0..4).fold(template.to_string(), |text, k| text.replace(&format!("\"id\":{}", 90 + k), &format!("\"id\":{}", size + k)).replace(&format!("\"ids\":[{}]", 90 + k), &format!("\"ids\":[{}]", size + k)));
    numbered.replace("VOCAB_GPT2", &gpt2).replace("MERGES_GPT2", &gpt2_merges).replace("VOCAB_META", &meta).replace("MERGES_META", &meta_merges).replace("VOCAB_SUBWORD", &subword).replace("MERGES_SUBWORD", &subword_merges)
}

struct Random(u64);

impl Random {
    fn below(&mut self, limit: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 20) as usize % limit
    }
}

const PIECES: &[&str] = &[
    "a", "b", "e", "h", "o", "l", "t", "n", "m", "A", "H", "L", "Z", "hola", "HOLA", "the", " the", "aba", "palabra", "NORMAL", "normal", "<|end|>", "<|begin|>", "<|fin|>", "<s>", "</s>", "<unk>", "[MASK]",
    " ", " ", " ", "  ", "\n", "\n\n", "\r\n", "\t", "\u{B}", "\u{C}", "\u{85}", "\u{A0}", "\u{2003}", "\u{3000}", "\u{200B}",
    "0", "1", "2", "12", "123", "1234567", "٣", "Ⅷ", "½", "'", "'s", "'S", "'t", "'re", "'ll", "'LL", "'d", "'ſ", ",", ".", "!", "?", "-", "(", ")", "«", "—", "…", "\"", "#", "$", "_",
    "é", "e\u{301}", "ñ", "n\u{303}", "ü", "\u{301}", "\u{323}\u{307}", "s\u{323}\u{307}", "s\u{307}\u{323}", "Å", "K", "ſ", "ß", "İ", "Σ", "ς", "ﬁ", "㎏",
    "가", "각", "\u{1100}\u{1161}", "\u{1100}\u{1161}\u{11A8}", "한국어", "日本語", "中文", "Русский", "العربية", "😀", "👍🏽", "\u{200D}", "🇪🇸", "\u{FFFD}", "ÿ", "À", "ø",
];

fn random_text(random: &mut Random) -> String {
    (0..random.below(14)).map(|_| PIECES[random.below(PIECES.len())]).collect()
}

/// Compares encoding and decoding with the library on many awkward texts.
fn compare(name: &str, json: &str, rounds: usize, failures: &mut Vec<String>) {
    let reference: Tokenizer = json.parse().unwrap_or_else(|e| panic!("{name}: {e}"));
    let mut tables = TokenizerTables::from_json(json, 1 << 14, 1 << 15).unwrap_or_else(|e| panic!("{name}: {e}"));
    let mut random = Random(0x1F2E_3D4C_5B6A_7988);
    for round in 0..rounds {
        let text = random_text(&mut random);
        let add_special = round % 2 == 1;
        tables.set_add_special(add_special);
        let expected = reference.encode(text.as_str(), add_special).unwrap().get_ids().to_vec();
        let got = tables.encode(&text).unwrap();
        if got != expected {
            failures.push(format!("{name} {text:?} (especiales: {add_special})\n  chip:       {got:?}\n  referencia: {expected:?}"));
            continue;
        }
        let written = String::from_utf8_lossy(&tables.decode(&expected)).into_owned();
        let decoded = reference.decode(&expected, true).unwrap();
        if written != decoded {
            failures.push(format!("{name} {text:?}\n  chip writes:  {written:?}\n  reference:    {decoded:?}"));
        }
    }
}

#[test]
fn every_style_matches_the_library() {
    let mut failures = Vec::new();
    for (name, template) in STYLES {
        compare(name, &style_json(template), 1500, &mut failures);
    }
    let shown: Vec<_> = failures.iter().take(12).cloned().collect();
    assert!(failures.is_empty(), "{} diferencias\n{}", failures.len(), shown.join("\n"));
}

#[test]
fn the_tokenizers_on_this_machine_match_the_library() {
    let mut failures = Vec::new();
    for path in local_tokenizers() {
        let json = std::fs::read_to_string(&path).unwrap();
        compare(&path.display().to_string(), &json, 400, &mut failures);
    }
    let shown: Vec<_> = failures.iter().take(12).cloned().collect();
    assert!(failures.is_empty(), "{} diferencias\n{}", failures.len(), shown.join("\n"));
}

#[test]
fn what_cannot_be_done_is_refused_with_the_reason() {
    let refused = |json: &str, word: &str| {
        let error = TokenizerTables::from_json(json, 64, 64).unwrap_err();
        assert!(error.contains(word), "{error}");
    };
    refused(r#"{"model":{"type":"Unigram","vocab":[]}}"#, "Unigram");
    refused(r#"{"normalizer":{"type":"Precompiled"},"model":{"type":"BPE","vocab":{},"merges":[]}}"#, "Precompiled");
    refused(r#"{"pre_tokenizer":{"type":"UnicodeScripts"},"model":{"type":"BPE","vocab":{},"merges":[]}}"#, "UnicodeScripts");
    refused(r#"{"model":{"type":"BPE","dropout":0.1,"vocab":{},"merges":[]}}"#, "dropout");
    refused(r#"{"pre_tokenizer":{"type":"Split","pattern":{"Regex":"(a)\\1"},"behavior":"Isolated"},"model":{"type":"BPE","vocab":{},"merges":[]}}"#, "\\1");
    refused(r#"{"pre_tokenizer":{"type":"Split","pattern":{"Regex":"(a*)*"},"behavior":"Isolated"},"model":{"type":"BPE","vocab":{},"merges":[]}}"#, "empty");
}

/// Text as far as it stays within this many tokens and the bytes set aside for them.
fn filled_to_context(reference: &Tokenizer, seed: &str, context: u64, text_bytes: usize) -> String {
    let (mut low, mut high) = (1, text_bytes / seed.len() + 1);
    while low + 1 < high {
        let middle = (low + high) / 2;
        let text = seed.repeat(middle);
        if text.len() <= text_bytes && reference.encode(text.as_str(), false).unwrap().len() as u64 <= context {
            low = middle;
        } else {
            high = middle;
        }
    }
    seed.repeat(low)
}

const TEXTURES: &[&str] = &[
    "The quick brown fox, who wasn't late, jumped over 12 lazy dogs.\n",
    "¿Cuál es la capital de Francia? Responde en una frase, por favor. ",
    "日本語のテキストと中文混合のサンプル文章です。",
    "한국어 문장을 여러 번 반복합니다. 안녕하세요. ",
    "Русский текст для проверки памяти токенизатора. ",
    "fn main() { let x = vec![1, 2, 3]; println!(\"{:?}\", x); }\r\n",
    "😀👍🏽🇪🇸 👨\u{200d}👩\u{200d}👧 ",
    "1234567890 3.14159 ",
    "cafe\u{301} n\u{303} e\u{301}e\u{301} ",
];

#[test]
fn a_prompt_as_long_as_the_context_fits_in_the_memory_set_aside() {
    const CONTEXT: u64 = 1024;
    let (text_bytes, arena_words) = tokenizer_rt::memory_for_context(CONTEXT);
    let mut failures = Vec::new();
    for path in local_tokenizers() {
        let reference = Tokenizer::from_file(&path).unwrap();
        let json = std::fs::read_to_string(&path).unwrap();
        let tables = TokenizerTables::from_json(&json, text_bytes, arena_words).unwrap();
        for seed in TEXTURES {
            let text = filled_to_context(&reference, seed, CONTEXT, text_bytes as usize);
            match tables.encode(&text) {
                Ok(tokens) if tokens == reference.encode(text.as_str(), false).unwrap().get_ids() => {}
                other => failures.push(format!("{} with {seed:?} ({} bytes): {}", path.display(), text.len(), other.map_or_else(|e| e, |_| "different tokens".to_string()))),
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn the_tables_stay_within_what_the_estimator_assumes() {
    let mut worst = 0f64;
    for path in local_tokenizers() {
        let tables = TokenizerTables::from_file(&path, 256).unwrap();
        let vocabulary = Tokenizer::from_file(&path).unwrap().get_vocab_size(true) as f64;
        let bits = tables.table_words() as f64 * 32.0 / vocabulary;
        eprintln!("{:>6.0} bits per token: {}", bits, path.display());
        worst = worst.max(bits);
    }
    assert!(worst <= arch::TOKENIZER_BITS_PER_TOKEN, "{worst} bits per token");
}

#[test]
fn running_out_of_memory_is_reported_not_misread() {
    let json = style_json(STYLES[0].1);
    let tables = TokenizerTables::from_json(&json, 1 << 14, 64).unwrap();
    assert!(tables.encode(&"hola ".repeat(500)).is_err());
}

#[test]
fn the_firmware_tokenizes_like_the_reference() {
    let Some(firmware) = build_firmware() else {
        eprintln!("skipped: the firmware is missing");
        return;
    };
    let mut failures = Vec::new();
    for path in local_tokenizers() {
        firmware_matches(&firmware, &path, &mut failures);
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn firmware_matches(firmware: &FirmwareImage, path: &Path, failures: &mut Vec<String>) {
    let reference = Tokenizer::from_file(path).unwrap();
    let tables = TokenizerTables::from_file(path, 8192).unwrap().to_words();
    eprintln!("tables: {} words", tables.len());

    let cell = CellConfig { weight_bits: 8, activation_bits: 16, accumulator_bits: 32, mac_stages: 1, adder_stages: 1, addr_bits: 3, coding: Default::default(), exponent_bits: 0, multiplier: netlist::multiplier::MultiplierKind::Array, fixed_weights: false, ecc: false, skip_zeros: false };
    let die = DieConfig { cells: 1, cell, scale_bits: 16, reg_bits: 12, index_bits: 18, microcode_bits: 2, broadcast_stages: 1, function_bits: 12, function_select_bits: 3, features: Default::default() };
    let code = Microprogram { words: Vec::new(), entries: vec![0, 1], parameters: vec![Parameter { register: 1, token: 1, position: 0, constant: 0, window_length: 0, window_slot: 0, window: 0 }, Parameter { register: 2, token: 0, position: 1, constant: 0, window_length: 0, window_slot: 0, window: 0 }], group: (0, 0), marks: Vec::new() };
    let setup = GenerationSetup { max_new: 0, greedy: true, limit: 1, temperature: 256, seed: 1, raw_tokens: false, stop: Vec::new(), draft_depth: 0, draft_logits: 0, think: None, think_budget: u32::MAX };
    let info = chip_info(&die, &code, 1, 248_077, &setup);
    let image = build_image(&firmware.program, &firmware.data, &info, &tables, &[]).unwrap();
    let soc = SocConfig { program_words: netlist::soc::PROGRAM_WORDS, data_bits: (tables.len() + (1 << 14)).next_power_of_two().ilog2(), fifo_bits: 6, links: Default::default() };
    let _ = ChipConfig { soc, die, sampler: SamplerConfig { logit_bits: 16, index_bits: 18, entries: 4, lanes: 1 } };

    // A prompt of several thousand tokens, which some models encode as a single piece.
    let long = "The quick brown fox, who wasn't late, jumped over 12 lazy dogs.\n".repeat(320);
    for text in TEXTS.iter().copied().chain([long.as_str()]) {
        let expected = reference.encode(text, false).unwrap().get_ids().to_vec();
        let tokens = Rc::new(RefCell::new(Vec::new()));
        let input = netlist::port::text(text);
        let sampler = SamplerModel { config: SamplerConfig { logit_bits: 16, index_bits: 18, entries: 4, lanes: 1 }, rng: 1, table: exponential_table() };
        let model = ChipModel::new(Recorder(tokens.clone()), code.clone(), sampler);
        let mut bus = SocBus::booted_with(soc, &image, &input, model).unwrap();
        let mut cpu = Rv32Iss::default();
        while !cpu.halted {
            cpu.step(&mut bus);
            assert!(cpu.cycle < 2_000_000_000, "the firmware does not finish with {text:?}");
        }
        let got = tokens.borrow().clone();
        eprintln!("{:>9} cycles, {:>3} tokens: {:?}", cpu.cycle, got.len(), &text[..text.len().min(40)]);
        if got != expected {
            failures.push(format!("{} {text:?}\n  firmware:   {got:?}\n  referencia: {expected:?}", path.display()));
        }
    }
}
