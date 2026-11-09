//! Compiles a `tokenizer.json` into the tables the chip's firmware runs from. What
//! the tables cannot express is refused here, with the reason, rather than giving a
//! model different tokens from those it was trained on.

use regex_syntax::hir::{Class, HirKind};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use tokenizer_rt::*;
use unicode_normalization_alignments::char as unicode;

const GPT2_PATTERN: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
const MAX_PROGRAM_WORDS: usize = 1 << 16;
const LAST_CHARACTER: u32 = 0x10FFFF;

/// Hash tables are kept two thirds full, which costs a probe or two per lookup.
fn hash_slots(entries: usize) -> usize {
    entries + entries / 2 + 1
}

/// The tables as the words of the firmware's memory.
#[derive(Clone, Debug, Default)]
pub struct TokenizerTables {
    words: Vec<u32>,
}

fn refuse<T>(what: &str) -> Result<T, String> {
    Err(format!("el tokenizador del chip no admite {what}"))
}

fn characters() -> impl Iterator<Item = char> {
    (0..=LAST_CHARACTER).filter_map(char::from_u32)
}

fn ranges_where(accepts: impl Fn(char) -> bool) -> Vec<(u32, u32)> {
    let mut ranges: Vec<(u32, u32)> = Vec::new();
    for c in characters().filter(|c| accepts(*c)) {
        let code = c as u32;
        match ranges.last_mut() {
            Some((_, end)) if *end + 1 == code => *end = code,
            _ => ranges.push((code, code)),
        }
    }
    ranges
}

#[derive(Clone, Debug)]
enum Node {
    Char(u32),
    Class(u32),
    Seq(Vec<Node>),
    Alt(Vec<Node>),
    Repeat { node: Box<Node>, min: u32, max: Option<u32>, greedy: bool },
    Look { node: Box<Node>, negative: bool },
    Assert(u32),
}

impl Node {
    fn nullable(&self) -> bool {
        match self {
            Node::Char(_) | Node::Class(_) => false,
            Node::Seq(list) => list.iter().all(Node::nullable),
            Node::Alt(list) => list.iter().any(Node::nullable),
            Node::Repeat { node, min, .. } => *min == 0 || node.nullable(),
            Node::Look { .. } | Node::Assert(_) => true,
        }
    }

    fn emit(&self, code: &mut Vec<u32>) -> Result<(), String> {
        if code.len() > MAX_PROGRAM_WORDS {
            return refuse("una expresión regular tan larga");
        }
        match self {
            Node::Char(value) => code.extend([R_CHAR, *value]),
            Node::Class(index) => code.extend([R_CLASS, *index]),
            Node::Seq(list) => list.iter().try_for_each(|n| n.emit(code))?,
            Node::Alt(list) => {
                let mut jumps = Vec::new();
                for (index, node) in list.iter().enumerate() {
                    if index + 1 == list.len() {
                        node.emit(code)?;
                        break;
                    }
                    let split = code.len();
                    code.extend([R_SPLIT, split as u32 + 3, 0]);
                    node.emit(code)?;
                    jumps.push(code.len() + 1);
                    code.extend([R_JUMP, 0]);
                    code[split + 2] = code.len() as u32;
                }
                let end = code.len() as u32;
                jumps.into_iter().for_each(|at| code[at] = end);
            }
            Node::Repeat { node, min, max, greedy } => {
                for _ in 0..*min {
                    node.emit(code)?;
                }
                let order = |body: u32, exit: u32| if *greedy { [body, exit] } else { [exit, body] };
                match max {
                    None => {
                        if node.nullable() {
                            return refuse("repetir sin límite algo que puede ser vacío");
                        }
                        let split = code.len();
                        code.extend([R_SPLIT, 0, 0]);
                        node.emit(code)?;
                        code.extend([R_JUMP, split as u32]);
                        let [first, second] = order(split as u32 + 3, code.len() as u32);
                        code[split + 1] = first;
                        code[split + 2] = second;
                    }
                    Some(max) => {
                        let mut splits = Vec::new();
                        for _ in *min..*max {
                            splits.push(code.len());
                            code.extend([R_SPLIT, 0, 0]);
                            node.emit(code)?;
                        }
                        let end = code.len() as u32;
                        for split in splits {
                            let [first, second] = order(split as u32 + 3, end);
                            code[split + 1] = first;
                            code[split + 2] = second;
                        }
                    }
                }
            }
            Node::Look { node, negative } => {
                let start = code.len();
                code.extend([R_LOOK, *negative as u32, 0]);
                node.emit(code)?;
                code.push(R_MATCH);
                code[start + 2] = code.len() as u32;
            }
            Node::Assert(operation) => code.push(*operation),
        }
        Ok(())
    }
}

struct Builder {
    words: Vec<u32>,
    blob: Vec<u8>,
    classes: Vec<Vec<(u32, u32)>>,
    class_numbers: HashMap<Vec<(u32, u32)>, u32>,
    forms: HashMap<String, u32>,
    lowercase: Option<(u32, u32)>,
}

/// Oniguruma's syntax as far as tokenizers use it. Character classes go to
/// `regex-syntax`, which supplies the Unicode tables.
struct Parser<'a> {
    text: Vec<char>,
    at: usize,
    case_insensitive: bool,
    builder: &'a mut Builder,
}

impl Parser<'_> {
    fn peek(&self) -> Option<char> {
        self.text.get(self.at).copied()
    }

    fn eat(&mut self, c: char) -> bool {
        let found = self.peek() == Some(c);
        self.at += found as usize;
        found
    }

    fn looking_at(&self, prefix: &str) -> bool {
        let wanted: Vec<char> = prefix.chars().collect();
        self.text[self.at..].starts_with(&wanted)
    }

    fn alternatives(&mut self) -> Result<Node, String> {
        let mut options = vec![self.sequence()?];
        while self.eat('|') {
            options.push(self.sequence()?);
        }
        Ok(if options.len() == 1 { options.remove(0) } else { Node::Alt(options) })
    }

    fn sequence(&mut self) -> Result<Node, String> {
        let mut items = Vec::new();
        while let Some(c) = self.peek() {
            if c == '|' || c == ')' {
                break;
            }
            let atom = self.atom()?;
            items.push(self.quantified(atom)?);
        }
        Ok(Node::Seq(items))
    }

    fn quantified(&mut self, atom: Node) -> Result<Node, String> {
        let (min, max) = match self.peek() {
            Some('?') => (0, Some(1)),
            Some('*') => (0, None),
            Some('+') => (1, None),
            Some('{') => match self.counted() {
                Some(bounds) => bounds,
                None => return Ok(atom),
            },
            _ => return Ok(atom),
        };
        self.at += 1;
        let greedy = !self.eat('?');
        if self.peek() == Some('+') {
            return refuse("cuantificadores posesivos en sus expresiones regulares");
        }
        if matches!(self.peek(), Some('*' | '{')) && self.counted_or_star() {
            return refuse("cuantificadores encadenados en sus expresiones regulares");
        }
        Ok(Node::Repeat { node: Box::new(atom), min, max, greedy })
    }

    fn counted_or_star(&self) -> bool {
        self.peek() == Some('*') || self.clone_counted().is_some()
    }

    fn clone_counted(&self) -> Option<(usize, u32, Option<u32>)> {
        let mut at = self.at + 1;
        let number = |at: &mut usize| -> Option<u32> {
            let begin = *at;
            while self.text.get(*at).is_some_and(char::is_ascii_digit) {
                *at += 1;
            }
            (begin != *at).then(|| self.text[begin..*at].iter().collect::<String>().parse().ok())?
        };
        let min = number(&mut at)?;
        let max = if self.text.get(at) == Some(&',') {
            at += 1;
            if self.text.get(at) == Some(&'}') { None } else { Some(number(&mut at)?) }
        } else {
            Some(min)
        };
        (self.text.get(at) == Some(&'}')).then_some((at, min, max))
    }

    /// Reads `{n}`, `{n,}` or `{n,m}`, leaving the closing brace to the caller.
    fn counted(&mut self) -> Option<(u32, Option<u32>)> {
        let (end, min, max) = self.clone_counted()?;
        self.at = end;
        Some((min, max))
    }

    fn group(&mut self) -> Result<Node, String> {
        let saved = self.case_insensitive;
        let mut close = true;
        let node = if self.looking_at("?:") {
            self.at += 2;
            self.alternatives()?
        } else if self.looking_at("?=") || self.looking_at("?!") {
            let negative = self.text[self.at + 1] == '!';
            self.at += 2;
            let inner = self.alternatives()?;
            if contains_look(&inner) {
                return refuse("miradas anidadas en sus expresiones regulares");
            }
            Node::Look { node: Box::new(inner), negative }
        } else if self.eat('?') {
            let mut enable = true;
            loop {
                match self.peek() {
                    Some('i') => self.case_insensitive = enable,
                    Some('-') => enable = false,
                    Some(':' | ')') => break,
                    _ => return refuse("esa construcción de sus expresiones regulares"),
                }
                self.at += 1;
            }
            if self.eat(')') {
                // Applies to the rest of the enclosing group.
                close = false;
                let rest = self.alternatives()?;
                if self.peek() != Some(')') {
                    self.case_insensitive = saved;
                }
                rest
            } else {
                self.at += 1;
                self.alternatives()?
            }
        } else {
            self.alternatives()?
        };
        if close {
            if !self.eat(')') {
                return Err("expresión regular sin cerrar".into());
            }
            self.case_insensitive = saved;
        }
        Ok(node)
    }

    fn set(&mut self, atom: &str) -> Result<Node, String> {
        let parsed = regex_syntax::ParserBuilder::new().case_insensitive(self.case_insensitive).build().parse(atom).map_err(|e| format!("expresión regular no admitida: {e}"))?;
        match parsed.kind() {
            HirKind::Class(Class::Unicode(class)) => Ok(Node::Class(self.builder.class(class.iter().map(|r| (r.start() as u32, r.end() as u32)).collect()))),
            HirKind::Literal(literal) => {
                let text = std::str::from_utf8(&literal.0).map_err(|e| e.to_string())?;
                let mut letters = text.chars();
                match (letters.next(), letters.next()) {
                    (Some(c), None) => Ok(Node::Char(c as u32)),
                    _ => Err(format!("expresión regular no admitida: {atom}")),
                }
            }
            _ => Err(format!("expresión regular no admitida: {atom}")),
        }
    }

    fn literal(&mut self, c: char) -> Result<Node, String> {
        if self.case_insensitive {
            self.set(&regex_syntax::escape(&c.to_string()))
        } else {
            Ok(Node::Char(c as u32))
        }
    }

    fn bracket(&mut self) -> Result<Node, String> {
        let start = self.at - 1;
        let mut depth = 1;
        if self.peek() == Some('^') {
            self.at += 1;
        }
        if self.peek() == Some(']') {
            self.at += 1;
        }
        while depth > 0 {
            match self.peek() {
                None => return Err("clase de caracteres sin cerrar".into()),
                Some('\\') => self.at += 2,
                Some('[') => {
                    depth += 1;
                    self.at += 1;
                }
                Some(']') => {
                    depth -= 1;
                    self.at += 1;
                }
                Some(_) => self.at += 1,
            }
        }
        let atom: String = self.text[start..self.at].iter().collect();
        self.set(&atom)
    }

    fn hexadecimal(&mut self, digits: usize) -> Result<char, String> {
        let end = (self.at + digits).min(self.text.len());
        let code: String = self.text[self.at..end].iter().collect();
        self.at = end;
        u32::from_str_radix(&code, 16).ok().and_then(char::from_u32).ok_or_else(|| format!("escape no válido: {code}"))
    }

    fn escape(&mut self) -> Result<Node, String> {
        let c = self.peek().ok_or("expresión regular que acaba en \\")?;
        self.at += 1;
        match c {
            's' | 'S' | 'd' | 'D' | 'w' | 'W' => self.set(&format!("\\{c}")),
            'p' | 'P' => {
                let start = self.at;
                if self.eat('{') {
                    while self.peek().is_some_and(|c| c != '}') {
                        self.at += 1;
                    }
                    self.at += 1;
                } else {
                    self.at += 1;
                }
                let name: String = self.text[start..self.at.min(self.text.len())].iter().collect();
                self.set(&format!("\\{c}{name}"))
            }
            'n' => self.literal('\n'),
            'r' => self.literal('\r'),
            't' => self.literal('\t'),
            'f' => self.literal('\x0C'),
            'v' => self.literal('\x0B'),
            'a' => self.literal('\x07'),
            'e' => self.literal('\x1B'),
            'x' if self.eat('{') => {
                let start = self.at;
                while self.peek().is_some_and(|c| c != '}') {
                    self.at += 1;
                }
                let code: String = self.text[start..self.at].iter().collect();
                self.at += 1;
                let value = u32::from_str_radix(&code, 16).ok().and_then(char::from_u32).ok_or("escape no válido")?;
                self.literal(value)
            }
            'x' => {
                let value = self.hexadecimal(2)?;
                self.literal(value)
            }
            'u' => {
                let value = self.hexadecimal(4)?;
                self.literal(value)
            }
            'A' => Ok(Node::Assert(R_TEXT_START)),
            'z' => Ok(Node::Assert(R_TEXT_END)),
            'Z' => Ok(Node::Assert(R_TEXT_END_NEWLINE)),
            c if c.is_ascii_alphanumeric() => refuse(&format!("\\{c} en sus expresiones regulares")),
            c => self.literal(c),
        }
    }

    fn atom(&mut self) -> Result<Node, String> {
        let c = self.peek().ok_or("expresión regular incompleta")?;
        self.at += 1;
        match c {
            '(' => self.group(),
            '[' => self.bracket(),
            '\\' => self.escape(),
            '.' => self.set("."),
            '^' => Ok(Node::Assert(R_LINE_START)),
            '$' => Ok(Node::Assert(R_LINE_END)),
            c => self.literal(c),
        }
    }
}

fn contains_look(node: &Node) -> bool {
    match node {
        Node::Look { .. } => true,
        Node::Seq(list) | Node::Alt(list) => list.iter().any(contains_look),
        Node::Repeat { node, .. } => contains_look(node),
        _ => false,
    }
}

fn split_mode(value: &Value) -> Result<u32, String> {
    match value.as_str() {
        Some("Removed") => Ok(SPLIT_REMOVED),
        Some("Isolated") => Ok(SPLIT_ISOLATED),
        Some("MergedWithPrevious") => Ok(SPLIT_PREVIOUS),
        Some("MergedWithNext") => Ok(SPLIT_NEXT),
        Some("Contiguous") => Ok(SPLIT_CONTIGUOUS),
        _ => refuse(&format!("el modo de corte {value}")),
    }
}

type Step = [u32; STEP_WORDS];

fn step(operation: u32, a: u32, b: u32, c: u32) -> Step {
    [operation, a, b, c]
}

impl Builder {
    fn place(&mut self, section: &[u32]) -> u32 {
        let at = self.words.len() as u32;
        self.words.extend_from_slice(section);
        at
    }

    fn string(&mut self, bytes: &[u8]) -> (u32, u32) {
        let at = self.blob.len() as u32;
        self.blob.extend_from_slice(bytes);
        (at, bytes.len() as u32)
    }

    fn class(&mut self, ranges: Vec<(u32, u32)>) -> u32 {
        if let Some(number) = self.class_numbers.get(&ranges) {
            return *number;
        }
        let number = self.classes.len() as u32;
        self.classes.push(ranges.clone());
        self.class_numbers.insert(ranges, number);
        number
    }

    fn program(&mut self, node: &Node) -> Result<u32, String> {
        let mut code = Vec::new();
        node.emit(&mut code)?;
        code.push(R_MATCH);
        Ok(self.place(&code))
    }

    fn regex(&mut self, pattern: &str) -> Result<u32, String> {
        let mut parser = Parser { text: pattern.chars().collect(), at: 0, case_insensitive: false, builder: self };
        let node = parser.alternatives()?;
        if parser.at != parser.text.len() {
            return Err(format!("expresión regular no admitida: {pattern}"));
        }
        self.program(&node)
    }

    fn literal(&mut self, text: &str) -> Result<u32, String> {
        self.program(&Node::Seq(text.chars().map(|c| Node::Char(c as u32)).collect()))
    }

    fn class_program(&mut self, accepts: impl Fn(char) -> bool) -> Result<u32, String> {
        let class = self.class(ranges_where(accepts));
        self.program(&Node::Class(class))
    }

    /// The pattern of a `Split` or `Replace`, in any of the forms the file can use.
    fn pattern(&mut self, value: &Value) -> Result<u32, String> {
        if let Some(text) = value.as_str().or_else(|| value["String"].as_str()) {
            if text.is_empty() {
                return refuse("un patrón vacío");
            }
            return self.literal(text);
        }
        match value["Regex"].as_str() {
            Some(pattern) => self.regex(pattern),
            None => refuse(&format!("el patrón {value}")),
        }
    }

    fn normal_form(&mut self, name: &str) -> u32 {
        if let Some(at) = self.forms.get(name) {
            return *at;
        }
        let compatible = name.starts_with("NFK");
        let mut sequences = Vec::new();
        let mut decompositions = Vec::new();
        let mut classes: Vec<(u32, u32, u32)> = Vec::new();
        let mut compositions = Vec::new();
        let hangul = 0xAC00..0xAC00 + 11172u32;
        for c in characters() {
            let code = c as u32;
            let class = unicode::canonical_combining_class(c) as u32;
            if class != 0 {
                match classes.last_mut() {
                    Some((_, end, same)) if *end + 1 == code && *same == class => *end = code,
                    _ => classes.push((code, code, class)),
                }
            }
            if hangul.contains(&code) {
                continue;
            }
            let mut parts = Vec::new();
            if compatible {
                unicode::decompose_compatible(c, |d| parts.push(d as u32));
            } else {
                unicode::decompose_canonical(c, |d| parts.push(d as u32));
            }
            if parts != [code] {
                decompositions.push((code, sequences.len() as u32, parts.len() as u32));
                sequences.extend_from_slice(&parts);
            }
        }
        let (firsts, seconds) = composition_parts();
        for first in &firsts {
            for second in &seconds {
                if let Some(result) = unicode::compose(*first, *second) {
                    compositions.push((*first as u32, *second as u32, result as u32));
                }
            }
        }
        compositions.sort_unstable();
        let sequence_at = self.place(&sequences);
        let table: Vec<u32> = decompositions.iter().flat_map(|(code, at, length)| [*code, sequence_at + at, *length]).collect();
        let decomposition_at = self.place(&table);
        let table: Vec<u32> = classes.iter().flat_map(|(a, b, c)| [*a, *b, *c]).collect();
        let class_at = self.place(&table);
        let table: Vec<u32> = compositions.iter().flat_map(|(a, b, c)| [*a, *b, *c]).collect();
        let composition_at = self.place(&table);
        let at = self.place(&[decomposition_at, decompositions.len() as u32, class_at, classes.len() as u32, composition_at, compositions.len() as u32, name.ends_with('C') as u32]);
        self.forms.insert(name.to_string(), at);
        at
    }

    fn lowercase_table(&mut self) -> (u32, u32) {
        if let Some(table) = self.lowercase {
            return table;
        }
        let mut table = Vec::new();
        for c in characters() {
            let lower: Vec<u32> = c.to_lowercase().map(|l| l as u32).collect();
            if lower != [c as u32] {
                table.extend([c as u32, lower[0], lower.get(1).copied().unwrap_or(0)]);
            }
        }
        let result = (self.place(&table), table.len() as u32 / 3);
        self.lowercase = Some(result);
        result
    }

    fn normalizer(&mut self, value: &Value, steps: &mut Vec<Step>) -> Result<(), String> {
        if value.is_null() {
            return Ok(());
        }
        let kind = value["type"].as_str().unwrap_or("");
        match kind {
            "Sequence" => {
                for item in value["normalizers"].as_array().ok_or("normalizador sin lista")? {
                    self.normalizer(item, steps)?;
                }
            }
            "NFC" | "NFD" | "NFKC" | "NFKD" => {
                let form = self.normal_form(kind);
                steps.push(step(S_NORMAL_FORM, form, 0, 0));
            }
            "Lowercase" => {
                let (table, count) = self.lowercase_table();
                steps.push(step(S_LOWERCASE, table, count, 0));
            }
            "Strip" => {
                let class = self.class(ranges_where(char::is_whitespace));
                steps.push(step(S_STRIP, value["strip_left"].as_bool().unwrap_or(true) as u32, value["strip_right"].as_bool().unwrap_or(true) as u32, class));
            }
            "StripAccents" => {
                let class = self.class(ranges_where(unicode::is_combining_mark));
                steps.push(step(S_REMOVE_CLASS, class, 0, 0));
            }
            "Replace" => {
                let pattern = self.pattern(&value["pattern"])?;
                let (at, length) = self.string(value["content"].as_str().unwrap_or("").as_bytes());
                steps.push(step(S_REPLACE, pattern, at, length));
            }
            "Prepend" => {
                let (at, length) = self.string(value["prepend"].as_str().unwrap_or("").as_bytes());
                steps.push(step(S_PREPEND, at, length, 0));
            }
            "ByteLevel" => steps.push(step(S_BYTES, 0, 0, 0)),
            other => return refuse(&format!("el normalizador «{other}»")),
        }
        Ok(())
    }

    fn pre_tokenizer(&mut self, value: &Value, steps: &mut Vec<Step>) -> Result<(), String> {
        if value.is_null() {
            return Ok(());
        }
        let kind = value["type"].as_str().unwrap_or("");
        match kind {
            "Sequence" => {
                for item in value["pretokenizers"].as_array().ok_or("pre-tokenizador sin lista")? {
                    self.pre_tokenizer(item, steps)?;
                }
            }
            "ByteLevel" => {
                if value["use_regex"].as_bool().unwrap_or(true) {
                    let pattern = self.regex(GPT2_PATTERN)?;
                    steps.push(step(S_SPLIT, pattern, SPLIT_ISOLATED, 0));
                }
                steps.push(step(S_BYTES, 0, 0, 0));
            }
            "Split" => {
                let pattern = self.pattern(&value["pattern"])?;
                let invert = if value["invert"].as_bool().unwrap_or(false) { SPLIT_INVERT } else { 0 };
                steps.push(step(S_SPLIT, pattern, split_mode(&value["behavior"])? | invert, 0));
            }
            "Digits" => {
                let pattern = self.class_program(char::is_numeric)?;
                let mode = if value["individual_digits"].as_bool().unwrap_or(false) { SPLIT_ISOLATED } else { SPLIT_CONTIGUOUS };
                steps.push(step(S_SPLIT, pattern, mode, 0));
            }
            "Whitespace" => {
                let pattern = self.regex(r"\w+|[^\w\s]+")?;
                steps.push(step(S_SPLIT, pattern, SPLIT_REMOVED | SPLIT_INVERT, 0));
            }
            "WhitespaceSplit" => {
                let pattern = self.class_program(char::is_whitespace)?;
                steps.push(step(S_SPLIT, pattern, SPLIT_REMOVED, 0));
            }
            "CharDelimiterSplit" => {
                let delimiter = value["delimiter"].as_str().and_then(|d| d.chars().next()).ok_or("delimitador vacío")?;
                let pattern = self.program(&Node::Char(delimiter as u32))?;
                steps.push(step(S_SPLIT, pattern, SPLIT_REMOVED, 0));
            }
            "Punctuation" => {
                let mode = if value["behavior"].is_null() { SPLIT_ISOLATED } else { split_mode(&value["behavior"])? };
                let pattern = self.punctuation()?;
                steps.push(step(S_SPLIT, pattern, mode, 0));
            }
            "BertPreTokenizer" => {
                let spaces = self.class_program(char::is_whitespace)?;
                steps.push(step(S_SPLIT, spaces, SPLIT_REMOVED, 0));
                let pattern = self.punctuation()?;
                steps.push(step(S_SPLIT, pattern, SPLIT_ISOLATED, 0));
            }
            "Metaspace" => {
                let replacement = value["replacement"].as_str().unwrap_or("▁");
                let space = self.literal(" ")?;
                let (at, length) = self.string(replacement.as_bytes());
                steps.push(step(S_REPLACE, space, at, length));
                let scheme = match (value["prepend_scheme"].as_str(), value["add_prefix_space"].as_bool()) {
                    (Some("first"), _) => META_FIRST,
                    (Some("never"), _) | (None, Some(false)) => META_NEVER,
                    _ => META_ALWAYS,
                };
                if scheme != META_NEVER {
                    steps.push(step(S_METASPACE, at, length, scheme));
                }
                if value["split"].as_bool().unwrap_or(true) {
                    let mark = self.literal(replacement)?;
                    steps.push(step(S_SPLIT, mark, SPLIT_NEXT, 0));
                }
            }
            other => return refuse(&format!("el pre-tokenizador «{other}»")),
        }
        Ok(())
    }

    fn punctuation(&mut self) -> Result<u32, String> {
        let parsed = regex_syntax::parse(r"[\p{P}!-/:-@\[-`{-~]").map_err(|e| e.to_string())?;
        let HirKind::Class(Class::Unicode(class)) = parsed.kind() else { return Err("clase de puntuación".into()) };
        let class = self.class(class.iter().map(|r| (r.start() as u32, r.end() as u32)).collect());
        self.program(&Node::Class(class))
    }

    fn class_of(&mut self, pattern: &str) -> Result<u32, String> {
        let parsed = regex_syntax::parse(pattern).map_err(|e| e.to_string())?;
        let HirKind::Class(Class::Unicode(class)) = parsed.kind() else { return Err("clase".into()) };
        Ok(self.class(class.iter().map(|r| (r.start() as u32, r.end() as u32)).collect()))
    }

    fn special_tokens(&self, processor: &Value, before: &mut Vec<u32>, after: &mut Vec<u32>) -> Result<(), String> {
        let id_of = |pair: &Value| pair[1].as_u64().map(|v| v as u32).ok_or("procesador sin identificador");
        match processor["type"].as_str() {
            None | Some("ByteLevel") => {}
            Some("Sequence") => {
                for item in processor["processors"].as_array().ok_or("procesador sin lista")? {
                    self.special_tokens(item, before, after)?;
                }
            }
            Some("BertProcessing" | "RobertaProcessing") => {
                before.push(id_of(&processor["cls"])?);
                after.push(id_of(&processor["sep"])?);
            }
            Some("TemplateProcessing") => {
                let mut seen = false;
                for item in processor["single"].as_array().ok_or("plantilla sin lista")? {
                    if item.get("Sequence").is_some() {
                        seen = true;
                        continue;
                    }
                    let name = item["SpecialToken"]["id"].as_str().ok_or("plantilla no válida")?;
                    let ids = processor["special_tokens"][name]["ids"].as_array().ok_or("plantilla sin el token")?;
                    let target = if seen { &mut *after } else { &mut *before };
                    target.extend(ids.iter().filter_map(Value::as_u64).map(|v| v as u32));
                }
            }
            Some(other) => return refuse(&format!("el post-procesador «{other}»")),
        }
        Ok(())
    }
}

/// The characters that can come first and second in a composition; Hangul is arithmetic.
fn composition_parts() -> (Vec<char>, Vec<char>) {
    let (mut firsts, mut seconds): (Vec<char>, Vec<char>) = (Vec::new(), Vec::new());
    for c in characters().filter(|c| !(0xAC00..0xAC00 + 11172).contains(&(*c as u32))) {
        let mut parts = Vec::new();
        unicode::decompose_canonical(c, |d| parts.push(d));
        if parts.len() >= 2 {
            // Each stage of a longer decomposition composes in turn, so composites can come first.
            firsts.push(c);
            firsts.extend(parts.iter().take(parts.len() - 1));
            seconds.extend(parts.iter().skip(1));
        }
    }
    firsts.sort_unstable();
    firsts.dedup();
    seconds.sort_unstable();
    seconds.dedup();
    (firsts, seconds)
}

/// Each token's output bytes, and how the start of the output is cleaned.
fn decoded(tokens: &mut [Vec<u8>], decoder: &Value, fused: &mut bool, leading: &mut u32, first_token: &mut bool) -> Result<(), String> {
    let gpt2: HashMap<char, u8> = (0..=255u8).map(|b| (char::from_u32(byte_character(b)).expect("carácter"), b)).collect();
    match decoder["type"].as_str() {
        None => {
            tokens.iter_mut().for_each(|t| t.insert(0, b' '));
            *leading += 1;
        }
        Some("Sequence") => {
            for item in decoder["decoders"].as_array().ok_or("decodificador sin lista")? {
                decoded(tokens, item, fused, leading, first_token)?;
            }
        }
        Some("ByteLevel") => {
            for token in tokens.iter_mut() {
                let text = String::from_utf8_lossy(token).into_owned();
                if let Some(bytes) = text.chars().map(|c| gpt2.get(&c).copied()).collect::<Option<Vec<u8>>>() {
                    *token = bytes;
                }
            }
            *fused = true;
        }
        Some("Fuse") => *fused = true,
        Some("Metaspace") => {
            let replacement = decoder["replacement"].as_str().unwrap_or("▁");
            let keeps = decoder["prepend_scheme"].as_str() == Some("never") || (decoder["prepend_scheme"].is_null() && decoder["add_prefix_space"].as_bool() == Some(false));
            tokens.iter_mut().for_each(|t| *t = String::from_utf8_lossy(t).replace(replacement, " ").into_bytes());
            *first_token |= !keeps;
        }
        Some("Replace") => {
            let content = decoder["content"].as_str().unwrap_or("");
            let Some(pattern) = decoder["pattern"].as_str().or_else(|| decoder["pattern"]["String"].as_str()) else {
                return refuse("un decodificador Replace con expresión regular");
            };
            tokens.iter_mut().for_each(|t| *t = String::from_utf8_lossy(t).replace(pattern, content).into_bytes());
        }
        Some("ByteFallback") => {
            for token in tokens.iter_mut() {
                if let Some(hex) = std::str::from_utf8(token).ok().and_then(|t| t.strip_prefix("<0x")).and_then(|t| t.strip_suffix('>')).filter(|h| h.len() == 2) {
                    if let Ok(byte) = u8::from_str_radix(hex, 16) {
                        *token = vec![byte];
                    }
                }
            }
        }
        Some("Strip") => {
            let content = decoder["content"].as_str().and_then(|c| c.chars().next()).unwrap_or(' ');
            let (start, stop) = (decoder["start"].as_u64().unwrap_or(0) as usize, decoder["stop"].as_u64().unwrap_or(0) as usize);
            if *fused {
                if content != ' ' || stop != 0 {
                    return refuse("recortar el final del texto ya unido");
                }
                *leading += start as u32;
            } else {
                for token in tokens.iter_mut() {
                    let letters: Vec<char> = String::from_utf8_lossy(token).chars().collect();
                    let from = letters.iter().take(start).take_while(|c| **c == content).count();
                    let to = letters.len() - letters[from..].iter().rev().take(stop).take_while(|c| **c == content).count();
                    *token = letters[from..to].iter().collect::<String>().into_bytes();
                }
            }
        }
        Some(other) => return refuse(&format!("el decodificador «{other}»")),
    }
    Ok(())
}

impl TokenizerTables {
    /// Compiles a model's tokenizer for a chip with this much context.
    pub fn from_file(path: &Path, context_tokens: u64) -> Result<Self, String> {
        let (text_bytes, arena_words) = memory_for_context(context_tokens);
        Self::from_json(&std::fs::read_to_string(path).map_err(|e| e.to_string())?, text_bytes, arena_words)
    }

    /// Compiles the text of a `tokenizer.json`, with room for that much text and work.
    pub fn from_json(text: &str, text_bytes: u32, arena_words: u32) -> Result<Self, String> {
        let json: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
        let model = &json["model"];
        let kind = model["type"].as_str().unwrap_or("");
        if kind != "BPE" && kind != "WordLevel" {
            return refuse(&format!("el modelo «{kind}»"));
        }
        if !model["dropout"].is_null() {
            return refuse("un modelo con dropout, que no es determinista");
        }
        let vocabulary = model["vocab"].as_object().ok_or("tokenizer.json sin vocabulario")?;
        let ids: HashMap<&str, u32> = vocabulary.iter().map(|(k, v)| (k.as_str(), v.as_u64().unwrap_or(0) as u32)).collect();

        let mut b = Builder { words: vec![0; HEADER_WORDS], blob: Vec::new(), classes: Vec::new(), class_numbers: HashMap::new(), forms: HashMap::new(), lowercase: None };
        let mut steps = Vec::new();
        b.normalizer(&json["normalizer"], &mut steps)?;
        let program: Vec<u32> = steps.iter().flatten().copied().chain([S_END; STEP_WORDS]).collect();
        b.words[F_NORMALIZER] = b.place(&program);
        let mut steps = Vec::new();
        b.pre_tokenizer(&json["pre_tokenizer"], &mut steps)?;
        let program: Vec<u32> = steps.iter().flatten().copied().chain([S_END; STEP_WORDS]).collect();
        b.words[F_PRE_TOKENIZER] = b.place(&program);

        let added_list = json["added_tokens"].as_array().cloned().unwrap_or_default();
        let flag = |token: &Value, key: &str| token[key].as_bool().unwrap_or(false);
        if added_list.iter().any(|t| flag(t, "single_word") || flag(t, "lstrip") || flag(t, "rstrip")) {
            b.words[F_WORD_CLASS] = b.class_of(r"\w")?;
            b.words[F_SPACE_CLASS] = b.class_of(r"\s")?;
        }

        let mut table = Vec::with_capacity(b.classes.len() * 2);
        for class in std::mem::take(&mut b.classes) {
            let flat: Vec<u32> = class.iter().flat_map(|(a, z)| [*a, *z]).collect();
            table.extend([b.place(&flat), class.len() as u32]);
        }
        b.words[F_CLASSES] = b.place(&table);

        // Normalized added tokens are matched after normalization, so they go through it too.
        let mut normalized_text: Vec<Vec<u8>> = Vec::new();
        for token in &added_list {
            let content = token["content"].as_str().unwrap_or("");
            let mut bytes = Vec::new();
            if flag(token, "normalized") {
                let mut temporary = b.words.clone();
                temporary[F_BLOB] = temporary.len() as u32;
                temporary.extend(blob_words(&b.blob));
                let mut arena = vec![0u32; (content.len() * 64).max(4096)];
                Tokenizer::new(&temporary).normalize(content.as_bytes(), &mut arena, &mut |byte| bytes.push(byte)).map_err(|_| "token añadido demasiado largo")?;
            }
            normalized_text.push(bytes);
        }
        let (mut raw, mut normalized) = (Vec::new(), Vec::new());
        for (token, processed) in added_list.iter().zip(&normalized_text) {
            let is_normalized = flag(token, "normalized");
            let bytes = if is_normalized { processed.as_slice() } else { token["content"].as_str().unwrap_or("").as_bytes() };
            let Some(first) = bytes.first() else { continue };
            let (at, length) = b.string(bytes);
            let flags = (flag(token, "single_word") as u32 * ADDED_SINGLE_WORD) | (flag(token, "lstrip") as u32 * ADDED_LSTRIP) | (flag(token, "rstrip") as u32 * ADDED_RSTRIP) | ((*first as u32) << ADDED_FIRST_SHIFT);
            let list = if is_normalized { &mut normalized } else { &mut raw };
            list.extend([token["id"].as_u64().unwrap_or(0) as u32, at, length, flags]);
        }
        for (table, field, count) in [(&raw, F_ADDED_RAW, F_ADDED_RAW_COUNT), (&normalized, F_ADDED_NORMALIZED, F_ADDED_NORMALIZED_COUNT)] {
            let mut bitmap = [0u32; ADDED_BITMAP_WORDS];
            for entry in table.chunks(4) {
                let first = entry[3] >> ADDED_FIRST_SHIFT;
                bitmap[first as usize / 32] |= 1 << (first % 32);
            }
            b.place(&bitmap);
            b.words[field] = b.place(table);
            b.words[count] = table.len() as u32 / 4;
        }

        let affix = |key: &str| model[key].as_str().unwrap_or("").to_string();
        let (prefix, suffix) = (affix("continuing_subword_prefix"), affix("end_of_word_suffix"));
        let is_bpe = kind == "BPE";
        let ignore_merges = is_bpe && model["ignore_merges"].as_bool().unwrap_or(false);

        // The model only ever looks up one character, unless it asks for whole pieces.
        let is_symbol = |text: &str| {
            let bare = text.strip_prefix(prefix.as_str()).filter(|_| !prefix.is_empty()).unwrap_or(text);
            let bare = bare.strip_suffix(suffix.as_str()).filter(|_| !suffix.is_empty()).unwrap_or(bare);
            bare.chars().count() == 1 || text.chars().count() == 1
        };
        let entries: Vec<(&str, u32)> = ids.iter().filter(|(text, _)| !is_bpe || ignore_merges || is_symbol(text)).map(|(text, id)| (*text, *id)).collect();
        let slots = hash_slots(entries.len());
        let seed = (0..).find(|seed| {
            let mut seen = HashSet::new();
            entries.iter().all(|(text, _)| seen.insert(hash_string(*seed, text.bytes())))
        }).unwrap_or(0);
        let mut vocabulary_table = vec![0u32; slots * 2];
        for (text, id) in &entries {
            let (first, second) = hash_string(seed, text.bytes());
            let mut slot = first as usize % slots;
            while vocabulary_table[2 * slot + 1] != 0 {
                slot = (slot + 1) % slots;
            }
            vocabulary_table[2 * slot] = second;
            vocabulary_table[2 * slot + 1] = id + 1;
        }
        b.words[F_SEED] = seed;
        b.words[F_VOCAB_SLOTS] = slots as u32;
        b.words[F_VOCAB] = b.place(&vocabulary_table);
        b.words[F_MODEL] = if is_bpe { MODEL_BPE } else { MODEL_WORD_LEVEL };
        b.words[F_IGNORE_MERGES] = ignore_merges as u32;

        let merges: Vec<(u32, u32, u32)> = model["merges"]
            .as_array()
            .map(|list| {
                list.iter()
                    .filter_map(|pair| {
                        // Older files write each pair as one string with a space between.
                        let (first, second) = match pair.as_str() {
                            Some(line) => line.split_once(' ')?,
                            None => (pair[0].as_str()?, pair[1].as_str()?),
                        };
                        let joined = format!("{first}{}", second.strip_prefix(prefix.as_str()).unwrap_or(second));
                        Some((*ids.get(first)?, *ids.get(second)?, *ids.get(joined.as_str())?))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let merge_slots = hash_slots(merges.len());
        let mut hash = vec![0u32; merge_slots];
        for (rank, (first, second, _)) in merges.iter().enumerate() {
            let mut slot = pair_slot(*first, *second) as usize % merge_slots;
            while hash[slot] != 0 {
                slot = (slot + 1) % merge_slots;
            }
            hash[slot] = rank as u32 + 1;
        }
        let entries: Vec<u32> = merges.iter().flat_map(|(first, second, result)| [*first, *second, *result]).collect();
        b.words[F_MERGE_SLOTS] = merge_slots as u32;
        b.words[F_MERGES] = b.place(&hash);
        b.words[F_MERGE_ENTRIES] = b.place(&entries);

        if is_bpe && model["byte_fallback"].as_bool().unwrap_or(false) {
            let table: Vec<u32> = (0..=255u8).map(|byte| ids.get(format!("<0x{byte:02X}>").as_str()).map_or(0, |id| id + 1)).collect();
            b.words[F_BYTE_FALLBACK] = b.place(&table);
        }
        b.words[F_FUSE_UNK] = model["fuse_unk"].as_bool().unwrap_or(false) as u32;
        b.words[F_UNK] = model["unk_token"].as_str().and_then(|u| ids.get(u)).map_or(0, |id| id + 1);
        if !is_bpe && b.words[F_UNK] == 0 {
            return refuse("un vocabulario sin token desconocido");
        }
        let (at, length) = b.string(prefix.as_bytes());
        b.words[F_PREFIX] = at;
        b.words[F_PREFIX_LEN] = length;
        let (at, length) = b.string(suffix.as_bytes());
        b.words[F_SUFFIX] = at;
        b.words[F_SUFFIX_LEN] = length;

        let (mut before, mut after) = (Vec::new(), Vec::new());
        b.special_tokens(&json["post_processor"], &mut before, &mut after)?;
        for (field, list) in [(F_POST_BEFORE, &before), (F_POST_AFTER, &after)] {
            let section: Vec<u32> = std::iter::once(list.len() as u32).chain(list.iter().copied()).collect();
            b.words[field] = b.place(&section);
        }

        let last_added = added_list.iter().filter_map(|t| t["id"].as_u64()).max().map_or(0, |id| id as usize + 1);
        let last_vocabulary = ids.values().max().map_or(0, |id| *id as usize + 1);
        let count = last_added.max(last_vocabulary);
        let mut strings = vec![Vec::new(); count];
        for (text, id) in &ids {
            strings[*id as usize] = text.as_bytes().to_vec();
        }
        for token in &added_list {
            strings[token["id"].as_u64().unwrap_or(0) as usize] = token["content"].as_str().unwrap_or("").as_bytes().to_vec();
        }
        let (mut fused, mut leading, mut first_token) = (false, 0, false);
        decoded(&mut strings, &json["decoder"], &mut fused, &mut leading, &mut first_token)?;
        // Special tokens write nothing, as when decoding with them skipped.
        for token in added_list.iter().filter(|t| flag(t, "special")) {
            strings[token["id"].as_u64().unwrap_or(0) as usize].clear();
        }
        let mut offsets = Vec::with_capacity(count + 1);
        for bytes in &strings {
            offsets.push(b.blob.len() as u32);
            b.blob.extend_from_slice(bytes);
        }
        offsets.push(b.blob.len() as u32);
        b.words[F_TOKENS] = count as u32;
        b.words[F_OFFSETS] = b.place(&offsets);
        b.words[F_LEADING_SPACES] = leading;
        b.words[F_FIRST_TOKEN_SPACES] = first_token as u32;

        let blob = blob_words(&b.blob);
        b.words[F_BLOB] = b.place(&blob);
        b.words[F_TEXT] = b.words.len() as u32;
        b.words[F_TEXT_BYTES] = text_bytes;
        b.words[F_ARENA_WORDS] = arena_words;
        Ok(Self { words: b.words })
    }

    /// Decides whether the post-processor's special tokens surround the text.
    pub fn set_add_special(&mut self, add: bool) {
        self.words[F_ADD_SPECIAL] = add as u32;
    }

    /// The words of the tables alone, without the room for text and work.
    pub fn table_words(&self) -> usize {
        self.words.len()
    }

    /// The tables, then the empty text buffer and working memory that follow them.
    pub fn to_words(&self) -> Vec<u32> {
        let mut words = self.words.clone();
        words.resize(words.len() + self.words[F_TEXT_BYTES].div_ceil(4) as usize + self.words[F_ARENA_WORDS] as usize, 0);
        words
    }

    /// The tokens the firmware produces for this text.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        let mut arena = vec![0u32; self.words[F_ARENA_WORDS] as usize];
        let mut tokens = Vec::new();
        Tokenizer::new(&self.words).encode(text.as_bytes(), &mut arena, &mut |id| tokens.push(id)).map_err(|_| "la memoria de trabajo del tokenizador no alcanza".to_string())?;
        Ok(tokens)
    }

    /// The bytes the firmware writes for these tokens.
    pub fn decode(&self, tokens: &[u32]) -> Vec<u8> {
        let tokenizer = Tokenizer::new(&self.words);
        let mut state = tokenizer.decoding();
        let mut bytes = Vec::new();
        for token in tokens {
            tokenizer.write_token(*token, &mut state, &mut |byte| bytes.push(byte));
        }
        bytes
    }
}

fn blob_words(blob: &[u8]) -> Vec<u32> {
    blob.chunks(4).map(|c| c.iter().rev().fold(0, |word, byte| word << 8 | *byte as u32)).collect()
}
