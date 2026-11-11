//! A reduced but complete die: cells with group scale and bias, the activation
//! network, a vector lane, a function table, a vector register file and a
//! microcoded sequencer. It executes programs built by `DieProgram`.

use crate::block::{Alu, Expr, Step, Var, FACTOR_FRACTION};
use crate::cell::{CellConfig, Gen};
use crate::rv32::Logic;
use crate::{Bit, Builder, Design};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};

const WIDE_EXTRA: u32 = 16;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct DieConfig {
    pub cells: u32,
    pub cell: CellConfig,
    pub scale_bits: u32,
    /// log2 of the number of vector registers and of their length.
    pub reg_bits: u32,
    pub index_bits: u32,
    pub microcode_bits: u32,
    pub broadcast_stages: u32,
    pub function_bits: u32,
    /// log2 of the number of function tables.
    pub function_select_bits: u32,
    /// What this die's fixed microprogram actually uses.
    #[serde(skip)]
    pub features: DieFeatures,
}

/// The parts of the general die that one microprogram needs. A die built with these
/// runs that microprogram and no other; `Default` keeps everything, for any program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DieFeatures {
    /// One bit per `OP_*` result that is ever selected.
    pub ops: u32,
    /// How many bits of each control field are ever set, in `field_layout` order.
    pub widths: [u8; FIELD_COUNT],
    /// Index registers in use, counting the always-zero first one.
    pub x_registers: u32,
    /// log2 of the elements an element-wise operation handles per cycle.
    pub lane_bits: u32,
    /// log2 of the vector units that run the same words on different data.
    pub cluster_bits: u32,
    /// log2 of the tokens each column of the key store holds; zero when keys are not in columns.
    pub key_shift: u32,
    /// Cycles a cell takes to apply a group's scale and bias, a few bits of each a cycle,
    /// while it adds up the next group: one is a whole multiplier for each, all at once.
    pub factor_steps: u32,
    /// The link to the next chip has a lane of its own for each vector lane, so a whole
    /// block crosses in one cycle.
    pub wide_link: bool,
    /// log2 of the words of vector memory, when each register takes only the words it
    /// uses: a table gives each its base. Zero gives every register `1 << index_bits`.
    pub vector_bits: u32,
    /// log2 of the groups whose scale and bias each cell keeps, when fewer than its
    /// weights; zero sizes those memories like the weights'.
    pub factor_bits: u32,
}

impl Default for DieFeatures {
    fn default() -> Self {
        Self { ops: u32::MAX, widths: [u8::MAX; FIELD_COUNT], x_registers: u32::MAX, lane_bits: 0, cluster_bits: 0, key_shift: 0, factor_steps: 1, wide_link: false, vector_bits: 0, factor_bits: 0 }
    }
}

impl DieFeatures {
    pub fn is_general(&self) -> bool {
        self.ops == u32::MAX
    }

    /// The same die with cells that take `steps` cycles to apply a group's factors.
    pub fn with_factor_steps(mut self, steps: u32) -> Self {
        self.factor_steps = steps.max(1);
        self
    }

    /// The same die with a link lane for each vector lane.
    pub fn with_wide_link(mut self, wide: bool) -> Self {
        self.wide_link = wide;
        self
    }

    /// The same die with `1 << bits` words of vector memory, handed out by a table.
    pub fn with_vector_bits(mut self, bits: u32) -> Self {
        self.vector_bits = bits;
        self
    }

    /// The same die with room for `1 << bits` groups of scale and bias in each cell.
    pub fn with_factor_bits(mut self, bits: u32) -> Self {
        self.factor_bits = bits;
        self
    }

    /// Lanes of the link to the next chip.
    pub fn link_lanes(&self) -> usize {
        if self.wide_link { 1 << self.lane_bits } else { 1 }
    }

    /// The same die with room for `1 << key_shift` tokens in each column of keys.
    pub fn with_key_shift(mut self, key_shift: u32) -> Self {
        self.key_shift = key_shift;
        self
    }

    /// The same die with several vector units.
    pub fn with_cluster_bits(mut self, cluster_bits: u32) -> Self {
        self.cluster_bits = cluster_bits;
        self
    }

    /// The same die with `lane_bits` of parallel vector lanes.
    pub fn with_lane_bits(mut self, lane_bits: u32) -> Self {
        self.lane_bits = lane_bits;
        self
    }

    /// The general die minus register loops, for programs written one register at a time.
    /// Whether the die can pipeline loops, which takes the two fields of a fixed operand.
    pub fn streams(&self) -> bool {
        self.widths[FIXED_A_FIELD] > 0
    }

    /// Whether the shifters can load straight from the read ports.
    pub fn shifts_now(&self) -> bool {
        self.widths[SHIFT_NOW_FIELD] > 0
    }

    /// Whether one group of the cells can follow another with no pause.
    pub fn feeds_cells(&self) -> bool {
        self.widths[CELL_FEED_FIELD] > 0
    }

    /// Whether a stream of blocks can step through a scalar operand one element a block
    /// and pile its rows up in one block of the wide memory.
    pub fn streams_rows(&self) -> bool {
        self.lane_bits > 0 && self.widths[A_SHIFT_FIELD] > 0 && self.widths[WIDE_CHAIN_FIELD] > 0
    }

    /// Whether the die can add whole rows into the wide memory.
    pub fn combines(&self) -> bool {
        self.widths[WIDE_ACCUMULATE_FIELD] > 0
    }

    pub fn without_register_windows() -> Self {
        let mut features = Self::default();
        features.widths[WINDOW_X_FIELD..].fill(0);
        features
    }

    /// Exactly what this microprogram needs.
    pub fn of(code: &Microprogram) -> Self {
        // Additions are also how the comparator and the sequencer's own results are made.
        let mut features = Self { ops: 1 << OP_ADD, widths: [0; FIELD_COUNT], x_registers: 1, lane_bits: 0, cluster_bits: 0, key_shift: 0, factor_steps: 1, wide_link: false, vector_bits: 0, factor_bits: 0 };
        let mut highest_x = 0;
        for word in &code.words {
            features.ops |= 1 << word.op;
            // Blocks that go to or come from the link need a link lane per vector lane.
            features.wide_link |= word.vector && (word.link_send || word.op == OP_LINK);
            for (width, value) in features.widths.iter_mut().zip(word.field_values()) {
                *width = (*width).max((u128::BITS - value.leading_zeros()) as u8);
            }
            highest_x = highest_x.max(word.offset_a).max(word.offset_b).max(word.offset_dst).max(word.x_dst).max(word.x_src).max(word.x_limit).max(word.window_x);
        }
        for parameter in &code.parameters {
            highest_x = highest_x.max(parameter.register);
        }
        // An immediate is signed: one more bit, so a positive one never reads as negative.
        if features.uses(OP_IMMEDIATE) {
            features.widths[TARGET_FIELD] += 1;
        }
        features.x_registers = highest_x + 1;
        features
    }

    /// What dies running either program need, for chips built identical to each other.
    pub fn union(self, other: Self) -> Self {
        let mut widths = self.widths;
        widths.iter_mut().zip(other.widths).for_each(|(a, b)| *a = (*a).max(b));
        Self { ops: self.ops | other.ops, widths, x_registers: self.x_registers.max(other.x_registers), lane_bits: self.lane_bits.max(other.lane_bits), cluster_bits: self.cluster_bits.max(other.cluster_bits), key_shift: self.key_shift.max(other.key_shift), factor_steps: self.factor_steps.max(other.factor_steps), wide_link: self.wide_link || other.wide_link, vector_bits: self.vector_bits.max(other.vector_bits), factor_bits: self.factor_bits.max(other.factor_bits) }
    }

    fn uses(&self, op: u32) -> bool {
        self.ops >> op & 1 == 1
    }
}

impl DieConfig {
    fn wide_bits(&self) -> u32 {
        self.cell.accumulator_bits + WIDE_EXTRA
    }
}

/// One horizontal microinstruction, repeated `repeat + 1` times.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct ControlWord {
    pub reg_a: u32,
    pub reg_b: u32,
    pub reg_dst: u32,
    /// See the `OP_*` constants.
    pub op: u32,
    pub shift: u32,
    pub function: u32,
    /// Address registers by literal index instead of by the counters.
    pub use_literal: bool,
    pub literal_a: u32,
    pub literal_b: u32,
    pub literal_dst: u32,
    pub load_shift_a: bool,
    pub load_shift_b: bool,
    pub load_shift_left: bool,
    pub magnitude_scan: bool,
    pub magnitude_clear: bool,
    pub accumulate: bool,
    pub accumulator_clear: bool,
    pub link_send: bool,
    pub wide_write: bool,
    pub wide_scan: bool,
    pub wide_clear: bool,
    pub load_wide_shift: bool,
    pub read_step: bool,
    pub read_reset: bool,
    pub write_en: bool,
    pub write_reset: bool,
    pub cell_run: bool,
    pub cell_clear: bool,
    pub cell_clear_group: bool,
    pub cell_flush: bool,
    pub chain_load: bool,
    pub chain_shift: bool,
    pub halt: bool,
    pub repeat: u32,
    /// Index registers added to each operand's literal; register 0 is always zero.
    pub offset_a: u32,
    pub offset_b: u32,
    pub offset_dst: u32,
    /// `index[x_dst] = index[x_src] + literal_b`.
    pub x_write: bool,
    pub x_dst: u32,
    pub x_src: u32,
    /// Jump to `target` unless the value just written equals `index[x_limit]`.
    pub branch: bool,
    pub x_limit: u32,
    pub target: u32,
    /// Adds `index[window_x]` to the marked operands' register numbers.
    pub window_x: u32,
    pub window_a: bool,
    pub window_b: bool,
    pub window_dst: bool,
    /// Runs on every lane at once, over a block of elements.
    pub vector: bool,
    /// Every lane takes the same single element for this operand.
    pub broadcast_a: bool,
    pub broadcast_b: bool,
    /// Moves every cell to the weight, or the group, at index A.
    pub cell_seek_weights: bool,
    pub cell_seek_factors: bool,
    /// The wide memory takes the next total of the result chain.
    pub wide_from_chain: bool,
    /// While the index counters stream, an operand that stays at its literal position.
    pub fixed_a: bool,
    pub fixed_b: bool,
    /// Each wide-memory element adds a product, at the read and the write of a stream.
    pub wide_accumulate: bool,
    /// The sum starts from zero instead of what the wide memory holds.
    pub wide_fresh: bool,
    /// The word repeats as many times as `index[x_limit]` says, not as its own field.
    pub repeat_x: bool,
    /// Operand A comes from this vector unit, plus one; zero means its own.
    pub from_cluster: u32,
    /// An index write only reaches this vector unit, plus one; zero means all of them.
    pub x_only: u32,
    /// Only the first `literal_b` vector units execute from now on.
    pub set_clusters: bool,
    /// The wide sum takes the total of what every vector unit holds at that place.
    pub wide_reduce: bool,
    pub max_scan: bool,
    pub max_clear: bool,
    pub max_out: bool,
    /// Each write goes a whole column further on, to store one token's key.
    pub write_spread: bool,
    /// The units add their wide memories in groups of `1 << reduce_group` neighbours.
    pub reduce_group: u32,
    /// The result is written in the same cycle it is computed, not one later.
    pub write_now: bool,
    /// Each unit also adds its own number times `literal_a` to an index it writes.
    pub x_scaled: bool,
    /// Non-zero: operand A comes from unit `index >> gather_shift`, so one stream
    /// collects a vector whose slices each unit computed.
    pub gather_shift: u32,
    /// With `x_scaled`, unit `k` adds `(k >> x_split) · literal_a` and
    /// `(k mod 2^x_split) · literal_dst`: several units per head, each on its own part.
    pub x_split: u32,
    /// The shifters load from the operands as they are read: A and left from A, B from B.
    pub shift_now: bool,
    /// The activation read now goes to the cells when it reaches them.
    pub cell_feed: bool,
    /// It is the last of a group: the cells apply the group's factors once it is in.
    pub cell_end: bool,
    /// Operand A walks the read counter shifted right this far, so a stream of blocks
    /// can take one element of A for each block of B.
    pub a_shift: u32,
    /// The wide sums add onto their own last total instead of what the memory holds, so
    /// a stream of rows piles up in one block.
    pub wide_chain: bool,
    /// Non-zero: the cells' groups are parts of quantisation groups of `1 << quant_bits`
    /// columns, and a unit's cells move on to the next scale and bias only where one ends.
    pub quant_bits: u32,
    /// The group ending now is a unit's last of the pass: its cells move on regardless.
    pub slice_end: bool,
    /// With `cell_seek_weights`: log2 of the codes the matrix keeps in each word of the cells.
    pub cell_packing: u32,
}

const REPEAT_BITS: u32 = 12;
pub const FIELD_COUNT: usize = 79;
const LINK_SEND_FIELD: usize = 17;
const WINDOW_X_FIELD: usize = 43;
const FIXED_A_FIELD: usize = 53;
/// The jump target, which an immediate step's compute cycle uses as its value instead.
const TARGET_FIELD: usize = 42;
const X_SPLIT_FIELD: usize = 70;
const SHIFT_NOW_FIELD: usize = 71;
const CELL_FEED_FIELD: usize = 72;
const A_SHIFT_FIELD: usize = 74;
const WIDE_CHAIN_FIELD: usize = 75;
const WIDE_ACCUMULATE_FIELD: usize = 55;
const MAX_SCAN_FIELD: usize = 62;

pub const OP_ADD: u32 = 0;
pub const OP_MUL: u32 = 1;
pub const OP_FUNCTION: u32 = 2;
pub const OP_CELLS: u32 = 3;
pub const OP_SUB: u32 = 4;
pub const OP_MAX: u32 = 5;
pub const OP_LZC: u32 = 6;
pub const OP_MOV: u32 = 7;
pub const OP_SHIFT_LEFT: u32 = 8;
pub const OP_ACCUMULATOR: u32 = 9;
pub const OP_IMMEDIATE: u32 = 10;
pub const OP_PARITY: u32 = 11;
pub const OP_LINK: u32 = 12;
pub const OP_WIDE_LZC: u32 = 13;
pub const OP_WIDE_OUT: u32 = 14;
pub const OP_EQUAL: u32 = 15;
const OP_BITS: u32 = 4;
const SHIFT_REG_BITS: u32 = 6;

/// Each field's name and how many bits of the word it takes on this die.
fn field_layout(c: &DieConfig) -> Vec<(&'static str, u32)> {
    full_layout(c).into_iter().zip(c.features.widths).map(|((name, width), used)| (name, width.min(used as u32))).collect()
}

fn full_layout(c: &DieConfig) -> Vec<(&'static str, u32)> {
    vec![
        ("reg_a", c.reg_bits),
        ("reg_b", c.reg_bits),
        ("reg_dst", c.reg_bits),
        ("op", OP_BITS),
        ("shift", 6),
        ("function", c.function_select_bits),
        ("use_literal", 1),
        ("literal_a", c.index_bits),
        ("literal_b", c.index_bits),
        ("literal_dst", c.index_bits),
        ("load_shift_a", 1),
        ("load_shift_b", 1),
        ("load_shift_left", 1),
        ("magnitude_scan", 1),
        ("magnitude_clear", 1),
        ("accumulate", 1),
        ("accumulator_clear", 1),
        ("link_send", 1),
        ("wide_write", 1),
        ("wide_scan", 1),
        ("wide_clear", 1),
        ("load_wide_shift", 1),
        ("read_step", 1),
        ("read_reset", 1),
        ("write_en", 1),
        ("write_reset", 1),
        ("cell_run", 1),
        ("cell_clear", 1),
        ("cell_clear_group", 1),
        ("cell_flush", 1),
        ("chain_load", 1),
        ("chain_shift", 1),
        ("halt", 1),
        ("repeat", REPEAT_BITS),
        ("offset_a", x_bits(c)),
        ("offset_b", x_bits(c)),
        ("offset_dst", x_bits(c)),
        ("x_write", 1),
        ("x_dst", x_bits(c)),
        ("x_src", x_bits(c)),
        ("branch", 1),
        ("x_limit", x_bits(c)),
        ("target", c.microcode_bits),
        ("window_x", x_bits(c)),
        ("window_a", 1),
        ("window_b", 1),
        ("window_dst", 1),
        ("vector", 1),
        ("broadcast_a", 1),
        ("broadcast_b", 1),
        ("cell_seek_weights", 1),
        ("cell_seek_factors", 1),
        ("wide_from_chain", 1),
        ("fixed_a", 1),
        ("fixed_b", 1),
        ("wide_accumulate", 1),
        ("wide_fresh", 1),
        ("repeat_x", 1),
        ("from_cluster", c.features.cluster_bits + 1),
        ("x_only", c.features.cluster_bits + 1),
        ("set_clusters", 1),
        ("wide_reduce", 1),
        ("max_scan", 1),
        ("max_clear", 1),
        ("max_out", 1),
        ("write_spread", 1),
        ("reduce_group", bits_for(c.features.cluster_bits + 1)),
        ("write_now", 1),
        ("x_scaled", 1),
        ("gather_shift", bits_for(c.index_bits)),
        ("x_split", bits_for(c.features.cluster_bits + 1)),
        ("shift_now", 1),
        ("cell_feed", 1),
        ("cell_end", 1),
        ("a_shift", bits_for(c.features.lane_bits)),
        ("wide_chain", 1),
        ("quant_bits", 3),
        ("slice_end", 1),
        ("cell_packing", crate::cell::bits_for_packing(crate::cell::MAX_PACKING)),
    ]
}

/// log2 of the index registers, which never outnumber the elements they address.
pub fn x_bits(c: &DieConfig) -> u32 {
    c.index_bits.min(5)
}

impl ControlWord {
    fn field_values(&self) -> [u128; FIELD_COUNT] {
        [
            self.reg_a as u128,
            self.reg_b as u128,
            self.reg_dst as u128,
            self.op as u128,
            self.shift as u128,
            self.function as u128,
            self.use_literal as u128,
            self.literal_a as u128,
            self.literal_b as u128,
            self.literal_dst as u128,
            self.load_shift_a as u128,
            self.load_shift_b as u128,
            self.load_shift_left as u128,
            self.magnitude_scan as u128,
            self.magnitude_clear as u128,
            self.accumulate as u128,
            self.accumulator_clear as u128,
            self.link_send as u128,
            self.wide_write as u128,
            self.wide_scan as u128,
            self.wide_clear as u128,
            self.load_wide_shift as u128,
            self.read_step as u128,
            self.read_reset as u128,
            self.write_en as u128,
            self.write_reset as u128,
            self.cell_run as u128,
            self.cell_clear as u128,
            self.cell_clear_group as u128,
            self.cell_flush as u128,
            self.chain_load as u128,
            self.chain_shift as u128,
            self.halt as u128,
            self.repeat as u128,
            self.offset_a as u128,
            self.offset_b as u128,
            self.offset_dst as u128,
            self.x_write as u128,
            self.x_dst as u128,
            self.x_src as u128,
            self.branch as u128,
            self.x_limit as u128,
            self.target as u128,
            self.window_x as u128,
            self.window_a as u128,
            self.window_b as u128,
            self.window_dst as u128,
            self.vector as u128,
            self.broadcast_a as u128,
            self.broadcast_b as u128,
            self.cell_seek_weights as u128,
            self.cell_seek_factors as u128,
            self.wide_from_chain as u128,
            self.fixed_a as u128,
            self.fixed_b as u128,
            self.wide_accumulate as u128,
            self.wide_fresh as u128,
            self.repeat_x as u128,
            self.from_cluster as u128,
            self.x_only as u128,
            self.set_clusters as u128,
            self.wide_reduce as u128,
            self.max_scan as u128,
            self.max_clear as u128,
            self.max_out as u128,
            self.write_spread as u128,
            self.reduce_group as u128,
            self.write_now as u128,
            self.x_scaled as u128,
            self.gather_shift as u128,
            self.x_split as u128,
            self.shift_now as u128,
            self.cell_feed as u128,
            self.cell_end as u128,
            self.a_shift as u128,
            self.wide_chain as u128,
            self.quant_bits as u128,
            self.slice_end as u128,
            self.cell_packing as u128,
        ]
    }

    /// The word as 32-bit pieces, least significant first.
    pub fn encode_words(&self, c: &DieConfig) -> Vec<u32> {
        let values = self.field_values();
        let mut words = vec![0u32; word_bits(c).div_ceil(32) as usize];
        let mut offset = 0usize;
        for ((_, width), value) in field_layout(c).iter().zip(values) {
            for bit in 0..*width as usize {
                if value >> bit & 1 == 1 {
                    words[(offset + bit) / 32] |= 1 << ((offset + bit) % 32);
                }
            }
            offset += *width as usize;
        }
        words
    }

    /// The word as one number, for the small dies whose words fit in 128 bits.
    pub fn encode(&self, c: &DieConfig) -> u128 {
        assert!(word_bits(c) <= 128, "la palabra no cabe en 128 bits");
        self.encode_words(c).iter().rev().fold(0, |word, piece| word << 32 | *piece as u128)
    }
}

/// The fields of a microcode word given as its bits, most significant first, as the
/// simulator shows the sequencer's `code`.
pub fn decode_word(bits: &str, c: &DieConfig) -> BTreeMap<&'static str, u128> {
    let low_first: Vec<bool> = bits.chars().rev().map(|b| b == '1').collect();
    let mut offset = 0usize;
    field_layout(c)
        .into_iter()
        .map(|(name, width)| {
            let value = (0..width as usize).filter(|bit| low_first.get(offset + bit) == Some(&true)).fold(0u128, |v, bit| v | 1 << bit);
            offset += width as usize;
            (name, value)
        })
        .collect()
}

/// What a microcode word makes the die do, in a sentence for someone following it.
pub fn describe_word(f: &BTreeMap<&'static str, u128>) -> String {
    let get = |name: &str| f.get(name).copied().unwrap_or(0);
    let on = |name: &str| get(name) != 0;
    if on("halt") {
        return "Fin de la rutina: el die avisa a la CPU de que ha terminado.".into();
    }
    let place = |reg: &str, literal: &str| if on("use_literal") { format!("r{}[{}]", get(reg), get(literal)) } else { format!("r{}", get(reg)) };
    let (a, b, dst) = (place("reg_a", "literal_a"), place("reg_b", "literal_b"), place("reg_dst", "literal_dst"));
    let mut parts: Vec<String> = Vec::new();
    if on("cell_seek_weights") || on("cell_seek_factors") {
        parts.push(format!("las celdas se colocan al principio de una matriz (pesos en la palabra {})", get("literal_a")));
    }
    if on("cell_run") || on("cell_feed") {
        parts.push(format!("las celdas reciben {a} por la red de activaciones, lo multiplican por su peso y lo acumulan"));
    }
    if on("cell_clear") || on("cell_clear_group") {
        parts.push("las celdas empiezan una suma nueva".into());
    }
    if on("cell_flush") || on("wide_write") || on("wide_reduce") {
        parts.push("las sumas pasan a la memoria ancha de cada unidad".into());
    }
    let written = on("write_en") || on("write_now");
    let what = match get("op") as u32 {
        OP_ADD => Some(format!("suma {a} + {b}")),
        OP_SUB => Some(format!("resta {a} − {b}")),
        OP_MUL => Some(format!("multiplica {a} × {b}")),
        OP_FUNCTION => {
            let name = match get("function") as u32 {
                crate::block::FN_EXP => "exponencial",
                crate::block::FN_SIGMOID => "sigmoide",
                crate::block::FN_RECIPROCAL => "inverso",
                crate::block::FN_INV_SQRT => "raíz inversa",
                crate::block::FN_SOFTPLUS => "softplus",
                crate::block::FN_GELU | crate::block::FN_GELU_TANH => "GELU",
                crate::block::FN_TANH => "tangente hiperbólica",
                _ => "función",
            };
            Some(format!("aplica la tabla de {name} a {a}"))
        }
        OP_CELLS => Some("recoge el resultado de las celdas".to_string()),
        OP_MAX => Some(format!("busca el máximo de {a}")),
        OP_LZC | OP_WIDE_LZC => Some(format!("cuenta los ceros a la izquierda de {a} para normalizarlo")),
        OP_MOV => Some(format!("copia {a}")),
        OP_SHIFT_LEFT => Some(format!("desplaza {a} a la izquierda")),
        OP_ACCUMULATOR => Some("lee el acumulador de la unidad".to_string()),
        OP_IMMEDIATE => Some("toma un número fijo de la propia instrucción".to_string()),
        OP_PARITY => Some(format!("calcula la paridad de {a}")),
        OP_LINK => Some("recibe una palabra del chip anterior".to_string()),
        OP_WIDE_OUT => Some("lee la memoria ancha y la redondea".to_string()),
        OP_EQUAL => Some(format!("compara {a} con {b}")),
        _ => None,
    };
    if let Some(what) = what {
        let lanes = if on("vector") { " en todos los carriles a la vez" } else { "" };
        parts.push(if written { format!("{what}{lanes} y lo guarda en {dst}") } else { format!("{what}{lanes}") });
    }
    if on("accumulate") {
        parts.push("lo suma al acumulador".into());
    }
    if on("link_send") {
        parts.push("lo envía al chip siguiente".into());
    }
    if get("repeat") > 0 {
        let times = get("repeat");
        parts.push(format!("repite el paso {times} {} más, avanzando por los elementos", if times == 1 { "vez" } else { "veces" }));
    }
    if on("branch") {
        parts.push(format!("si el contador x{} no ha llegado a su límite, vuelve a la palabra {}", get("x_src"), get("target")));
    }
    if parts.is_empty() {
        return "Espera: deja pasar un ciclo mientras termina un resultado anterior.".into();
    }
    let mut text = parts.join("; ");
    if let Some(first) = text.get(..1) {
        text = first.to_uppercase() + &text[1..];
    }
    text + "."
}

pub fn word_bits(c: &DieConfig) -> u32 {
    field_layout(c).iter().map(|(_, w)| w).sum()
}

fn bus_or(b: &mut Builder, a: Bit, x: Bit, name: &str) -> Bit {
    let y = b.wire(name, 1)[0];
    b.or_to(a, x, y);
    y
}

fn and_not(b: &mut Builder, enable: Bit, value: Bit, name: &str) -> Bit {
    let inverted = b.wire(&format!("{name}_n"), 1)[0];
    b.not_to(value, inverted);
    let y = b.wire(name, 1)[0];
    b.and_to(enable, inverted, y);
    y
}

/// Bits of the step between registers a loop visits, as a power of two, in each entry of
/// the register table.
pub const BASE_SHIFT_BITS: u32 = 5;

/// The pins of one lane of the link to the neighbouring chips.
struct LinkPins {
    tx_valid: Vec<Bit>,
    tx_bits: Vec<Bit>,
    tx_ready: Vec<Bit>,
    rx_valid: Vec<Bit>,
    rx_bits: Vec<Bit>,
    rx_ready: Vec<Bit>,
    tx_ack: Vec<Bit>,
    rx_ack: Vec<Bit>,
}

impl LinkPins {
    /// The pins of a lane named `prefix`, for words of `width` bits.
    fn new(b: &mut Builder, prefix: &str, width: u32) -> Self {
        let bits = crate::link::wire_bits(width);
        let ack = crate::link::ACK_BITS as u32;
        Self {
            tx_valid: b.output(&format!("{prefix}_tx_valid"), 1),
            tx_bits: b.output(&format!("{prefix}_tx_bits"), bits),
            tx_ready: b.input(&format!("{prefix}_tx_ready"), 1),
            rx_valid: b.input(&format!("{prefix}_rx_valid"), 1),
            rx_bits: b.input(&format!("{prefix}_rx_bits"), bits),
            rx_ready: b.output(&format!("{prefix}_rx_ready"), 1),
            tx_ack: b.input(&format!("{prefix}_tx_ack"), ack),
            rx_ack: b.output(&format!("{prefix}_rx_ack"), ack),
        }
    }
}

/// Loadable counter: `clear` wins, then `advance` increments.
/// The bits a count up to `value` needs.
fn bits_for(value: u32) -> u32 {
    (u32::BITS - value.leading_zeros()).max(1)
}

/// Steps whose result can be written as it is computed: nothing reads it through the
/// result register afterwards, and no other unit has to see it on the way.
/// Register stages in the tree that finds the largest element across the lanes: one after
/// every three of its levels but the last, so the largest of a block reaches `best` that
/// many cycles later.
fn max_scan_stages(lane_bits: u32) -> usize {
    lane_bits.saturating_sub(1) as usize / 3
}

/// Register stages in the network that finds the largest operand over the units: the last
/// cycle has one of its levels, after the comparison it feeds, and the others two each.
fn max_stages(cluster_bits: u32) -> usize {
    cluster_bits as usize / 2
}

fn writes_at_once(s: &Alu) -> bool {
    // The largest over a group of units takes the network's stages, and a product is
    // rounded in the cycle after it is computed.
    if (s.op == OP_MAX && s.reduce_group > 0) || s.op == OP_MUL {
        return false;
    }
    let late_shift = (s.load_shift_a || s.load_shift_b || s.load_shift_left || s.load_wide_shift) && !s.shift_now;
    let later = late_shift || s.link_send || s.wide_write;
    !later && !matches!(s.op, OP_FUNCTION | OP_LINK | OP_CELLS)
}

/// Whether `next` may read in the cycle `previous` computes in: it reads nothing
/// `previous` writes, clears nothing `previous` updates, and the fields both cycles use
/// agree. Addresses only matter when they are presented, so those may differ.
fn can_follow(previous: &Alu, next: &Alu, lanes: u32) -> bool {
    // A clear in the read cycle loses only what `previous` would gather in that cycle.
    let clears = (next.magnitude_clear && previous.magnitude_scan)
        || (next.max_clear && previous.max_scan)
        || (next.accumulator_clear && previous.accumulate)
        || (next.wide_clear && previous.wide_scan);
    let windowed = |s: &Alu| s.window_a || s.window_b || s.window_dst;
    // Lanes, broadcasts and the unit to read from act when the data arrives, so the
    // shared word keeps those of `previous`.
    let agree = (!windowed(previous) && !windowed(next) || previous.window_x == next.window_x)
        // Only an immediate carries a literal, and it carries it in the jump target.
        && (previous.literal.is_none() || previous.op == OP_IMMEDIATE);
    let span = |vector: bool| if vector { lanes } else { 1 };
    let touches = |(reg, at): (u32, u32), offset: u32, window: bool, vector: bool| {
        let (written, from) = previous.dst;
        if reg != written {
            return false;
        }
        if offset != 0 || previous.offset_dst != 0 || window || previous.window_dst {
            return true;
        }
        let (a, b) = (span(vector), span(previous.vector));
        at < from + b && from < at + a
    };
    // Reads of a unit's own registers take what the step before is writing from the banks.
    let forwarded = next.from_cluster == 0;
    let depends = previous.write && !forwarded && (touches(next.a, next.offset_a, next.window_a, next.vector) || touches(next.b, next.offset_b, next.window_b, next.vector));
    !clears && (agree || reads_nothing(next)) && !depends
}

/// Steps that read no register, so their read cycle may carry any address fields.
fn reads_nothing(s: &Alu) -> bool {
    matches!(s.op, OP_IMMEDIATE | OP_LZC | OP_WIDE_LZC) && !s.magnitude_scan && !s.max_scan && !s.accumulate && !s.wide_scan
}

/// When `literals[k] = literals[0] + Σ_j bit_j(k) · d_j` for strides `d_j`, the same as
/// `literals[0] + Σ_s (k >> s) · (d_s - 2 · d_(s-1))`: those terms that are not zero.
fn bit_linear(literals: &[u32], mask: u64, bits: u32) -> Option<Vec<(u32, u64)>> {
    let first = literals[0] as u64;
    let stride = |j: u32| literals.get(1usize << j).map_or(0, |l| (*l as u64).wrapping_sub(first) & mask);
    let strides: Vec<u64> = (0..bits).map(stride).collect();
    let fits = literals.iter().enumerate().all(|(k, literal)| {
        let sum = (0..bits).filter(|j| k >> j & 1 == 1).fold(first, |sum, j| sum.wrapping_add(strides[j as usize]));
        sum & mask == *literal as u64 & mask
    });
    fits.then(|| (0..bits).filter_map(|s| {
        let below = if s == 0 { 0 } else { strides[s as usize - 1].wrapping_mul(2) };
        let factor = strides[s as usize].wrapping_sub(below) & mask;
        (factor != 0).then_some((s, factor))
    }).collect())
}

/// An immediate, kept in `index_bits` like any index, as the jump target carries it: the
/// same signed value in the target's own width.
fn immediate_target(literal: Option<u32>, c: DieConfig) -> u32 {
    let Some(value) = literal else { return 0 };
    let shift = 32 - c.index_bits;
    let signed = ((value << shift) as i32) >> shift;
    signed as u32 & ((1u32 << c.microcode_bits) - 1)
}

/// Whether `bits` holds at least `value`.
fn at_least(l: &mut Logic, bits: &[Bit], value: u32) -> Bit {
    bits.iter().enumerate().fold(Bit::One, |reached, (i, bit)| if value >> i & 1 == 1 { l.and(*bit, reached) } else { l.or(*bit, reached) })
}

/// An index that moves a whole block of elements at a time while `block` is set.
fn index_counter(gen: &mut Gen, b: &mut Builder, (name, wires): (&str, usize), width: u32, (clear, advance): (Bit, Bit), block: Bit, lane_bits: usize) -> Vec<Bit> {
    if lane_bits == 0 {
        return counter(gen, b, name, width, clear, advance);
    }
    let mut l = Logic { b, next: wires };
    let value = l.fresh_bus(width as usize);
    let by_block = l.and(advance, block);
    let single = l.not(block);
    let by_one = l.and(advance, single);
    let mut step = vec![Bit::Zero; width as usize];
    (step[0], step[lane_bits]) = (by_one, by_block);
    let next = l.adder(&value, &step, Bit::Zero).0;
    let keep = l.not(clear);
    let held: Vec<Bit> = next.iter().map(|bit| l.and(keep, *bit)).collect();
    l.connect_flops(&held, &value);
    value
}

fn counter(gen: &mut Gen, b: &mut Builder, name: &str, width: u32, clear: Bit, advance: Bit) -> Vec<Bit> {
    let module = gen.address_counter(width);
    let value = b.wire(name, width);
    b.inst(&format!("{name}_counter"), &module, &[("clear", vec![clear]), ("advance", vec![advance]), ("address", value.clone())]);
    value
}

impl Gen {
    /// A cell plus its scale and bias path: at each group's end the raw sum is
    /// scaled, the group's activation sum times the bias is added, and both clear.
    fn scaled_cell(&mut self, c: &DieConfig) -> String {
        let name = "scaled_cell".to_string();
        if self.modules.contains_key(&name) {
            return name;
        }
        // The reduced die writes its weights one at a time, so it has no protected words.
        let cfg = CellConfig { ecc: false, ..c.cell };
        let (a, acc, s, wide) = (cfg.activation_bits, cfg.accumulator_bits, c.scale_bits, c.wide_bits());
        let compute = self.cell(cfg, true);
        let scale_mul = self.multiplier(s, acc, 1);
        let adder = self.ripple_adder(acc);
        let reg_acc = self.register(acc);
        let wide_adder = self.ripple_adder(wide);
        let reg_wide = self.register(wide);
        let mux_wide = self.mux_bus(wide);

        let mut b = Builder::new("scaled_cell", "Celda con escala y sesgo", "Multiplica sus pesos por la activación y, al final de cada grupo, aplica la escala y el sesgo que comparte ese grupo.");
        let clear = b.input("clear", 1)[0];
        let clear_group = b.input("clear_group", 1)[0];
        let flush = b.input("flush", 1)[0];
        let advance = b.input("advance", 1)[0];
        let serial = (c.features.factor_steps > 1).then(|| (b.input("factor_first", 1)[0], b.input("factor_done", 1)[0]));
        let seek_weights = b.input("seek_weights", 1);
        let seek_factors = b.input("seek_factors", 1)[0];
        let seek_addr = b.input("seek_addr", cfg.addr_bits);
        let load = (!cfg.fixed_weights).then(|| (b.input("load_en", 1)[0], b.input("load_scale", 1)[0], b.input("load_addr", cfg.addr_bits), b.input("load_data", s.max(cfg.weight_bits))));
        let run = b.input("run", 1)[0];
        let x = b.input("x", a);
        let total = b.output("total", wide);

        b.group("Celda de cálculo");
        let group_sum = b.wire("group_sum", acc);
        // A flush takes the group's sum and starts the next group with whatever product
        // arrives in that cycle, so groups can follow each other with no gap.
        let mut compute_pins = vec![("clear", vec![clear]), ("clear_total", vec![clear_group]), ("restart", vec![flush]), ("run", vec![run]), ("x", x.clone()), ("total", group_sum.clone()), ("seek", seek_weights), ("seek_addr", seek_addr.clone())];
        if cfg.most_packing() > 0 {
            compute_pins.push(("packing", b.input("packing", crate::cell::bits_for_packing(cfg.most_packing()))));
        }
        if let Some((load_en, load_scale, load_addr, load_data)) = &load {
            let load_weight = and_not(&mut b, *load_en, *load_scale, "load_weight");
            compute_pins.extend([("load_en", vec![load_weight]), ("load_addr", load_addr.clone()), ("load_data", load_data[..cfg.weight_bits as usize].to_vec())]);
        }
        b.inst("compute", &compute, &compute_pins);

        b.group("Suma de activaciones");
        // The activations are delayed to arrive with their products, so a flush splits
        // both sums at the same place.
        let reg_x = self.register(a);
        let (mut x_late, mut run_late) = (x.clone(), run);
        for stage in 0..cfg.latency() as usize - 1 {
            let q = b.wire(&format!("x_late{stage}"), a);
            b.inst(&format!("x_late_reg{stage}"), &reg_x, &[("d", x_late), ("q", q.clone())]);
            x_late = q;
            let r = b.wire(&format!("run_late{stage}"), 1)[0];
            b.dff_to(run_late, r);
            run_late = r;
        }
        let sign = *x_late.last().expect("activación vacía");
        let x_wide: Vec<Bit> = (0..acc as usize).map(|i| x_late.get(i).copied().unwrap_or(sign)).collect();
        let x_sum = b.wire("x_sum", acc);
        let x_gated = b.wire("x_gated", acc);
        for i in 0..acc as usize {
            b.and_to(run_late, x_wide[i], x_gated[i]);
        }
        let x_kept = b.wire("x_kept", acc);
        let x_keep = b.wire("x_keep_on_flush", 1)[0];
        b.not_to(flush, x_keep);
        for i in 0..acc as usize {
            b.and_to(x_keep, x_sum[i], x_kept[i]);
        }
        let x_next = b.wire("x_next", acc);
        let unused = b.wire("x_carry", 1);
        b.inst("x_adder", &adder, &[("a", x_kept), ("b", x_gated), ("cin", vec![Bit::Zero]), ("sum", x_next.clone()), ("cout", unused)]);
        let keep = b.wire("x_keep", 1)[0];
        let reset = bus_or(&mut b, clear, clear_group, "x_reset");
        b.not_to(reset, keep);
        let x_d = b.wire("x_d", acc);
        for i in 0..acc as usize {
            b.and_to(keep, x_next[i], x_d[i]);
        }
        b.inst("x_sum_reg", &reg_acc, &[("d", x_d), ("q", x_sum.clone())]);

        b.group("Escala y sesgo");
        let factor_load = load.map(|(load_en, load_scale, load_addr, load_data)| {
            let write_scale = b.wire("write_factors", 1)[0];
            b.and_to(load_en, load_scale, write_scale);
            FactorLoad { write_scale, load_data, load_addr }
        });
        self.finish_scaled_cell(b, c, &scale_mul, &wide_adder, &reg_wide, &mux_wide, SumParts { group_sum, x_sum, load: factor_load, clear, flush, advance, serial, total, clear_group, seek_factors, seek_addr })
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_scaled_cell(&mut self, mut b: Builder, c: &DieConfig, scale_mul: &str, wide_adder: &str, reg_wide: &str, mux_wide: &str, p: SumParts) -> String {
        let cfg = c.cell;
        let (acc, s, wide) = (cfg.accumulator_bits, c.scale_bits, c.wide_bits());

        // Scale and bias alternate in memory; a flush takes the scale first.
        b.group("Escala y sesgo");
        let group = b.wire("group_index", cfg.addr_bits);
        let module_counter = self.seek_counter(cfg.addr_bits);
        b.inst("group_index_counter", &module_counter, &[("clear", vec![p.clear]), ("advance", vec![p.advance]), ("seek", vec![p.seek_factors]), ("target", p.seek_addr.clone()), ("address", group.clone())]);
        let read_addr = b.wire("factors_raddr", cfg.addr_bits);
        for i in 0..cfg.addr_bits as usize {
            b.xor_to(group[i], Bit::Zero, read_addr[i]);
        }
        let scale = b.wire("scale", s);
        let bias = b.wire("bias", s);
        // Groups are far fewer than weights: the memories of the factors take only the
        // address bits they need.
        let factor_bits = match c.features.factor_bits {
            0 => cfg.addr_bits,
            bits => bits.min(cfg.addr_bits),
        };
        let read_addr = read_addr[..factor_bits as usize].to_vec();
        match &p.load {
            Some(load) => {
                // Host addresses are 2·group for the scale and 2·group + 1 for the bias.
                let mut write_addr: Vec<Bit> = load.load_addr[1..].to_vec();
                write_addr.push(Bit::Zero);
                write_addr.truncate(factor_bits as usize);
                let write_bias = b.wire("write_bias", 1)[0];
                b.and_to(load.write_scale, load.load_addr[0], write_bias);
                let write_scale_only = and_not(&mut b, load.write_scale, load.load_addr[0], "write_scale_only");
                b.ram2p("scale_mem", factor_bits, s, &[("raddr", read_addr.clone()), ("waddr", write_addr.clone()), ("we", vec![write_scale_only]), ("wdata", load.load_data[..s as usize].to_vec()), ("rdata", scale.clone())]);
                b.ram2p("bias_mem", factor_bits, s, &[("raddr", read_addr), ("waddr", write_addr), ("we", vec![write_bias]), ("wdata", load.load_data[..s as usize].to_vec()), ("rdata", bias.clone())]);
            }
            None => {
                b.rom("scale_mem", factor_bits, s, &[("addr", read_addr.clone()), ("rdata", scale.clone())]);
                b.rom("bias_mem", factor_bits, s, &[("addr", read_addr), ("rdata", bias.clone())]);
            }
        }

        let (sum_scaled, x_biased, update) = match p.serial {
            None => {
                let sum_scaled = b.wire("sum_scaled", acc + s);
                b.inst("scale_times_sum", scale_mul, &[("weight", scale), ("x", p.group_sum.clone()), ("product", sum_scaled.clone())]);
                let x_biased = b.wire("x_biased", acc + s);
                b.inst("bias_times_x", scale_mul, &[("weight", bias), ("x", p.x_sum.clone()), ("product", x_biased.clone())]);
                (sum_scaled, x_biased, p.flush)
            }
            Some((first, done)) => {
                // The flush keeps the group's sums and factors; then each cycle takes the next
                // few bits of each factor, highest first, while the cell adds up the next group.
                let steps = c.features.factor_steps;
                let chunk = s.div_ceil(steps);
                let chunk_mul = self.multiplier(chunk + 1, acc, 1);
                let sum_scaled = self.serial_product(&mut b, "scale", (&scale, &p.group_sum), (p.flush, first), (steps, chunk), &chunk_mul);
                let x_biased = self.serial_product(&mut b, "bias", (&bias, &p.x_sum), (p.flush, first), (steps, chunk), &chunk_mul);
                (sum_scaled, x_biased, done)
            }
        };

        b.group("Total de la fila");
        let extend = |v: &[Bit]| -> Vec<Bit> {
            let sign = *v.last().expect("vector vacío");
            (0..wide as usize).map(|i| v.get(i).copied().unwrap_or(sign)).collect()
        };
        // The total and both terms meet in a carry-save step with one fast adder.
        let _ = wide_adder;
        let next = Logic { b: &mut b, next: 59_500 }.sum_many(vec![p.total.clone(), extend(&sum_scaled), extend(&x_biased)]);
        let chosen = b.wire("total_chosen", wide);
        b.inst("total_select", mux_wide, &[("s", vec![update]), ("a", p.total.clone()), ("b", next), ("y", chosen.clone())]);
        let not_clear = b.wire("total_keep", 1)[0];
        let total_clear = bus_or(&mut b, p.clear, p.clear_group, "total_clear");
        b.not_to(total_clear, not_clear);
        let d = b.wire("total_d", wide);
        for i in 0..wide as usize {
            b.and_to(not_clear, chosen[i], d[i]);
        }
        b.inst("total_reg", reg_wide, &[("d", d), ("q", p.total)]);
        self.add(b.finish())
    }

    /// `factor · value`, a few bits of the factor a cycle from the top, its sign with the
    /// first: kept at the flush, done `steps` cycles after it.
    #[allow(clippy::too_many_arguments)]
    fn serial_product(&mut self, b: &mut Builder, name: &str, (factor, value): (&[Bit], &[Bit]), (flush, first): (Bit, Bit), (steps, chunk): (u32, u32), chunk_mul: &str) -> Vec<Bit> {
        let mut l = Logic { b, next: 0 };
        l.b.set_prefix(&format!("{name}_"));
        let (padded, width) = ((steps * chunk) as usize, value.len());
        let top = *factor.last().expect("factor vacío");
        let wide_factor: Vec<Bit> = (0..padded).map(|i| factor.get(i).copied().unwrap_or(top)).collect();
        // The factor shifts up a chunk a cycle; the value holds still.
        let held_factor = l.fresh_bus(padded);
        let shifted: Vec<Bit> = (0..padded).map(|i| if i < chunk as usize { Bit::Zero } else { held_factor[i - chunk as usize] }).collect();
        let next_factor = l.mux_bus(flush, &shifted, &wide_factor);
        l.connect_flops(&next_factor, &held_factor);
        let held_value = l.fresh_bus(width);
        let kept = l.mux_bus(flush, &held_value, value);
        l.connect_flops(&kept, &held_value);
        // The highest chunk carries the sign; the others are positive.
        let piece: Vec<Bit> = held_factor[padded - chunk as usize..].to_vec();
        let sign = l.and(first, *piece.last().expect("trozo vacío"));
        let signed_piece: Vec<Bit> = piece.iter().copied().chain(std::iter::once(sign)).collect();
        let product = l.fresh_bus(chunk as usize + 1 + width);
        l.b.inst("chunk_times_value", chunk_mul, &[("weight", signed_piece), ("x", held_value.clone()), ("product", product.clone())]);
        let total_width = padded + width;
        let sum = l.fresh_bus(total_width);
        let up: Vec<Bit> = (0..total_width).map(|i| if i < chunk as usize { Bit::Zero } else { sum[i - chunk as usize] }).collect();
        let restart = l.not(first);
        let before: Vec<Bit> = up.iter().map(|bit| l.and(*bit, restart)).collect();
        let product_sign = *product.last().expect("producto vacío");
        let extended: Vec<Bit> = (0..total_width).map(|i| product.get(i).copied().unwrap_or(product_sign)).collect();
        let next = l.adder(&before, &extended, Bit::Zero).0;
        l.connect_flops(&next, &sum);
        l.b.set_prefix("");
        // The cycle after the last, the register holds the whole product.
        let _ = next;
        sum
    }

    /// What the wide memory holds plus a product, or the total over the units, registered for its write.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn wide_sum(&mut self, b: &mut Builder, (name, adder, register, mux): (&str, &str, &str, &str), held: &[Bit], product: &[Bit], (fresh, chain): (Bit, Bit), (bits, fraction): (u32, usize), across: Option<(Bit, &[Bit])>) -> Vec<Bit> {
        let q = b.wire(&format!("{name}_q"), bits);
        // Chained, the sum goes on from its own last total.
        let base = b.wire(&format!("{name}_base"), bits);
        b.inst(&format!("{name}_chain"), mux, &[("s", vec![chain]), ("a", held.to_vec()), ("b", q.clone()), ("y", base.clone())]);
        let keep = b.wire(&format!("{name}_keep"), 1)[0];
        b.not_to(fresh, keep);
        let start = b.wire(&format!("{name}_start"), bits);
        for (from, to) in base.iter().zip(&start) {
            b.and_to(keep, *from, *to);
        }
        let sign = *product.last().expect("producto vacío");
        let extended: Vec<Bit> = std::iter::repeat_n(Bit::Zero, fraction).chain((0..bits as usize - fraction).map(|i| product.get(i).copied().unwrap_or(sign))).collect();
        let sum = b.wire(&format!("{name}_d"), bits);
        let carry = b.wire(&format!("{name}_carry"), 1);
        b.inst(name, adder, &[("a", start), ("b", extended), ("cin", vec![Bit::Zero]), ("sum", sum.clone()), ("cout", carry)]);
        let chosen = match across {
            Some((on, total)) => {
                let y = b.wire(&format!("{name}_chosen"), bits);
                b.inst(&format!("{name}_select"), mux, &[("s", vec![on]), ("a", sum), ("b", total.to_vec()), ("y", y.clone())]);
                y
            }
            None => sum,
        };
        b.inst(&format!("{name}_reg"), register, &[("d", chosen), ("q", q.clone())]);
        q
    }

    pub(crate) fn die(&mut self, c: &DieConfig) -> String {
        let cfg = c.cell;
        let (a, wide) = (cfg.activation_bits, c.wide_bits());
        let word = word_bits(c);
        let addr_bits = c.reg_bits + c.index_bits;
        let cell = self.scaled_cell(c);
        let reg_a = self.register(a);
        let reg_wide = self.register(wide);
        let mux_wide = self.mux_bus(wide);
        let mux_a = self.mux_bus(a);
        let mux_addr = self.mux_bus(addr_bits);
        // Every lane multiplies with a tree, so the clock does not wait on a chain of adders.
        let mult = self.tree_multiplier(crate::multiplier::MultiplierKind::Wallace, a, a, 1);
        let add = self.ripple_adder(a);

        let mut b = Builder::new("llm2chip_die", "Die", "Secuenciador, banco de vectores, unidad vectorial, tabla de funciones, red de activaciones y celdas.");
        let start = b.input("start", 1)[0];
        let entry = b.input("entry", c.microcode_bits);
        let param_write = b.input("param_write", 1)[0];
        let param_index = b.input("param_index", x_bits(c));
        let param_data = b.input("param_data", c.index_bits);
        let load_code = b.input("load_code", 1)[0];
        let code_addr = b.input("code_addr", c.microcode_bits);
        let code_data = b.input("code_data", word);
        let load_vec = b.input("load_vec", 1)[0];
        let vec_addr = b.input("vec_addr", addr_bits);
        let vec_data = b.input("vec_data", a);
        let load_fn = b.input("load_fn", 1)[0];
        let fn_addr = b.input("fn_addr", c.function_bits + c.function_select_bits);
        let cell_ports = (!cfg.fixed_weights).then(|| (b.input("load_cell", c.cells), b.input("load_scale", 1)[0], b.input("cell_addr", cfg.addr_bits)));
        let cell_data = b.input("cell_data", c.scale_bits.max(cfg.weight_bits).max(a));
        let halted = b.output("halted", 1)[0];
        let vec_out = b.output("vec_out", a);
        let link_tx_valid = b.output("link_tx_valid", 1);
        let link_tx_bits = b.output("link_tx_bits", crate::link::wire_bits(a));
        let link_tx_ready = b.input("link_tx_ready", 1);
        let link_rx_valid = b.input("link_rx_valid", 1);
        let link_rx_bits = b.input("link_rx_bits", crate::link::wire_bits(a));
        let link_rx_ready = b.output("link_rx_ready", 1);
        let link_tx_ack = b.input("link_tx_ack", crate::link::ACK_BITS as u32);
        let link_rx_ack = b.output("link_rx_ack", crate::link::ACK_BITS as u32);
        let link_rx_stall = b.input("link_rx_stall", 1);
        // One of each for every lane of the link, so the board can count them all.
        let link_corrected = b.output("link_corrected", c.features.link_lanes() as u32);
        let link_failed = b.output("link_failed", c.features.link_lanes() as u32);
        let link_in = b.wire("link_in", a);
        let link_stall = b.wire("link_stall", 1)[0];
        // A wide link has a lane of its own for each further vector lane, with its pins.
        let link_lanes = c.features.link_lanes();
        let extra_link: Vec<LinkPins> = (1..link_lanes).map(|k| LinkPins::new(&mut b, &format!("link{k}"), a)).collect();
        let link_lane_in: Vec<Vec<Bit>> = (1..link_lanes).map(|k| b.wire(&format!("link_in{k}"), a)).collect();
        let link_lane_out: Vec<Vec<Bit>> = (1..link_lanes).map(|k| b.wire(&format!("link_out{k}"), a)).collect();

        // --- Sequencer ---
        b.group("Secuenciador");
        let pc = b.wire("pc", c.microcode_bits);
        let pc_next = b.wire("pc_next", c.microcode_bits);
        let code = b.wire("code", word);
        // Words wider than one memory macro are split across several side by side.
        for (part, (stored, written)) in code.chunks(64).zip(code_data.chunks(64)).enumerate() {
            b.ram2p(&format!("microcode_{part}"), c.microcode_bits, stored.len() as u32, &[("raddr", pc_next.clone()), ("waddr", code_addr.clone()), ("we", vec![load_code]), ("wdata", written.to_vec()), ("rdata", stored.to_vec())]);
        }

        let mut fields: BTreeMap<&str, Vec<Bit>> = BTreeMap::new();
        let mut offset = 0usize;
        for ((name, stored), (_, full)) in field_layout(c).into_iter().zip(full_layout(c)) {
            // What this microprogram never sets reads as zero, and the logic behind it goes.
            let mut raw: Vec<Bit> = code[offset..offset + stored as usize].to_vec();
            raw.resize(full as usize, Bit::Zero);
            fields.insert(name, raw);
            offset += stored as usize;
        }

        // `running` gates every control line so the idle machine does nothing.
        let halt_flag = b.wire("halted_q", 1)[0];
        let started = b.wire("start_n", 1)[0];
        b.not_to(start, started);
        let not_halted = b.wire("not_halted", 1)[0];
        b.not_to(halt_flag, not_halted);
        let running = b.wire("running", 1)[0];
        let unstalled = b.wire("not_stalled", 1)[0];
        b.not_to(link_stall, unstalled);
        let active = b.wire("sequencer_active", 1)[0];
        b.and_to(not_halted, started, active);
        b.and_to(active, unstalled, running);
        let gated = |b: &mut Builder, name: &str, bit: Bit| -> Bit {
            let y = b.wire(name, 1)[0];
            b.and_to(running, bit, y);
            y
        };
        let halt_now = gated(&mut b, "ctl_halt", fields["halt"][0]);
        let set_halt = bus_or(&mut b, halt_flag, halt_now, "halt_d_raw");
        let halt_d = b.wire("halt_d", 1)[0];
        b.and_to(set_halt, started, halt_d);
        b.dff_to(halt_d, halt_flag);
        b.xor_to(halt_flag, Bit::Zero, halted);

        // Repeat counter: counts up to the word's repeat field, then the PC moves on.
        // A word may repeat as often as an index register says, in elements or in blocks.
        let repeat_limit = b.wire("repeat_limit", c.index_bits);
        let lane_shift = c.features.lane_bits as usize;
        let repeat_bits = REPEAT_BITS.max(c.index_bits);
        let repeat_module = self.address_counter(repeat_bits);
        let repeat = b.wire("repeat_count", repeat_bits);
        let word_done = b.wire("word_done", 1)[0];
        let clear_repeat = bus_or(&mut b, word_done, start, "repeat_clear");
        b.inst("repeat_counter", &repeat_module, &[("clear", vec![clear_repeat]), ("advance", vec![running]), ("address", repeat.clone())]);
        let target: Vec<Bit> = {
            let mut l = Logic { b: &mut b, next: 49_000 };
            let from_register = fields["repeat_x"][0];
            let shifted: Vec<Bit> = (0..repeat_bits as usize).map(|i| repeat_limit.get(i + lane_shift).copied().unwrap_or(Bit::Zero)).collect();
            let plain: Vec<Bit> = (0..repeat_bits as usize).map(|i| repeat_limit.get(i).copied().unwrap_or(Bit::Zero)).collect();
            let counted = l.mux_bus(fields["vector"][0], &plain, &shifted);
            let literal: Vec<Bit> = (0..repeat_bits as usize).map(|i| fields["repeat"].get(i).copied().unwrap_or(Bit::Zero)).collect();
            l.mux_bus(from_register, &literal, &counted)
        };
        let equal = {
            let mut l = Logic { b: &mut b, next: 58_900 };
            let differ: Vec<Bit> = (0..repeat_bits as usize).map(|i| l.xor(repeat[i], target[i])).collect();
            let any = l.any(&differ);
            l.not(any)
        };
        b.and_to(equal, running, word_done);

        // The memory reads the next address, so its output always matches the PC.
        let pc_adder = self.ripple_adder(c.microcode_bits);
        let pc_sum = b.wire("pc_sum", c.microcode_bits);
        let pc_carry = b.wire("pc_carry", 1);
        b.inst("pc_adder", &pc_adder, &[("a", pc.clone()), ("b", vec![Bit::Zero; c.microcode_bits as usize]), ("cin", vec![word_done]), ("sum", pc_sum.clone()), ("cout", pc_carry)]);
        let pc_reg = self.register(c.microcode_bits);
        b.inst("pc_reg", &pc_reg, &[("d", pc_next.clone()), ("q", pc.clone())]);

        let ctl = |name: &'static str, b: &mut Builder, fields: &BTreeMap<&str, Vec<Bit>>| -> Bit {
            let out = b.wire(&format!("ctl_{name}"), 1)[0];
            b.and_to(running, fields[name][0], out);
            out
        };
        let cell_seek_weights = ctl("cell_seek_weights", &mut b, &fields);
        // How many codes each word holds in the matrix the cells were last pointed at.
        let cell_packing = {
            let mut l = Logic { b: &mut b, next: 59_600 };
            let width = crate::cell::bits_for_packing(cfg.most_packing()) as usize;
            let mode = l.fresh_bus(width);
            let field: Vec<Bit> = (0..width).map(|i| fields["cell_packing"].get(i).copied().unwrap_or(Bit::Zero)).collect();
            let next = l.mux_bus(cell_seek_weights, &mode, &field);
            l.connect_flops(&next, &mode);
            mode
        };
        let cell_seek_factors = ctl("cell_seek_factors", &mut b, &fields);
        let cell_clear = ctl("cell_clear", &mut b, &fields);
        let cell_clear_group = ctl("cell_clear_group", &mut b, &fields);
        // An activation read with `cell_feed` reaches the cells `x_delay` cycles later, and
        // the last of a group is in their totals `latency` cycles after that: delay lines
        // run and flush them, so the next group's reads follow with no pause.
        let (cell_run, cell_flush, cell_end_now, cell_flush_now) = {
            let delay = |b: &mut Builder, name: &str, from: Bit, cycles: usize| -> Bit {
                let mut bit = from;
                for stage in 0..cycles {
                    let q = b.wire(&format!("{name}_late{stage}"), 1)[0];
                    b.dff_to(bit, q);
                    bit = q;
                }
                bit
            };
            let x_delay = 1 + c.broadcast_stages as usize;
            let feed = ctl("cell_feed", &mut b, &fields);
            let end = ctl("cell_end", &mut b, &fields);
            let run_late = delay(&mut b, "cell_feed", feed, x_delay);
            let flush_late = delay(&mut b, "cell_end", end, x_delay + cfg.latency() as usize);
            let run = ctl("cell_run", &mut b, &fields);
            let flush = ctl("cell_flush", &mut b, &fields);
            (bus_or(&mut b, run, run_late, "cell_run_any"), bus_or(&mut b, flush, flush_late, "cell_flush_any"), end, flush)
        };
        let chain_load = ctl("chain_load", &mut b, &fields);
        let chain_shift = ctl("chain_shift", &mut b, &fields);
        // Cells that apply their factors over several cycles hear when each flush started.
        let factor_steps = c.features.factor_steps as usize;
        let factor_pins: Vec<(&str, Vec<Bit>)> = if factor_steps > 1 {
            let mut bit = cell_flush;
            let mut late = Vec::with_capacity(factor_steps);
            for stage in 0..factor_steps {
                let q = b.wire(&format!("factor_late{stage}"), 1)[0];
                b.dff_to(bit, q);
                late.push(q);
                bit = q;
            }
            let q = b.wire("factor_late_done", 1)[0];
            b.dff_to(late[factor_steps - 1], q);
            vec![("factor_first", vec![late[0]]), ("factor_done", vec![q])]
        } else {
            Vec::new()
        };

        // Every vector unit runs the same words. Each has its own registers and index
        // registers, so inside a loop over heads each one works on a different head.
        let clusters = 1usize << c.features.cluster_bits;
        let lane_bits = c.features.lane_bits as usize;
        let lanes = 1usize << lane_bits;
        let cluster_banks: Vec<Vec<Vec<Bit>>> = (0..clusters).map(|k| (0..lanes).map(|lane| b.wire(&format!("cluster{k}_bank{lane}"), a)).collect()).collect();
        let gather_index = b.wire("gather_index", c.index_bits);
        let (foreign, shared_banks): (Bit, Vec<Vec<Bit>>) = {
            let mut l = Logic { b: &mut b, next: 50_000 };
            // In a gathering stream the block being read says which unit has it.
            let gathering = l.any(&fields["gather_shift"]);
            let mut owner = gather_index.clone();
            for (level, select) in fields["gather_shift"].iter().enumerate() {
                let step = 1usize << level;
                let moved: Vec<Bit> = (0..owner.len()).map(|i| owner.get(i + step).copied().unwrap_or(Bit::Zero)).collect();
                owner = l.mux_bus(*select, &owner, &moved);
            }
            let unit_bits = c.features.cluster_bits as usize;
            // The unit named with a read gives the data that arrives a cycle later, so the
            // cycle a step computes in can already carry the next step's read.
            let picked: Vec<Bit> = (0..clusters as u32).map(|k| {
                let now = l.equals(&fields["from_cluster"], k + 1);
                let named = l.fresh();
                l.connect_flops(&[now], &[named]);
                let owns = l.equals(&owner[..unit_bits.min(owner.len())], k);
                let owns = l.and(owns, gathering);
                l.or(named, owns)
            }).collect();
            let foreign = l.any(&picked);
            let banks = (0..lanes).map(|lane| (0..a as usize).map(|bit| {
                let terms: Vec<Bit> = picked.iter().zip(&cluster_banks).map(|(on, banks)| l.and(*on, banks[lane][bit])).collect();
                l.any(&terms)
            }).collect()).collect();
            (foreign, banks)
        };
        let unit_chains: Vec<Vec<Vec<Bit>>> = (0..clusters).map(|k| (0..lanes).map(|lane| b.wire(&format!("unit{k}_chain{lane}"), wide)).collect()).collect();
        let wide_bits = 2 * a + 16 + FACTOR_FRACTION;
        let unit_wide: Vec<Vec<Vec<Bit>>> = (0..clusters).map(|k| (0..lanes).map(|lane| b.wire(&format!("unit{k}_wide{lane}"), wide_bits)).collect()).collect();
        let unit_wide_magnitude: Vec<Vec<Bit>> = (0..clusters).map(|k| b.wire(&format!("unit{k}_wide_magnitude"), 2 * a + 15)).collect();
        let group_wide_magnitude: Vec<Vec<Bit>> = (0..clusters).map(|k| b.wire(&format!("unit{k}_group_wide_magnitude"), 2 * a + 15)).collect();
        let unit_accumulator: Vec<Vec<Bit>> = (0..clusters).map(|k| b.wire(&format!("unit{k}_accumulator"), 2 * a + 16)).collect();
        let group_accumulator: Vec<Vec<Bit>> = (0..clusters).map(|k| b.wire(&format!("unit{k}_group_accumulator"), 2 * a + 16)).collect();
        let unit_magnitude: Vec<Vec<Bit>> = (0..clusters).map(|k| b.wire(&format!("unit{k}_magnitude"), a - 1)).collect();
        let group_magnitude: Vec<Vec<Bit>> = (0..clusters).map(|k| b.wire(&format!("unit{k}_group_magnitude"), a - 1)).collect();
        // The largest operand A over a group, for a maximum that every unit of it agrees on.
        let shares_max = clusters > 1 && c.features.uses(OP_MAX);
        let unit_max: Vec<Vec<Bit>> = (0..clusters).map(|k| b.wire(&format!("unit{k}_max"), a)).collect();
        let group_max: Vec<Vec<Bit>> = (0..clusters).map(|k| b.wire(&format!("unit{k}_group_max"), a)).collect();
        let reduced: Vec<Vec<Vec<Bit>>> = (0..clusters).map(|k| (0..lanes).map(|lane| b.wire(&format!("unit{k}_reduced{lane}"), wide_bits)).collect()).collect();
        // What bit `j` of a unit's number adds to an index written with `x_scaled`.
        let bit_strides: Vec<Vec<Bit>> = {
            b.set_prefix("xsplit_");
            let width = c.index_bits as usize;
            let mut l = Logic { b: &mut b, next: 0 };
            let unit_bits = c.features.cluster_bits as usize;
            let options = 1usize << fields["x_split"].len();
            let strides = (0..unit_bits).map(|j| {
                let shifted = |literal: &[Bit], by: usize| -> Vec<Bit> { (0..width).map(|i| if i < by { Bit::Zero } else { literal.get(i - by).copied().unwrap_or(Bit::Zero) }).collect() };
                let leaves: Vec<Vec<Bit>> = (0..options).map(|split| if split <= j { shifted(&fields["literal_a"], j - split) } else if split <= unit_bits { shifted(&fields["literal_dst"], j) } else { vec![Bit::Zero; width] }).collect();
                l.select(&fields["x_split"], &leaves)
            }).collect();
            b.set_prefix("");
            strides
        };
        let global_running = running;
        // With compact vector memory, a table gives each register where it starts and the
        // step between the registers a loop over them visits; it is read once a cycle for
        // every unit, by the register numbers in the word or those the host asks for.
        let compact = c.features.vector_bits > 0;
        let mem_bits = if compact { c.features.vector_bits } else { addr_bits };
        let host_reads_top = bus_or(&mut b, start, halt_flag, "host_reads_top");
        let bases: Option<[(Vec<Bit>, Vec<Bit>); 3]> = compact.then(|| {
            let entry_bits = (mem_bits + BASE_SHIFT_BITS) as usize;
            let load_base = b.input("load_base", 1)[0];
            let base_addr = b.input("base_addr", c.reg_bits);
            let base_data = b.input("base_data", mem_bits + BASE_SHIFT_BITS);
            b.group("Tabla de registros");
            let mut l = Logic { b: &mut b, next: 70_000 };
            let entries: Vec<Vec<Bit>> = (0..1u32 << c.reg_bits)
                .map(|e| {
                    let q = l.fresh_bus(entry_bits);
                    let here = l.equals(&base_addr, e);
                    let write = l.and(load_base, here);
                    let next = l.mux_bus(write, &q, &base_data);
                    l.connect_flops(&next, &q);
                    q
                })
                .collect();
            let vec_reg = vec_addr[c.index_bits as usize..].to_vec();
            let key_a = l.mux_bus(host_reads_top, &fields["reg_a"].iter().copied().chain(std::iter::repeat(Bit::Zero)).take(c.reg_bits as usize).collect::<Vec<_>>(), &vec_reg);
            let key_w = l.mux_bus(load_vec, &fields["reg_dst"].iter().copied().chain(std::iter::repeat(Bit::Zero)).take(c.reg_bits as usize).collect::<Vec<_>>(), &vec_reg);
            let key_b: Vec<Bit> = fields["reg_b"].iter().copied().chain(std::iter::repeat(Bit::Zero)).take(c.reg_bits as usize).collect();
            [key_a, key_b, key_w].map(|key| {
                let entry = l.select(&key, &entries);
                (entry[..mem_bits as usize].to_vec(), entry[mem_bits as usize..].to_vec())
            })
        });
        let mut first_unit = None;
        let mut unit_ports = Vec::with_capacity(clusters);
        for (cluster, unit_banks) in cluster_banks.iter().enumerate() {
            b.set_prefix(&if clusters > 1 { format!("c{cluster}_") } else { String::new() });
            let enabled = b.wire("unit_enabled", 1)[0];
            let running = if clusters > 1 {
                let mut l = Logic { b: &mut b, next: 51_000 };
                let set_now = l.and(global_running, fields["set_clusters"][0]);
                let counted = at_least(&mut l, &fields["literal_b"], cluster as u32 + 1);
                // With a field of the unit's number given, only those below a limit take part.
                let width = c.features.cluster_bits as usize + 1;
                let leaves: Vec<Vec<Bit>> = (0..1usize << fields["x_split"].len()).map(|shift| (0..width).map(|i| if (cluster >> shift.min(31)) >> i & 1 == 1 { Bit::One } else { Bit::Zero }).collect()).collect();
                let moved = l.select(&fields["x_split"], &leaves);
                let field: Vec<Bit> = moved.iter().enumerate().map(|(i, bit)| {
                    let within = at_least(&mut l, &fields["literal_dst"], i as u32 + 1);
                    l.and(*bit, within)
                }).collect();
                let limit: Vec<Bit> = (0..width).map(|i| fields["literal_a"].get(i).copied().unwrap_or(Bit::Zero)).collect();
                let inverted: Vec<Bit> = limit.iter().map(|bit| l.not(*bit)).collect();
                let reaches = l.adder(&field, &inverted, Bit::One).1;
                let below = l.not(reaches);
                let free = l.any(&fields["literal_dst"]);
                let free = l.not(free);
                let in_slot = l.or(free, below);
                let wanted = l.and(counted, in_slot);
                let next = l.mux(set_now, enabled, wanted);
                let d = l.or(next, start);
                l.connect_flops(&[d], &[enabled]);
                l.and(global_running, enabled)
            } else {
                b.or_to(Bit::One, Bit::Zero, enabled);
                global_running
            };
            let ctl = |name: &'static str, b: &mut Builder, fields: &BTreeMap<&str, Vec<Bit>>| -> Bit {
                let out = b.wire(&format!("ctl_{name}"), 1)[0];
                b.and_to(running, fields[name][0], out);
                out
            };
            let read_step = ctl("read_step", &mut b, &fields);
            let read_reset = ctl("read_reset", &mut b, &fields);
            let write_en = ctl("write_en", &mut b, &fields);
            let wide_write = ctl("wide_write", &mut b, &fields);
            let wide_from_chain = ctl("wide_from_chain", &mut b, &fields);
            let write_reset = ctl("write_reset", &mut b, &fields);
            // Loops and everything that depends on the token or the position live here.
            b.group("Registros de índice");
            let x_write = ctl("x_write", &mut b, &fields);
            let branch = ctl("branch", &mut b, &fields);
            let x_adder = self.ripple_adder(c.index_bits);
            let (offset_a, offset_b, offset_dst);
            let (window_a, window_b, window_dst);
            {
                let width = c.index_bits as usize;
                let mut l = Logic { b: &mut b, next: 0 };
                let stored: Vec<Vec<Bit>> = (0..1usize << x_bits(c)).map(|i| if i == 0 || i as u32 >= c.features.x_registers { vec![Bit::Zero; width] } else { l.fresh_bus(width) }).collect();
                offset_a = l.select(&fields["offset_a"], &stored);
                offset_b = l.select(&fields["offset_b"], &stored);
                offset_dst = l.select(&fields["offset_dst"], &stored);
                let window = l.select(&fields["window_x"], &stored);
                let window: Vec<Bit> = (0..c.reg_bits as usize).map(|i| window.get(i).copied().unwrap_or(Bit::Zero)).collect();
                let gate = |l: &mut Logic, on: Bit| -> Vec<Bit> { window.iter().map(|bit| l.and(*bit, on)).collect() };
                window_a = gate(&mut l, fields["window_a"][0]);
                window_b = gate(&mut l, fields["window_b"][0]);
                window_dst = gate(&mut l, fields["window_dst"][0]);
                let source = l.select(&fields["x_src"], &stored);
                let limit = l.select(&fields["x_limit"], &stored);
                if cluster == 0 {
                    for (from, to) in limit.iter().zip(&repeat_limit) {
                        l.b.or_to(*from, Bit::Zero, *to);
                    }
                }
                let sum = l.fresh_bus(width);
                let carry = l.fresh();
                // Unit `k` adds `k · literal_a` as well, so one word gives every unit its own index.
                let step: Vec<Bit> = if cluster == 0 {
                    fields["literal_b"].clone()
                } else {
                    let scaled = l.and(fields["x_scaled"][0], Bit::One);
                    let mut total: Vec<Bit> = fields["literal_b"].clone();
                    for (bit, stride) in bit_strides.iter().enumerate() {
                        if cluster >> bit & 1 == 1 {
                            let moved: Vec<Bit> = stride.iter().map(|s| l.and(*s, scaled)).collect();
                            total = l.adder(&total, &moved, Bit::Zero).0;
                        }
                    }
                    total
                };
                l.b.inst("x_step", &x_adder, &[("a", source), ("b", step), ("cin", vec![Bit::Zero]), ("sum", sum.clone()), ("cout", vec![carry])]);

                let target_register = l.mux_bus(param_write, &fields["x_dst"], &param_index);
                let data = l.mux_bus(param_write, &sum, &param_data);
                let reaches = {
                    let open = l.equals(&fields["x_only"], 0);
                    let mine = l.equals(&fields["x_only"], cluster as u32 + 1);
                    l.or(open, mine)
                };
                let local_write = l.and(x_write, reaches);
                let any_write = l.or(param_write, local_write);
                for (i, q) in stored.iter().enumerate().take(c.features.x_registers as usize).skip(1) {
                    let selected = l.equals(&target_register, i as u32);
                    let write = l.and(selected, any_write);
                    let d = l.mux_bus(write, q, &data);
                    l.connect_flops(&d, q);
                }

                let differs: Vec<Bit> = sum.iter().zip(&limit).map(|(a, b)| l.xor(*a, *b)).collect();
                let not_finished = l.any(&differs);
                let branch_now = l.and(branch, word_done);
                let taken = l.and(branch_now, not_finished);
                let followed = l.mux_bus(taken, &pc_sum, &fields["target"]);
                let chosen = l.mux_bus(start, &followed, &entry);
                if cluster == 0 {
                    for (from, to) in chosen.iter().zip(&pc_next) {
                        l.b.or_to(*from, Bit::Zero, *to);
                    }
                }
            }

            // --- Vector registers ---
            b.group("Banco de vectores");
            let read_index = index_counter(self, &mut b, ("read_index", 46_000), c.index_bits, (read_reset, read_step), fields["vector"][0], lane_shift);
            if cluster == 0 {
                // The data arriving now was read with the index of the cycle before.
                let mut l = Logic { b: &mut b, next: 56_000 };
                l.connect_flops(&read_index, &gather_index);
            }
            let chain_write = b.wire("chain_write", 1)[0];
            let counted_write = bus_or(&mut b, wide_from_chain, fields["wide_accumulate"][0], "counted_wide_write");
            b.and_to(wide_write, counted_write, chain_write);
            let write_step = bus_or(&mut b, write_en, chain_write, "write_step");
            let block_write = bus_or(&mut b, fields["vector"][0], fields["wide_from_chain"][0], "block_write");
            let write_index = index_counter(self, &mut b, ("write_index", 47_000), c.index_bits, (write_reset, write_step), block_write, lane_shift);
            let join = |reg: &[Bit], index: &[Bit]| -> Vec<Bit> { index.iter().chain(reg).copied().collect() };
            let use_literal = fields["use_literal"][0];
            let mux_index = self.mux_bus(c.index_bits);
            let pick = |b: &mut Builder, name: &str, fixed: Bit, counter: &[Bit], literal: &[Bit]| -> Vec<Bit> {
                let y = b.wire(name, c.index_bits);
                let select = bus_or(b, use_literal, fixed, &format!("{name}_literal"));
                b.inst(&format!("{name}_select"), &mux_index, &[("s", vec![select]), ("a", counter.to_vec()), ("b", literal.to_vec()), ("y", y.clone())]);
                y
            };
            // A can walk the counter more slowly: one of its elements for each block of B.
            let read_index_a = {
                let mut l = Logic { b: &mut b, next: 57_000 };
                let mut index = read_index.clone();
                for (level, bit) in fields["a_shift"].iter().enumerate() {
                    let by = 1usize << level;
                    let shifted: Vec<Bit> = (0..index.len()).map(|i| index.get(i + by).copied().unwrap_or(Bit::Zero)).collect();
                    index = l.mux_bus(*bit, &index, &shifted);
                }
                index
            };
            let plain_a = pick(&mut b, "plain_a", fields["fixed_a"][0], &read_index_a, &fields["literal_a"]);
            let plain_b = pick(&mut b, "plain_b", fields["fixed_b"][0], &read_index, &fields["literal_b"]);
            let plain_w = pick(&mut b, "plain_w", Bit::Zero, &write_index, &fields["literal_dst"]);
            let plain_w = if c.features.key_shift > 0 {
                let shift = c.features.key_shift as usize;
                let spread: Vec<Bit> = (0..c.index_bits as usize).map(|i| if i < shift { Bit::Zero } else { write_index[i - shift] }).collect();
                let y = b.wire("plain_w_spread", c.index_bits);
                b.inst("plain_w_spread_select", &mux_index, &[("s", vec![fields["write_spread"][0]]), ("a", plain_w), ("b", spread), ("y", y.clone())]);
                y
            } else {
                plain_w
            };
            let offset = |b: &mut Builder, name: &str, plain: &[Bit], by: &[Bit]| -> Vec<Bit> {
                let sum = b.wire(name, c.index_bits);
                let carry = b.wire(&format!("{name}_carry"), 1);
                b.inst(&format!("{name}_offset"), &x_adder, &[("a", plain.to_vec()), ("b", by.to_vec()), ("cin", vec![Bit::Zero]), ("sum", sum.clone()), ("cout", carry)]);
                sum
            };
            let index_a = offset(&mut b, "index_a", &plain_a, &offset_a);
            let index_b = offset(&mut b, "index_b", &plain_b, &offset_b);
            let index_w = offset(&mut b, "index_w", &plain_w, &offset_dst);
            // Only a program that loops over registers pays for adding to their numbers.
            let windowed = c.features.widths[WINDOW_X_FIELD] > 0;
            let register_adder = self.ripple_adder(c.reg_bits);
            let register = |b: &mut Builder, name: &str, plain: &[Bit], by: &[Bit]| -> Vec<Bit> {
                if !windowed {
                    return plain.to_vec();
                }
                let sum = b.wire(name, c.reg_bits);
                let carry = b.wire(&format!("{name}_carry"), 1);
                b.inst(&format!("{name}_window"), &register_adder, &[("a", plain.to_vec()), ("b", by.to_vec()), ("cin", vec![Bit::Zero]), ("sum", sum.clone()), ("cout", carry)]);
                sum
            };
            let host_reading = bus_or(&mut b, start, halt_flag, "host_reads");
            let (addr_b, waddr, raddr_a) = match &bases {
                // Base, plus the loop's register step, plus the index; the host gives its
                // own register and index and no step.
                Some([(base_a, shift_a), (base_b, shift_b), (base_w, shift_w)]) => {
                    let mut l = Logic { b: &mut b, next: 71_000 };
                    let vec_index = &vec_addr[..c.index_bits as usize];
                    let mut physical = |base: &[Bit], shift: &[Bit], window: &[Bit], index: &[Bit], host: Bit| -> Vec<Bit> {
                        let index = l.mux_bus(host, index, vec_index);
                        let alone = l.not(host);
                        let window: Vec<Bit> = window.iter().map(|bit| l.and(*bit, alone)).collect();
                        let mut step: Vec<Bit> = (0..mem_bits as usize).map(|i| window.get(i).copied().unwrap_or(Bit::Zero)).collect();
                        for (level, bit) in shift.iter().enumerate() {
                            let by = 1usize << level;
                            let moved: Vec<Bit> = (0..step.len()).map(|i| if i >= by { step[i - by] } else { Bit::Zero }).collect();
                            step = l.mux_bus(*bit, &step, &moved);
                        }
                        let index: Vec<Bit> = (0..mem_bits as usize).map(|i| index.get(i).copied().unwrap_or(Bit::Zero)).collect();
                        let start = l.adder(base, &step, Bit::Zero).0;
                        l.adder(&start, &index, Bit::Zero).0
                    };
                    let raddr_a = physical(base_a, shift_a, &window_a, &index_a, host_reading);
                    let addr_b = physical(base_b, shift_b, &window_b, &index_b, Bit::Zero);
                    let waddr = physical(base_w, shift_w, &window_dst, &index_w, load_vec);
                    let named = |l: &mut Logic, name: &str, bits: Vec<Bit>| {
                        let wire = l.b.wire(name, mem_bits);
                        for (from, to) in bits.iter().zip(&wire) {
                            l.b.or_to(*from, Bit::Zero, *to);
                        }
                        wire
                    };
                    (named(&mut l, "addr_b", addr_b), named(&mut l, "waddr", waddr), named(&mut l, "raddr_a", raddr_a))
                }
                None => {
                    let number_a = register(&mut b, "reg_a_final", &fields["reg_a"], &window_a);
                    let number_b = register(&mut b, "reg_b_final", &fields["reg_b"], &window_b);
                    let number_w = register(&mut b, "reg_dst_final", &fields["reg_dst"], &window_dst);
                    let addr_a = join(&number_a, &index_a);
                    let addr_b = join(&number_b, &index_b);
                    let addr_w = join(&number_w, &index_w);
                    let waddr = b.wire("waddr", addr_bits);
                    b.inst("waddr_select", &mux_addr, &[("s", vec![load_vec]), ("a", addr_w), ("b", vec_addr.clone()), ("y", waddr.clone())]);
                    let raddr_a = b.wire("raddr_a", addr_bits);
                    b.inst("raddr_select", &mux_addr, &[("s", vec![host_reading]), ("a", addr_a), ("b", vec_addr.clone()), ("y", raddr_a.clone())]);
                    (addr_b, waddr, raddr_a)
                }
            };

            let write_data = b.wire("write_data", a);
            let wdata = b.wire("wdata", a);
            // A simple step writes its result as soon as it has it.
            let direct_result = b.wire("direct_result", a);
            let write_now = ctl("write_now", &mut b, &fields);
            let finished = b.wire("finished_result", a);
            b.inst("finished_select", &mux_a, &[("s", vec![write_now]), ("a", write_data.clone()), ("b", direct_result.clone()), ("y", finished.clone())]);
            b.inst("wdata_select", &mux_a, &[("s", vec![load_vec]), ("a", finished), ("b", vec_data.clone()), ("y", wdata.clone())]);
            let any_write = bus_or(&mut b, write_en, write_now, "any_write");
            let we = bus_or(&mut b, any_write, load_vec, "vec_we");

            let lane_products: Vec<Vec<Bit>> = (0..lanes).map(|lane| b.wire(&format!("lane_product{lane}"), 2 * a)).collect();
            // Every lane's product is kept a cycle: products are rounded, and added up, in the
            // cycle after the one they are computed in.
            let held_products: Vec<Vec<Bit>> = (0..lanes).map(|lane| b.wire(&format!("lane_product_held{lane}"), 2 * a)).collect();
            // In a block of a variable-length scan, the last one may be only partly used.
            let scanned_lane: Vec<Bit> = {
                let mut l = Logic { b: &mut b, next: 49_500 };
                let last_block = l.and(fields["repeat_x"][0], word_done);
                let whole = l.not(last_block);
                (0..lanes as u32).map(|lane| {
                    let reached = at_least(&mut l, &repeat_limit[..lane_bits], lane);
                    l.or(whole, reached)
                }).collect()
            };
            // Elements sit in as many banks as lanes, by the low bits of their index, so a
            // whole block can be read and written at once and a single element still can.
            let vector_flag = fields["vector"][0];
            let lane_results: Vec<Vec<Bit>> = (0..lanes).map(|lane| if lane == 0 { write_data.clone() } else { b.wire(&format!("lane{lane}_result"), a) }).collect();
            // What each lane computes this cycle, for a block written as it is computed.
            let lane_direct: Vec<Vec<Bit>> = (0..lanes).map(|lane| if lane == 0 { direct_result.clone() } else { b.wire(&format!("lane{lane}_direct"), a) }).collect();
            let (mut banks_a, mut banks_b) = (Vec::with_capacity(lanes), Vec::with_capacity(lanes));
            {
                let bank_bits = mem_bits - lane_bits as u32;
                let mut l = Logic { b: &mut b, next: 40_000 };
                let written_together = l.and(any_write, vector_flag);
                // A read in the cycle of a write to the same place gets the old value from the
                // memory; each bank keeps what it wrote a cycle and gives that instead, so a
                // step can read what the one before it is writing, a block or one element.
                let same_place = |l: &mut Logic, addr: &[Bit]| -> Bit {
                    let differ: Vec<Bit> = addr[lane_bits..].iter().zip(&waddr[lane_bits..]).map(|(x, y)| l.xor(*x, *y)).collect();
                    let any = l.any(&differ);
                    l.not(any)
                };
                let (same_a, same_b) = (same_place(&mut l, &raddr_a), same_place(&mut l, &addr_b));
                for (lane, (late, direct)) in lane_results.iter().zip(&lane_direct).enumerate() {
                    let here = l.equals(&waddr[..lane_bits], lane as u32);
                    let alone = l.and(we, here);
                    let masked = l.and(written_together, scanned_lane[lane]);
                    let write = l.or(alone, masked);
                    let data = if lane == 0 {
                        wdata.clone()
                    } else {
                        let result = l.mux_bus(write_now, late, direct);
                        l.mux_bus(written_together, &wdata, &result)
                    };
                    let (name_a, name_b) = if lanes == 1 { ("vectors_a".to_string(), "vectors_b".to_string()) } else { (format!("vectors_a{lane}"), format!("vectors_b{lane}")) };
                    let (read_this_a, read_this_b) = (l.b.wire(&format!("{name_a}_data"), a), l.b.wire(&format!("{name_b}_data"), a));
                    l.b.ram2p(&name_a, bank_bits, a, &[("raddr", raddr_a[lane_bits..].to_vec()), ("waddr", waddr[lane_bits..].to_vec()), ("we", vec![write]), ("wdata", data.clone()), ("rdata", read_this_a.clone())]);
                    l.b.ram2p(&name_b, bank_bits, a, &[("raddr", addr_b[lane_bits..].to_vec()), ("waddr", waddr[lane_bits..].to_vec()), ("we", vec![write]), ("wdata", data.clone()), ("rdata", read_this_b.clone())]);
                    let written = l.fresh_bus(a as usize);
                    l.connect_flops(&data, &written);
                    let take = |l: &mut Logic, same: Bit, memory: Vec<Bit>| -> Vec<Bit> {
                        let now = l.and(write, same);
                        let late = l.fresh();
                        l.connect_flops(&[now], &[late]);
                        l.mux_bus(late, &memory, &written)
                    };
                    let (own_a, own_b) = (take(&mut l, same_a, read_this_a), take(&mut l, same_b, read_this_b));
                    banks_a.push(own_a);
                    banks_b.push(own_b);
                }
            }
            // Reading a register of another unit takes its lanes instead of this unit's.
            let banks_a: Vec<Vec<Bit>> = {
                let mut l = Logic { b: &mut b, next: 52_000 };
                banks_a.iter().enumerate().map(|(lane, own)| {
                    for (from, to) in own.iter().zip(&unit_banks[lane]) {
                        l.b.or_to(*from, Bit::Zero, *to);
                    }
                    l.mux_bus(foreign, own, &shared_banks[lane])
                }).collect()
            };
            let read_a = b.wire("read_a", a);
            let read_b = b.wire("read_b", a);
            {
                let mut l = Logic { b: &mut b, next: 45_000 };
                let (bank_a, bank_b) = (l.fresh_bus(lane_bits), l.fresh_bus(lane_bits));
                l.connect_flops(&raddr_a[..lane_bits], &bank_a);
                l.connect_flops(&addr_b[..lane_bits], &bank_b);
                let (chosen_a, chosen_b) = (l.select(&bank_a, &banks_a), l.select(&bank_b, &banks_b));
                for (from, to) in chosen_a.iter().zip(&read_a).chain(chosen_b.iter().zip(&read_b)) {
                    l.b.or_to(*from, Bit::Zero, *to);
                }
            }
            for i in 0..a as usize {
                if cluster == 0 {
                    b.xor_to(read_a[i], Bit::Zero, vec_out[i]);
                }
            }

            // --- Vector unit ---
            b.group("Desplazamientos dinámicos");
            let load_shift_a = ctl("load_shift_a", &mut b, &fields);
            let load_shift_b = ctl("load_shift_b", &mut b, &fields);
            let load_shift_left = ctl("load_shift_left", &mut b, &fields);
            let shift_mux = self.mux_bus(SHIFT_REG_BITS);
            let shift_reg = self.register(SHIFT_REG_BITS);
            let held_shift = |b: &mut Builder, name: &str, load: Bit, data: &[Bit]| -> Vec<Bit> {
                let q = b.wire(name, SHIFT_REG_BITS);
                let d = b.wire(&format!("{name}_d"), SHIFT_REG_BITS);
                b.inst(&format!("{name}_select"), &shift_mux, &[("s", vec![load]), ("a", q.clone()), ("b", data.to_vec()), ("y", d.clone())]);
                b.inst(&format!("{name}_reg"), &shift_reg, &[("d", d), ("q", q.clone())]);
                q
            };
            // Loaded from the result as it is written, or straight from the operands read.
            // A move loads what it reads; any other step loads its own result as it computes
            // it, so a count of leading zeros sets the shifter in the same step.
            let (from_a, from_b) = {
                let now = ctl("shift_now", &mut b, &fields);
                let mut l = Logic { b: &mut b, next: 46_000 };
                let bits = SHIFT_REG_BITS as usize;
                let low = &write_data[..bits];
                let moving = l.equals(&fields["op"], OP_MOV);
                let at_once = l.mux_bus(moving, &direct_result[..bits], &read_a[..bits]);
                (l.mux_bus(now, low, &at_once), l.mux_bus(now, low, &read_b[..bits]))
            };
            let shift_a = held_shift(&mut b, "shift_a", load_shift_a, &from_a);
            let shift_b = held_shift(&mut b, "shift_b", load_shift_b, &from_b);
            let shift_left = held_shift(&mut b, "shift_left", load_shift_left, &from_a);
            let aligned_a = self.rounded_right(&mut b, "align_a", &read_a, &shift_a, a as usize);
            let aligned_b = self.rounded_right(&mut b, "align_b", &read_b, &shift_b, a as usize);

            b.group("Unidad vectorial");
            let product = b.wire("product", 2 * a);
            b.inst("multiplier", &mult, &[("weight", read_a.clone()), ("x", read_b.clone()), ("product", product.clone())]);
            Logic { b: &mut b, next: 59_300 }.connect_flops(&product, &held_products[0]);
            let op = fields["op"].clone();
            // The adder subtracts for OP_SUB, whose bit 2 is set.
            let subtract = op[2];
            let flipped = b.wire("b_flipped", a);
            for i in 0..a as usize {
                b.xor_to(aligned_b[i], subtract, flipped[i]);
            }
            let sum = b.wire("sum", a);
            let unused = b.wire("sum_carry", 1);
            b.inst("adder", &add, &[("a", aligned_a.clone()), ("b", flipped), ("cin", vec![subtract]), ("sum", sum.clone()), ("cout", unused)]);
            let shifted = self.rounded_right(&mut b, "product_shift", &held_products[0], &fields["shift"], a as usize);

            b.group("Comparador");
            // A unit sitting out a loop offers the smallest value, so it never wins.
            let max_a = if shares_max {
                let smallest: Vec<Bit> = (0..a as usize).map(|i| if i + 1 == a as usize { Bit::One } else { Bit::Zero }).collect();
                b.inst("max_offer", &mux_a, &[("s", vec![enabled]), ("a", smallest), ("b", aligned_a.clone()), ("y", unit_max[cluster].clone())]);
                // Over a group the network gives it, some cycles later; alone, A itself.
                let mut l = Logic { b: &mut b, next: 59_100 };
                let grouped = l.any(&fields["reduce_group"]);
                l.mux_bus(grouped, &aligned_a, &group_max[cluster])
            } else {
                aligned_a.clone()
            };
            let inverted = b.wire("cmp_inverted", a);
            for i in 0..a as usize {
                b.not_to(aligned_b[i], inverted[i]);
            }
            let difference = b.wire("difference", a);
            let unused = b.wire("cmp_carry", 1);
            b.inst("comparator", &add, &[("a", max_a.clone()), ("b", inverted), ("cin", vec![Bit::One]), ("sum", difference.clone()), ("cout", unused)]);
            let top = a as usize - 1;
            let signs_differ = b.wire("signs_differ", 1)[0];
            b.xor_to(max_a[top], aligned_b[top], signs_differ);
            let a_is_less = b.wire("a_is_less", 1)[0];
            b.mux_to(signs_differ, difference[top], max_a[top], a_is_less);
            let maximum = b.wire("maximum", a);
            b.inst("max_select", &mux_a, &[("s", vec![a_is_less]), ("a", max_a), ("b", aligned_b), ("y", maximum.clone())]);

            b.group("Detector de magnitud");
            let scan = ctl("magnitude_scan", &mut b, &fields);
            let magnitude_clear = ctl("magnitude_clear", &mut b, &fields);
            let magnitude = b.wire("magnitude", a);
            let keep_magnitude = b.wire("magnitude_keep", 1)[0];
            b.not_to(magnitude_clear, keep_magnitude);
            let magnitude_d = b.wire("magnitude_d", a);
            let block_scan = b.wire("magnitude_block", 1)[0];
            b.and_to(scan, vector_flag, block_scan);
            for i in 0..a as usize {
                let differs = b.wire(&format!("mag_differs{i}"), 1)[0];
                if i == top {
                    b.xor_to(Bit::Zero, Bit::Zero, differs);
                } else {
                    // A scan over a block looks at every lane, in a tree; the others count only then.
                    let mut l = Logic { b: &mut b, next: 59_200 };
                    let mut bits = vec![l.xor(read_a[i], read_a[top])];
                    for (lane, bank) in banks_a.iter().enumerate().skip(1) {
                        let here = l.xor(bank[i], bank[top]);
                        let live = l.and(block_scan, scanned_lane[lane]);
                        bits.push(l.and(live, here));
                    }
                    let any = l.any(&bits);
                    l.b.or_to(any, Bit::Zero, differs);
                }
                let seen = b.wire(&format!("mag_seen{i}"), 1)[0];
                b.and_to(scan, differs, seen);
                let merged = b.wire(&format!("mag_merged{i}"), 1)[0];
                b.or_to(magnitude[i], seen, merged);
                b.and_to(keep_magnitude, merged, magnitude_d[i]);
            }
            b.inst("magnitude_reg", &reg_a, &[("d", magnitude_d), ("q", magnitude.clone())]);
            // A unit sitting out a loop does not count in its group.
            for (from, to) in magnitude[..top].iter().zip(&unit_magnitude[cluster]) {
                b.and_to(*from, enabled, *to);
            }
            let seen_magnitude: Vec<Bit> = if clusters > 1 { group_magnitude[cluster].clone() } else { magnitude[..top].to_vec() };
            let leading = leading_zeros(&mut b, "lzc", &seen_magnitude, a);

            let (best, best_used) = if c.features.widths[MAX_SCAN_FIELD] > 0 {
                b.group("Mayor valor");
                let best = b.wire("best", a);
                let (scan_best, clear_best) = (ctl("max_scan", &mut b, &fields), ctl("max_clear", &mut b, &fields));
                let mut l = Logic { b: &mut b, next: 55_000 };
                let top = a as usize - 1;
                let larger = |l: &mut Logic, x: &[Bit], y: &[Bit]| -> Vec<Bit> {
                    let inverted: Vec<Bit> = y.iter().map(|bit| l.not(*bit)).collect();
                    let difference = l.adder(x, &inverted, Bit::One).0;
                    let signs_differ = l.xor(x[top], y[top]);
                    let less = l.mux(signs_differ, difference[top], x[top]);
                    l.mux_bus(less, x, y)
                };
                // The lanes meet in a tree; a lane left out offers lane 0's value again.
                let block = l.and(vector_flag, scan_best);
                let mut level: Vec<Vec<Bit>> = vec![read_a.clone()];
                for (lane, bank) in banks_a.iter().enumerate().skip(1) {
                    let taken = l.and(block, scanned_lane[lane]);
                    level.push(l.mux_bus(taken, &read_a, bank));
                }
                let mut depth = 0;
                let mut scanning = scan_best;
                while level.len() > 1 {
                    level = level.chunks(2).map(|pair| if pair.len() == 2 { larger(&mut l, &pair[0], &pair[1]) } else { pair[0].clone() }).collect();
                    depth += 1;
                    // A register after every three levels but the last, as `max_scan_stages` counts.
                    if depth % 3 == 0 && level.len() > 1 {
                        level = level.iter().map(|value| {
                            let held = l.fresh_bus(value.len());
                            l.connect_flops(value, &held);
                            held
                        }).collect();
                        let late = l.fresh();
                        l.connect_flops(&[scanning], &[late]);
                        scanning = late;
                    }
                }
                let largest = level.pop().expect("al menos un carril");
                let updated = larger(&mut l, &best, &largest);
                let kept = l.mux_bus(scanning, &best, &updated);
                let lowest: Vec<Bit> = (0..a as usize).map(|i| if i == top { Bit::One } else { Bit::Zero }).collect();
                let next = l.mux_bus(clear_best, &kept, &lowest);
                l.connect_flops(&next, &best);
                (best, true)
            } else {
                (Vec::new(), false)
            };

            b.group("Acumulador vectorial");
            let accumulator_bits = 2 * a + 16;
            let acc_reg = self.register(accumulator_bits);
            let acc_mux = self.mux_bus(accumulator_bits);
            let accumulate = ctl("accumulate", &mut b, &fields);
            let accumulator_clear = ctl("accumulator_clear", &mut b, &fields);
            let accumulator = b.wire("accumulator", accumulator_bits);
            // The products are kept a cycle; then every lane's, and the sum so far, meet in
            // a carry-save tree with one fast adder at the end.
            let (acc_sum, accumulating) = {
                let mut l = Logic { b: &mut b, next: 54_000 };
                let late = |l: &mut Logic, bits: &[Bit]| -> Vec<Bit> {
                    let held = l.fresh_bus(bits.len());
                    l.connect_flops(bits, &held);
                    held
                };
                let accumulating = late(&mut l, &[accumulate])[0];
                let block = late(&mut l, &[vector_flag])[0];
                let mut operands = vec![accumulator.clone()];
                for lane in 0..lanes {
                    let held = &held_products[lane];
                    let sign = held[2 * a as usize - 1];
                    let counted = if lane == 0 { Bit::One } else {
                        let taken = late(&mut l, &[scanned_lane[lane]])[0];
                        l.and(block, taken)
                    };
                    operands.push((0..accumulator_bits as usize).map(|i| l.and(held.get(i).copied().unwrap_or(sign), counted)).collect());
                }
                (l.sum_many(operands), accumulating)
            };
            let acc_next = b.wire("acc_next", accumulator_bits);
            b.inst("acc_select", &acc_mux, &[("s", vec![accumulating]), ("a", accumulator.clone()), ("b", acc_sum), ("y", acc_next.clone())]);
            let keep_acc = b.wire("acc_keep", 1)[0];
            b.not_to(accumulator_clear, keep_acc);
            let acc_d = b.wire("acc_d", accumulator_bits);
            for i in 0..accumulator_bits as usize {
                b.and_to(keep_acc, acc_next[i], acc_d[i]);
            }
            b.inst("accumulator_reg", &acc_reg, &[("d", acc_d), ("q", accumulator.clone())]);
            let acc_shifted = barrel_right(&mut b, &acc_mux, "acc_shift", &accumulator, &fields["shift"]);

            // Each output's full-precision sum waits here until the largest one is known.
            b.group("Memoria ancha");
            let wide_scan = ctl("wide_scan", &mut b, &fields);
            let wide_clear = ctl("wide_clear", &mut b, &fields);
            let load_wide_shift = ctl("load_wide_shift", &mut b, &fields);
            // The wide memory keeps the cells' fractional bits until every unit's part is
            // added, so the result rounds once, as a single unit would.
            let fraction = FACTOR_FRACTION as usize;
            let wide_mux = self.mux_bus(wide_bits);
            let wide_adder = self.ripple_adder(wide_bits);
            let wide_register = self.register(wide_bits);
            let wide_stored = b.wire("wide_stored", wide_bits);
            let wide_read: Vec<Bit> = wide_stored[fraction..].to_vec();
            let chain_out = unit_chains[cluster][0].clone();
            let chain_shifted = barrel_right(&mut b, &mux_wide, "chain_shift", &chain_out, &fields["shift"]);
            for (from, to) in accumulator.iter().zip(&unit_accumulator[cluster]) {
                b.and_to(*from, enabled, *to);
            }
            let stored_sum: Vec<Bit> = if clusters > 1 { group_accumulator[cluster].clone() } else { accumulator.clone() };
            let from_accumulator: Vec<Bit> = std::iter::repeat_n(Bit::Zero, fraction).chain(stored_sum.iter().copied()).collect();
            let wide_data = b.wire("wide_data", wide_bits);
            let from_outside = b.wire("wide_outside", wide_bits);
            // Chains drain through the network that adds the units, which keeps a unit's own
            // total when no level of it is on.
            b.inst("wide_data_select", &wide_mux, &[("s", vec![wide_from_chain]), ("a", from_accumulator), ("b", reduced[cluster][0].clone()), ("y", from_outside.clone())]);
            let combines = c.features.combines();
            let lane_sums: Vec<Vec<Bit>> = (0..lanes).map(|lane| b.wire(&format!("wide_sum{lane}"), wide_bits)).collect();
            if combines {
                b.inst("wide_sum_select", &wide_mux, &[("s", vec![fields["wide_accumulate"][0]]), ("a", from_outside), ("b", lane_sums[0].clone()), ("y", wide_data.clone())]);
            } else {
                for (from, to) in from_outside.iter().zip(&wide_data) {
                    b.or_to(*from, Bit::Zero, *to);
                }
            }
            // A combination walks the wide memory with the counter, whatever operand A reads.
            let wide_raddr = if combines {
                let y = b.wire("wide_raddr", c.index_bits);
                b.inst("wide_raddr_select", &mux_index, &[("s", vec![fields["wide_accumulate"][0]]), ("a", index_a.clone()), ("b", plain_b.clone()), ("y", y.clone())]);
                y
            } else {
                index_a.clone()
            };
            let mut wide_banks = Vec::with_capacity(lanes);
            {
                let mut l = Logic { b: &mut b, next: 48_000 };
                let by_block = l.or(vector_flag, wide_from_chain);
                let together = l.and(wide_write, by_block);
                for (lane, sum) in lane_sums.iter().enumerate() {
                    let here = l.equals(&index_w[..lane_bits], lane as u32);
                    let alone = l.and(wide_write, here);
                    let write = l.or(alone, together);
                    let data = if lane == 0 {
                        wide_data.clone()
                    } else {
                        let block = l.mux_bus(wide_from_chain, sum, &reduced[cluster][lane]);
                        l.mux_bus(together, &wide_data, &block)
                    };
                    let name = if lanes == 1 { "wide_memory".to_string() } else { format!("wide_memory{lane}") };
                    let read = l.b.wire(&format!("{name}_data"), wide_bits);
                    l.b.ram2p(&name, c.index_bits - lane_bits as u32, wide_bits, &[("raddr", wide_raddr[lane_bits..].to_vec()), ("waddr", index_w[lane_bits..].to_vec()), ("we", vec![write]), ("wdata", data), ("rdata", read.clone())]);
                    for (from, to) in read.iter().zip(&unit_wide[cluster][lane]) {
                        l.b.or_to(*from, Bit::Zero, *to);
                    }
                    wide_banks.push(read);
                }
                let bank = l.fresh_bus(lane_bits);
                l.connect_flops(&wide_raddr[..lane_bits], &bank);
                let chosen = l.select(&bank, &wide_banks);
                for (from, to) in chosen.iter().zip(&wide_stored) {
                    l.b.or_to(*from, Bit::Zero, *to);
                }
            }
            let wide_fresh = (fields["wide_fresh"][0], fields["wide_chain"][0]);
            let across_units = (clusters > 1).then_some(fields["wide_reduce"][0]);
            if combines {
                let sum = self.wide_sum(&mut b, ("wide_sum_adder", &wide_adder, &wide_register, &wide_mux), &wide_stored, &product, wide_fresh, (wide_bits, fraction), across_units.map(|on| (on, reduced[cluster][0].as_slice())));
                for (from, to) in sum.iter().zip(&lane_sums[0]) {
                    b.or_to(*from, Bit::Zero, *to);
                }
            }
            let wide_top = accumulator_bits as usize - 1;
            let wide_magnitude = b.wire("wide_magnitude", accumulator_bits - 1);
            let keep_wide = b.wire("wide_keep", 1)[0];
            b.not_to(wide_clear, keep_wide);
            let wide_magnitude_d = b.wire("wide_magnitude_d", accumulator_bits - 1);
            for i in 0..wide_top {
                let differs = {
                    // Every lane at once, in a tree.
                    let mut l = Logic { b: &mut b, next: 53_000 + i * 2048 };
                    let mut bits = vec![l.xor(wide_read[i], wide_read[wide_top])];
                    for (lane, bank) in wide_banks.iter().enumerate().skip(1) {
                        let (value, sign) = (bank[fraction + i], bank[fraction + wide_top]);
                        let here = l.xor(value, sign);
                        let block = l.and(wide_scan, vector_flag);
                        let live = l.and(block, scanned_lane[lane]);
                        bits.push(l.and(live, here));
                    }
                    l.any(&bits)
                };
                let seen = b.wire(&format!("wide_seen{i}"), 1)[0];
                b.and_to(wide_scan, differs, seen);
                let merged = b.wire(&format!("wide_merged{i}"), 1)[0];
                b.or_to(wide_magnitude[i], seen, merged);
                b.and_to(keep_wide, merged, wide_magnitude_d[i]);
            }
            let wide_magnitude_reg = self.register(accumulator_bits - 1);
            b.inst("wide_magnitude_reg", &wide_magnitude_reg, &[("d", wide_magnitude_d), ("q", wide_magnitude.clone())]);
            for (from, to) in wide_magnitude.iter().zip(&unit_wide_magnitude[cluster]) {
                b.and_to(*from, enabled, *to);
            }
            let seen_wide: Vec<Bit> = if clusters > 1 { group_wide_magnitude[cluster].clone() } else { wide_magnitude.clone() };
            let wide_leading = leading_zeros(&mut b, "wide_lzc", &seen_wide, a);
            let wide_shift = held_shift(&mut b, "wide_shift", load_wide_shift, &from_a);
            let wide_shifted = self.rounded_right(&mut b, "wide_right", &wide_read, &wide_shift, a as usize);

            b.group("Desplazamiento a la izquierda");
            let left_bits = 2 * a;
            let left_mux = self.mux_bus(left_bits);
            let raw_sign = read_a[top];
            let widened: Vec<Bit> = (0..left_bits as usize).map(|i| read_a.get(i).copied().unwrap_or(raw_sign)).collect();
            let moved_left = barrel_left(&mut b, &left_mux, "shift_left_barrel", &widened, &shift_left);
            let saturated = saturate(&mut b, &moved_left, a);

            b.group("Tabla de funciones");
            let table_out = b.wire("table_out", a);
            let mut table_index: Vec<Bit> = read_a[(a - c.function_bits) as usize..].to_vec();
            table_index.extend_from_slice(&fields["function"]);
            b.ram2p("function_table", c.function_bits + c.function_select_bits, a, &[("raddr", table_index), ("waddr", fn_addr.clone()), ("we", vec![load_fn]), ("wdata", cell_data[..a as usize].to_vec()), ("rdata", table_out.clone())]);

            // --- Result selection ---
            b.group("Selección de resultado");
            // An immediate's value rides in the jump target, which no step that computes needs,
            // so the next step's read can share its cycle.
            let immediate_sign = *fields["target"].last().expect("destino vacío");
            let immediate: Vec<Bit> = (0..a as usize).map(|i| fields["target"].get(i).copied().unwrap_or(immediate_sign)).collect();
            let mut parity = vec![Bit::Zero; a as usize];
            parity[0] = read_a[0];
            // All-ones-but-the-sign when the two aligned operands match, zero otherwise.
            let any_difference = Logic { b: &mut b, next: 58_950 }.any(&difference);
            let same = b.wire("eq_same", 1)[0];
            b.not_to(any_difference, same);
            // Only the first match in a scan counts, so ties cannot select extra elements.
            let seen = b.wire("eq_seen", 1)[0];
            let first_match = and_not(&mut b, same, seen, "eq_first");
            let mut is_equal = Bit::One;
            for (i, bit) in op.iter().enumerate() {
                let next = b.wire(&format!("is_eq{i}"), 1)[0];
                b.and_to(is_equal, *bit, next);
                is_equal = next;
            }
            let now_seen = b.wire("eq_now", 1)[0];
            b.and_to(is_equal, first_match, now_seen);
            let seen_any = bus_or(&mut b, seen, now_seen, "eq_any_seen");
            let keep_seen = and_not(&mut b, seen_any, magnitude_clear, "eq_seen_d");
            b.dff_to(keep_seen, seen);
            let mut equal = vec![first_match; a as usize];
            equal[a as usize - 1] = Bit::Zero;
            let zero_leaf = vec![Bit::Zero; a as usize];
            let chosen_leaf = |op: u32, leaf: Vec<Bit>| if c.features.uses(op) { leaf } else { zero_leaf.clone() };
            let leaves: Vec<Vec<Bit>> = vec![
                sum.clone(),
                chosen_leaf(OP_MUL, shifted),
                chosen_leaf(OP_FUNCTION, table_out),
                chosen_leaf(OP_CELLS, chain_shifted[..a as usize].to_vec()),
                chosen_leaf(OP_SUB, sum),
                chosen_leaf(OP_MAX, maximum),
                chosen_leaf(OP_LZC, leading),
                chosen_leaf(OP_MOV, aligned_a),
                chosen_leaf(OP_SHIFT_LEFT, saturated),
                chosen_leaf(OP_ACCUMULATOR, acc_shifted[..a as usize].to_vec()),
                chosen_leaf(OP_IMMEDIATE, immediate),
                chosen_leaf(OP_PARITY, parity),
                chosen_leaf(OP_LINK, link_in.clone()),
                chosen_leaf(OP_WIDE_LZC, wide_leading),
                chosen_leaf(OP_WIDE_OUT, wide_shifted),
                chosen_leaf(OP_EQUAL, equal),
            ];
            let result = mux_tree(&mut b, &mux_a, "op", leaves, &op);
            let result = if best_used {
                let y = b.wire("result_or_best", a);
                b.inst("best_select", &mux_a, &[("s", vec![fields["max_out"][0]]), ("a", result), ("b", best.clone()), ("y", y.clone())]);
                y
            } else {
                result
            };
            for (from, to) in result.iter().zip(&direct_result) {
                b.or_to(*from, Bit::Zero, *to);
            }
            // While the link holds the sequencer back, the result in flight must stay put.
            let result_held = b.wire("result_held", a);
            b.inst("result_hold", &mux_a, &[("s", vec![link_stall]), ("a", result), ("b", write_data.clone()), ("y", result_held.clone())]);
            b.inst("result_reg", &reg_a, &[("d", result_held), ("q", write_data.clone())]);

            // Every further lane repeats the element-wise operations on its own bank.
            for lane in 1..lanes {
                let name = |part: &str| format!("lane{lane}_{part}");
                b.group(&format!("Carril {lane}"));
                let operand_a = {
                    let y = b.wire(&name("a"), a);
                    b.inst(&name("a_select"), &mux_a, &[("s", vec![fields["broadcast_a"][0]]), ("a", banks_a[lane].clone()), ("b", read_a.clone()), ("y", y.clone())]);
                    y
                };
                let operand_b = {
                    let y = b.wire(&name("b"), a);
                    b.inst(&name("b_select"), &mux_a, &[("s", vec![fields["broadcast_b"][0]]), ("a", banks_b[lane].clone()), ("b", read_b.clone()), ("y", y.clone())]);
                    y
                };
                let lane_a = self.rounded_right(&mut b, &name("align_a"), &operand_a, &shift_a, a as usize);
                let lane_b = self.rounded_right(&mut b, &name("align_b"), &operand_b, &shift_b, a as usize);
                let product = lane_products[lane].clone();
                b.inst(&name("multiplier"), &mult, &[("weight", operand_a.clone()), ("x", operand_b.clone()), ("product", product.clone())]);
                Logic { b: &mut b, next: 59_400 }.connect_flops(&product, &held_products[lane]);
                let flipped = b.wire(&name("flipped"), a);
                for i in 0..a as usize {
                    b.xor_to(lane_b[i], subtract, flipped[i]);
                }
                let sum = b.wire(&name("sum"), a);
                let unused = b.wire(&name("carry"), 1);
                b.inst(&name("adder"), &add, &[("a", lane_a.clone()), ("b", flipped), ("cin", vec![subtract]), ("sum", sum.clone()), ("cout", unused)]);
                let shifted = self.rounded_right(&mut b, &name("product_shift"), &held_products[lane], &fields["shift"], a as usize);
                let table = b.wire(&name("table"), a);
                let mut index: Vec<Bit> = operand_a[(a - c.function_bits) as usize..].to_vec();
                index.extend_from_slice(&fields["function"]);
                b.ram2p(&name("function_table"), c.function_bits + c.function_select_bits, a, &[("raddr", index), ("waddr", fn_addr.clone()), ("we", vec![load_fn]), ("wdata", cell_data[..a as usize].to_vec()), ("rdata", table.clone())]);
                let raw_sign = operand_a[top];
                let widened: Vec<Bit> = (0..left_bits as usize).map(|i| operand_a.get(i).copied().unwrap_or(raw_sign)).collect();
                let moved_left = barrel_left(&mut b, &left_mux, &name("shift_left"), &widened, &shift_left);
                let saturated = saturate(&mut b, &moved_left, a);
                let mut leaves = vec![zero_leaf.clone(); 16];
                leaves[OP_ADD as usize] = sum.clone();
                    let wide_out = self.rounded_right(&mut b, &name("wide_right"), &wide_banks[lane][fraction..], &wide_shift, a as usize);
            let link_leaf = link_lane_in.get(lane - 1).cloned().unwrap_or_else(|| zero_leaf.clone());
            for (op, leaf) in [(OP_MUL, shifted), (OP_FUNCTION, table), (OP_SUB, sum), (OP_MOV, lane_a), (OP_SHIFT_LEFT, saturated), (OP_WIDE_OUT, wide_out), (OP_LINK, link_leaf)] {
                    if c.features.uses(op) {
                        leaves[op as usize] = leaf;
                    }
                }
                if combines {
                    let (adder_name, sum) = (name("wide_sum"), &lane_sums[lane]);
                    let total = self.wide_sum(&mut b, (&adder_name, &wide_adder, &wide_register, &wide_mux), &wide_banks[lane], &product, wide_fresh, (wide_bits, fraction), across_units.map(|on| (on, reduced[cluster][lane].as_slice())));
                    for (from, to) in total.iter().zip(sum) {
                        b.or_to(*from, Bit::Zero, *to);
                    }
                }
                let chosen = mux_tree(&mut b, &mux_a, &name("op"), leaves, &op);
                for (from, to) in chosen.iter().zip(&lane_direct[lane]) {
                    b.or_to(*from, Bit::Zero, *to);
                }
                let held = b.wire(&name("held"), a);
                b.inst(&name("hold"), &mux_a, &[("s", vec![link_stall]), ("a", chosen), ("b", lane_results[lane].clone()), ("y", held.clone())]);
                b.inst(&name("result_reg"), &reg_a, &[("d", held), ("q", lane_results[lane].clone())]);
            }

            if cluster == 0 {
                first_unit = Some(write_data);
                for (from, to) in lane_results[1..].iter().zip(&link_lane_out) {
                    for (bit, out) in from.iter().zip(to) {
                        b.or_to(*bit, Bit::Zero, *out);
                    }
                }
            }
            unit_ports.push((read_a, index_a));
        }
        b.set_prefix("");
        let write_data = first_unit.expect("al menos una unidad");
        {
            // Each level adds a unit's value to its neighbour's, so after `g` levels every
            // unit in a group of `1 << g` holds the group's total.
            let mut l = Logic { b: &mut b, next: 60_000 };
            let levels = c.features.cluster_bits as usize;
            let on: Vec<Bit> = (0..levels).map(|level| at_least(&mut l, &fields["reduce_group"], level as u32 + 1)).collect();
            let mut seen: Vec<Vec<Bit>> = unit_magnitude.clone();
            for (level, active) in on.iter().enumerate() {
                seen = (0..clusters).map(|k| {
                    let merged: Vec<Bit> = seen[k].iter().zip(&seen[k ^ (1 << level)]).map(|(x, y)| l.or(*x, *y)).collect();
                    l.mux_bus(*active, &seen[k], &merged)
                }).collect();
            }
            for (value, output) in seen.iter().zip(&group_magnitude) {
                for (from, to) in value.iter().zip(output) {
                    l.b.or_to(*from, Bit::Zero, *to);
                }
            }
            if shares_max {
                let top = a as usize - 1;
                let mut largest: Vec<Vec<Bit>> = unit_max.clone();
                for (level, active) in on.iter().enumerate() {
                    largest = (0..clusters).map(|k| {
                        let (x, y) = (&largest[k], &largest[k ^ (1 << level)]);
                        let inverted: Vec<Bit> = y.iter().map(|bit| l.not(*bit)).collect();
                        let difference = l.adder(x, &inverted, Bit::One).0;
                        let signs_differ = l.xor(x[top], y[top]);
                        let less = l.mux(signs_differ, difference[top], x[top]);
                        let larger = l.mux_bus(less, x, y);
                        l.mux_bus(*active, x, &larger)
                    }).collect();
                    // Registers as `max_stages` counts them, the last before the last level.
                    if (on.len() - 1 - level) % 2 == 1 {
                        largest = largest.iter().map(|value| {
                            let held = l.fresh_bus(value.len());
                            l.connect_flops(value, &held);
                            held
                        }).collect();
                    }
                }
                for (value, output) in largest.iter().zip(&group_max) {
                    for (from, to) in value.iter().zip(output) {
                        l.b.or_to(*from, Bit::Zero, *to);
                    }
                }
            }
            let mut wide_seen: Vec<Vec<Bit>> = unit_wide_magnitude.clone();
            let mut sums: Vec<Vec<Bit>> = unit_accumulator.clone();
            for (level, active) in on.iter().enumerate() {
                wide_seen = (0..clusters).map(|k| {
                    let merged: Vec<Bit> = wide_seen[k].iter().zip(&wide_seen[k ^ (1 << level)]).map(|(x, y)| l.or(*x, *y)).collect();
                    l.mux_bus(*active, &wide_seen[k], &merged)
                }).collect();
                sums = (0..clusters).map(|k| {
                    let added = l.adder(&sums[k], &sums[k ^ (1 << level)], Bit::Zero).0;
                    l.mux_bus(*active, &sums[k], &added)
                }).collect();
            }
            for (values, outputs) in [(&wide_seen, &group_wide_magnitude), (&sums, &group_accumulator)] {
                for (value, output) in values.iter().zip(outputs) {
                    for (from, to) in value.iter().zip(output) {
                        l.b.or_to(*from, Bit::Zero, *to);
                    }
                }
            }
            // While the chains drain, the network adds what they give out instead, so every
            // unit stores the total over all of them in the same cycle.
            let draining = fields["wide_from_chain"][0];
            for lane in 0..lanes {
                let mut values: Vec<Vec<Bit>> = unit_wide.iter().zip(&unit_chains).map(|(unit, chains)| {
                    let sign = *chains[lane].last().expect("cadena vacía");
                    let chain: Vec<Bit> = (0..wide_bits as usize).map(|i| chains[lane].get(i).copied().unwrap_or(sign)).collect();
                    l.mux_bus(draining, &unit[lane], &chain)
                }).collect();
                for (level, active) in on.iter().enumerate() {
                    values = (0..clusters).map(|k| {
                        let sum = l.adder(&values[k], &values[k ^ (1 << level)], Bit::Zero).0;
                        l.mux_bus(*active, &values[k], &sum)
                    }).collect();
                }
                for (unit, value) in values.iter().enumerate() {
                    for (from, to) in value.iter().zip(&reduced[unit][lane]) {
                        l.b.or_to(*from, Bit::Zero, *to);
                    }
                }
            }
        }
        let op = fields["op"].clone();

        // The sequencer waits when it needs a word that has not arrived or cannot send one.
        b.group("Enlace entre dies");
        let link_send = ctl("link_send", &mut b, &fields);
        let has_link = c.features.uses(OP_LINK) || c.features.widths[LINK_SEND_FIELD] > 0;
        let endpoint = self.link_endpoint(a);
        let can_send = b.wire("link_can_send", 1)[0];
        let has_data = b.wire("link_has_data", 1)[0];
        let link_pop = b.wire("link_pop", 1)[0];
        {
            let mut l = Logic { b: &mut b, next: 20_000 };
            let reads_link = l.equals(&op, OP_LINK);
            let empty = l.not(has_data);
            let starved = l.and(reads_link, empty);
            let blocked = {
                let full = l.not(can_send);
                l.and(fields["link_send"][0], full)
            };
            let stall = l.or(starved, blocked);
            l.b.or_to(stall, Bit::Zero, link_stall);
            let pop = l.and(reads_link, running);
            l.b.or_to(pop, Bit::Zero, link_pop);
        }
        if has_link {
            // Every lane of the link sends and takes a word together: a block goes when all
            // can send, and is taken when all have one.
            let lanes_in = std::iter::once(LinkPins { tx_valid: link_tx_valid, tx_bits: link_tx_bits, tx_ready: link_tx_ready, rx_valid: link_rx_valid, rx_bits: link_rx_bits, rx_ready: link_rx_ready, tx_ack: link_tx_ack, rx_ack: link_rx_ack.clone() }).chain(extra_link);
            let data_out = std::iter::once(write_data).chain(link_lane_out);
            let data_in = std::iter::once(link_in).chain(link_lane_in);
            let (mut sendable, mut ready, mut corrected, mut failed) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for (k, ((pins, out), into)) in lanes_in.zip(data_out).zip(data_in).enumerate() {
                let name = if k == 0 { "link".to_string() } else { format!("link{k}") };
                let (lane_can, lane_has, lane_corrected, lane_failed) = (b.wire(&format!("{name}_can_send"), 1)[0], b.wire(&format!("{name}_has_data"), 1)[0], b.wire(&format!("{name}_corrected"), 1)[0], b.wire(&format!("{name}_failed"), 1)[0]);
                b.inst(&name, &endpoint, &[
                    ("tx_send", vec![link_send]),
                    ("tx_data", out),
                    ("tx_can_send", vec![lane_can]),
                    ("pin_tx_valid", pins.tx_valid),
                    ("pin_tx_bits", pins.tx_bits),
                    ("pin_tx_ready", pins.tx_ready),
                    ("pin_rx_valid", pins.rx_valid),
                    ("pin_rx_bits", pins.rx_bits),
                    ("pin_rx_ready", pins.rx_ready),
                    ("pin_tx_ack", pins.tx_ack),
                    ("pin_rx_ack", pins.rx_ack),
                    ("pin_rx_stall", link_rx_stall.clone()),
                    ("core_stall", vec![Bit::Zero]),
                    ("rx_pop", vec![link_pop]),
                    ("rx_data", into),
                    ("rx_has_data", vec![lane_has]),
                    ("corrected", vec![lane_corrected]),
                    ("failed", vec![lane_failed]),
                ]);
                sendable.push(lane_can);
                ready.push(lane_has);
                corrected.push(lane_corrected);
                failed.push(lane_failed);
            }
            let mut l = Logic { b: &mut b, next: 21_000 };
            let (all_can, all_have) = (l.all(&sendable), l.all(&ready));
            l.b.or_to(all_can, Bit::Zero, can_send);
            l.b.or_to(all_have, Bit::Zero, has_data);
            for (lane, (fixed, lost)) in corrected.iter().zip(&failed).enumerate() {
                l.b.or_to(*fixed, Bit::Zero, link_corrected[lane]);
                l.b.or_to(*lost, Bit::Zero, link_failed[lane]);
            }
        } else {
            // A die that never talks to a neighbour has no endpoint at all.
            b.or_to(Bit::One, Bit::Zero, can_send);
            b.or_to(Bit::Zero, Bit::Zero, has_data);
            for bit in link_in.iter().chain(&link_tx_valid).chain(&link_tx_bits).chain(&link_rx_ready).chain(&link_rx_ack).chain(&link_corrected).chain(&link_failed) {
                b.or_to(Bit::Zero, Bit::Zero, *bit);
            }
        }

        // --- Activation network and cells, one array per vector unit ---
        let per_unit = (c.cells as usize / clusters).max(1);
        for (unit, (read_a, index_a)) in unit_ports.into_iter().enumerate() {
            b.group("Red de activaciones");
            let mut x_bus = read_a;
            for stage in 0..c.broadcast_stages {
                let q = b.wire(&format!("x_bus{unit}_{stage}"), a);
                b.inst(&format!("x_stage{unit}_{stage}"), &reg_a, &[("d", x_bus), ("q", q.clone())]);
                x_bus = q;
            }
            let cell_seek_addr: Vec<Bit> = (0..cfg.addr_bits as usize).map(|i| index_a.get(i).copied().unwrap_or(Bit::Zero)).collect();
            // A group of the cells that ends a quantisation group, which this unit's column
            // shows, or ends the unit's share of the pass, moves its cells' factors on.
            let advance = {
                let mut l = Logic { b: &mut b, next: 58_000 + unit * 64 };
                let quant = &fields["quant_bits"];
                let mut done = Bit::One;
                for (i, bit) in index_a.iter().enumerate().take((1 << quant.len()) - 1) {
                    let within = at_least(&mut l, quant, i as u32 + 1);
                    let outside = l.not(within);
                    let ok = l.or(*bit, outside);
                    done = l.and(done, ok);
                }
                let ends = l.or(done, fields["slice_end"][0]);
                let mut late = l.and(cell_end_now, ends);
                for _ in 0..1 + c.broadcast_stages as usize + cfg.latency() as usize {
                    let q = l.fresh();
                    l.connect_flops(&[late], &[q]);
                    late = q;
                }
                l.or(late, cell_flush_now)
            };
            let mut chain_in = vec![vec![Bit::Zero; wide as usize]; lanes];
            for row in (0..per_unit).rev() {
                let i = unit * per_unit + row;
                let lane = row % lanes;
                b.group("Celdas");
                let total = b.wire(&format!("cell{i}_total"), wide);
                let mut cell_pins = vec![("clear", vec![cell_clear]), ("clear_group", vec![cell_clear_group]), ("flush", vec![cell_flush]), ("advance", vec![advance]), ("run", vec![cell_run]), ("x", x_bus.clone()), ("total", total.clone()), ("seek_weights", vec![cell_seek_weights]), ("seek_factors", vec![cell_seek_factors]), ("seek_addr", cell_seek_addr.clone())];
                cell_pins.extend(factor_pins.iter().cloned());
                if cfg.most_packing() > 0 {
                    cell_pins.push(("packing", cell_packing.clone()));
                }
                if let Some((load_cell, load_scale, cell_addr)) = &cell_ports {
                    cell_pins.extend([("load_en", vec![load_cell[i]]), ("load_scale", vec![*load_scale]), ("load_addr", cell_addr.clone()), ("load_data", cell_data[..c.scale_bits.max(cfg.weight_bits) as usize].to_vec())]);
                }
                b.inst(&format!("cell{i}"), &cell, &cell_pins);

                b.group("Cadena de resultados");
                let chosen = b.wire(&format!("chain{i}_load"), wide);
                b.inst(&format!("chain{i}_select"), &mux_wide, &[("s", vec![chain_load]), ("a", chain_in[lane].clone()), ("b", total), ("y", chosen.clone())]);
                let q = b.wire(&format!("chain{i}"), wide);
                let keep = bus_or(&mut b, chain_load, chain_shift, &format!("chain{i}_update"));
                let held = b.wire(&format!("chain{i}_held"), wide);
                b.inst(&format!("chain{i}_hold"), &mux_wide, &[("s", vec![keep]), ("a", q.clone()), ("b", chosen), ("y", held.clone())]);
                b.inst(&format!("chain{i}_reg"), &reg_wide, &[("d", held), ("q", q.clone())]);
                chain_in[lane] = q;
            }
            for (lane, output) in unit_chains[unit].iter().enumerate() {
                for (from, to) in chain_in[lane].iter().zip(output) {
                    b.or_to(*from, Bit::Zero, *to);
                }
            }
        }
        self.add(b.finish())
    }
}

struct FactorLoad {
    write_scale: Bit,
    load_data: Vec<Bit>,
    load_addr: Vec<Bit>,
}

struct SumParts {
    group_sum: Vec<Bit>,
    x_sum: Vec<Bit>,
    /// Absent when the factors are fixed at manufacture.
    load: Option<FactorLoad>,
    clear: Bit,
    flush: Bit,
    /// Moves on to the next scale and bias: with each flush, unless a group is in parts.
    advance: Bit,
    /// With factors applied over several cycles: the first of them, and the last, when
    /// the products are complete and go into the total.
    serial: Option<(Bit, Bit)>,
    total: Vec<Bit>,
    clear_group: Bit,
    seek_factors: Bit,
    seek_addr: Vec<Bit>,
}

/// Arithmetic right shift by the 6-bit `amount`, from selectors.
fn barrel_right(b: &mut Builder, mux: &str, name: &str, value: &[Bit], amount: &[Bit]) -> Vec<Bit> {
    let width = value.len();
    let sign = *value.last().expect("vector vacío");
    let mut current = value.to_vec();
    for (level, select) in amount.iter().enumerate() {
        let shift = 1usize << level;
        let moved: Vec<Bit> = (0..width).map(|i| current.get(i + shift).copied().unwrap_or(sign)).collect();
        let out = b.wire(&format!("{name}_{level}"), width as u32);
        b.inst(&format!("{name}_level{level}"), mux, &[("s", vec![*select]), ("a", current), ("b", moved), ("y", out.clone())]);
        current = out;
    }
    current
}

impl Gen {
    /// Arithmetic right shift to `width` bits that rounds to nearest. The largest
    /// positive value does not round up, since it would wrap around to negative.
    fn rounded_right(&mut self, b: &mut Builder, name: &str, value: &[Bit], amount: &[Bit], width: usize) -> Vec<Bit> {
        let mut extended = vec![Bit::Zero];
        extended.extend_from_slice(value);
        let mux = self.mux_bus(extended.len() as u32);
        let shifted = barrel_right(b, &mux, name, &extended, amount);
        let truncated = shifted[1..=width].to_vec();
        let largest = {
            let mut l = Logic { b, next: 59_000 };
            let positive = l.not(truncated[width - 1]);
            let mut ones: Vec<Bit> = truncated[..width - 1].to_vec();
            ones.push(positive);
            l.all(&ones)
        };
        let carry = and_not(b, shifted[0], largest, &format!("{name}_carry"));
        let adder = self.ripple_adder(width as u32);
        let rounded = b.wire(&format!("{name}_rounded"), width as u32);
        let unused = b.wire(&format!("{name}_cout"), 1);
        b.inst(&format!("{name}_round"), &adder, &[("a", truncated), ("b", vec![Bit::Zero; width]), ("cin", vec![carry]), ("sum", rounded.clone()), ("cout", unused)]);
        rounded
    }
}

/// Arithmetic-free left shift by the 6-bit `amount`, filling with zeros.
pub(crate) fn barrel_left(b: &mut Builder, mux: &str, name: &str, value: &[Bit], amount: &[Bit]) -> Vec<Bit> {
    let width = value.len();
    let mut current = value.to_vec();
    for (level, select) in amount.iter().enumerate() {
        let shift = 1usize << level;
        let moved: Vec<Bit> = (0..width).map(|i| if i >= shift { current[i - shift] } else { Bit::Zero }).collect();
        let out = b.wire(&format!("{name}_{level}"), width as u32);
        b.inst(&format!("{name}_level{level}"), mux, &[("s", vec![*select]), ("a", current), ("b", moved), ("y", out.clone())]);
        current = out;
    }
    current
}

/// Narrows a double-width value to `width` bits, clamping to the largest magnitude.
fn saturate(b: &mut Builder, value: &[Bit], width: u32) -> Vec<Bit> {
    let top = value.len() - 1;
    let sign = value[top];
    let mut overflow = Bit::Zero;
    for (i, bit) in value.iter().enumerate().take(top).skip(width as usize - 1) {
        let differs = b.wire(&format!("sat_differs{i}"), 1)[0];
        b.xor_to(*bit, sign, differs);
        let next = b.wire(&format!("sat_any{i}"), 1)[0];
        b.or_to(overflow, differs, next);
        overflow = next;
    }
    let not_sign = b.wire("sat_not_sign", 1)[0];
    b.not_to(sign, not_sign);
    let out = b.wire("saturated", width);
    for i in 0..width as usize {
        let clamped = if i == width as usize - 1 { sign } else { not_sign };
        b.mux_to(overflow, value[i], clamped, out[i]);
    }
    out
}

/// How many leading zeros `bits` has, from the most significant bit, as a `width`-bit value.
pub(crate) fn leading_zeros(b: &mut Builder, name: &str, bits: &[Bit], width: u32) -> Vec<Bit> {
    let count_bits = (usize::BITS - bits.len().leading_zeros()) as usize;
    let mut result: Vec<Bit> = bits.len().to_le_bytes().iter().flat_map(|byte| (0..8).map(move |i| if byte >> i & 1 == 1 { Bit::One } else { Bit::Zero })).take(count_bits).collect();
    // From the least significant bit up, each set bit overrides the earlier answer.
    for (index, bit) in bits.iter().enumerate() {
        let count = bits.len() - 1 - index;
        let chosen = b.wire(&format!("{name}_{index}"), count_bits as u32);
        for k in 0..count_bits {
            let literal = if count >> k & 1 == 1 { Bit::One } else { Bit::Zero };
            b.mux_to(*bit, result[k], literal, chosen[k]);
        }
        result = chosen;
    }
    (0..width as usize).map(|i| result.get(i).copied().unwrap_or(Bit::Zero)).collect()
}

/// Selects one of `leaves` with the binary `select`, one level of selectors per bit.
pub(crate) fn mux_tree(b: &mut Builder, mux: &str, prefix: &str, leaves: Vec<Vec<Bit>>, select: &[Bit]) -> Vec<Bit> {
    let width = leaves[0].len() as u32;
    let mut level = leaves;
    for (depth, bit) in select.iter().enumerate() {
        level = level
            .chunks(2)
            .enumerate()
            .map(|(i, pair)| {
                let y = b.wire(&format!("{prefix}_mux{depth}_{i}"), width);
                b.inst(&format!("{prefix}_select{depth}_{i}"), mux, &[("s", vec![*bit]), ("a", pair[0].clone()), ("b", pair[1].clone()), ("y", y.clone())]);
                y
            })
            .collect();
    }
    level.remove(0)
}

impl Gen {
    /// Several dies in a ring, each link pipelined by `link_stages` registers.
    fn package(&mut self, c: &DieConfig, dies: u32, link_stages: u32) -> String {
        let die = self.die(c);
        let bits = crate::link::wire_bits(c.cell.activation_bits);
        let reg = self.register(bits);
        let template = self.modules[&die].clone();
        let mut b = Builder::new("llm2chip_package", "Paquete", "Varios dies conectados en anillo: cada uno pasa su vector de activación al siguiente.");

        let mut wires: Vec<std::collections::HashMap<String, Vec<Bit>>> = Vec::new();
        for d in 0..dies {
            let mut pins = std::collections::HashMap::new();
            for port in &template.ports {
                let width = template.nets[port.net].width;
                let pin = port.name.split_once('_').map_or("", |(_, pin)| pin);
                let on_ring = port.name.starts_with("link") && ["tx_valid", "tx_bits", "tx_ready", "tx_ack", "rx_valid", "rx_bits", "rx_ready", "rx_ack"].contains(&pin);
                let bits = if on_ring {
                    b.wire(&format!("d{d}_{}", port.name), width)
                } else if port.dir == crate::Dir::Input {
                    b.input(&format!("d{d}_{}", port.name), width)
                } else {
                    b.output(&format!("d{d}_{}", port.name), width)
                };
                pins.insert(port.name.clone(), bits);
            }
            wires.push(pins);
        }
        for d in 0..dies as usize {
            b.group("Dies");
            let conns: Vec<(&str, Vec<Bit>)> = template.ports.iter().map(|p| (p.name.as_str(), wires[d][&p.name].clone())).collect();
            b.inst(&format!("die{d}"), &die, &conns);

            b.group("Enlaces");
            let target = (d + 1) % dies as usize;
            for lane in 0..c.features.link_lanes() {
                let p = if lane == 0 { "link".to_string() } else { format!("link{lane}") };
                let pin = |name: &str| format!("{p}_{name}");
                let mut data = wires[d][&pin("tx_bits")].clone();
                let mut valid = wires[d][&pin("tx_valid")][0];
                let mut ready = wires[target][&pin("rx_ready")][0];
                for stage in 0..link_stages {
                    let q = b.wire(&format!("link{d}_{lane}_stage{stage}"), bits);
                    b.inst(&format!("link{d}_{lane}_reg{stage}"), &reg, &[("d", data), ("q", q.clone())]);
                    data = q;
                    let (next_valid, next_ready) = (b.wire(&format!("link{d}_{lane}_valid{stage}"), 1)[0], b.wire(&format!("link{d}_{lane}_ready{stage}"), 1)[0]);
                    b.dff_to(valid, next_valid);
                    b.dff_to(ready, next_ready);
                    (valid, ready) = (next_valid, next_ready);
                }
                for (i, bit) in data.iter().enumerate() {
                    b.xor_to(*bit, Bit::Zero, wires[target][&pin("rx_bits")][i]);
                }
                b.xor_to(valid, Bit::Zero, wires[target][&pin("rx_valid")][0]);
                b.xor_to(ready, Bit::Zero, wires[d][&pin("tx_ready")][0]);
                for (from, to) in wires[target][&pin("rx_ack")].iter().zip(&wires[d][&pin("tx_ack")]) {
                    b.xor_to(*from, Bit::Zero, *to);
                }
            }
        }
        self.add(b.finish())
    }
}

/// Several reduced dies joined in a ring by pipelined links.
pub fn package_design(config: DieConfig, dies: u32, link_stages: u32) -> Design {
    let mut gen = Gen { modules: BTreeMap::new() };
    let top = gen.package(&config, dies, link_stages);
    Design { top, modules: gen.modules }
}

/// The complete reduced die.
pub fn die_design(config: DieConfig) -> Design {
    let mut gen = Gen { modules: BTreeMap::new() };
    let top = gen.die(&config);
    let mut design = Design { top, modules: gen.modules };
    if !config.features.is_general() {
        crate::optimize::trim_top(&mut design);
    }
    design
}

/// Lays operations out on a per-cycle timeline and compresses it into microcode.
pub struct DieProgram {
    config: DieConfig,
    cycles: Vec<ControlWord>,
    /// Cycles that a jump or a start can land on, which therefore cannot be merged away.
    labels: BTreeSet<usize>,
    entries: Vec<usize>,
    /// Index register holding each run-time quantity.
    parameters: Vec<Expr>,
    scoped: HashMap<Expr, u32>,
    active: Vec<Var>,
    in_use: BTreeSet<u32>,
    /// The first cycle after the last group fed to the cells has reached their totals.
    cells_settle: usize,
    /// Blocks of totals loaded into the result chains and not yet moved to the wide memory:
    /// the next pass of the cells moves them while it computes.
    drain_pending: u32,
    /// The last two-cycle step and the cycle it computes in, which the next step's read
    /// may share when it does not depend on it.
    last_short: Option<(usize, Alu)>,
    /// The first cycle a step may read the largest element found so far in: the tree
    /// across the lanes delivers each block's a few cycles after it computes.
    max_ready: usize,
    /// The first cycle a step may read the accumulator in: products reach it a cycle
    /// after they are computed.
    acc_ready: usize,
}

/// One index register the CPU sets before starting the die.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Parameter {
    pub register: u32,
    pub token: u32,
    pub position: u32,
    pub constant: u32,
    /// Multiplies how many of the last `window` tokens exist.
    pub window_length: u32,
    /// Multiplies this position's place in a ring of `window` entries.
    pub window_slot: u32,
    pub window: u32,
}

impl Microprogram {
    /// The cycles one routine takes, which depend only on the parameters.
    pub fn cycles(&self, config: &DieConfig, routine: usize, token: u32, position: u32) -> u64 {
        let mask = (1u32 << config.index_bits) - 1;
        let mut index = vec![0u32; 1 << x_bits(config)];
        for parameter in &self.parameters {
            index[parameter.register as usize] = parameter.value(token, position) & mask;
        }
        let (mut pc, mut cycles) = (self.entries[routine] as usize, 0u64);
        loop {
            let word = &self.words[pc];
            cycles += if word.repeat_x {
                let blocks = if word.vector { config.features.lane_bits } else { 0 };
                (index[word.x_limit as usize] >> blocks) as u64 + 1
            } else {
                word.repeat as u64 + 1
            };
            if word.halt {
                return cycles;
            }
            let mut next = pc + 1;
            if word.x_write {
                let value = index[word.x_src as usize].wrapping_add(word.literal_b) & mask;
                if word.x_dst != 0 && word.x_only <= 1 {
                    index[word.x_dst as usize] = value;
                }
                if word.branch && value != index[word.x_limit as usize] {
                    next = word.target as usize;
                }
            }
            pc = next;
        }
    }
}

impl Parameter {
    pub fn value(&self, token: u32, position: u32) -> u32 {
        let windowed = if self.window > 0 { self.window_length.wrapping_mul((position + 1).min(self.window)).wrapping_add(self.window_slot.wrapping_mul(position % self.window)) } else { 0 };
        self.token.wrapping_mul(token).wrapping_add(self.position.wrapping_mul(position)).wrapping_add(self.constant).wrapping_add(windowed)
    }
}

/// Microcode loaded once, with the entry points and parameters that drive it.
#[derive(Clone, Debug, Serialize)]
pub struct Microprogram {
    pub words: Vec<ControlWord>,
    pub entries: Vec<u32>,
    pub parameters: Vec<Parameter>,
}

const LANE_LATENCY: usize = 2;

fn visit_indices(steps: &[Step], f: &mut impl FnMut(Expr)) {
    for step in steps {
        match step {
            Step::One(alu) => {
                f(alu.a.1);
                f(alu.b.1);
                f(alu.dst.1);
                f(alu.window);
            }
            Step::Repeat { body, .. } => visit_indices(body, f),
            Step::CombineRows { weights, start, window, .. } => {
                f(weights.1);
                f(*start);
                f(*window);
            }
            Step::Combine { weights, start, width, window, .. } => {
                f(*width);
                f(weights.1);
                f(*start);
                f(*window);
            }
            Step::Stream { body, count, .. } => {
                f(*count);
                f(body.a.1);
                f(body.b.1);
                f(body.dst.1);
                f(body.window);
            }
            Step::CellSeek { weights, factors, .. } => {
                f(*weights);
                f(*factors);
            }
            Step::Clustered { body, .. } => visit_indices(body, f),
            Step::CellGroup { start, .. } => f(*start),
            Step::WideReduce { rows, .. } => f(*rows),
            Step::CellPass | Step::CellDrain { .. } => {}
        }
    }
}

fn only_parameters(e: &Expr) -> bool {
    e.variables().all(|(v, _)| matches!(v, Var::Token | Var::Position | Var::WindowLength(_) | Var::WindowSlot(_)))
}

impl DieProgram {
    pub fn new(config: DieConfig) -> Self {
        // The microcode read port takes one cycle to deliver the first word.
        let cycles = vec![ControlWord::default()];
        Self { config, cycles, labels: BTreeSet::from([0]), entries: vec![0], parameters: Vec::new(), scoped: HashMap::new(), active: Vec::new(), in_use: BTreeSet::new(), cells_settle: 0, drain_pending: 0, last_short: None, max_ready: 0, acc_ready: 0 }
    }

    fn slot(&mut self, at: usize) -> &mut ControlWord {
        if self.cycles.len() <= at {
            self.cycles.resize(at + 1, ControlWord::default());
        }
        &mut self.cycles[at]
    }

    fn end(&self) -> usize {
        self.cycles.len()
    }

    /// Ends the current routine and starts another that the CPU can enter separately.
    pub fn next_entry(&mut self) {
        let halt_at = self.end();
        self.slot(halt_at).halt = true;
        let start = self.end();
        self.slot(start);
        self.labels.insert(start);
        self.entries.push(start);
    }

    fn parameter(&mut self, value: Expr) -> u32 {
        let found = self.parameters.iter().position(|p| *p == value).unwrap_or_else(|| {
            self.parameters.push(value);
            self.parameters.len() - 1
        });
        let register = found as u32 + 1;
        assert!(!self.in_use.contains(&register), "faltan registros de índice");
        register
    }

    fn allocate(&mut self) -> u32 {
        let register = ((self.parameters.len() as u32 + 1)..1 << x_bits(&self.config)).rev().find(|r| !self.in_use.contains(r)).expect("faltan registros de índice");
        self.in_use.insert(register);
        register
    }

    fn release(&mut self, register: u32) {
        self.in_use.remove(&register);
    }

    fn set_index(&mut self, dst: u32, src: u32, value: u32) {
        let at = self.end();
        let mask = (1u64 << self.config.index_bits) - 1;
        let w = self.slot(at);
        w.x_write = true;
        w.x_dst = dst;
        w.x_src = src;
        w.literal_b = (value as u64 & mask) as u32;
    }

    /// Splits an index into its literal part and the register holding the rest.
    fn resolve(&mut self, index: Expr) -> (u32, u32) {
        let variable = index.without_constant();
        let register = if variable.is_constant() {
            0
        } else if only_parameters(&variable) {
            self.parameter(variable)
        } else {
            *self.scoped.get(&variable).expect("índice sin registro asignado")
        };
        (index.constant(), register)
    }

    fn compile_loop(&mut self, variable: Var, count: Expr, body: &[Step]) {
        assert!(only_parameters(&count), "el número de vueltas solo puede depender de los parámetros");
        // A loop may take over a pending drain only if every turn leaves the same one behind.
        let lanes = 1u32 << self.config.features.lane_bits;
        let carries = matches!((body.first(), body.last()), (Some(Step::CellPass), Some(Step::CellDrain { rows })) if rows.div_ceil(lanes) == self.drain_pending);
        if !carries {
            self.finish_drain();
        }
        let pending = self.drain_pending;
        let counter = self.allocate();
        let limit = self.allocate();
        self.set_index(counter, 0, 0);
        if count.is_constant() {
            self.set_index(limit, 0, count.constant());
        } else {
            let source = self.parameter(count);
            self.set_index(limit, source, 0);
        }

        let mut keys = BTreeSet::new();
        let active = self.active.clone();
        visit_indices(body, &mut |index| {
            let key = index.restricted(|v| v == variable || active.contains(&v) || matches!(v, Var::Token | Var::Position | Var::WindowLength(_) | Var::WindowSlot(_)));
            if key.coefficient(variable) != 0 {
                keys.insert(ExprKey(key));
            }
        });

        // Each expression that moves with this loop gets a register stepped once per turn.
        let mut stepped = Vec::new();
        for ExprKey(key) in keys {
            let stride = key.coefficient(variable);
            if key == Expr::var(variable) {
                self.scoped.insert(key, counter);
                continue;
            }
            let base = key.restricted(|v| v != variable);
            let source = self.resolve(base).1;
            let register = self.allocate();
            self.set_index(register, source, 0);
            self.scoped.insert(key, register);
            stepped.push((key, register, stride));
        }

        let start = self.end();
        self.labels.insert(start);
        self.active.push(variable);
        self.assemble(body);
        self.active.pop();
        // A group fed at the end of the last turn is still settling after the jump back.
        let fed_inside = self.cells_settle > self.end();
        assert_eq!(self.drain_pending, pending, "cada vuelta deja el mismo vaciado pendiente");

        for (_, register, stride) in &stepped {
            self.set_index(*register, *register, *stride);
        }
        let last = self.end() - 1;
        let w = self.cycles[last];
        let can_merge = last >= start && !w.x_write && !w.branch && w.literal_b == 0 && w.target == 0;
        let at = if can_merge { last } else { self.end() };
        let w = self.slot(at);
        w.x_write = true;
        w.x_dst = counter;
        w.x_src = counter;
        w.literal_b = 1;
        w.branch = true;
        w.x_limit = limit;
        w.target = start as u32;
        self.last_short = None;
        if fed_inside {
            self.cells_settle = self.end() + self.feed_latency();
        }

        for (key, register, _) in stepped {
            self.scoped.remove(&key);
            self.release(register);
        }
        self.scoped.remove(&Expr::var(variable));
        self.release(limit);
        self.release(counter);
    }

    /// Lays out steps and loops, turning every variable index into an index register.
    pub fn assemble(&mut self, steps: &[Step]) {
        for step in steps {
            if !matches!(step, Step::CellPass | Step::CellGroup { .. } | Step::CellDrain { .. } | Step::Repeat { .. }) {
                self.finish_drain();
            }
            match step {
                Step::One(symbolic) => {
                    let (a, offset_a) = self.resolve(symbolic.a.1);
                    let (b, offset_b) = self.resolve(symbolic.b.1);
                    let (dst, offset_dst) = self.resolve(symbolic.dst.1);
                    let mut bound = symbolic.map_indices(|_| 0);
                    (bound.a.1, bound.b.1, bound.dst.1) = (a, b, dst);
                    (bound.offset_a, bound.offset_b, bound.offset_dst) = (offset_a, offset_b, offset_dst);
                    if symbolic.window_a || symbolic.window_b || symbolic.window_dst {
                        bound.window_x = self.resolve(symbolic.window).1;
                    }
                    self.alu(&bound);
                }
                Step::Repeat { variable, count, body } => self.compile_loop(*variable, *count, body),
                Step::Stream { variable, count, in_elements, body } => self.stream(*variable, *count, *in_elements, body),
                Step::Combine { .. } => self.combine(step),
                Step::CombineRows { .. } => self.combine_rows(step),
                Step::Clustered { vars, assignments, slots, body } => self.clustered(vars, assignments, *slots, body),
                Step::CellSeek { weights, factors, packing } => self.cell_seek(*weights, *factors, *packing),
                Step::CellPass => self.cell_pass(),
                Step::CellGroup { src, length, start, quant_bits, last } => self.cell_group(*src, *length, *start, (*quant_bits, *last)),
                Step::WideReduce { rows, group } => self.wide_reduce(*rows, *group),
                Step::CellDrain { rows } => self.cell_drain(*rows),
            }
        }
    }

    /// The offset register for an operand that moves with the index counters.
    fn moving_offset(&mut self, index: Expr, temporaries: &mut Vec<u32>) -> u32 {
        let (literal, register) = self.resolve(index);
        if literal == 0 {
            return register;
        }
        let temporary = self.allocate();
        temporaries.push(temporary);
        self.set_index(temporary, register, literal);
        temporary
    }

    /// One element per cycle: each word reads one, computes the last and writes an earlier one.
    fn stream(&mut self, variable: Var, count: Expr, in_elements: bool, step: &Alu<Expr>) {
        let moves = |index: Expr| index.coefficient(variable) != 0;
        let mut temporaries = Vec::new();
        let place = |this: &mut Self, index: Expr, temporaries: &mut Vec<u32>| {
            if moves(index) { (0, this.moving_offset(index.without(variable), temporaries)) } else { this.resolve(index) }
        };
        let (literal_a, offset_a) = place(self, step.a.1, &mut temporaries);
        let (literal_b, offset_b) = place(self, step.b.1, &mut temporaries);
        let offset_dst = self.moving_offset(step.dst.1.without(variable), &mut temporaries);
        let window_x = if step.window_a || step.window_b || step.window_dst { self.resolve(step.window).1 } else { 0 };
        let (fixed_a, fixed_b) = (!moves(step.a.1), !moves(step.b.1));
        // A broadcast operand that moves one element for each block of the others.
        let lane_bits = self.config.features.lane_bits;
        let a_shift = if step.vector && lane_bits > 0 && step.a.1.coefficient(variable) == 1 { lane_bits } else { 0 };
        // The function table reads a cycle later, and a product is rounded a cycle later.
        let extra = usize::from(step.op == OP_FUNCTION || step.op == OP_MUL);
        let written = 2 + extra;
        let by_blocks = count.is_constant() && !in_elements;
        let n = if by_blocks { count.constant() as usize } else { usize::MAX / 2 };

        let word = |k: usize| {
            let mut w = ControlWord { vector: step.vector, window_x, from_cluster: step.from_cluster, gather_shift: step.gather_shift, ..Default::default() };
            if k < n {
                w.read_step = true;
                (w.reg_a, w.literal_a, w.offset_a, w.window_a, w.fixed_a, w.a_shift) = (step.a.0, literal_a, offset_a, step.window_a, fixed_a, a_shift);
                (w.reg_b, w.literal_b, w.offset_b, w.window_b, w.fixed_b) = (step.b.0, literal_b, offset_b, step.window_b, fixed_b);
            }
            if (1..n + 1 + extra).contains(&k) {
                (w.op, w.shift, w.function, w.magnitude_scan, w.accumulate, w.wide_scan, w.max_scan) = (step.op, step.shift, step.function, step.magnitude_scan && k <= n, step.accumulate && k <= n, step.wide_scan && k <= n, step.max_scan && k <= n);
                (w.broadcast_a, w.broadcast_b) = (step.broadcast_a, step.broadcast_b);
            }
            if (written..n + written).contains(&k) {
                (w.reg_dst, w.offset_dst, w.window_dst, w.write_en, w.write_spread) = (step.dst.0, offset_dst, step.window_dst, step.write, step.write_spread);
            }
            w
        };
        if by_blocks {
            self.emit_stream(n, written, true, word);
        } else {
            let leading = if step.write { written } else { 1 };
            self.emit_variable_stream(count, leading, &mut temporaries, word);
        }
        // The last block computes no later than the cycle after the stream's words.
        if step.max_scan {
            self.max_ready = self.end() + max_scan_stages(self.config.features.lane_bits) + 1;
        }
        if step.accumulate {
            self.acc_ready = self.end() + 2;
        }
        for temporary in temporaries {
            self.release(temporary);
        }
    }

    /// A stream whose length is only known at run time. What runs off the end is read
    /// and computed but never written, so the words need no tail.
    fn emit_variable_stream(&mut self, count: Expr, leading: usize, temporaries: &mut Vec<u32>, word: impl Fn(usize) -> ControlWord) {
        let (literal, register) = self.resolve(count);
        let last = self.allocate();
        temporaries.push(last);
        self.set_index(last, register, literal.wrapping_sub(1));
        let start = self.end();
        let w = self.slot(start);
        (w.read_reset, w.write_reset) = (true, true);
        for k in 0..leading {
            let at = self.end();
            *self.slot(at) = word(k);
        }
        let at = self.end();
        *self.slot(at) = ControlWord { repeat_x: true, x_limit: last, ..word(leading) };
    }

    /// Lays out the words of a stream of `n` elements whose results land `written` later.
    fn emit_stream(&mut self, n: usize, written: usize, reset: bool, word: impl Fn(usize) -> ControlWord) {
        if reset {
            let start = self.end();
            let w = self.slot(start);
            (w.read_reset, w.write_reset) = (true, true);
        }
        let (from, until) = if n > written { (written, n) } else { (n + written, n + written) };
        for k in 0..from.min(n + written) {
            let at = self.end();
            *self.slot(at) = word(k);
        }
        // The words in the middle are all the same, so they are written once.
        let mut left = until - from;
        while left > 0 {
            let turns = left.min(1 << REPEAT_BITS);
            let at = self.end();
            *self.slot(at) = ControlWord { repeat: turns as u32 - 1, ..word(from) };
            left -= turns;
        }
        for k in until.max(from.min(n + written))..n + written {
            let at = self.end();
            *self.slot(at) = word(k);
        }
    }

    /// Rows of one block, each times its own weight, added up into the first block of the
    /// wide memory: one row a cycle, the sum carried in the adders' register.
    fn combine_rows(&mut self, step: &Step) {
        let Step::CombineRows { weights, matrix, start, rows, window, window_a, window_b } = *step else {
            unreachable!("solo se llama con una combinación de filas");
        };
        let mut temporaries = Vec::new();
        let offset_a = self.moving_offset(weights.1, &mut temporaries);
        let offset_b = self.moving_offset(start, &mut temporaries);
        let window_x = if window_a || window_b { self.resolve(window).1 } else { 0 };
        let lane_bits = self.config.features.lane_bits;
        let n = rows as usize;
        let word = |k: usize| {
            let mut w = ControlWord { vector: true, window_x, wide_accumulate: true, ..Default::default() };
            if k < n {
                w.read_step = true;
                (w.reg_a, w.offset_a, w.window_a, w.a_shift) = (weights.0, offset_a, window_a, lane_bits);
                (w.reg_b, w.offset_b, w.window_b) = (matrix, offset_b, window_b);
            }
            if (1..n + 1).contains(&k) {
                (w.broadcast_a, w.wide_fresh, w.wide_chain) = (true, k == 1, k > 1);
            }
            w.wide_write = k == n + 1;
            w
        };
        self.emit_stream(n, 2, true, word);
        for temporary in temporaries {
            self.release(temporary);
        }
    }

    /// Adds a whole row, times one weight, to the sums in the wide memory.
    fn combine(&mut self, step: &Step) {
        let Step::Combine { weights, matrix, start, width, fresh, window, window_a, window_b, moving } = *step else {
            unreachable!("solo se llama con una combinación");
        };
        let mut temporaries = Vec::new();
        let (literal_a, offset_a) = if moving { (0, self.moving_offset(weights.1, &mut temporaries)) } else { self.resolve(weights.1) };
        let offset_b = self.moving_offset(start, &mut temporaries);
        let window_x = if window_a || window_b { self.resolve(window).1 } else { 0 };
        let lane_bits = self.config.features.lane_bits;
        let lanes = 1u32 << lane_bits;
        let aligned = start.is_multiple_of(lanes);
        if !width.is_constant() {
            let vector = aligned && lanes > 1;
            let word = |k: usize| {
                let mut w = ControlWord { vector, window_x, wide_accumulate: true, ..Default::default() };
                w.read_step = true;
                (w.reg_a, w.literal_a, w.offset_a, w.window_a, w.fixed_a) = (weights.0, literal_a, offset_a, window_a, !moving);
                (w.reg_b, w.offset_b, w.window_b) = (matrix, offset_b, window_b);
                if k >= 1 {
                    (w.broadcast_a, w.wide_fresh) = (vector && !moving, fresh);
                }
                w.wide_write = k >= 2;
                w
            };
            self.emit_variable_stream(width, 2, &mut temporaries, word);
            for temporary in temporaries {
                self.release(temporary);
            }
            return;
        }
        let width = width.constant();
        let (blocks, rest) = if width.is_multiple_of(lanes) && aligned { (width / lanes, 0) } else { (0, width) };
        let word = |k: usize, n: usize, vector: bool| {
            let mut w = ControlWord { vector, window_x, wide_accumulate: true, ..Default::default() };
            if k < n {
                w.read_step = true;
                (w.reg_a, w.literal_a, w.offset_a, w.window_a, w.fixed_a) = (weights.0, literal_a, offset_a, window_a, !moving);
                (w.reg_b, w.offset_b, w.window_b) = (matrix, offset_b, window_b);
            }
            if (1..n + 1).contains(&k) {
                (w.broadcast_a, w.wide_fresh) = (vector && !moving, fresh);
            }
            w.wide_write = (2..n + 2).contains(&k);
            w
        };
        let reset = rest == 0;
        if blocks > 0 {
            self.emit_stream(blocks as usize, 2, reset, |k| word(k, blocks as usize, true));
        }
        if rest > 0 {
            self.emit_stream(rest as usize, 2, !reset, |k| word(k, rest as usize, false));
        }
        for temporary in temporaries {
            self.release(temporary);
        }
    }

    /// Runs the same words on several vector units, each with its own loop indices.
    fn clustered(&mut self, vars: &[Var], assignments: &[Vec<u32>], slots: Option<(u32, u32, u32)>, body: &[Step]) {
        let units = 1u32 << self.config.features.cluster_bits;
        let at = self.end();
        let w = self.slot(at);
        (w.set_clusters, w.literal_b) = (true, assignments.len() as u32);
        if let Some((shift, bits, limit)) = slots {
            (w.x_split, w.literal_dst, w.literal_a) = (shift, bits, limit);
        }

        let mut keys = BTreeSet::new();
        let active = self.active.clone();
        visit_indices(body, &mut |index| {
            let key = index.restricted(|v| vars.contains(&v) || active.contains(&v) || matches!(v, Var::Token | Var::Position | Var::WindowLength(_) | Var::WindowSlot(_)));
            if vars.iter().any(|v| key.coefficient(*v) != 0) {
                keys.insert(ExprKey(key));
            }
        });
        let mut registers = Vec::new();
        for ExprKey(key) in keys {
            let register = self.allocate();
            let source = self.resolve(key.restricted(|v| !vars.contains(&v))).1;
            let mask = (1u64 << self.config.index_bits) - 1;
            let literals: Vec<u32> = assignments.iter().map(|values| vars.iter().zip(values).map(|(v, value)| key.coefficient(*v).wrapping_mul(*value)).fold(0u32, u32::wrapping_add)).collect();
            // Unit `k` gets `first + (k >> split) · outer + (k mod 2^split) · inner`.
            let spaced = |split: u32| -> Option<(u64, u64)> {
                let at = |unit: usize| literals.get(unit).map(|l| (*l as u64).wrapping_sub(literals[0] as u64) & mask);
                let outer = at(1 << split).unwrap_or(0);
                let inner = if split == 0 { 0 } else { at(1).unwrap_or(0) };
                let fits = literals.iter().enumerate().all(|(unit, literal)| {
                    let expected = (literals[0] as u64).wrapping_add((unit as u64 >> split).wrapping_mul(outer)).wrapping_add((unit as u64 & ((1 << split) - 1)).wrapping_mul(inner));
                    (*literal as u64 & mask) == expected & mask
                });
                fits.then_some((outer, inner))
            };
            let splits = if self.config.features.widths[X_SPLIT_FIELD] == 0 { 0 } else { self.config.features.cluster_bits };
            if let Some((split, (outer, inner))) = (0..=splits).find_map(|split| spaced(split).map(|strides| (split, strides))) {
                // Evenly spaced across the units, or across heads and their parts: a
                // single word writes them all.
                self.set_index(register, source, literals[0]);
                let last = self.end() - 1;
                let w = &mut self.cycles[last];
                (w.x_scaled, w.literal_a, w.x_split, w.literal_dst) = (true, outer as u32, split, inner as u32);
            } else if let Some(terms) = bit_linear(&literals, mask, self.config.features.cluster_bits).filter(|terms| splits > 0 && terms.len() < literals.len()) {
                // Linear in each bit of the unit's number: unit `k` gets `first` plus a
                // multiple of `k >> s` for a few shifts `s`, a word each.
                self.set_index(register, source, literals[0]);
                for (shift, factor) in terms {
                    self.set_index(register, register, 0);
                    let last = self.end() - 1;
                    let w = &mut self.cycles[last];
                    (w.x_scaled, w.literal_a, w.x_split) = (true, factor as u32, shift);
                }
            } else {
                for (unit, literal) in literals.iter().enumerate() {
                    self.set_index(register, source, *literal);
                    let last = self.end() - 1;
                    self.cycles[last].x_only = unit as u32 + 1;
                }
            }
            self.scoped.insert(key, register);
            registers.push((key, register));
        }
        self.active.extend_from_slice(vars);
        self.assemble(body);
        self.finish_drain();
        self.active.truncate(active.len());
        for (key, register) in registers {
            self.scoped.remove(&key);
            self.release(register);
        }
        let at = self.end();
        let w = self.slot(at);
        (w.set_clusters, w.literal_b) = (true, units);
    }

    /// Cycles from feeding the last activation of a group until the totals have it.
    fn feed_latency(&self) -> usize {
        1 + self.config.broadcast_stages as usize + self.config.cell.latency() as usize + 1 + self.factor_wait()
    }

    /// Cycles the last group's factors take after its flush, beyond a whole multiplier's.
    fn factor_wait(&self) -> usize {
        match self.config.features.factor_steps {
            1 => 0,
            steps => steps as usize + 1,
        }
    }

    /// The first free cycle once the cells have taken everything fed to them.
    fn after_cells(&self) -> usize {
        self.end().max(self.cells_settle)
    }

    /// Points every cell at the first weight and the first group of a matrix.
    fn cell_seek(&mut self, weights: Expr, factors: Expr, packing: u32) {
        let (literal, offset) = self.resolve(weights);
        let t = self.after_cells();
        let w = self.slot(t);
        (w.use_literal, w.literal_a, w.offset_a, w.cell_seek_weights, w.write_reset, w.cell_packing) = (true, literal, offset, true, true, packing);
        let (literal, offset) = self.resolve(factors);
        let w = self.slot(t + 1);
        (w.use_literal, w.literal_a, w.offset_a, w.cell_seek_factors) = (true, literal, offset, true);
        self.slot(t + 2);
    }

    /// Starts one round of the cells over the input vector.
    fn cell_pass(&mut self) {
        let t = self.after_cells();
        let w = self.slot(t);
        (w.read_reset, w.cell_clear_group) = (true, true);
        self.drain_into(t);
    }

    /// Moves one pending block of totals from the chains into the wide memory at `t`.
    fn drain_into(&mut self, t: usize) {
        if self.drain_pending > 0 {
            let (vector, units) = (self.config.features.lane_bits > 0, self.config.features.cluster_bits);
            let w = self.slot(t);
            (w.wide_write, w.wide_from_chain, w.chain_shift) = (true, true, true);
            w.vector |= vector && !w.read_step;
            // Every unit holds part of the same rows: the drain adds them up on the way.
            w.reduce_group = units;
            self.drain_pending -= 1;
        }
    }

    /// Moves whatever the chains still hold.
    fn finish_drain(&mut self) {
        while self.drain_pending > 0 {
            let t = self.end();
            self.drain_into(t);
        }
    }

    /// Streams the next `length` activations through the cells, then applies their factors.
    fn cell_group(&mut self, src: u32, length: u32, first: Expr, (quant_bits, last): (u32, bool)) {
        let offset = self.resolve(first).1;
        let c = self.config;
        if c.features.feeds_cells() && length >= 2 {
            // The delay lines run and flush the cells, so the next group can start at once.
            let start = self.end();
            for i in 0..length as usize {
                let w = self.slot(start + i);
                (w.read_step, w.reg_a, w.offset_a, w.cell_feed) = (true, src, offset, true);
                w.cell_end = i + 1 == length as usize;
                if w.cell_end {
                    (w.quant_bits, w.slice_end) = (quant_bits, last && quant_bits > 0);
                }
                self.drain_into(start + i);
            }
            self.cells_settle = start + length as usize - 1 + self.feed_latency();
            return;
        }
        let x_delay = 1 + c.broadcast_stages as usize;
        let start = self.end();
        for i in 0..length as usize {
            let w = self.slot(start + i);
            w.read_step = true;
            (w.reg_a, w.offset_a) = (src, offset);
            self.slot(start + i + x_delay).cell_run = true;
        }
        let settled = start + length as usize + x_delay + c.cell.latency() as usize + 1;
        self.slot(settled).cell_flush = true;
        let wait = self.factor_wait();
        self.slot(settled + 1 + wait);
    }

    /// Moves the totals of the first `rows` cells into the wide memory, a block at a time.
    fn cell_drain(&mut self, rows: u32) {
        let lane_bits = self.config.features.lane_bits;
        let lanes = 1u32 << lane_bits;
        if self.config.features.feeds_cells() {
            // Load the chains now; the next pass moves them while it computes.
            self.finish_drain();
            let t = self.after_cells();
            self.slot(t).chain_load = true;
            self.drain_pending = rows.div_ceil(lanes);
            return;
        }
        let t = self.after_cells();
        self.slot(t).chain_load = true;
        for i in 0..rows.div_ceil(lanes) as usize {
            let w = self.slot(t + 1 + i);
            (w.wide_write, w.wide_from_chain, w.chain_shift, w.vector) = (true, true, true, lane_bits > 0);
        }
        let end = self.end();
        self.slot(end);
    }

    /// Replaces each sum in the wide memory by the total over all the vector units.
    fn wide_reduce(&mut self, rows: Expr, group: u32) {
        let vector = self.config.features.lane_bits > 0;
        let mut temporaries = Vec::new();
        let word = |k: usize| {
            let mut w = ControlWord { vector, wide_accumulate: true, read_step: true, reduce_group: group, ..Default::default() };
            w.wide_reduce = k >= 1;
            w.wide_write = k >= 2;
            w
        };
        self.emit_variable_stream(rows, 2, &mut temporaries, word);
        for temporary in temporaries {
            self.release(temporary);
        }
    }

    /// One single-element operation. Nothing overlaps, so each takes a few cycles.
    pub fn alu(&mut self, step: &Alu) {
        self.wait_for_max(step);
        if self.config.features.streams() && writes_at_once(step) {
            return self.short_alu(step);
        }
        let immediate = immediate_target(step.literal, self.config);
        let t0 = self.end();
        let extra = if step.op == OP_FUNCTION || step.op == OP_MUL {
            1
        } else if step.op == OP_MAX && step.reduce_group > 0 {
            max_stages(self.config.features.cluster_bits)
        } else {
            0
        };
        {
            let w = self.slot(t0);
            w.use_literal = true;
            w.reg_a = step.a.0;
            w.literal_a = step.a.1;
            w.offset_a = step.offset_a;
            w.reg_b = step.b.0;
            w.literal_b = step.b.1;
            w.offset_b = step.offset_b;
            (w.window_x, w.window_a, w.window_b) = (step.window_x, step.window_a, step.window_b);
            (w.vector, w.broadcast_a, w.broadcast_b) = (step.vector, step.broadcast_a, step.broadcast_b);
            w.from_cluster = step.from_cluster;
            w.magnitude_clear = step.magnitude_clear;
            w.max_clear = step.max_clear;
            w.accumulator_clear = step.accumulator_clear;
            w.wide_clear = step.wide_clear;
        }
        {
            let w = self.slot(t0 + 1);
            w.use_literal = true;
            w.reg_a = step.a.0;
            w.literal_a = step.a.1;
            w.offset_a = step.offset_a;
            w.reg_b = step.b.0;
            w.literal_b = step.b.1;
            w.target = immediate;
            w.offset_b = step.offset_b;
            (w.window_x, w.window_a, w.window_b) = (step.window_x, step.window_a, step.window_b);
            (w.vector, w.broadcast_a, w.broadcast_b) = (step.vector, step.broadcast_a, step.broadcast_b);
            w.from_cluster = step.from_cluster;
            w.op = step.op;
            w.shift = step.shift;
            w.function = step.function;
            w.magnitude_scan = step.magnitude_scan;
            w.reduce_group = step.reduce_group;
            (w.max_scan, w.max_out) = (step.max_scan, step.max_out);
            w.reduce_group = step.reduce_group;
            w.accumulate = step.accumulate;
            w.wide_scan = step.wide_scan;
            w.wide_write = step.wide_write;
            if step.wide_write {
                w.literal_dst = step.dst.1;
                w.offset_dst = step.offset_dst;
            }
        }
        for more in 0..extra {
            let w = self.slot(t0 + 2 + more);
            (w.reduce_group, w.shift) = (step.reduce_group, step.shift);
            w.use_literal = true;
            w.reg_a = step.a.0;
            w.literal_a = step.a.1;
            w.offset_a = step.offset_a;
            // B too, which a comparison at the end of the network still reads.
            (w.reg_b, w.literal_b, w.offset_b) = (step.b.0, step.b.1, step.offset_b);
            (w.window_x, w.window_a, w.window_b) = (step.window_x, step.window_a, step.window_b);
            (w.vector, w.broadcast_a, w.broadcast_b) = (step.vector, step.broadcast_a, step.broadcast_b);
            w.from_cluster = step.from_cluster;
            w.op = step.op;
            w.function = step.function;
        }
        let t = t0 + 2 + extra;
        let w = self.slot(t);
        w.use_literal = true;
        w.reg_dst = step.dst.0;
        w.literal_dst = step.dst.1;
        w.offset_dst = step.offset_dst;
        (w.window_x, w.window_dst) = (step.window_x, step.window_dst);
        w.vector = step.vector;
        w.write_en = step.write;
        w.load_shift_a = step.load_shift_a;
        w.load_shift_b = step.load_shift_b;
        w.load_shift_left = step.load_shift_left;
        w.link_send = step.link_send;
        w.load_wide_shift = step.load_wide_shift;
    }

    /// A step that needs nothing after its result: read, then compute and write. Its read
    /// shares the previous step's compute cycle when nothing ties the two together.
    fn short_alu(&mut self, step: &Alu) {
        let lanes = 1u32 << self.config.features.lane_bits;
        let immediate = immediate_target(step.literal, self.config);
        let t0 = match self.last_short {
            // A word that steps or jumps on the indices keeps its own literal B.
            Some((at, previous)) if at + 1 == self.end() && !self.labels.contains(&self.end()) && !self.cycles[at].x_write && can_follow(&previous, step, lanes) => at,
            _ => self.end(),
        };
        self.last_short = Some((t0 + 1, *step));
        // Sharing a cycle, a step that reads nothing leaves that cycle's fields as they are.
        let shared = t0 < self.end();
        if !shared || !reads_nothing(step) {
            let w = self.slot(t0);
            w.use_literal = true;
            (w.reg_a, w.literal_a, w.offset_a) = (step.a.0, step.a.1, step.offset_a);
            (w.reg_b, w.literal_b, w.offset_b) = (step.b.0, step.b.1, step.offset_b);
            (w.window_x, w.window_a, w.window_b) = (step.window_x, step.window_a, step.window_b);
            if !shared {
                (w.vector, w.broadcast_a, w.broadcast_b) = (step.vector, step.broadcast_a, step.broadcast_b);
            }
            w.from_cluster = step.from_cluster;
            w.magnitude_clear |= step.magnitude_clear;
            w.max_clear |= step.max_clear;
            w.accumulator_clear |= step.accumulator_clear;
            w.wide_clear |= step.wide_clear;
        }
        let w = self.slot(t0 + 1);
        w.use_literal = true;
        (w.reg_a, w.literal_a, w.offset_a) = (step.a.0, step.a.1, step.offset_a);
        (w.reg_b, w.literal_b, w.offset_b, w.target) = (step.b.0, step.b.1, step.offset_b, immediate);
        (w.window_x, w.window_a, w.window_b, w.window_dst) = (step.window_x, step.window_a, step.window_b, step.window_dst);
        (w.vector, w.broadcast_a, w.broadcast_b) = (step.vector, step.broadcast_a, step.broadcast_b);
        w.from_cluster = step.from_cluster;
        (w.op, w.shift, w.function) = (step.op, step.shift, step.function);
        (w.magnitude_scan, w.reduce_group, w.max_scan, w.max_out) = (step.magnitude_scan, step.reduce_group, step.max_scan, step.max_out);
        (w.accumulate, w.wide_scan) = (step.accumulate, step.wide_scan);
        (w.reg_dst, w.literal_dst, w.offset_dst, w.write_now) = (step.dst.0, step.dst.1, step.offset_dst, step.write);
        (w.load_shift_a, w.load_shift_b, w.load_shift_left, w.shift_now) = (step.load_shift_a, step.load_shift_b, step.load_shift_left, step.shift_now);
        w.load_wide_shift = step.load_wide_shift;
    }

    /// A step that takes the largest element found so far reads it once the tree across
    /// the lanes has delivered the last block's; one that adds to it is remembered.
    fn wait_for_max(&mut self, step: &Alu) {
        if step.max_out {
            // The step computes no earlier than the end, even sharing a cycle to read.
            while self.end() < self.max_ready {
                let at = self.end();
                self.slot(at);
                self.last_short = None;
            }
        }
        if step.max_scan {
            // Computing at most two words after the end, as a step of three cycles does.
            self.max_ready = self.end() + 2 + max_scan_stages(self.config.features.lane_bits) + 1;
        }
        // The accumulator is read to store it or to shift it out.
        if step.wide_write || step.op == OP_ACCUMULATOR {
            while self.end() < self.acc_ready {
                let at = self.end();
                self.slot(at);
                self.last_short = None;
            }
        }
        if step.accumulate {
            self.acc_ready = self.end() + 3;
        }
    }

    /// Leaves the die idle for `cycles`, for example while a link settles.
    pub fn wait(&mut self, cycles: usize) {
        let at = self.end() + cycles;
        self.slot(at);
    }

    /// `dst[i] = f(a[i], b[i])` for one of the lane's operations, over `len` values.
    fn lane_op(&mut self, op: u32, shift: u32, a: u32, b: u32, dst: u32, len: usize) {
        // The function table adds one cycle of synchronous read, and a product one of rounding.
        let extra = usize::from(op == OP_FUNCTION || op == OP_MUL);
        let start = self.end();
        self.slot(start).read_reset = true;
        self.slot(start).write_reset = true;
        for i in 0..len {
            let w = self.slot(start + 1 + i);
            w.read_step = true;
            w.reg_a = a;
            w.reg_b = b;
        }
        // Address presented at t, memory output at t+1, result register at t+2.
        for i in 0..len {
            let w = self.slot(start + 1 + i + 1 + extra);
            w.reg_a = a;
            w.reg_b = b;
            w.op = op;
            w.shift = shift;
        }
        for i in 0..len {
            let w = self.slot(start + 1 + i + LANE_LATENCY + extra);
            w.reg_dst = dst;
            w.write_en = true;
        }
    }

    pub fn add(&mut self, a: u32, b: u32, dst: u32, len: usize) {
        self.lane_op(0, 0, a, b, dst, len);
    }

    pub fn mul(&mut self, a: u32, b: u32, shift: u32, dst: u32, len: usize) {
        self.lane_op(1, shift, a, b, dst, len);
    }

    pub fn function(&mut self, a: u32, dst: u32, len: usize) {
        self.lane_op(2, 0, a, a, dst, len);
    }

    /// Clears cell addresses and totals; matrices are then consumed in program order.
    pub fn restart_cells(&mut self) {
        let at = self.end();
        self.slot(at).cell_clear = true;
        self.slot(at + 1);
    }

    /// `dst = M · src`, with `groups` groups of `group` columns, one row per cell.
    pub fn matvec(&mut self, src: u32, dst: u32, group: usize, groups: usize, shift: u32) {
        let c = self.config;
        let rows = c.cells as usize;
        let x_delay = 1 + c.broadcast_stages as usize;
        let mut clock = self.end();
        self.slot(clock).read_reset = true;
        self.slot(clock).cell_clear_group = true;
        clock += 1;

        for _ in 0..groups {
            let start = clock;
            for i in 0..group {
                let w = self.slot(start + i);
                w.read_step = true;
                w.reg_a = src;
                self.slot(start + i + x_delay).cell_run = true;
            }
            // Wait for the cell pipeline, then flush the group into the row total.
            let settled = start + group + x_delay + c.cell.latency() as usize + 1;
            self.slot(settled).cell_flush = true;
            // The factor memories need the new group index before the next flush.
            clock = settled + 2;
        }

        self.slot(clock).chain_load = true;
        self.slot(clock).write_reset = true;
        clock += 1;
        for i in 0..rows {
            // Chain output is registered into the result register, then written.
            self.slot(clock + i).op = 3;
            self.slot(clock + i).shift = shift;
            self.slot(clock + i).chain_shift = true;
            self.slot(clock + i + 1).write_en = true;
            self.slot(clock + i + 1).reg_dst = dst;
        }
        self.slot(clock + rows + 1);
    }

    fn compress(&mut self) -> (Vec<ControlWord>, HashMap<usize, u32>) {
        let halt_at = self.end();
        self.slot(halt_at).halt = true;
        let mut words: Vec<ControlWord> = Vec::new();
        let mut address = HashMap::new();
        for (cycle, word) in self.cycles.iter().enumerate() {
            if !self.labels.contains(&cycle) {
                if let Some(last) = words.last_mut() {
                    let mut candidate = *word;
                    candidate.repeat = last.repeat;
                    let effect = word.halt || word.x_write || word.branch;
                    if word.repeat == 0 && !word.repeat_x && !last.repeat_x && !effect && candidate.encode_words(&self.config) == last.encode_words(&self.config) && last.repeat < (1 << REPEAT_BITS) - 1 {
                        last.repeat += 1;
                        continue;
                    }
                }
            }
            address.insert(cycle, words.len() as u32);
            words.push(*word);
        }
        for word in &mut words {
            if word.branch {
                word.target = address[&(word.target as usize)];
            }
        }
        (words, address)
    }

    /// Compressed microcode, ending in a halt word.
    pub fn finish(mut self) -> Vec<ControlWord> {
        self.compress().0
    }

    /// The finished program, with what the CPU needs to run each routine.
    pub fn finish_program(mut self) -> Microprogram {
        let (words, address) = self.compress();
        let entries = self.entries.iter().map(|cycle| address[cycle]).collect();
        let parameters = self
            .parameters
            .iter()
            .enumerate()
            .map(|(i, value)| {
                let mut parameter = Parameter { register: i as u32 + 1, token: value.coefficient(Var::Token), position: value.coefficient(Var::Position), constant: value.constant(), window_length: 0, window_slot: 0, window: 0 };
                for (variable, factor) in value.variables() {
                    match variable {
                        Var::WindowLength(n) => (parameter.window_length, parameter.window) = (factor, n),
                        Var::WindowSlot(n) => (parameter.window_slot, parameter.window) = (factor, n),
                        _ => {}
                    }
                }
                parameter
            })
            .collect();
        Microprogram { words, entries, parameters }
    }
}

/// Lets expressions be collected in a stable order.
#[derive(PartialEq, Eq)]
struct ExprKey(Expr);

impl Ord for ExprKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let terms = |e: &Expr| e.variables().collect::<Vec<_>>();
        terms(&self.0).cmp(&terms(&other.0)).then(self.0.constant().cmp(&other.0.constant()))
    }
}

impl PartialOrd for ExprKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
