//! The standard test port: a TAP controller with identification, bypass, access to the
//! debug counters and control of the memories' self-test.

use crate::cell::Gen;
use crate::rv32::{constant, Logic};
use crate::{Bit, Builder};

pub const IDCODE: u32 = 0x1C0A_1001;
pub const IR_IDCODE: u32 = 0b0001;
pub const IR_DEBUG_SELECT: u32 = 0b0010;
pub const IR_DEBUG_VALUE: u32 = 0b0011;
pub const IR_SELF_TEST: u32 = 0b0100;
pub const IR_BYPASS: u32 = 0b1111;

const TEST_LOGIC_RESET: u32 = 0;
const CAPTURE_DR: u32 = 3;
const SHIFT_DR: u32 = 4;
const UPDATE_DR: u32 = 8;
const CAPTURE_IR: u32 = 10;
const SHIFT_IR: u32 = 11;
const UPDATE_IR: u32 = 15;
/// Each state's successor with TMS low and with TMS high, in the order of IEEE 1149.1.
const TRANSITIONS: [(u32, u32); 16] = [(1, 0), (1, 2), (3, 9), (4, 5), (4, 5), (6, 8), (6, 7), (4, 8), (1, 2), (10, 0), (11, 12), (11, 12), (13, 15), (13, 14), (11, 15), (1, 2)];

impl Gen {
    pub(crate) fn jtag_tap(&mut self) -> String {
        let name = "jtag_tap".to_string();
        if self.modules.contains_key(&name) {
            return name;
        }
        let mut b = Builder::new(&name, "Puerto JTAG", "Controlador TAP estándar: identificación del chip, bypass, lectura de contadores y autotest de memorias.");
        let tck = b.input("tck", 1)[0];
        let tms = b.input("tms", 1)[0];
        let tdi = b.input("tdi", 1)[0];
        let debug_value = b.input("debug_value", 32);
        let test_status = b.input("test_status", 2);
        let tdo = b.output("tdo", 1)[0];
        let debug_select = b.output("debug_select", 3);
        let debug_override = b.output("debug_override", 1)[0];
        let test_enable = b.output("test_enable", 1)[0];
        {
            let mut l = Logic { b: &mut b, next: 0 };
            // The test port runs on its own clock: its pins pass through two flip-flops each
            // before the chip's clock reads them, and the chip's clock must be several times
            // faster than TCK.
            let mut synchronized = |bit: Bit| {
                let (first, second) = (l.fresh(), l.fresh());
                l.connect_flops(&[bit], &[first]);
                l.connect_flops(&[first], &[second]);
                second
            };
            let (tck, tms, tdi) = (synchronized(tck), synchronized(tms), synchronized(tdi));
            let previous = l.fresh();
            l.connect_flops(&[tck], &[previous]);
            let (not_previous, not_tck) = (l.not(previous), l.not(tck));
            let rising = l.and(tck, not_previous);
            let falling = l.and(not_tck, previous);
            let on_edge = |l: &mut Logic, enable: Bit, d: &[Bit], q: &[Bit]| {
                let kept = l.mux_bus(enable, q, d);
                l.connect_flops(&kept, q);
            };

            l.b.group("Estados");
            let state = l.fresh_bus(4);
            let leaves: Vec<Vec<Bit>> = TRANSITIONS.iter().map(|(low, high)| l.mux_bus(tms, &constant(*low, 4), &constant(*high, 4))).collect();
            let next = l.select(&state, &leaves);
            on_edge(&mut l, rising, &next, &state);
            let reset = l.equals(&state, TEST_LOGIC_RESET);
            let (capture_dr, shift_dr, update_dr) = (l.equals(&state, CAPTURE_DR), l.equals(&state, SHIFT_DR), l.equals(&state, UPDATE_DR));
            let (capture_ir, shift_ir, update_ir) = (l.equals(&state, CAPTURE_IR), l.equals(&state, SHIFT_IR), l.equals(&state, UPDATE_IR));

            l.b.group("Instrucción");
            let shift_register = l.fresh_bus(4);
            let instruction = l.fresh_bus(4);
            let moved: Vec<Bit> = (0..4).map(|i| if i == 3 { tdi } else { shift_register[i + 1] }).collect();
            let captured = l.mux_bus(capture_ir, &shift_register, &constant(IR_IDCODE, 4));
            let shifted = l.mux_bus(shift_ir, &captured, &moved);
            on_edge(&mut l, rising, &shifted, &shift_register);
            let updated = l.mux_bus(update_ir, &instruction, &shift_register);
            let initial = l.mux_bus(reset, &updated, &constant(IR_IDCODE, 4));
            on_edge(&mut l, rising, &initial, &instruction);
            let is = |l: &mut Logic, code: u32| l.equals(&instruction, code);
            let (idcode, select, value, self_test) = (is(&mut l, IR_IDCODE), is(&mut l, IR_DEBUG_SELECT), is(&mut l, IR_DEBUG_VALUE), is(&mut l, IR_SELF_TEST));

            l.b.group("Registro de datos");
            let wide = l.or(idcode, value);
            let short = l.or(select, self_test);
            let data = l.fresh_bus(32);
            let enable_q = l.fresh();
            let (selected, override_q) = (l.fresh_bus(3), l.fresh());
            let status: Vec<Bit> = vec![enable_q, test_status[0], test_status[1]];
            let padded: Vec<Bit> = status.iter().copied().chain(std::iter::repeat_n(Bit::Zero, 29)).collect();
            let identity = l.mux_bus(idcode, &constant(0, 32), &constant(IDCODE, 32));
            let with_value = l.mux_bus(value, &identity, &debug_value);
            let capture_value = l.mux_bus(self_test, &with_value, &padded);
            let top = |l: &mut Logic, i: usize| -> Bit {
                // The bit at the top of the active length takes TDI; the others take their neighbour.
                let (is_top_wide, is_top_short, is_top_one) = (i == 31, i == 2, i == 0);
                let above = data.get(i + 1).copied().unwrap_or(Bit::Zero);
                let mut bit = above;
                if is_top_wide {
                    bit = l.mux(wide, Bit::Zero, tdi);
                }
                if is_top_short {
                    bit = l.mux(short, bit, tdi);
                }
                if is_top_one {
                    let single = {
                        let any = l.or(wide, short);
                        l.not(any)
                    };
                    bit = l.mux(single, bit, tdi);
                }
                bit
            };
            let shifted: Vec<Bit> = (0..32).map(|i| top(&mut l, i)).collect();
            let captured = l.mux_bus(capture_dr, &data, &capture_value);
            let chosen = l.mux_bus(shift_dr, &captured, &shifted);
            on_edge(&mut l, rising, &chosen, &data);

            let writes_select = l.and(update_dr, select);
            let new_select = l.mux_bus(writes_select, &selected, &data[..3]);
            on_edge(&mut l, rising, &new_select, &selected);
            let hold_override = l.or(override_q, writes_select);
            let clear_override = l.not(reset);
            let next_override = l.and(hold_override, clear_override);
            on_edge(&mut l, rising, &[next_override], &[override_q]);
            let writes_test = l.and(update_dr, self_test);
            let new_enable = l.mux_bus(writes_test, &[enable_q], &data[..1]);
            let cleared = l.and(new_enable[0], clear_override);
            on_edge(&mut l, rising, &[cleared], &[enable_q]);

            l.b.group("Salida");
            let shifting = l.or(shift_dr, shift_ir);
            let source = l.mux(shift_ir, data[0], shift_register[0]);
            let held = l.fresh();
            let gated = l.and(shifting, source);
            on_edge(&mut l, falling, &[gated], &[held]);
            l.b.or_to(held, Bit::Zero, tdo);
            for (from, to) in selected.iter().zip(&debug_select) {
                l.b.or_to(*from, Bit::Zero, *to);
            }
            l.b.or_to(override_q, Bit::Zero, debug_override);
            l.b.or_to(enable_q, Bit::Zero, test_enable);
        }
        self.add(b.finish())
    }
}
