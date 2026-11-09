//! What keeps a chip with billions of memory bits working: error correction on stored
//! words, a march test that finds bad rows at start-up, and spare rows that replace them.

use crate::cell::Gen;
use crate::rv32::{constant, Logic};
use crate::{Bit, Builder, Design};
use std::collections::BTreeMap;

/// Check bits for `data_bits` of data: a Hamming code plus one parity bit.
pub fn check_bits(data_bits: u32) -> u32 {
    let mut hamming = 1;
    while (1u64 << hamming) < data_bits as u64 + hamming as u64 + 1 {
        hamming += 1;
    }
    hamming + 1
}

/// The code position of each data bit: positions that are not powers of two.
fn data_positions(data_bits: u32) -> Vec<u32> {
    (1..).filter(|p: &u32| !p.is_power_of_two()).take(data_bits as usize).collect()
}

/// The check bits of a word, the Hamming ones first and overall parity last.
pub fn encode(data: u128, data_bits: u32) -> u32 {
    let hamming = check_bits(data_bits) - 1;
    let mut check = 0u32;
    for (i, position) in data_positions(data_bits).iter().enumerate() {
        if data >> i & 1 == 1 {
            check ^= position;
        }
    }
    let parity = (data.count_ones() + check.count_ones()) & 1;
    check | parity << hamming
}

/// Index bits of a data bit in a word of `lanes` weights of `weight_bits` each.
fn lane_index_bits(lanes: u32, weight_bits: u32) -> (u32, u32) {
    (lanes.ilog2(), u32::BITS - (weight_bits.max(1) - 1).leading_zeros())
}

/// Check bits of a word stored as lanes: the index of a bit, two markers and a parity.
/// One more than plain Hamming, in exchange for correcting a lane without the rest.
pub fn lane_check_bits(lanes: u32, weight_bits: u32) -> u32 {
    let (lane, bit) = lane_index_bits(lanes, weight_bits);
    lane + bit + 3
}

/// What a data bit contributes to the check bits: its own index, so the syndrome of a
/// single error reads as lane and bit. All columns have odd weight of at least three,
/// which keeps them apart from each other, from errors in the check bits and from pairs.
fn lane_column(lane: u32, bit: u32, lanes: u32, weight_bits: u32) -> u32 {
    let (lane_bits, bit_bits) = lane_index_bits(lanes, weight_bits);
    let index = lane << bit_bits | bit;
    let width = lane_bits + bit_bits;
    index | 0b11 << width | (index.count_ones() & 1 ^ 1) << (width + 2)
}

/// The check bits of a word of `lanes` weights, its first lane in the low bits.
pub fn lane_encode(word: u128, lanes: u32, weight_bits: u32) -> u32 {
    let mut check = 0;
    for lane in 0..lanes {
        for bit in 0..weight_bits {
            if word >> (lane * weight_bits + bit) & 1 == 1 {
                check ^= lane_column(lane, bit, lanes, weight_bits);
            }
        }
    }
    check
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Decoded {
    pub data: u128,
    pub corrected: bool,
    pub failed: bool,
}

/// Corrects one wrong bit and detects two.
pub fn decode(data: u128, stored: u32, data_bits: u32) -> Decoded {
    let hamming = check_bits(data_bits) - 1;
    let fresh = encode(data, data_bits);
    let syndrome = (fresh ^ stored) & ((1 << hamming) - 1);
    let odd = ((data.count_ones() + (stored & ((1 << hamming) - 1)).count_ones() + (stored >> hamming & 1)) & 1) == 1;
    match (syndrome, odd) {
        (0, false) => Decoded { data, corrected: false, failed: false },
        (0, true) => Decoded { data, corrected: true, failed: false },
        (s, true) => match data_positions(data_bits).iter().position(|p| *p == s) {
            Some(bit) => Decoded { data: data ^ (1 << bit), corrected: true, failed: false },
            None if s.is_power_of_two() => Decoded { data, corrected: true, failed: false },
            None => Decoded { data, corrected: false, failed: true },
        },
        _ => Decoded { data, corrected: false, failed: true },
    }
}

fn parity_of(l: &mut Logic, bits: &[Bit]) -> Bit {
    bits.iter().copied().reduce(|a, b| l.xor(a, b)).unwrap_or(Bit::Zero)
}

/// The Hamming check bits of `data`, as gates.
fn hamming_bits(l: &mut Logic, data: &[Bit], hamming: u32) -> Vec<Bit> {
    let positions = data_positions(data.len() as u32);
    (0..hamming)
        .map(|k| {
            let covered: Vec<Bit> = data.iter().zip(&positions).filter(|(_, p)| *p >> k & 1 == 1).map(|(bit, _)| *bit).collect();
            parity_of(l, &covered)
        })
        .collect()
}

impl Gen {
    pub(crate) fn secded_encoder(&mut self, data_bits: u32) -> String {
        let name = format!("secded_encoder_{data_bits}");
        if self.modules.contains_key(&name) {
            return name;
        }
        let hamming = check_bits(data_bits) - 1;
        let mut b = Builder::new(&name, "Codificador ECC", "Calcula los bits de comprobación que se guardan junto a cada palabra.");
        let data = b.input("data", data_bits);
        let check = b.output("check", hamming + 1);
        {
            let mut l = Logic { b: &mut b, next: 0 };
            let bits = hamming_bits(&mut l, &data, hamming);
            let everything: Vec<Bit> = data.iter().chain(&bits).copied().collect();
            let parity = parity_of(&mut l, &everything);
            for (from, to) in bits.iter().chain([&parity]).zip(&check) {
                l.b.or_to(*from, Bit::Zero, *to);
            }
        }
        self.add(b.finish())
    }

    pub(crate) fn secded_decoder(&mut self, data_bits: u32) -> String {
        let name = format!("secded_decoder_{data_bits}");
        if self.modules.contains_key(&name) {
            return name;
        }
        let hamming = check_bits(data_bits) - 1;
        let mut b = Builder::new(&name, "Corrector ECC", "Corrige un bit erróneo en cada palabra leída y avisa si hay dos.");
        let data = b.input("data", data_bits);
        let stored = b.input("check", hamming + 1);
        let corrected = b.output("corrected", data_bits);
        let fixed = b.output("fixed", 1)[0];
        let failed = b.output("failed", 1)[0];
        {
            let mut l = Logic { b: &mut b, next: 0 };
            let fresh = hamming_bits(&mut l, &data, hamming);
            let syndrome: Vec<Bit> = fresh.iter().zip(&stored).map(|(a, c)| l.xor(*a, *c)).collect();
            let everything: Vec<Bit> = data.iter().chain(&stored).copied().collect();
            let odd = parity_of(&mut l, &everything);
            let any = l.any(&syndrome);
            for (i, position) in data_positions(data_bits).iter().enumerate() {
                let hit = l.equals(&syndrome, *position);
                let flip = l.and(hit, odd);
                let out = l.xor(data[i], flip);
                l.b.or_to(out, Bit::Zero, corrected[i]);
            }
            // Two errors leave even parity with a non-zero syndrome; so does an impossible syndrome.
            let even = l.not(odd);
            let double = l.and(any, even);
            let valid: Vec<Bit> = data_positions(data_bits).iter().copied().chain((0..hamming).map(|k| 1 << k)).map(|p| l.equals(&syndrome, p)).collect();
            let known = l.any(&valid);
            let unknown = l.not(known);
            let wild = {
                let t = l.and(any, unknown);
                l.and(t, odd)
            };
            let bad = l.or(double, wild);
            l.b.or_to(bad, Bit::Zero, failed);
            let good = l.not(bad);
            let single = l.and(odd, good);
            l.b.or_to(single, Bit::Zero, fixed);
        }
        self.add(b.finish())
    }

    /// Corrects only the lane being read: the check is computed on the whole word, but
    /// turning its syndrome into a correction takes a handful of gates per weight bit.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn lane_corrector(&mut self, lanes: u32, weight_bits: u32) -> String {
        let name = format!("lane_corrector_{lanes}_{weight_bits}");
        if self.modules.contains_key(&name) {
            return name;
        }
        let (lane_bits, bit_bits) = lane_index_bits(lanes, weight_bits);
        let width = lane_bits + bit_bits;
        let total = lane_check_bits(lanes, weight_bits);
        let word = lanes * weight_bits;
        let mut b = Builder::new(&name, "Corrector ECC por carril", "Comprueba la palabra entera y corrige solo el peso que se está leyendo.");
        let data = b.input("data", word);
        let stored = b.input("check", total);
        let lane_now = b.input("lane", lane_bits);
        let weight = b.output("weight", weight_bits);
        let fixed = b.output("fixed", 1)[0];
        let failed = b.output("failed", 1)[0];
        {
            let mut l = Logic { b: &mut b, next: 0 };
            let covered = |k: u32| -> Vec<Bit> {
                let mut bits = Vec::new();
                for lane in 0..lanes {
                    for bit in 0..weight_bits {
                        if lane_column(lane, bit, lanes, weight_bits) >> k & 1 == 1 {
                            bits.push(data[(lane * weight_bits + bit) as usize]);
                        }
                    }
                }
                bits
            };
            let mut fresh: Vec<Bit> = (0..width).map(|k| parity_of(&mut l, &covered(k))).collect();
            // Both markers cover every data bit, so one parity tree serves them.
            let markers = parity_of(&mut l, &covered(width));
            fresh.extend([markers, markers]);
            fresh.push(parity_of(&mut l, &covered(width + 2)));
            let syndrome: Vec<Bit> = fresh.iter().zip(&stored).map(|(a, c)| l.xor(*a, *c)).collect();
            let odd = parity_of(&mut l, &syndrome);
            let marked = l.and(syndrome[width as usize], syndrome[width as usize + 1]);
            let same_lane: Vec<Bit> = syndrome[bit_bits as usize..width as usize].iter().zip(&lane_now).map(|(a, c)| {
                let differs = l.xor(*a, *c);
                l.not(differs)
            }).collect();
            let in_lane = same_lane.iter().fold(l.and(marked, odd), |all, bit| l.and(all, *bit));
            let leaves: Vec<Vec<Bit>> = data.chunks(weight_bits as usize).map(|lane| lane.to_vec()).collect();
            let raw = l.select(&lane_now, &leaves);
            for (bit, value) in raw.iter().enumerate() {
                let here = l.equals(&syndrome[..bit_bits as usize], bit as u32);
                let flip = l.and(in_lane, here);
                let out = l.xor(*value, flip);
                l.b.or_to(out, Bit::Zero, weight[bit]);
            }
            // A lone set bit is an error in the check bits themselves, which harms nothing.
            let (mut one, mut two) = (Bit::Zero, Bit::Zero);
            for bit in &syndrome {
                let both = l.and(one, *bit);
                two = l.or(two, both);
                one = l.or(one, *bit);
            }
            let single_check = {
                let not_two = l.not(two);
                l.and(one, not_two)
            };
            let in_range = (weight_bits..1 << bit_bits).fold(Bit::One, |all, unused| {
                let bad = l.equals(&syndrome[..bit_bits as usize], unused);
                let fine = l.not(bad);
                l.and(all, fine)
            });
            let column = l.and(marked, in_range);
            let known = l.or(column, single_check);
            let even = l.not(odd);
            let double = l.and(one, even);
            let unknown = l.not(known);
            let wild = {
                let t = l.and(one, odd);
                l.and(t, unknown)
            };
            let bad = l.or(double, wild);
            l.b.or_to(bad, Bit::Zero, failed);
            let good = l.not(bad);
            let single = {
                let t = l.and(one, odd);
                l.and(t, good)
            };
            l.b.or_to(single, Bit::Zero, fixed);
        }
        self.add(b.finish())
    }

    /// March C−: finds stuck bits and most coupling faults in about 11 cycles per word.
    fn march_test(&mut self, addr_bits: u32, width: u32) -> String {
        let name = format!("march_test_{addr_bits}_{width}");
        if self.modules.contains_key(&name) {
            return name;
        }
        let a = addr_bits as usize;
        let mut b = Builder::new(&name, "Autotest de memoria", "Escribe y lee toda la memoria en secuencia ascendente y descendente para encontrar bits defectuosos.");
        let start = b.input("start", 1)[0];
        let rdata = b.input("mem_rdata", width);
        let mem_addr = b.output("mem_addr", addr_bits);
        let mem_we = b.output("mem_we", 1)[0];
        let mem_wdata = b.output("mem_wdata", width);
        let done_out = b.output("done", 1)[0];
        let fail_out = b.output("fail", 1)[0];
        let fail_addr = b.output("fail_addr", addr_bits);
        {
            let mut l = Logic { b: &mut b, next: 0 };
            l.b.group("Secuencia");
            let (running, sub, failed, finished) = (l.fresh(), l.fresh(), l.fresh(), l.fresh());
            let phase = l.fresh_bus(3);
            let address = l.fresh_bus(a);
            let writes_only = l.equals(&phase, 0);
            let reads = l.not(writes_only);
            let last_phase = l.equals(&phase, 5);
            let writes = l.not(last_phase);
            let (p3, p4, p2) = (l.equals(&phase, 3), l.equals(&phase, 4), l.equals(&phase, 2));
            let down = l.or(p3, p4);
            let value_out = {
                let p1 = l.equals(&phase, 1);
                l.or(p1, p3)
            };
            let expected = l.or(p2, p4);
            let not_sub = l.not(sub);
            let compare = l.and(running, sub);
            let settled = {
                let t = l.or(writes_only, sub);
                l.and(t, running)
            };
            let we = {
                let t = l.and(writes, settled);
                let plain = l.not(reads);
                let u = l.or(plain, sub);
                l.and(t, u)
            };
            l.b.or_to(we, Bit::Zero, mem_we);
            for out in &mem_wdata {
                l.b.or_to(value_out, Bit::Zero, *out);
            }
            for (from, to) in address.iter().zip(&mem_addr) {
                l.b.or_to(*from, Bit::Zero, *to);
            }

            l.b.group("Comparación");
            let wrong: Vec<Bit> = rdata.iter().map(|bit| l.xor(*bit, expected)).collect();
            let any_wrong = l.any(&wrong);
            let bad_now = {
                let t = l.and(compare, reads);
                l.and(t, any_wrong)
            };

            l.b.group("Direcciones");
            let last_address = {
                let literals: Vec<Bit> = address.iter().map(|bit| l.xor(*bit, down)).collect();
                l.all(&literals)
            };
            let step_on = l.and(settled, last_address);
            let holds = {
                let phase_ends = l.or(p2, p4);
                l.and(phase_ends, step_on)
            };
            let moves = {
                let free = l.not(holds);
                l.and(settled, free)
            };
            let mut borrow_or_carry = moves;
            let stepped: Vec<Bit> = address
                .iter()
                .map(|bit| {
                    let out = l.xor(*bit, borrow_or_carry);
                    let propagate = l.xor(*bit, down);
                    borrow_or_carry = l.and(propagate, borrow_or_carry);
                    out
                })
                .collect();
            let zero_address = constant(0, a);
            let next_address = l.mux_bus(start, &stepped, &zero_address);
            l.connect_flops(&next_address, &address);

            l.b.group("Secuencia");
            let phase_step = {
                let more = l.not(last_phase);
                l.and(step_on, more)
            };
            let mut carry = phase_step;
            let advanced: Vec<Bit> = phase
                .iter()
                .map(|bit| {
                    let out = l.xor(*bit, carry);
                    carry = l.and(*bit, carry);
                    out
                })
                .collect();
            let zero_phase = constant(0, 3);
            let next_phase = l.mux_bus(start, &advanced, &zero_phase);
            l.connect_flops(&next_phase, &phase);

            let sub_toggle = l.and(running, reads);
            let next_sub = {
                let t = l.and(sub_toggle, not_sub);
                let keep = l.not(start);
                l.and(t, keep)
            };
            l.connect_flops(&[next_sub], &[sub]);

            let test_over = {
                let t = l.and(step_on, last_phase);
                l.or(t, bad_now)
            };
            let not_start = l.not(start);
            let next_running = {
                let more = l.not(test_over);
                let t = l.and(running, more);
                l.or(start, t)
            };
            l.connect_flops(&[next_running], &[running]);
            let next_finished = {
                let t = l.or(finished, test_over);
                l.and(t, not_start)
            };
            l.connect_flops(&[next_finished], &[finished]);
            l.b.or_to(finished, Bit::Zero, done_out);
            let next_failed = {
                let t = l.or(failed, bad_now);
                l.and(t, not_start)
            };
            l.connect_flops(&[next_failed], &[failed]);
            l.b.or_to(failed, Bit::Zero, fail_out);

            let first = {
                let already = l.not(failed);
                l.and(bad_now, already)
            };
            let held = l.fresh_bus(a);
            let next_held = l.mux_bus(first, &held, &address);
            l.connect_flops(&next_held, &held);
            for (from, to) in held.iter().zip(&fail_addr) {
                l.b.or_to(*from, Bit::Zero, *to);
            }
        }
        self.add(b.finish())
    }

    /// A memory that tests itself and replaces up to `spares` bad rows with spare ones.
    fn repairable_memory(&mut self, addr_bits: u32, width: u32, spares: u32) -> String {
        let name = format!("repairable_memory_{addr_bits}_{width}_{spares}");
        if self.modules.contains_key(&name) {
            return name;
        }
        let march = self.march_test(addr_bits, width);
        let (a, s) = (addr_bits as usize, spares as usize);
        let spare_bits = (usize::BITS - (s.max(2) - 1).leading_zeros()) as usize;
        let mut b = Builder::new(&name, "Memoria reparable", "Comprueba sus filas al arrancar y sustituye las defectuosas por filas de reserva.");
        let start = b.input("test_start", 1)[0];
        let addr = b.input("addr", addr_bits);
        let we = b.input("we", 1)[0];
        let wdata = b.input("wdata", width);
        let rdata = b.output("rdata", width);
        let done_out = b.output("test_done", 1)[0];
        let failed_out = b.output("test_failed", 1)[0];
        let used_out = b.output("spares_used", spare_bits as u32 + 1);

        let (march_addr, march_we, march_wdata) = (b.wire("march_addr", addr_bits), b.wire("march_we", 1)[0], b.wire("march_wdata", width));
        let (march_done, march_fail, fail_addr) = (b.wire("march_done", 1)[0], b.wire("march_fail", 1)[0], b.wire("fail_addr", addr_bits));
        let restart = b.wire("restart", 1)[0];
        b.group("Autotest");
        b.inst("march", &march, &[("start", vec![restart]), ("mem_rdata", rdata.clone()), ("mem_addr", march_addr.clone()), ("mem_we", vec![march_we]), ("mem_wdata", march_wdata.clone()), ("done", vec![march_done]), ("fail", vec![march_fail]), ("fail_addr", fail_addr.clone())]);
        let (physical, chosen_we, chosen_wdata);
        {
            let mut l = Logic { b: &mut b, next: 0 };
            l.b.group("Control");
            let (testing, finished, out_of_spares, launch) = (l.fresh(), l.fresh(), l.fresh(), l.fresh());
            let used = l.fresh_bus(spare_bits + 1);
            let needs_repair = {
                // The cycle after a repair the march test still shows its old result.
                let settled = l.not(launch);
                let t = l.and(testing, march_done);
                let u = l.and(t, march_fail);
                l.and(u, settled)
            };
            let full = l.equals(&used, spares);
            let repair = {
                let room = l.not(full);
                l.and(needs_repair, room)
            };
            let give_up = l.and(needs_repair, full);
            let passed = {
                let clean = l.not(march_fail);
                let settled = l.not(launch);
                let t = l.and(testing, march_done);
                let u = l.and(t, clean);
                l.and(u, settled)
            };
            let begin = l.or(start, launch);
            l.b.or_to(begin, Bit::Zero, restart);
            l.connect_flops(&[repair], &[launch]);
            let ending = l.or(give_up, passed);
            let next_testing = {
                let more = l.not(ending);
                let t = l.and(testing, more);
                l.or(start, t)
            };
            l.connect_flops(&[next_testing], &[testing]);
            let not_start = l.not(start);
            let next_finished = {
                let t = l.or(finished, ending);
                l.and(t, not_start)
            };
            l.connect_flops(&[next_finished], &[finished]);
            l.b.or_to(finished, Bit::Zero, done_out);
            let next_out = {
                let t = l.or(out_of_spares, give_up);
                l.and(t, not_start)
            };
            l.connect_flops(&[next_out], &[out_of_spares]);
            l.b.or_to(out_of_spares, Bit::Zero, failed_out);
            let mut carry = repair;
            let bumped: Vec<Bit> = used
                .iter()
                .map(|bit| {
                    let out = l.xor(*bit, carry);
                    carry = l.and(*bit, carry);
                    out
                })
                .collect();
            let zero = constant(0, spare_bits + 1);
            let next_used = l.mux_bus(start, &bumped, &zero);
            l.connect_flops(&next_used, &used);
            for (from, to) in used.iter().zip(&used_out) {
                l.b.or_to(*from, Bit::Zero, *to);
            }

            l.b.group("Filas de reserva");
            let logical = l.mux_bus(testing, &addr, &march_addr);
            let mut mapped = logical.clone();
            let mut to_spare = Bit::Zero;
            for spare in 0..s {
                let valid = l.fresh();
                let stored = l.fresh_bus(a);
                let this_one = l.equals(&used[..spare_bits + 1], spare as u32);
                let write = l.and(repair, this_one);
                let next_valid = {
                    let t = l.or(valid, write);
                    l.and(t, not_start)
                };
                l.connect_flops(&[next_valid], &[valid]);
                let next_stored = l.mux_bus(write, &stored, &fail_addr);
                l.connect_flops(&next_stored, &stored);
                let same: Vec<Bit> = logical.iter().zip(&stored).map(|(x, y)| { let d = l.xor(*x, *y); l.not(d) }).collect();
                let equal = l.all(&same);
                let hit = l.and(equal, valid);
                let target = constant(spare as u32, a);
                mapped = l.mux_bus(hit, &mapped, &target);
                to_spare = l.or(to_spare, hit);
            }
            physical = mapped.iter().copied().chain([to_spare]).collect::<Vec<Bit>>();
            chosen_we = l.mux(testing, we, march_we);
            chosen_wdata = l.mux_bus(testing, &wdata, &march_wdata);
        }
        b.group("Memoria");
        b.sram("cells", addr_bits + 1, width, &[("addr", physical), ("we", vec![chosen_we]), ("wdata", chosen_wdata), ("rdata", rdata)]);
        self.add(b.finish())
    }
}

pub fn secded_design(data_bits: u32, decoder: bool) -> Design {
    let mut gen = Gen { modules: BTreeMap::new() };
    let top = if decoder { gen.secded_decoder(data_bits) } else { gen.secded_encoder(data_bits) };
    Design { top, modules: gen.modules }
}

pub fn repairable_memory_design(addr_bits: u32, width: u32, spares: u32) -> Design {
    let mut gen = Gen { modules: BTreeMap::new() };
    let top = gen.repairable_memory(addr_bits, width, spares);
    Design { top, modules: gen.modules }
}
