//! Generates the compute cell: weight memory, multiplier and accumulator.
//! Weights and activations are signed two's complement.

use crate::format::WeightCoding;
use crate::multiplier::MultiplierKind;
use crate::{Bit, Builder, Design, Module};
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct CellConfig {
    pub weight_bits: u32,
    pub activation_bits: u32,
    pub accumulator_bits: u32,
    pub mac_stages: u32,
    pub adder_stages: u32,
    /// log2 of the number of weights the cell's memory holds.
    pub addr_bits: u32,
    /// How the stored codes are turned into numbers. See `format`.
    pub coding: WeightCoding,
    /// Float: exponent bits. Posit: `es`. Logarithmic: fraction bits. Otherwise unused.
    pub exponent_bits: u32,
    pub multiplier: MultiplierKind,
    /// The weights are fixed at manufacture: the cell reads a ROM and has no way to write it.
    pub fixed_weights: bool,
}

impl CellConfig {
    /// Cycles from presenting an input until the accumulator has absorbed it.
    pub fn latency(&self) -> u32 {
        let table = (self.coding == WeightCoding::Codebook) as u32;
        1 + table + (self.mac_stages.min(self.significand_bits()).max(1) - 1) + (self.adder_stages.max(1) - 1) + 1
    }

    pub(crate) fn checked(self) -> Self {
        let weight_bits = self.weight_bits.clamp(2, 32);
        Self {
            weight_bits,
            activation_bits: self.activation_bits.clamp(2, 64),
            accumulator_bits: self.accumulator_bits.clamp(2, 128),
            mac_stages: self.mac_stages.clamp(1, 6),
            adder_stages: self.adder_stages.clamp(1, 6),
            addr_bits: self.addr_bits.clamp(1, 20),
            coding: self.coding,
            exponent_bits: self.coding.legal_exponent_bits(weight_bits, self.exponent_bits),
            multiplier: self.multiplier,
            fixed_weights: self.fixed_weights,
        }
    }
}

pub(crate) struct Gen {
    pub(crate) modules: BTreeMap<String, Module>,
}

impl Gen {
    pub(crate) fn add(&mut self, module: Module) -> String {
        let name = module.name.clone();
        self.modules.insert(name.clone(), module);
        name
    }

    pub(crate) fn full_adder(&mut self) -> String {
        if self.modules.contains_key("full_adder") {
            return "full_adder".into();
        }
        let mut b = Builder::new("full_adder", "Sumador completo", "Suma tres bits y produce el bit de suma y el de acarreo.");
        let a = b.input("a", 1)[0];
        let x = b.input("b", 1)[0];
        let cin = b.input("cin", 1)[0];
        let sum = b.output("sum", 1)[0];
        let cout = b.output("cout", 1)[0];
        let p = b.wire("p", 1)[0];
        let g1 = b.wire("g1", 1)[0];
        let g2 = b.wire("g2", 1)[0];
        b.group("Suma");
        b.xor_to(a, x, p);
        b.xor_to(p, cin, sum);
        b.group("Acarreo");
        b.and_to(a, x, g1);
        b.and_to(p, cin, g2);
        b.or_to(g1, g2, cout);
        self.add(b.finish())
    }

    pub(crate) fn ripple_adder(&mut self, width: u32) -> String {
        let name = format!("ripple_adder_{width}");
        if self.modules.contains_key(&name) {
            return name;
        }
        let fa = self.full_adder();
        let mut b = Builder::new(&name, &format!("Sumador de {width} bits"), "Cadena de sumadores completos: cada acarreo alimenta al siguiente bit.");
        let a = b.input("a", width);
        let x = b.input("b", width);
        let cin = b.input("cin", 1)[0];
        let sum = b.output("sum", width);
        let cout = b.output("cout", 1)[0];
        b.group("Cadena de acarreo");
        let carries = b.wire("carry", width.saturating_sub(1));
        let mut carry = cin;
        for i in 0..width as usize {
            let out = if i + 1 == width as usize { cout } else { carries[i] };
            b.inst(&format!("fa{i}"), &fa, &[("a", vec![a[i]]), ("b", vec![x[i]]), ("cin", vec![carry]), ("sum", vec![sum[i]]), ("cout", vec![out])]);
            carry = out;
        }
        self.add(b.finish())
    }

    pub(crate) fn register(&mut self, width: u32) -> String {
        let name = format!("register_{width}");
        if self.modules.contains_key(&name) {
            return name;
        }
        let mut b = Builder::new(&name, &format!("Registro de {width} bits"), "Guarda el valor de entrada en cada flanco de reloj.");
        let d = b.input("d", width);
        let q = b.output("q", width);
        b.group("Biestables");
        for i in 0..width as usize {
            b.dff_to(d[i], q[i]);
        }
        self.add(b.finish())
    }

    pub(crate) fn mux_bus(&mut self, width: u32) -> String {
        let name = format!("mux_bus_{width}");
        if self.modules.contains_key(&name) {
            return name;
        }
        let mut b = Builder::new(&name, &format!("Selector de {width} bits"), "Deja pasar «a» si la selección vale 0 y «b» si vale 1.");
        let s = b.input("s", 1)[0];
        let a = b.input("a", width);
        let x = b.input("b", width);
        let y = b.output("y", width);
        b.group("Selectores");
        for i in 0..width as usize {
            b.mux_to(s, a[i], x[i], y[i]);
        }
        self.add(b.finish())
    }

    /// `value & enable`, or for the sign row `!value & enable` so the row subtracts.
    fn partial_product(&mut self, width: u32, negate: bool) -> String {
        let name = format!("partial_product_{width}{}", if negate { "_neg" } else { "" });
        if self.modules.contains_key(&name) {
            return name;
        }
        let what = if negate {
            "Fila de signo: si el bit del peso vale 1, invierte el valor para restarlo."
        } else {
            "Deja pasar el valor desplazado si el bit del peso vale 1."
        };
        let mut b = Builder::new(&name, "Producto parcial", what);
        let enable = b.input("enable", 1)[0];
        let value = b.input("value", width);
        let out = b.output("out", width);
        b.group("Puertas");
        for i in 0..width as usize {
            if negate {
                let inverted = b.wire(&format!("inv{i}"), 1)[0];
                b.not_to(value[i], inverted);
                b.and_to(enable, inverted, out[i]);
            } else {
                b.and_to(enable, value[i], out[i]);
            }
        }
        self.add(b.finish())
    }

    pub(crate) fn multiplier(&mut self, weight_bits: u32, x_bits: u32, stages: u32) -> String {
        let width = weight_bits + x_bits;
        let stages = stages.min(weight_bits).max(1);
        let name = format!("multiplier_{weight_bits}x{x_bits}_s{stages}");
        if self.modules.contains_key(&name) {
            return name;
        }
        let adder = self.ripple_adder(width);
        let plain = self.partial_product(width, false);
        let negate = self.partial_product(width, true);
        let weight_reg = self.register(weight_bits);
        let x_reg = self.register(x_bits);
        let sum_reg = self.register(width);

        let mut b = Builder::new(
            &name,
            &format!("Multiplicador con signo {weight_bits} × {x_bits}"),
            "Suma una fila desplazada por cada bit del peso. La última fila resta, porque el bit alto tiene signo negativo.",
        );
        let weight_in = b.input("weight", weight_bits);
        let x_in = b.input("x", x_bits);
        let product = b.output("product", width);

        let cuts: Vec<u32> = (1..stages).map(|k| weight_bits * k / stages).collect();
        let mut weight = weight_in;
        let mut x = x_in;
        let mut running = vec![Bit::Zero; width as usize];

        for row in 0..weight_bits {
            b.group(&format!("Fila {row}"));
            let last = row + 1 == weight_bits;
            let sign = *x.last().expect("activación sin bits");
            let shifted: Vec<Bit> = (0..width as usize)
                .map(|i| {
                    let i = i as i64 - row as i64;
                    if i < 0 {
                        Bit::Zero
                    } else {
                        x.get(i as usize).copied().unwrap_or(sign)
                    }
                })
                .collect();
            let selected = weight[row as usize];
            let partial = b.wire(&format!("pp{row}"), width);
            b.inst(&format!("partial{row}"), if last { &negate } else { &plain }, &[("enable", vec![selected]), ("value", shifted), ("out", partial.clone())]);
            let sum = b.wire(&format!("sum{row}"), width);
            let unused = b.wire(&format!("carry_out{row}"), 1);
            let cin = if last { selected } else { Bit::Zero };
            b.inst(&format!("adder{row}"), &adder, &[("a", running), ("b", partial), ("cin", vec![cin]), ("sum", sum.clone()), ("cout", unused)]);
            running = sum;

            // A pipeline cut stores everything the later rows still need.
            if cuts.contains(&(row + 1)) && !last {
                b.group(&format!("Etapa tras fila {row}"));
                let q_sum = b.wire(&format!("sum{row}_q"), width);
                b.inst(&format!("sum_reg{row}"), &sum_reg, &[("d", running), ("q", q_sum.clone())]);
                running = q_sum;
                let q_weight = b.wire(&format!("weight{row}_q"), weight_bits);
                b.inst(&format!("weight_reg{row}"), &weight_reg, &[("d", weight), ("q", q_weight.clone())]);
                weight = q_weight;
                let q_x = b.wire(&format!("x{row}_q"), x_bits);
                b.inst(&format!("x_reg{row}"), &x_reg, &[("d", x), ("q", q_x.clone())]);
                x = q_x;
            }
        }

        b.group("Resultado");
        for (i, bit) in running.iter().enumerate() {
            b.xor_to(*bit, Bit::Zero, product[i]);
        }
        self.add(b.finish())
    }

    /// `fast` swaps the ripple adder for a parallel-prefix one.
    pub(crate) fn accumulator(&mut self, width: u32, fast: bool) -> String {
        let name = format!("accumulator_{width}{}", if fast { "_fast" } else { "" });
        if self.modules.contains_key(&name) {
            return name;
        }
        let adder = if fast { self.prefix_adder(width) } else { self.ripple_adder(width) };
        let mux = self.mux_bus(width);
        let reg = self.register(width);
        let mut b = Builder::new(&name, &format!("Acumulador de {width} bits"), "Suma cada producto válido al total. «clear» lo devuelve a cero.");
        let mut sum_ports: Vec<(&str, Vec<Bit>)> = Vec::new();
        let clear = b.input("clear", 1)[0];
        let valid = b.input("valid", 1)[0];
        let value = b.input("value", width);
        let total = b.output("total", width);
        let not_clear = b.wire("not_clear", 1)[0];
        let sum = b.wire("sum", width);
        let unused = b.wire("carry_out", 1);
        let chosen = b.wire("chosen", width);
        let next = b.wire("next", width);

        b.group("Suma");
        sum_ports.extend([("a", total.clone()), ("b", value), ("sum", sum.clone())]);
        if !fast {
            sum_ports.extend([("cin", vec![Bit::Zero]), ("cout", unused)]);
        }
        b.inst("adder", &adder, &sum_ports);
        b.group("Decisión");
        b.inst("select", &mux, &[("s", vec![valid]), ("a", total.clone()), ("b", sum), ("y", chosen.clone())]);
        b.not_to(clear, not_clear);
        for i in 0..width as usize {
            b.and_to(not_clear, chosen[i], next[i]);
        }
        b.group("Registro");
        b.inst("total_reg", &reg, &[("d", next), ("q", total)]);
        self.add(b.finish())
    }

    pub(crate) fn address_counter(&mut self, width: u32) -> String {
        let name = format!("address_counter_{width}");
        if self.modules.contains_key(&name) {
            return name;
        }
        let adder = self.ripple_adder(width);
        let reg = self.register(width);
        let mut b = Builder::new(&name, &format!("Contador de direcciones de {width} bits"), "Avanza a la siguiente posición de memoria en cada multiplicación.");
        let clear = b.input("clear", 1)[0];
        let advance = b.input("advance", 1)[0];
        let address = b.output("address", width);
        let not_clear = b.wire("not_clear", 1)[0];
        let sum = b.wire("sum", width);
        let unused = b.wire("carry_out", 1);
        let next = b.wire("next", width);

        b.group("Incremento");
        b.inst("adder", &adder, &[("a", address.clone()), ("b", vec![Bit::Zero; width as usize]), ("cin", vec![advance]), ("sum", sum.clone()), ("cout", unused)]);
        b.group("Puesta a cero");
        b.not_to(clear, not_clear);
        for i in 0..width as usize {
            b.and_to(not_clear, sum[i], next[i]);
        }
        b.group("Registro");
        b.inst("address_reg", &reg, &[("d", next), ("q", address)]);
        self.add(b.finish())
    }

    pub(crate) fn cell(&mut self, c: CellConfig) -> String {
        let (w, a, acc) = (c.weight_bits, c.activation_bits, c.accumulator_bits);
        let significand = c.significand_bits();
        let product_bits = significand + a;
        let mac_stages = c.mac_stages.min(significand).max(1);
        let counter = self.address_counter(c.addr_bits);
        let x_reg = self.register(a);
        let multiplier = match c.multiplier {
            MultiplierKind::Array => self.multiplier(significand, a, mac_stages),
            kind => self.tree_multiplier(kind, significand, a, mac_stages),
        };
        let acc_reg = self.register(acc);
        let accumulator = self.accumulator(acc, c.multiplier != MultiplierKind::Array);

        let mut b = Builder::new(
            "compute_cell",
            "Celda de cálculo",
            "Guarda sus pesos, multiplica cada uno por la activación que llega y acumula la suma.",
        );
        let clear = b.input("clear", 1)[0];
        let clear_total = b.input("clear_total", 1)[0];
        let load_ports = (!c.fixed_weights).then(|| (b.input("load_en", 1), b.input("load_addr", c.addr_bits), b.input("load_data", w)));
        let run = b.input("run", 1)[0];
        let x = b.input("x", a);
        let total = b.output("total", acc);
        let table_ports = (c.coding == WeightCoding::Codebook && !c.fixed_weights).then(|| (b.input("table_en", 1)[0], b.input("table_addr", w), b.input("table_data", significand)));
        let exp_base = c.shifts().then(|| b.input("exp_base", c.exponent_width()));

        b.group("Direcciones");
        let address = b.wire("address", c.addr_bits);
        b.inst("counter", &counter, &[("clear", vec![clear]), ("advance", vec![run]), ("address", address.clone())]);
        let weight = b.wire("weight", w);
        match load_ports {
            Some((load_en, load_addr, load_data)) => {
                let addr_mux = self.mux_bus(c.addr_bits);
                let mem_addr = b.wire("mem_addr", c.addr_bits);
                b.inst("address_select", &addr_mux, &[("s", load_en.clone()), ("a", address), ("b", load_addr), ("y", mem_addr.clone())]);
                b.group("Memoria de pesos");
                b.sram("weight_sram", c.addr_bits, w, &[("addr", mem_addr), ("we", load_en), ("wdata", load_data), ("rdata", weight.clone())]);
            }
            None => {
                b.group("Memoria de pesos");
                b.rom("weight_rom", c.addr_bits, w, &[("addr", address), ("rdata", weight.clone())]);
            }
        }

        b.group("Decodificación del formato");
        let decoded = self.decode(&mut b, &c, weight, table_ports);

        b.group("Alineación");
        let x_aligned = b.wire("x_aligned", a);
        b.inst("x_reg", &x_reg, &[("d", x), ("q", x_aligned.clone())]);
        let mut valid = b.wire("valid0", 1)[0];
        b.dff_to(run, valid);
        let x_aligned = if c.coding == WeightCoding::Codebook {
            let second = b.wire("x_table", a);
            b.inst("x_table_reg", &x_reg, &[("d", x_aligned), ("q", second.clone())]);
            let next = b.wire("valid_table", 1)[0];
            b.dff_to(valid, next);
            valid = next;
            second
        } else {
            x_aligned
        };
        let window = exp_base.map(|base| self.exponent_window(&mut b, &c, &decoded.exponent, &base));

        b.group("Multiplicador");
        let product = b.wire("product", product_bits);
        b.inst("multiplier", &multiplier, &[("weight", decoded.significand), ("x", x_aligned), ("product", product.clone())]);
        let mut window = window;
        for stage in 1..mac_stages {
            let next = b.wire(&format!("valid_mul{stage}"), 1)[0];
            b.dff_to(valid, next);
            valid = next;
            window = window.map(|(amount, under)| {
                let q = b.wire(&format!("amount_mul{stage}"), amount.len() as u32);
                for (d, out) in amount.iter().zip(&q) {
                    b.dff_to(*d, *out);
                }
                let q_under = b.wire(&format!("under_mul{stage}"), 1)[0];
                b.dff_to(under, q_under);
                (q, q_under)
            });
        }

        b.group("Etapas de suma");
        let sign = *product.last().expect("producto sin bits");
        let mut value: Vec<Bit> = (0..acc as usize).map(|i| product.get(i).copied().unwrap_or(sign)).collect();
        if let Some((amount, under)) = window {
            value = self.place_product(&mut b, acc, value, &amount, under);
        }
        for stage in 1..c.adder_stages {
            let q = b.wire(&format!("value_q{stage}"), acc);
            b.inst(&format!("value_reg{stage}"), &acc_reg, &[("d", value), ("q", q.clone())]);
            value = q;
            let next = b.wire(&format!("valid_add{stage}"), 1)[0];
            b.dff_to(valid, next);
            valid = next;
        }

        b.group("Acumulador");
        let clear_acc = b.wire("clear_acc", 1)[0];
        b.or_to(clear, clear_total, clear_acc);
        b.inst("accumulator", &accumulator, &[("clear", vec![clear_acc]), ("valid", vec![valid]), ("value", value), ("total", total)]);
        self.add(b.finish())
    }
}

/// The complete cell netlist for a configuration.
pub fn cell_design(config: CellConfig) -> Design {
    let mut gen = Gen { modules: BTreeMap::new() };
    let top = gen.cell(config.checked());
    Design { top, modules: gen.modules }
}
