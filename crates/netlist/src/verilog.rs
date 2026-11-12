//! Structural Verilog-2001, accepted by FPGA and ASIC tools alike.

use crate::{Bit, Design, Dir, InstKind, Module, Prim};
use std::fmt::Write;

const LIBRARY: &str = "\
module llm2chip_dff (input clk, input d, output reg q);
  initial q = 1'b0;
  always @(posedge clk) q <= d;
endmodule

module llm2chip_sram #(parameter ADDR_BITS = 8, parameter WIDTH = 8) (
  input clk, input [ADDR_BITS-1:0] addr, input we,
  input [WIDTH-1:0] wdata, output reg [WIDTH-1:0] rdata
);
  reg [WIDTH-1:0] memory [0:(1<<ADDR_BITS)-1];
  integer i;
  initial begin
    rdata = 0;
    for (i = 0; i < (1<<ADDR_BITS); i = i + 1) memory[i] = 0;
  end
  always @(posedge clk) begin
    rdata <= memory[addr];
    if (we) memory[addr] <= wdata;
  end
endmodule

module llm2chip_rom #(parameter ADDR_BITS = 8, parameter WIDTH = 8) (
  input clk, input [ADDR_BITS-1:0] addr, output reg [WIDTH-1:0] rdata
);
  reg [WIDTH-1:0] memory [0:(1<<ADDR_BITS)-1];
  integer i;
  initial begin
    rdata = 0;
    for (i = 0; i < (1<<ADDR_BITS); i = i + 1) memory[i] = 0;
  end
  always @(posedge clk) rdata <= memory[addr];
endmodule

module llm2chip_ram2p #(parameter ADDR_BITS = 8, parameter WIDTH = 8) (
  input clk, input [ADDR_BITS-1:0] raddr, input [ADDR_BITS-1:0] waddr, input we,
  input [WIDTH-1:0] wdata, output reg [WIDTH-1:0] rdata
);
  reg [WIDTH-1:0] memory [0:(1<<ADDR_BITS)-1];
  integer i;
  initial begin
    rdata = 0;
    for (i = 0; i < (1<<ADDR_BITS); i = i + 1) memory[i] = 0;
  end
  always @(posedge clk) begin
    rdata <= memory[raddr];
    if (we) memory[waddr] <= wdata;
  end
endmodule
";

fn bit_text(m: &Module, bit: &Bit) -> String {
    match bit {
        Bit::Zero => "1'b0".into(),
        Bit::One => "1'b1".into(),
        Bit::Net(id, index) => {
            let net = &m.nets[*id];
            if net.width == 1 {
                net.name.clone()
            } else {
                format!("{}[{index}]", net.name)
            }
        }
    }
}

fn bus_text(m: &Module, bits: &[Bit]) -> String {
    if bits.len() == 1 {
        return bit_text(m, &bits[0]);
    }
    let parts: Vec<String> = bits.iter().rev().map(|b| bit_text(m, b)).collect();
    format!("{{{}}}", parts.join(", "))
}

fn range(width: u32) -> String {
    if width > 1 {
        format!("[{}:0] ", width - 1)
    } else {
        String::new()
    }
}

fn emit_module(out: &mut String, m: &Module) {
    let needs_clock = true;
    let _ = writeln!(out, "// {}: {}", m.title, m.description);
    let mut ports: Vec<String> = Vec::new();
    if needs_clock {
        ports.push("input clk".into());
    }
    for p in &m.ports {
        let dir = if p.dir == Dir::Input { "input" } else { "output" };
        ports.push(format!("{dir} {}{}", range(m.nets[p.net].width), p.name));
    }
    let _ = writeln!(out, "module {} (\n  {}\n);", m.name, ports.join(",\n  "));

    for (id, net) in m.nets.iter().enumerate() {
        if m.ports.iter().all(|p| p.net != id) {
            let _ = writeln!(out, "  wire {}{};", range(net.width), net.name);
        }
    }

    let mut group = "";
    for inst in &m.instances {
        if inst.group != group {
            group = &inst.group;
            let _ = writeln!(out, "\n  // {group}");
        }
        let pin = |port: &str| bus_text(m, &inst.conns[port]);
        match &inst.kind {
            InstKind::Prim(Prim::And2) => {
                let _ = writeln!(out, "  and {} ({}, {}, {});", inst.name, pin("y"), pin("a"), pin("b"));
            }
            InstKind::Prim(Prim::Or2) => {
                let _ = writeln!(out, "  or {} ({}, {}, {});", inst.name, pin("y"), pin("a"), pin("b"));
            }
            InstKind::Prim(Prim::Xor2) => {
                let _ = writeln!(out, "  xor {} ({}, {}, {});", inst.name, pin("y"), pin("a"), pin("b"));
            }
            InstKind::Prim(Prim::Not) => {
                let _ = writeln!(out, "  not {} ({}, {});", inst.name, pin("y"), pin("a"));
            }
            InstKind::Prim(Prim::Mux2) => {
                let _ = writeln!(out, "  assign {} = {} ? {} : {};", pin("y"), pin("s"), pin("b"), pin("a"));
            }
            InstKind::Prim(Prim::Dff) => {
                let _ = writeln!(out, "  llm2chip_dff {} (.clk(clk), .d({}), .q({}));", inst.name, pin("d"), pin("q"));
            }
            InstKind::Prim(Prim::Sram { addr_bits, width }) => {
                let _ = writeln!(
                    out,
                    "  llm2chip_sram #(.ADDR_BITS({addr_bits}), .WIDTH({width})) {} (.clk(clk), .addr({}), .we({}), .wdata({}), .rdata({}));",
                    inst.name,
                    pin("addr"),
                    pin("we"),
                    pin("wdata"),
                    pin("rdata")
                );
            }
            InstKind::Prim(Prim::Rom { addr_bits, width }) => {
                let _ = writeln!(out, "  llm2chip_rom #(.ADDR_BITS({addr_bits}), .WIDTH({width})) {} (.clk(clk), .addr({}), .rdata({}));", inst.name, pin("addr"), pin("rdata"));
            }
            InstKind::Prim(Prim::Ram2p { addr_bits, width }) => {
                let _ = writeln!(
                    out,
                    "  llm2chip_ram2p #(.ADDR_BITS({addr_bits}), .WIDTH({width})) {} (.clk(clk), .raddr({}), .waddr({}), .we({}), .wdata({}), .rdata({}));",
                    inst.name,
                    pin("raddr"),
                    pin("waddr"),
                    pin("we"),
                    pin("wdata"),
                    pin("rdata")
                );
            }
            InstKind::Module(child) => {
                let conns: Vec<String> = inst.conns.iter().map(|(port, bits)| format!(".{port}({})", bus_text(m, bits))).collect();
                let _ = writeln!(out, "  {child} {} (.clk(clk), {});", inst.name, conns.join(", "));
            }
        }
    }
    let _ = writeln!(out, "endmodule\n");
}

/// The whole design as one Verilog file, children before their parents.
pub fn to_verilog(design: &Design) -> String {
    write_design(design, false)
}

/// The same design for an FPGA: each multiplier that finishes in one cycle is written as a
/// signed product, which the tools map onto the device's multiplier blocks. Every kind of
/// multiplier the generator builds computes exactly that product.
pub fn to_verilog_for_fpga(design: &Design) -> String {
    write_design(design, true)
}

/// A one-cycle multiplier, `weight × x` with both signed, written so tools infer it.
fn behavioural_multiplier(m: &Module) -> Option<String> {
    let name = m.name.strip_prefix("wallace_").or_else(|| m.name.strip_prefix("booth_")).unwrap_or(&m.name);
    let shape = name.strip_prefix("multiplier_")?.strip_suffix("_s1")?;
    let (weight, x) = shape.split_once('x')?;
    let (weight, x): (u32, u32) = (weight.parse().ok()?, x.parse().ok()?);
    let mut ports: Vec<&str> = m.ports.iter().map(|p| p.name.as_str()).filter(|p| *p != "clk").collect();
    ports.sort_unstable();
    (ports == ["product", "weight", "x"]).then(|| {
        format!(
            "module {} (\n  input clk,\n  input [{}:0] weight,\n  input [{}:0] x,\n  output [{}:0] product\n);\n  assign product = $signed(weight) * $signed(x);\nendmodule\n\n",
            m.name,
            weight - 1,
            x - 1,
            weight + x - 1
        )
    })
}

/// An adder written as a sum, so the FPGA tools put it on their carry chains instead of
/// in a chain of look-up tables. Ports `a`, `b`, optional `cin` and `cout`, and `sum`.
fn behavioural_adder(m: &Module) -> Option<String> {
    let width: u32 = m.name.strip_prefix("ripple_adder_").or_else(|| m.name.strip_prefix("prefix_adder_"))?.parse().ok()?;
    let mut ports: Vec<&str> = m.ports.iter().map(|p| p.name.as_str()).filter(|p| *p != "clk").collect();
    ports.sort_unstable();
    let top = width - 1;
    match ports.as_slice() {
        ["a", "b", "cin", "cout", "sum"] => Some(format!(
            "module {} (\n  input clk,\n  input [{top}:0] a,\n  input [{top}:0] b,\n  input cin,\n  output [{top}:0] sum,\n  output cout\n);\n  assign {{cout, sum}} = a + b + cin;\nendmodule\n\n",
            m.name
        )),
        ["a", "b", "sum"] => Some(format!("module {} (\n  input clk,\n  input [{top}:0] a,\n  input [{top}:0] b,\n  output [{top}:0] sum\n);\n  assign sum = a + b;\nendmodule\n\n", m.name)),
        _ => None,
    }
}

/// A table of flip-flops read without the clock, written as a memory so the FPGA tools put
/// it in their distributed RAM, one copy per read port, instead of selecting among
/// thousands of flip-flops. Ports `we`, `waddr`, `wdata`, and `keyN` and `entryN` per read.
fn behavioural_register_table(m: &Module) -> Option<String> {
    let shape = m.name.strip_prefix("register_table_")?;
    let (size, reads) = shape.split_once("_r")?;
    let (addr_bits, width) = size.split_once('x')?;
    let (addr_bits, width, reads): (u32, u32, u32) = (addr_bits.parse().ok()?, width.parse().ok()?, reads.parse().ok()?);
    let mut text = format!("module {} (\n  input clk,\n  input we,\n  input [{a}:0] waddr,\n  input [{w}:0] wdata", m.name, a = addr_bits - 1, w = width - 1);
    for i in 0..reads {
        let _ = write!(text, ",\n  input [{}:0] key{i},\n  output [{}:0] entry{i}", addr_bits - 1, width - 1);
    }
    let _ = write!(text, "\n);\n  reg [{}:0] entries [0:{}];\n  integer i;\n  initial for (i = 0; i < {}; i = i + 1) entries[i] = 0;\n  always @(posedge clk) if (we) entries[waddr] <= wdata;\n", width - 1, (1u64 << addr_bits) - 1, 1u64 << addr_bits);
    for i in 0..reads {
        let _ = writeln!(text, "  assign entry{i} = entries[key{i}];");
    }
    text.push_str("endmodule\n\n");
    Some(text)
}

fn write_design(design: &Design, fpga: bool) -> String {
    let mut out = String::from("// Generado por llm2chip.\n\n");
    out.push_str(LIBRARY);
    out.push('\n');

    fn visit<'a>(design: &'a Design, name: &str, seen: &mut Vec<&'a str>) {
        let module = &design.modules[name];
        for inst in &module.instances {
            if let InstKind::Module(child) = &inst.kind {
                visit(design, child, seen);
            }
        }
        if !seen.contains(&module.name.as_str()) {
            seen.push(module.name.as_str());
        }
    }
    let mut order = Vec::new();
    visit(design, &design.top, &mut order);
    for name in order {
        let module = &design.modules[name];
        match fpga.then(|| behavioural_multiplier(module).or_else(|| behavioural_adder(module)).or_else(|| behavioural_register_table(module))).flatten() {
            Some(text) => out.push_str(&text),
            None => emit_module(&mut out, module),
        }
    }
    out
}

/// Input values for one clock cycle of a generated testbench.
pub struct TestStep {
    pub inputs: Vec<(String, u128)>,
}

/// One word of a ROM, as the mask would fix it. `path` is relative to the top module.
#[derive(Clone, Debug)]
pub struct RomWord {
    pub path: String,
    pub address: usize,
    pub value: u128,
}

/// A testbench that applies `steps` and prints `watch` after every rising edge.
pub fn verilog_testbench(design: &Design, steps: &[TestStep], watch: &str) -> String {
    verilog_testbench_with_rom(design, &[], steps, watch)
}

/// The same, with the contents of the design's ROMs written before the first clock.
pub fn verilog_testbench_with_rom(design: &Design, rom: &[RomWord], steps: &[TestStep], watch: &str) -> String {
    let top = &design.modules[&design.top];
    let mut out = String::from("`timescale 1ns/1ns\nmodule testbench;\n  reg clk = 0;\n");
    let mut pins = vec![".clk(clk)".to_string()];
    for p in &top.ports {
        let kind = if p.dir == Dir::Input { "reg" } else { "wire" };
        let init = if p.dir == Dir::Input { " = 0" } else { "" };
        let _ = writeln!(out, "  {kind} {}{}{init};", range(top.nets[p.net].width), p.name);
        pins.push(format!(".{}({})", p.name, p.name));
    }
    let _ = writeln!(out, "  {} dut ({});", top.name, pins.join(", "));
    out.push_str("  initial begin\n");
    if !rom.is_empty() {
        out.push_str("    #0;\n");
    }
    for word in rom {
        let _ = writeln!(out, "    dut.{}.memory[{}] = {};", word.path.replace('/', "."), word.address, word.value);
    }
    for step in steps {
        for (name, value) in &step.inputs {
            let width = top.ports.iter().find(|p| &p.name == name).map_or(128, |p| top.nets[p.net].width);
            let masked = if width >= 128 { *value } else { value & ((1u128 << width) - 1) };
            let _ = writeln!(out, "    {name} = {width}'d{masked};");
        }
        let _ = writeln!(out, "    #1 clk = 1; #1 $display(\"%0d\", {watch}); clk = 0; #1;");
    }
    out.push_str("    $finish;\n  end\nendmodule\n");
    out
}
