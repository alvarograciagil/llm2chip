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
        emit_module(&mut out, &design.modules[name]);
    }
    out
}

/// Input values for one clock cycle of a generated testbench.
pub struct TestStep {
    pub inputs: Vec<(String, u128)>,
}

/// A testbench that applies `steps` and prints `watch` after every rising edge.
pub fn verilog_testbench(design: &Design, steps: &[TestStep], watch: &str) -> String {
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
