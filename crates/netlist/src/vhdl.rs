//! Structural VHDL-2008, for tools and teams that prefer it to Verilog.

use crate::{Bit, Design, Dir, InstKind, Module, Prim};
use std::fmt::Write;

const LIBRARY: &str = "\
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity llm2chip_dff is
  port (clk : in std_logic; d : in std_logic; q : out std_logic := '0');
end entity;

architecture rtl of llm2chip_dff is
begin
  process (clk) begin
    if rising_edge(clk) then q <= d; end if;
  end process;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity llm2chip_ram2p is
  generic (ADDR_BITS : natural; WIDTH : natural);
  port (
    clk : in std_logic;
    raddr, waddr : in std_logic_vector(ADDR_BITS - 1 downto 0);
    we : in std_logic;
    wdata : in std_logic_vector(WIDTH - 1 downto 0);
    rdata : out std_logic_vector(WIDTH - 1 downto 0) := (others => '0')
  );
end entity;

architecture rtl of llm2chip_ram2p is
  type memory_t is array (0 to 2 ** ADDR_BITS - 1) of std_logic_vector(WIDTH - 1 downto 0);
  signal memory : memory_t := (others => (others => '0'));
begin
  process (clk) begin
    if rising_edge(clk) then
      rdata <= memory(to_integer(unsigned(raddr)));
      if we = '1' then memory(to_integer(unsigned(waddr))) <= wdata; end if;
    end if;
  end process;
end architecture;

";

/// VHDL reserves many short words that are natural net names here.
fn name(raw: &str) -> String {
    format!("n_{raw}")
}

fn bit_text(m: &Module, bit: &Bit) -> String {
    match bit {
        Bit::Zero => "'0'".into(),
        Bit::One => "'1'".into(),
        Bit::Net(id, index) => format!("{}({index})", name(&m.nets[*id].name)),
    }
}

fn bus_text(m: &Module, bits: &[Bit]) -> String {
    if bits.len() == 1 {
        return format!("(0 => {})", bit_text(m, &bits[0]));
    }
    let parts: Vec<String> = bits.iter().rev().map(|b| bit_text(m, b)).collect();
    parts.join(" & ")
}

fn vector(width: u32) -> String {
    format!("std_logic_vector({} downto 0)", width - 1)
}

/// Associates an output port with its signals, bit by bit unless it is one whole net.
fn output_association(m: &Module, formal: &str, bits: &[Bit]) -> String {
    if let Some(Bit::Net(id, 0)) = bits.first() {
        let whole = m.nets[*id].width as usize == bits.len() && bits.iter().enumerate().all(|(i, b)| *b == Bit::Net(*id, i as u32));
        if whole {
            return format!("{formal} => {}", name(&m.nets[*id].name));
        }
    }
    let parts: Vec<String> = bits.iter().enumerate().map(|(i, b)| format!("{formal}({i}) => {}", bit_text(m, b))).collect();
    parts.join(", ")
}

fn emit(out: &mut String, design: &Design, m: &Module) {
    let _ = writeln!(out, "-- {}: {}\nlibrary ieee;\nuse ieee.std_logic_1164.all;\n", m.title, m.description);
    let mut ports = vec!["clk : in std_logic".to_string()];
    for p in &m.ports {
        let dir = if p.dir == Dir::Input { "in" } else { "buffer" };
        ports.push(format!("{} : {dir} {}", name(&p.name), vector(m.nets[p.net].width)));
    }
    let _ = writeln!(out, "entity {} is\n  port (\n    {}\n  );\nend entity;\n", m.name, ports.join(";\n    "));
    let _ = writeln!(out, "architecture structural of {} is", m.name);
    for (id, net) in m.nets.iter().enumerate() {
        if m.ports.iter().all(|p| p.net != id) {
            let _ = writeln!(out, "  signal {} : {};", name(&net.name), vector(net.width));
        }
    }
    let _ = writeln!(out, "begin");
    for inst in &m.instances {
        let label = format!("u_{}", inst.name);
        let one = |port: &str| bit_text(m, &inst.conns[port][0]);
        let bus = |port: &str| bus_text(m, &inst.conns[port]);
        match &inst.kind {
            InstKind::Prim(Prim::And2) => {
                let _ = writeln!(out, "  {} <= {} and {};", one("y"), one("a"), one("b"));
            }
            InstKind::Prim(Prim::Or2) => {
                let _ = writeln!(out, "  {} <= {} or {};", one("y"), one("a"), one("b"));
            }
            InstKind::Prim(Prim::Xor2) => {
                let _ = writeln!(out, "  {} <= {} xor {};", one("y"), one("a"), one("b"));
            }
            InstKind::Prim(Prim::Not) => {
                let _ = writeln!(out, "  {} <= not {};", one("y"), one("a"));
            }
            InstKind::Prim(Prim::Mux2) => {
                let _ = writeln!(out, "  {} <= {} when {} = '1' else {};", one("y"), one("b"), one("s"), one("a"));
            }
            InstKind::Prim(Prim::Dff) => {
                let _ = writeln!(out, "  {label}: entity work.llm2chip_dff port map (clk => clk, d => {}, q => {});", one("d"), one("q"));
            }
            InstKind::Prim(Prim::Ram2p { addr_bits, width }) => {
                let _ = writeln!(
                    out,
                    "  {label}: entity work.llm2chip_ram2p generic map ({addr_bits}, {width}) port map (clk => clk, raddr => {}, waddr => {}, we => {}, wdata => {}, {});",
                    bus("raddr"), bus("waddr"), one("we"), bus("wdata"), output_association(m, "rdata", &inst.conns["rdata"])
                );
            }
            InstKind::Prim(Prim::Rom { addr_bits, width }) => {
                // Read-only: the contents are for the foundry's ROM macro that replaces it.
                let _ = writeln!(
                    out,
                    "  {label}: entity work.llm2chip_ram2p generic map ({addr_bits}, {width}) port map (clk => clk, raddr => {}, waddr => (others => '0'), we => '0', wdata => (others => '0'), {});",
                    bus("addr"), output_association(m, "rdata", &inst.conns["rdata"])
                );
            }
            InstKind::Prim(Prim::Sram { addr_bits, width }) => {
                // A single-port memory is the two-port one with its addresses tied.
                let _ = writeln!(
                    out,
                    "  {label}: entity work.llm2chip_ram2p generic map ({addr_bits}, {width}) port map (clk => clk, raddr => {}, waddr => {}, we => {}, wdata => {}, {});",
                    bus("addr"), bus("addr"), one("we"), bus("wdata"), output_association(m, "rdata", &inst.conns["rdata"])
                );
            }
            InstKind::Module(child) => {
                let module = &design.modules[child];
                let conns: Vec<String> = module
                    .ports
                    .iter()
                    .map(|p| {
                        let bits = &inst.conns[&p.name];
                        if p.dir == Dir::Input {
                            format!("{} => {}", name(&p.name), bus_text(m, bits))
                        } else {
                            output_association(m, &name(&p.name), bits)
                        }
                    })
                    .collect();
                let _ = writeln!(out, "  {label}: entity work.{child} port map (clk => clk, {});", conns.join(", "));
            }
        }
    }
    let _ = writeln!(out, "end architecture;\n");
}

/// The whole design as one VHDL file, children before their parents.
pub fn to_vhdl(design: &Design) -> String {
    let mut out = String::from("-- Generated by llm2chip.\n\n");
    out.push_str(LIBRARY);

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
    for module in order {
        emit(&mut out, design, &design.modules[module]);
    }
    out
}

/// A testbench that applies `steps` and fails unless `watch` ends as `expected`.
pub fn vhdl_testbench(design: &Design, steps: &[crate::TestStep], watch: &str, expected: u64) -> String {
    let top = &design.modules[&design.top];
    let bits_of = |value: u128, width: u32| -> String { (0..width).rev().map(|i| if value >> i & 1 == 1 { '1' } else { '0' }).collect() };
    let mut out = String::from("library ieee;\nuse ieee.std_logic_1164.all;\n\nentity testbench is\nend entity;\n\narchitecture sim of testbench is\n  signal clk : std_logic := '0';\n");
    let mut pins = vec!["clk => clk".to_string()];
    for p in &top.ports {
        let width = top.nets[p.net].width;
        let init = if p.dir == Dir::Input { " := (others => '0')" } else { "" };
        let _ = writeln!(out, "  signal {} : {}{init};", name(&p.name), vector(width));
        pins.push(format!("{} => {}", name(&p.name), name(&p.name)));
    }
    let _ = writeln!(out, "begin\n  dut: entity work.{} port map ({});\n  process begin", top.name, pins.join(", "));
    for step in steps {
        for (port, value) in &step.inputs {
            let width = top.ports.iter().find(|p| &p.name == port).map_or(1, |p| top.nets[p.net].width);
            let _ = writeln!(out, "    {} <= \"{}\";", name(port), bits_of(*value, width));
        }
        out.push_str("    wait for 1 ns; clk <= '1'; wait for 1 ns; clk <= '0'; wait for 1 ns;\n");
    }
    let width = top.ports.iter().find(|p| p.name == watch).map_or(1, |p| top.nets[p.net].width);
    let _ = writeln!(out, "    assert {} = \"{}\" report \"resultado distinto\" severity failure;", name(watch), bits_of(expected as u128, width));
    out.push_str("    std.env.finish;\n  end process;\nend architecture;\n");
    out
}
