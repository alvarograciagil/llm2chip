//! A ready-to-use project around a design: RTL, testbench, constraints and tool scripts.

use crate::{to_verilog, to_vhdl, verilog_testbench, Design, TestStep};
use std::collections::BTreeMap;

pub struct ProjectOptions<'a> {
    pub name: &'a str,
    pub description: &'a str,
    pub clock_ns: f64,
    pub watch: &'a str,
    pub report: String,
}

fn sdc(clock_ns: f64) -> String {
    format!("# Reloj objetivo del diseño.\ncreate_clock -name clk -period {clock_ns:.3} [get_ports clk]\nset_input_delay -clock clk 0.1 [remove_from_collection [all_inputs] [get_ports clk]]\nset_output_delay -clock clk 0.1 [all_outputs]\n")
}

fn yosys_asic(top: &str) -> String {
    format!("# Síntesis genérica: yosys -s synth/asic.ys\nread_verilog rtl/{top}.v\nsynth -top {top} -flatten -noabc\nabc -g AND,OR,XOR,MUX\nopt_clean\nstat\nwrite_verilog -noattr build/{top}_gates.v\n")
}

fn yosys_fpga(top: &str) -> String {
    format!("# Síntesis para FPGA UltraScale+: yosys -s synth/fpga.ys\nread_verilog rtl/{top}.v\nsynth_xilinx -top {top} -family xcup -flatten\nstat\nwrite_json build/{top}_xilinx.json\n")
}

fn vivado(top: &str, clock_ns: f64) -> String {
    format!(
        "# vivado -mode batch -source synth/vivado.tcl\nread_verilog rtl/{top}.v\nread_xdc constr/clock.xdc\nsynth_design -top {top} -part xcu250-figd2104-2L-e\nreport_utilization -file build/utilization.rpt\nreport_timing_summary -file build/timing.rpt\n# Reloj objetivo: {clock_ns:.3} ns\n"
    )
}

fn xdc(clock_ns: f64) -> String {
    format!("create_clock -name clk -period {clock_ns:.3} [get_ports clk]\n")
}

fn makefile(top: &str) -> String {
    format!(
        "# Requiere Icarus Verilog para `sim` y Yosys para `synth` y `fpga`.\n.PHONY: sim synth fpga\n\nbuild:\n\tmkdir -p build\n\nsim: build\n\tiverilog -g2005 -s testbench -o build/tb.vvp rtl/{top}.v tb/testbench.v\n\tvvp build/tb.vvp\n\nsynth: build\n\tyosys -s synth/asic.ys\n\nfpga: build\n\tyosys -s synth/fpga.ys\n"
    )
}

fn readme(options: &ProjectOptions<'_>, top: &str) -> String {
    format!(
        "# {name}\n\n{description}\n\nGenerado por llm2chip.\n\n## Archivos\n\n- `rtl/{top}.v` y `rtl/{top}.vhd`: el circuito completo en puertas, en Verilog-2001 y VHDL-2008.\n- `tb/testbench.v`: carga pesos, ejecuta y muestra `{watch}` en cada ciclo.\n- `constr/`: reloj objetivo de {clock:.3} ns, en SDC y XDC.\n- `synth/`: guiones para Yosys y Vivado.\n- `report.json`: parámetros, puertas y latencia.\n\n## Uso\n\n```bash\nmake sim\nmake synth\nmake fpga\n```\n\nLas memorias se describen como matrices de registros. Para una tecnología concreta, sustituye `llm2chip_sram` y `llm2chip_ram2p` por las macros de memoria de esa fundición.\n",
        name = options.name,
        description = options.description,
        watch = options.watch,
        clock = options.clock_ns,
    )
}

/// Every file of the project, keyed by its path inside the bundle.
pub fn project_files(design: &Design, steps: &[TestStep], options: &ProjectOptions<'_>) -> BTreeMap<String, String> {
    let top = design.top.clone();
    let mut files = BTreeMap::new();
    files.insert(format!("rtl/{top}.v"), to_verilog(design));
    files.insert(format!("rtl/{top}.vhd"), to_vhdl(design));
    files.insert("tb/testbench.v".to_string(), verilog_testbench(design, steps, options.watch));
    files.insert("constr/clock.sdc".to_string(), sdc(options.clock_ns));
    files.insert("constr/clock.xdc".to_string(), xdc(options.clock_ns));
    files.insert("synth/asic.ys".to_string(), yosys_asic(&top));
    files.insert("synth/fpga.ys".to_string(), yosys_fpga(&top));
    files.insert("synth/vivado.tcl".to_string(), vivado(&top, options.clock_ns));
    files.insert("Makefile".to_string(), makefile(&top));
    files.insert("README.md".to_string(), readme(options, &top));
    files.insert("report.json".to_string(), options.report.clone());
    files
}
