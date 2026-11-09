//! The RV32IM core in gates must match the instruction-set simulator cycle for cycle.

use netlist::rv32::{core_design, Retired, Rv32Bus, Rv32Iss};
use netlist::Simulator;

struct Memory {
    program: Vec<u32>,
    data: Vec<u32>,
}

const DATA_BASE: u32 = 0x1000_0000;

impl Memory {
    fn slot(&self, address: u32) -> usize {
        ((address - DATA_BASE) / 4) as usize
    }
}

impl Rv32Bus for Memory {
    fn fetch(&mut self, address: u32) -> u32 {
        self.program.get((address / 4) as usize).copied().unwrap_or(0x0010_0073)
    }

    fn read(&mut self, address: u32) -> u32 {
        self.data[self.slot(address)]
    }

    fn write(&mut self, address: u32, data: u32, strobe: u8) {
        let slot = self.slot(address);
        for byte in 0..4 {
            if strobe >> byte & 1 == 1 {
                self.data[slot] = (self.data[slot] & !(0xFF << (8 * byte))) | (data & (0xFF << (8 * byte)));
            }
        }
    }
}

struct Noise(u64);

impl Noise {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }

    fn below(&mut self, limit: u32) -> u32 {
        self.next() % limit
    }
}

fn r_type(funct7: u32, rs2: u32, rs1: u32, funct3: u32, rd: u32, opcode: u32) -> u32 {
    funct7 << 25 | rs2 << 20 | rs1 << 15 | funct3 << 12 | rd << 7 | opcode
}

fn i_type(imm: i32, rs1: u32, funct3: u32, rd: u32, opcode: u32) -> u32 {
    ((imm as u32) & 0xFFF) << 20 | rs1 << 15 | funct3 << 12 | rd << 7 | opcode
}

fn s_type(imm: i32, rs2: u32, rs1: u32, funct3: u32) -> u32 {
    let imm = imm as u32 & 0xFFF;
    (imm >> 5) << 25 | rs2 << 20 | rs1 << 15 | funct3 << 12 | (imm & 31) << 7 | 0b0100011
}

fn b_type(imm: i32, rs2: u32, rs1: u32, funct3: u32) -> u32 {
    let imm = imm as u32;
    (imm >> 12 & 1) << 31 | (imm >> 5 & 63) << 25 | rs2 << 20 | rs1 << 15 | funct3 << 12 | (imm >> 1 & 15) << 8 | (imm >> 11 & 1) << 7 | 0b1100011
}

fn j_type(imm: i32, rd: u32) -> u32 {
    let imm = imm as u32;
    (imm >> 20 & 1) << 31 | (imm >> 1 & 0x3FF) << 21 | (imm >> 11 & 1) << 20 | (imm >> 12 & 0xFF) << 12 | rd << 7 | 0b1101111
}

const BASE_REGISTER: u32 = 31;
const EBREAK: u32 = 0x0010_0073;

fn random_program(noise: &mut Noise, length: usize) -> Vec<u32> {
    let mut code = vec![0x1000_0000 | BASE_REGISTER << 7 | 0b0110111];
    for rd in 1..31 {
        code.push(noise.next() & 0xFFFF_F000 | rd << 7 | 0b0110111);
        code.push(i_type(noise.below(4096) as i32 - 2048, rd, 0, rd, 0b0010011));
    }
    for _ in 0..length {
        let (rd, rs1, rs2) = (noise.below(31), noise.below(31), noise.below(31));
        let imm = noise.below(4096) as i32 - 2048;
        let word_offset = (noise.below(32) * 4) as i32;
        let instruction = match noise.below(12) {
            0 => r_type(if noise.below(2) == 0 { 0 } else { 32 }, rs2, rs1, 0, rd, 0b0110011),
            1 => r_type(0, rs2, rs1, [1, 2, 3, 4, 6, 7][noise.below(6) as usize], rd, 0b0110011),
            2 => r_type(if noise.below(2) == 0 { 0 } else { 32 }, rs2, rs1, 5, rd, 0b0110011),
            3 => i_type(imm, rs1, [0, 2, 3, 4, 6, 7][noise.below(6) as usize], rd, 0b0010011),
            4 => r_type(0, noise.below(32), rs1, 1, rd, 0b0010011),
            5 => r_type(if noise.below(2) == 0 { 0 } else { 32 }, noise.below(32), rs1, 5, rd, 0b0010011),
            6 => r_type(1, rs2, rs1, noise.below(8), rd, 0b0110011),
            7 => {
                let (funct3, step) = [(0, 1), (1, 2), (2, 4), (4, 1), (5, 2)][noise.below(5) as usize];
                i_type(word_offset + (noise.below(4 / step) * step) as i32, BASE_REGISTER, funct3, rd, 0b0000011)
            }
            8 => {
                let (funct3, step) = [(0, 1), (1, 2), (2, 4)][noise.below(3) as usize];
                s_type(word_offset + (noise.below(4 / step) * step) as i32, rs2, BASE_REGISTER, funct3)
            }
            9 => b_type(8, rs2, rs1, [0, 1, 4, 5, 6, 7][noise.below(6) as usize]),
            10 => j_type(8, rd),
            _ => noise.next() & 0xFFFF_F000 | rd << 7 | if noise.below(2) == 0 { 0b0110111 } else { 0b0010111 },
        };
        code.push(instruction);
    }
    code.push(EBREAK);
    code
}

fn iss_trace(program: &[u32], data: &[u32]) -> (Vec<Retired>, Vec<u32>, u64) {
    let mut bus = Memory { program: program.to_vec(), data: data.to_vec() };
    let mut cpu = Rv32Iss::default();
    let mut trace = Vec::new();
    while !cpu.halted {
        trace.extend(cpu.step(&mut bus));
    }
    (trace, bus.data, cpu.cycle)
}

fn gate_trace(program: &[u32], data: &[u32], limit: u64) -> (Vec<Retired>, Vec<u32>, u64) {
    let mut sim = Simulator::new(&core_design()).unwrap();
    let mut bus = Memory { program: program.to_vec(), data: data.to_vec() };
    sim.set("reset", 1).unwrap();
    sim.tick();
    sim.set("reset", 0).unwrap();
    let (mut instruction, mut loaded) = (bus.fetch(0), 0);
    let mut trace = Vec::new();
    for cycle in 0..limit {
        sim.set("imem_rdata", instruction as u64).unwrap();
        sim.set("dmem_rdata", loaded as u64).unwrap();
        sim.settle();
        if sim.get("halted").unwrap() == 1 {
            return (trace, bus.data, cycle + 1);
        }
        if sim.get("rf_we").unwrap() == 1 {
            trace.push(Retired { cycle, rd: sim.get("rf_addr").unwrap() as u32, value: sim.get("rf_wdata").unwrap() as u32 });
        }
        let pc = sim.get("imem_addr").unwrap() as u32;
        let address = sim.get("dmem_addr").unwrap() as u32;
        let strobe = sim.get("dmem_we").unwrap() as u8;
        if sim.get("dmem_re").unwrap() == 1 {
            loaded = bus.read(address & !3);
        }
        if strobe != 0 {
            bus.write(address & !3, sim.get("dmem_wdata").unwrap() as u32, strobe);
        }
        instruction = bus.fetch(pc);
        sim.tick();
    }
    panic!("la CPU no se detuvo en {limit} ciclos");
}

#[test]
fn gates_match_the_instruction_simulator_on_random_programs() {
    let mut noise = Noise(2026);
    for round in 0..6 {
        let program = random_program(&mut noise, 150);
        let data: Vec<u32> = (0..64).map(|_| noise.next()).collect();
        let (expected, expected_data, cycles) = iss_trace(&program, &data);
        let (got, got_data, got_cycles) = gate_trace(&program, &data, cycles + 16);
        for (index, (a, b)) in expected.iter().zip(&got).enumerate() {
            assert_eq!(a, b, "ronda {round}, escritura {index}");
        }
        assert_eq!(expected.len(), got.len(), "ronda {round}");
        assert_eq!(expected_data, got_data, "ronda {round}");
        assert_eq!(cycles, got_cycles, "ronda {round}");
    }
}

#[test]
fn division_follows_the_riscv_corner_cases() {
    let cases = [(7, 0, 4), (7, 0, 5), (7, 0, 6), (7, 0, 7), (0x8000_0000, u32::MAX, 4), (0x8000_0000, u32::MAX, 6), ((-7i32) as u32, 2, 4), ((-7i32) as u32, 2, 6), (u32::MAX, u32::MAX, 1), (u32::MAX, 2, 2), (0x8000_0000, 0x8000_0000, 3)];
    for (a, b, funct3) in cases {
        let load = |rd: u32, value: u32| {
            let low = (value as i32) << 20 >> 20;
            let high = value.wrapping_sub(low as u32) & 0xFFFF_F000;
            [high | rd << 7 | 0b0110111, i_type(low, rd, 0, rd, 0b0010011)]
        };
        let mut program = load(1, a).to_vec();
        program.extend(load(2, b));
        program.push(r_type(1, 2, 1, funct3, 3, 0b0110011));
        program.push(EBREAK);
        let (expected, _, cycles) = iss_trace(&program, &[]);
        let (got, _, got_cycles) = gate_trace(&program, &[], cycles + 16);
        assert_eq!(expected, got, "{a:#x} op{funct3} {b:#x}");
        assert_eq!(cycles, got_cycles);
    }
}
