type Params = Record<string, unknown>;

interface Base {
  key: string;
  label: string;
  help?: string;
  /** Hidden while it would have no effect on the design. */
  when?: (p: Params) => boolean;
}

export type Field = Base &
  (
    | { kind: "range"; min: number; max: number; step?: number; log?: boolean; unit?: string; integer?: boolean }
    | { kind: "select"; options: [string, string][] }
    | { kind: "preset"; sections: PresetSection[] }
    | { kind: "toggle" }
    | { kind: "text"; placeholder?: string }
  );

export interface PresetSection {
  title: string;
  presets: { label: string; values: Params }[];
}

export interface Group {
  title: string;
  open: boolean;
  fields: Field[];
}

const range = (from: number, to: number) => Array.from({ length: to - from + 1 }, (_, i) => from + i);
const fixed = (bits: number): Params => ({ weight_coding: "fixed", weight_bits: bits, use_model_mix: false, group_size: 0 });
const float = (bits: number, exponent: number, group = 0, fine = false): Params => ({
  weight_coding: "float",
  weight_bits: bits,
  weight_exponent_bits: exponent,
  use_model_mix: false,
  group_size: group,
  weight_bias: false,
  weight_fine_scale: fine,
  ...(group > 0 ? { scale_bits: 8 } : {}),
});
const affine = (bits: number, group: number, bias: boolean): Params => ({ weight_coding: "affine", weight_bits: bits, use_model_mix: false, group_size: group, weight_bias: bias });

export const WEIGHT_PRESETS: PresetSection[] = [
  {
    title: "Grouped affine (MLX)",
    presets: [
      { label: "Mix of 4 and 8 bits", values: { weight_coding: "affine", weight_bits: 4, use_model_mix: true, group_size: 32, weight_bias: true } },
      ...range(2, 8).map((bits) => ({ label: `MLX ${bits} bits`, values: affine(bits, bits <= 5 ? 32 : 64, true) })),
    ],
  },
  {
    title: "GGUF",
    presets: [
      { label: "Q2_K", values: affine(2, 16, false) },
      { label: "Q3_K", values: affine(3, 16, false) },
      { label: "Q4_0", values: affine(4, 32, false) },
      { label: "Q4_1", values: affine(4, 32, true) },
      { label: "Q5_0", values: affine(5, 32, false) },
      { label: "Q5_1", values: affine(5, 32, true) },
      { label: "Q6_K", values: affine(6, 16, false) },
      { label: "Q8_0", values: affine(8, 32, false) },
    ],
  },
  { title: "Fixed point (Q)", presets: range(2, 32).map((bits) => ({ label: `Q${bits}`, values: fixed(bits) })) },
  {
    title: "Floating point",
    presets: [
      { label: "FP8 (E4M3)", values: float(8, 4) },
      { label: "FP8 (E5M2)", values: float(8, 5) },
      { label: "FP16", values: float(16, 5) },
      { label: "BF16", values: float(16, 8) },
      { label: "FP24", values: float(24, 7) },
      { label: "FP32", values: float(32, 8) },
    ],
  },
  {
    title: "Microscaling",
    presets: [
      { label: "MXFP4 (E2M1 / 32)", values: float(4, 2, 32) },
      { label: "NVFP4 (E2M1 / 16, fine scale)", values: float(4, 2, 16, true) },
      { label: "MXFP6 (E2M3 / 32)", values: float(6, 2, 32) },
      { label: "MXFP6 (E3M2 / 32)", values: float(6, 3, 32) },
      { label: "MXFP8 (E4M3 / 32)", values: float(8, 4, 32) },
      { label: "MXFP8 (E5M2 / 32)", values: float(8, 5, 32) },
    ],
  },
  {
    title: "Posit",
    presets: [
      { label: "Posit8 es1", values: { ...float(8, 1), weight_coding: "posit" } },
      { label: "Posit8 es2", values: { ...float(8, 2), weight_coding: "posit" } },
      { label: "Posit16 es1", values: { ...float(16, 1), weight_coding: "posit" } },
      { label: "Posit16 es2", values: { ...float(16, 2), weight_coding: "posit" } },
      { label: "Posit32 es2", values: { ...float(32, 2), weight_coding: "posit" } },
    ],
  },
  {
    title: "Logarithmic",
    presets: [
      { label: "LNS 8 (4 fraction bits)", values: { ...float(8, 4), weight_coding: "log" } },
      { label: "LNS 12 (6 fraction bits)", values: { ...float(12, 6), weight_coding: "log" } },
      { label: "LNS 16 (8 fraction bits)", values: { ...float(16, 8), weight_coding: "log" } },
    ],
  },
  {
    title: "Codebook",
    presets: [2, 3, 4, 6, 8].map((bits) => ({ label: `${bits} bits (${2 ** bits} values)`, values: { ...fixed(bits), weight_coding: "codebook" } })),
  },
];

export function matchingPreset(params: Params): string | null {
  for (const section of WEIGHT_PRESETS) {
    for (const preset of section.presets) {
      if (Object.entries(preset.values).every(([key, value]) => params[key] === value)) return preset.label;
    }
  }
  return null;
}

const usesHbm = (p: Params) => p.weight_backend === "hbm4" || p.kv_backend === "hbm4";
const usesExternal = (p: Params) => p.weight_backend === "external_ddr" || p.kv_backend === "external_ddr";

export const PARAM_GROUPS: Group[] = [
  {
    title: "Goal",
    open: true,
    fields: [
      { kind: "range", key: "max_params_g", label: "Max parameters", min: 0.0001, max: 100000, log: true, unit: "G", help: "Size of the largest model the chip can hold, up to 100 trillion parameters (100,000 G). Sizes the weight memory." },
      { kind: "range", key: "context_tokens", label: "Context", min: 128, max: 1048576, log: true, integer: true, unit: "tokens", help: "Up to 1 M tokens. The KV cache grows linearly with the context and is the block that most limits speed; from about 32 K it is worth moving it to HBM4." },
      { kind: "toggle", key: "mtp_enabled", label: "MTP support", help: "Adds the multi-token prediction (MTP) head: an extra layer that guesses the next tokens while the main model checks the previous ones. With a single user the dies wait a lot, and this work fills that gap: when it guesses right, several tokens come out per pass." },
      { kind: "range", key: "mtp_depth", label: "MTP depth", min: 1, max: 3, step: 1, integer: true, help: "How many tokens the MTP head runs ahead each round. More depth gives more tokens per pass when it guesses right, but each level guesses less often (in Qwen, around 94 %, 86 % and 74 %) and needs state kept to undo.", when: (p) => p.mtp_enabled === true },
      { kind: "toggle", key: "include_vision", label: "Vision (image analysis)", help: "The chip reads images: reserves area for the image encoder (a ViT of about 400 M parameters in Qwen) that turns each image into tokens for the language model, so it can describe them or answer about them. It does not draw: that is Draws images." },
      { kind: "toggle", key: "draw_images", label: "Draws images (text to image)", help: "The chip also draws: a model (Show-o) draws and retouches pictures from a text, on a die of its own. It is not vision, which analyses images. The floorplan and the estimate count it: a reference die measured with Show-o's transformer, its embedding table in external memory and MAGVIT-v2 on the host. Off, nothing of drawing is in the chip or the app." },
      { kind: "toggle", key: "speech_in", label: "Listens (speech to text)", help: "The chip also listens: a speech recogniser (Moonshine) turns a spoken question into text for the language model, on dies of its own. The floorplan and the estimate count them: reference dies until they are planned with your models in the Listen view. Off, nothing of listening is in the chip or the app." },
      { kind: "toggle", key: "speech_out", label: "Speaks (text to speech)", help: "The chip also speaks: a speech writer (OuteTTS) and a sound decoder (DAC) say the answer aloud, on dies of their own. The floorplan and the estimate count them: reference dies until they are planned with your models in the Speak view. Off, nothing of speaking is in the chip or the app." },
      { kind: "text", key: "system_prompt", label: "System prompt", placeholder: "You are a voice assistant. Answer in one short sentence.", help: "A fixed text the chip reads before every question, such as how it should answer; a voice assistant needs one to answer briefly. It goes in the chip's description, in the image: burnt into ROM with the weights, or loaded with them at power-up, where a new image changes it. The chip reads it when a conversation starts, after power-up or the clear pin, while it waits for the first question; the turns after go on from its cache, and it takes that much of the context. Empty, there is none." },
    ],
  },
  {
    title: "Organization",
    open: true,
    fields: [
      { kind: "range", key: "target_tok_s", label: "Target speed", min: 0, max: 40000, step: 100, integer: true, unit: "tok/s", help: "With the model's size and the clock, sets how many multipliers are needed. With 0 you set them by hand." },
      { kind: "range", key: "macs_per_die", when: (p) => !Number(p.target_tok_s), label: "Multipliers per die", min: 2, max: 2000000, log: true, integer: true, help: "With a single user only one die works at a time, so this value sets the speed." },
      { kind: "range", key: "freq_ghz", label: "Frequency", min: 0.02, max: 3.4, log: true, unit: "GHz", help: "The die's clock. Its slowest path between registers limits it: the multiplier and the adder tree (more stages shorten them) and the node's gate delay. A faster clock gives more tok/s with the same logic, but more power." },
      { kind: "select", key: "die_mode", label: "Die type", options: [["custom", "Tailored to this model"], ["identical", "All alike (expandable)"]], help: "Either way it is the same chip repeated, loading its part of the model from the image. To measure sizes the die for this model's part; expandable reserves room (layers, width, vocabulary) to connect more chips for larger models. Only with the weights burnt into ROM is each die different." },
      { kind: "range", key: "max_layers", label: "Max layers", min: 1, max: 1024, log: true, integer: true, help: "The layers come from each model: Qwen3.8 has 64 and Llama 3.1 8B, 32. This limit reserves their state in each die.", when: (p) => p.die_mode === "identical" },
      { kind: "range", key: "max_hidden_size", label: "Max hidden size", min: 64, max: 65536, log: true, integer: true, help: "Largest width of the vector that runs through the layers (5,120 in Qwen3.8-27B). With identical dies, it sets the width of buffers and links for any model loaded.", when: (p) => p.die_mode === "identical" },
      { kind: "range", key: "max_vocabulary", label: "Max vocabulary", min: 256, max: 4194304, log: true, integer: true, help: "Distinct tokens the largest model takes (248,320 in Qwen). Sizes lm_head, the sampling unit and the tokenizer tables.", when: (p) => p.die_mode === "identical" },
      { kind: "range", key: "dies", label: "Dies", min: 0, max: 32, step: 1, integer: true, help: "How many dies the model is spread over. With 0 the fewest is chosen with which each die fits the maximum area. With a single user the dies work in turns, so more dies give more memory but not more speed on their own." },
      { kind: "select", key: "chip_split", label: "Chip split", options: [["layers", "By layers, in a chain"], ["tensor", "By tensor, all at once"], ["mixed", "Mixed: tensor groups in a chain"]], help: "How several chips share the model. By layers: each chip keeps whole layers and passes the hidden vector to the next; little communication, but with a single user one chip works at a time. By tensor: each chip keeps a part of every layer and all of them compute each layer at once; much faster for one user, but they add up their results twice per layer and need fast links, inside the package. Mixed: groups of chips in tensor joined in a chain, so each group fits in a package.", when: (p) => Number(p.dies) !== 1 },
      { kind: "range", key: "tensor_group", label: "Chips per group", min: 1, max: 32, step: 1, integer: true, help: "In the mixed split, how many chips share each layer. The groups are joined in a chain. With 1 it is the split by layers; with all the chips, by tensor. A group should fit in a package, joined by UCIe.", when: (p) => p.chip_split === "mixed" },
      { kind: "range", key: "die_max_mm2", label: "Max die area", min: 10, max: 858, step: 1, integer: true, unit: "mm²", help: "Maximum area of each die. The physical limit is the scanner's reticle (26 × 33 mm, about 858 mm²). Large dies yield less per wafer, so a smaller limit sometimes pays." },
    ],
  },
  {
    title: "Memory",
    open: true,
    fields: [
      { kind: "select", key: "weight_backend", label: "Weights", options: [["sram", "SRAM (swappable model)"], ["rom", "ROM (model etched in)"], ["stacked_dram", "Stacked DRAM"], ["hbm4", "HBM4"], ["external_ddr", "External memory"]], help: "With SRAM the chip takes other models. With ROM the weights are burnt in at manufacture: weight writing and its load time go away, but changing model needs new masks." },
      { kind: "select", key: "kv_backend", label: "Cache and state", options: [["sram", "SRAM"], ["hbm4", "HBM4"], ["external_ddr", "External memory"]], help: "Where the attention's KV cache and DeltaNet's recurrent state live. In SRAM they are fastest but take area that grows with the context; outside, that memory's bandwidth limits speed." },
      { kind: "range", key: "hbm_stacks", label: "HBM4 stacks", min: 1, max: 16, step: 1, integer: true, help: "Each HBM4 stack adds capacity and bandwidth, and a PHY of about 12 mm² on the die that connects it.", when: usesHbm },
      { kind: "range", key: "stacked_dram_tbps", label: "DRAM bandwidth", min: 10, max: 400, log: true, unit: "TB/s", help: "What the DRAM stacked on the die delivers per second. With the weights out of SRAM, it sets the speed: each token reads every weight once.", when: (p) => p.weight_backend === "stacked_dram" },
      { kind: "range", key: "ddr_gbps", label: "External bandwidth", min: 0.01, max: 400, log: true, unit: "GB/s", help: "What the external memory (DDR, LPDDR) delivers per second. With the weights outside, tok/s cannot exceed this value divided by the model's size.", when: usesExternal },
      { kind: "range", key: "ddr_capacity_gb", label: "External capacity", min: 0.004, max: 256, log: true, unit: "GB", help: "External memory available: it must hold what is taken off the chip (weights, KV cache or both).", when: usesExternal },
    ],
  },
  {
    title: "Arithmetic",
    open: true,
    fields: [
      { kind: "preset", key: "weight_coding_preset", label: "Weight format", sections: WEIGHT_PRESETS, help: "The circuit adapts: the memory, the multiplier and the decoder of each cell change." },
      { kind: "select", key: "weight_coding", label: "Coding", options: [["affine", "Grouped affine"], ["fixed", "Fixed point"], ["float", "Floating point"], ["posit", "Posit"], ["log", "Logarithmic"], ["codebook", "Codebook"]], help: "How each stored code is turned into a number. Each family generates its own decoder." },
      { kind: "range", key: "weight_bits", label: "Weight bits", min: 2, max: 32, step: 1, integer: true, help: "Bits stored per weight, from 2 to 32. Sets the weight memory, which dominates the area, and the size of each cell's multiplier. With fewer bits more model fits but precision is lost." },
      { kind: "range", key: "weight_exponent_bits", label: "Exponent bits", min: 1, max: 15, step: 1, integer: true, help: "In floating point, the exponent bits. In posit, “es”. In logarithmic, the fraction bits.", when: (p) => ["float", "posit", "log"].includes(String(p.weight_coding)) },
      { kind: "toggle", key: "use_model_mix", label: "Keep 8-bit tensors", help: "Keeps the embeddings and the output head at 8 bits and the rest at 4. For Qwen it reproduces the mix of its released weights, which also keeps out_proj and the MLP of the last layers at 8. The cells keep two 4-bit codes in each 8-bit word.", when: (p) => p.weight_coding === "affine" },
      { kind: "toggle", key: "weight_bias", label: "Bias per group", help: "Whether each group also keeps a bias (weight = code · scale + bias). MLX and Q4_1 do; Q4_0 only keeps a scale. The bias costs memory and an addition per group.", when: (p) => p.weight_coding === "affine" },
      { kind: "toggle", key: "weight_fine_scale", label: "Fine scale per group", help: "A floating-point scale like NVFP4 needs a multiplier. A power of two like MXFP4 only a shifter, but it loses precision.", when: (p) => p.weight_coding === "float" && Number(p.group_size) > 0 },
      { kind: "range", key: "group_size", label: "Weights per group", min: 0, max: 256, step: 8, integer: true, help: "Consecutive weights that share scale and bias (32 in MLX's Qwen). Small groups give more precision but more factors to keep and apply. With 0 there are no groups.", when: (p) => p.weight_coding === "affine" || p.weight_coding === "float" },
      { kind: "range", key: "scale_bits", label: "Scale and bias bits", min: 8, max: 32, step: 1, integer: true, help: "Precision with which each cell keeps the scale and bias of each group. More bits: less error, and more memory and scale multiplier.", when: (p) => Number(p.group_size) > 0 && (p.weight_coding === "affine" || p.weight_coding === "float") },
      { kind: "range", key: "activation_bits", label: "Activation bits", min: 8, max: 32, step: 1, integer: true, help: "Width of the values that travel through the chip between operations. Sets the width of the multipliers, the activation network and the vector memory: among what weighs most in area and precision." },
      { kind: "range", key: "accumulator_bits", label: "Accumulator bits", min: 16, max: 128, step: 1, integer: true, help: "Width of each cell's accumulator, which adds the products of weight and activation. If short, the sums overflow; if the format needs more, the chip uses the width needed." },
      { kind: "range", key: "delta_state_bits", label: "DeltaNet state bits", min: 8, max: 32, step: 1, integer: true, help: "Width of DeltaNet's recurrent state, a matrix per head updated every token. The reference uses 32 bits; reducing it saves a lot of memory, but the error builds up over the context and it is the largest numerical risk." },
      { kind: "range", key: "kv_bits", label: "KV cache bits", min: 4, max: 32, step: 1, integer: true, help: "Width of each key and value stored per token. The cache grows with the context, so with long contexts it is the memory that takes most after the weights." },
    ],
  },
  {
    title: "Pipeline stages",
    open: false,
    fields: [
      { kind: "toggle", key: "auto_circuit", label: "Custom circuit", help: "The generator tries every way of building the cell for these parameters, measures its real circuit and keeps the best." },
      { kind: "select", key: "circuit_goal", label: "Prioritize", options: [["balanced", "Area and energy"], ["area", "Area"], ["energy", "Energy"]], help: "What is minimised among the cell designs that reach the clock asked for: area, energy per token or a balance of both.", when: (p) => Boolean(p.auto_circuit) },
      { kind: "select", key: "multiplier_kind", when: (p) => !p.auto_circuit, label: "Multiplier type", options: [["wallace", "Wallace"], ["booth", "Booth radix-4"], ["array", "Array"], ["multiples", "Precomputed multiples"]], help: "Wallace and Booth reduce the products with carry-free adders and are much faster. The array is simpler and more compact at low clocks." },
      { kind: "range", key: "mac_stages", when: (p) => !p.auto_circuit, label: "Multiplier", min: 1, max: 6, step: 1, integer: true, help: "More stages allow a higher clock in exchange for area and latency." },
      { kind: "range", key: "factor_steps", label: "Scale and bias cycles", min: 1, max: 16, step: 1, integer: true, help: "Each cell applies a group's scale and bias over this many cycles, a few bits a cycle, while it already adds up the next group. With 1 each cell needs a whole multiplier; with 8 one eight times smaller is enough, with no loss of speed if the group has at least 8 weights.", when: (p) => Number(p.group_size) > 0 },
      { kind: "range", key: "adder_stages", when: (p) => !p.auto_circuit, label: "Adder tree", min: 1, max: 6, step: 1, integer: true, help: "Registers inside the tree that adds the cells' products. More stages allow a higher clock in exchange for latency and flip-flops." },
      { kind: "range", key: "noc_stages", label: "Activation network", min: 1, max: 4, step: 1, integer: true, help: "Registers in the network that sends each activation to every cell. In large dies the wires are long: more stages keep them from limiting the clock." },
    ],
  },
  {
    title: "Vector units",
    open: false,
    fields: [
      { kind: "range", key: "vector_lanes_per_die", label: "Lanes per die", min: 1, max: 512000, log: true, integer: true, help: "Lanes of the vector units, which multiply activation by activation: attention Q·K, delta rule, gates and norms. Few lanes lengthen the parts that are not matrix products." },
      { kind: "range", key: "vector_units_per_die", label: "Vector units per die", min: 1, max: 256, log: true, integer: true, help: "Copies of the vector unit: each computes one attention or DeltaNet head at a time." },
      { kind: "range", key: "sampler_lanes", label: "Logits per cycle when sampling", min: 1, max: 4096, log: true, integer: true, help: "How many logits the sampling unit compares at once. With few, picking the token holds the chip back." },
      { kind: "range", key: "sampler_entries", label: "Sampling candidates", min: 1, max: 1024, log: true, integer: true, help: "How many of the best tokens the sampling unit keeps: the largest top-k it takes. Each candidate costs registers and cycles when picking." },
      { kind: "range", key: "vector_bits", label: "Bits per lane", min: 8, max: 32, step: 1, integer: true, help: "Width of the vector units' multipliers, which work activation by activation (attention, delta rule, norms). Their area grows with its square." },
      { kind: "range", key: "function_units_per_die", label: "Function units", min: 1, max: 128, step: 1, integer: true, help: "Tables that compute non-linear functions (exp, SiLU, sigmoid, inverse square root…) by looking up and interpolating. More units apply the function to more elements at once." },
      { kind: "range", key: "function_table_bits", label: "Table resolution", min: 4, max: 20, step: 1, integer: true, help: "Entries of each function table, in powers of two: 12 is 4,096. More entries reduce the error of exp, SiLU or the inverse square root in exchange for memory in each unit." },
    ],
  },
  {
    title: "System",
    open: false,
    fields: [
      { kind: "select", key: "link", label: "Die-to-die link", options: [["auto", "Automatic by package"], ["ucie", "UCIe"], ["serdes", "Board SerDes"], ["pins", "I/O pins"]], help: "UCIe joins dies inside one package: fast (about 10 ns per hop) and small. SerDes joins them across the board: slow (about 150 ns) and larger. I/O pins are all that chips from an open shuttle have, each in its own package: 16 pins each way at 50 MHz and about 60 ns per hop. Automatic fills packages and joins them with SerDes, except on an open shuttle, which uses the pins." },
      { kind: "range", key: "link_words", label: "Link width", min: 1, max: 4096, log: true, integer: true, unit: "activations/cycle", help: "How many activations each link passes per cycle. The hidden vector crosses every boundary between dies on every token: a narrow link adds thousands of cycles per boundary; a wide one needs more UCIe or SerDes modules." },
      { kind: "toggle", key: "host_uart", label: "Serial port (UART)", help: "Adds a serial port to the host port, beside its byte port: a byte in ten bits, and a clear-to-send line that tells the host to wait while the receive queue fills or another link is inside a message. It carries the same messages, and the chip answers on the link the last message came from. A board's USB serial port is the usual one, with hardware flow control." },
      { kind: "range", key: "uart_baud", label: "Serial port rate", min: 9600, max: 12000000, log: true, integer: true, unit: "baud", help: "Bits a second on the serial port: a byte takes ten. Text needs little; speech needs about 640,000 baud to listen and 1,920,000 to speak.", when: (p) => p.host_uart === true },
      { kind: "toggle", key: "host_qspi", label: "Quad SPI", help: "Adds a quad SPI target to the host port: four data lines, a clock, a select and an interrupt line that says the chip has something to send. The host writes, reads and asks how much room and how much waiting there is. Its clock may be up to an eighth of the chip's, so it carries a byte every 16 cycles: some 60 MB/s at 1 GHz." },
      { kind: "toggle", key: "host_i2c", label: "I2C", help: "Adds an I2C target to the host port: two open-drain lines shared with other chips, at its own address, and an interrupt line that says the chip has something to send. A write's bytes go in, each refused if the port cannot take it, so the host sends it again; a read gives first the room in the receive queue and what waits to be sent. It is slow (a byte in nine clocks, 1 MHz at most) but every microcontroller has it." },
      { kind: "range", key: "i2c_khz", label: "I2C clock", min: 100, max: 1000, log: true, integer: true, unit: "kHz", help: "The I2C bus's clock: 100 kHz standard, 400 fast, 1,000 Fast-mode Plus. A byte takes nine clocks, so 1 MHz carries 111 KB/s: enough for text and to listen, not to speak. The chip's clock must be at least twenty times faster.", when: (p) => p.host_i2c === true },
      { kind: "select", key: "load_interface", label: "Weight loading", options: [["nvme", "NVMe"], ["pcie5", "PCIe 5.0 x16"], ["uart", "UART"]], help: "Where the weights arrive from when the chip powers up: NVMe, PCIe 5.0 or UART. It only affects start-up time, not generation speed.", when: (p) => p.weight_backend !== "rom" },
    ],
  },
];

export const FIELDS: Field[] = PARAM_GROUPS.flatMap((g) => g.fields);
export const fieldLabel = (key: string) => FIELDS.find((f) => f.key === key)?.label ?? key;
