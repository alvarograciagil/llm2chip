/** Parameters that are not sliders of the designer: what the gate generator derives on its
 * own, what the sizing search takes, and what each generation is asked for. */
export interface DocEntry {
  name: string;
  text: string;
}

export interface DocSection {
  id: string;
  title: string;
  intro: string;
  entries: DocEntry[];
}

export const EXTRA_DOCS: DocSection[] = [
  {
    id: "generator",
    title: "Gate generator",
    intro:
      "The circuit that is simulated gate by gate and exported to Verilog comes from these parameters. Most are not chosen by hand: the generator derives them from the model and the microprogram so the die has just what it needs.",
    entries: [
      { name: "Cells per die", text: "Each cell has its weight memory and its multiplier. More cells split each matrix-vector product among more multipliers: fewer cycles per token and more area. There must be at least as many cells per unit as lanes." },
      { name: "Lanes per unit", text: "Elements a vector unit processes per cycle in element-wise operations (sums, norms, functions). In powers of two." },
      { name: "Vector units", text: "Copies of the vector unit that run the same instructions on different data: each takes an attention or DeltaNet head, or a piece of a vector. They are joined by a tree network that adds and finds maxima." },
      { name: "Depth of each cell's memory", text: "Weight words and scale-and-bias groups each cell keeps. It comes from spreading all the chip's matrices over its cells." },
      { name: "Codes per word", text: "The cell's word is as wide as the model's widest code (8 bits in MLX's Qwen). Each matrix keeps 1, 2 or 4 codes per word according to its own width (q8, q4, q2), and the microcode tells the cell how many when it points at it. It is turned off with ECC, which protects whole words." },
      { name: "Scale and bias cycles (K)", text: "Each cell applies a group's scale and bias in K cycles, a few bits per cycle, while it adds up the next group. It shrinks the scale multiplier to 1/K of its size; it only costs cycles if the group has fewer than K weights." },
      { name: "Multipliers", text: "The cells use radix-4 Booth (fewer partial products) and the lanes Wallace (a tree without carry propagation). There are also array (compact, slow) and precomputed multiples. Adders of 8 bits or more are Brent-Kung prefix adders." },
      { name: "Vector registers and index", text: "The microprogram names each piece of data by register and index. A table, loaded with the image, gives each register its base in the vector memory and the step of the loops that go through registers by number; the compiler works out how far each register reaches in its routines, so each one takes only what it uses. Without the table, each register would reserve as much as the longest." },
      { name: "Microcode", text: "The die's fixed program, with one entry point per routine (token, logits, MTP head…). Each field of the control word has only the bits the program uses, and each operation that does not appear is not generated." },
      { name: "Key columns", text: "With several units, the KV cache's keys are kept in columns of 2^n tokens so each unit scores its share of the tokens at once." },
      { name: "Function tables", text: "Index bits of each table (resolution) and how many tables there are (exp, SiLU, sigmoid, softplus, inverse square root, tanh…)." },
      { name: "Weights in ROM", text: "With the model burnt in at the factory, weight writing and loading disappear, and the memory is denser, but changing the model takes other masks." },
      { name: "ECC in cells", text: "Corrects one bit and detects two per weight word. It costs about 430 gate equivalents per cell, plus check bits." },
      { name: "Skip zeros", text: "Freezes the multiplier's input when the weight is zero so it does not switch: saves energy, not area." },
      { name: "Control CPU", text: "A RISC-V RV32IM with its program and data memory: boots the chip, loads the image with CRC, tokenizes, drives the die through its registers, samples, and applies the stop list and the reasoning budget." },
      { name: "Sampling unit", text: "Logits it compares per cycle and candidates it keeps (the largest top-k). It picks the token with temperature using a table of exponentials." },
      { name: "Chip-to-chip link", text: "Each word travels with a sequence number and an error-correcting code; the receiver has a queue and asks for retries, so the board can delay words without breaking anything." },
      { name: "Wide link", text: "With several chips, the link has a lane of its own for each vector lane, each with its sequence, its error correction and its queue, like UCIe lanes: a whole block crosses in one cycle. The hidden vector goes from one chip to the next in tens of cycles instead of thousands. It is only generated if the microprogram sends blocks." },
    ],
  },
  {
    id: "split",
    title: "Chip split",
    intro:
      "When the model does not fit in one chip, or more speed is wanted, there are three ways to split it. It is chosen with \u201cChip split\u201d and, in the mixed split, \u201cChips per group\u201d.",
    entries: [
      { name: "By layers, in a chain", text: "Each chip keeps a stretch of consecutive layers. For each token, the first does its layers and passes the hidden vector (a few thousand values) to the second, and so on to the last, which returns it to the first for the logits. Advantages: little communication, which tolerates slow links across the board; no synchronization within a layer; each chip keeps only its layers' weights and cache; reading a prompt, they all work at once, each on a token. Drawback: generating for a single user only one chip works at a time, so more chips give more memory, not more speed. The gate generator implements it and tests it in gates with two chips over a noisy link." },
      { name: "By tensor, all at once", text: "Each chip keeps part of every matrix and of every group of heads, and they all compute each layer at once. After the attention and after the MLP of each layer they add up their partial results in a ring: 2·(N−1) steps, each with an N-th of the vector. Advantage: with one user the speed grows almost with the number of chips. Drawbacks: many synchronizations per token, which need fast, low-latency links, in practice inside the same package; a prompt is not read in a chain. The estimator models the most efficient variant, which also splits the attention heads. The gate generator implements a simpler variant today, \u201cby rows\u201d: each chip keeps an equal band of rows of each matrix and, after each product, the chips pass their bands round a ring until all have the whole vector. Before narrowing each product they agree on the shift over the link (two words per chip), so the result is bit for bit that of a single chip; the test in gates checks it with two chips and noisy links. Attention, DeltaNet and the norms are computed whole by each chip: the weights are split, but the KV cache and the state are repeated in all of them, and more moves over the link than in the efficient variant." },
      { name: "Mixed", text: "Groups of chips that share each layer by tensor, joined in a chain. It is what large systems use: each group inside a package, with UCIe, and the groups across the board. With a group of 1 chip it is the split by layers; with all in one group, the split by tensor. In gates, the last chip of each group closes the group's ring with a link back to the first, and the hidden vector passes to the next group over the usual link. A boot image carries a plan for each number of chips: those that do not form whole groups are split by layers." },
    ],
  },
  {
    id: "sizing",
    title: "Sizing",
    intro:
      "The search tries die shapes (cells, lanes, units, number of chips), generates the real microprogram for each one and counts its cycles at the last position of the context, so the tok/s come from the circuit and are guaranteed with the context full.",
    entries: [
      { name: "Target tok/s", text: "The minimum speed with the context full. The search keeps the die with the least silicon that reaches it." },
      { name: "Time to first token", text: "The maximum after a prompt of a given length. With several chips, the prompt advances in a chain: each chip works on a different token at once." },
      { name: "Context", text: "Tokens that fit at once. It sets the KV cache, the RoPE table and the position where the tok/s are measured." },
      { name: "Chips", text: "How many identical chips the layers are split over. They are cut so the chip that needs the most memory has as little as possible: the first one also keeps lm_head, so it carries fewer layers." },
      { name: "Maximum clock", text: "Each die runs at its highest frequency according to its measured critical path, or at this one if it is lower." },
      { name: "Process", text: "Delay of a full adder, area of a NAND2 gate with its wiring, area of an SRAM bit and its periphery, and maximum die size (reticle: 858 mm² in N6). A larger die cannot be made and is discarded." },
      { name: "Most efficient and balance", text: "From the speed-area frontier, the point with the least area per tok/s is proposed, and the last one where each extra tok/s costs no more than a given multiple of the mean area per tok/s." },
    ],
  },
  {
    id: "generation",
    title: "Generation",
    intro: "Chosen with each request; they do not change the hardware.",
    entries: [
      { name: "Max tokens", text: "How many tokens it generates at most; it also stops at any token on the model's stop list." },
      { name: "Temperature", text: "0 always picks the most likely. Higher spreads the probability over more candidates." },
      { name: "Top-k", text: "Among how many of the best candidates it draws; it cannot exceed the sampling unit's candidates." },
      { name: "Think before answering", text: "With the model's chat template, leaves its reasoning block open; without it, the block arrives closed and it answers directly. It is prompt text." },
      { name: "Reasoning budget", text: "How many tokens it may reason before the firmware closes the reasoning for it. The opening and closing tokens come from each model's tokenizer." },
      { name: "MTP depth", text: "How many tokens the multi-token prediction head runs ahead; how many it gets right is measured." },
      { name: "Fidelity", text: "How much of the chip is simulated: real firmware instruction by instruction, sampling unit, tokenizer in the firmware and image loading by transactions." },
    ],
  },
];
