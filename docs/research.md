# Research: chips for LLMs

Reference notes for calibrating the estimator and deciding what is worth building. Manufacturers' figures are their own claims unless stated otherwise.

## Reference chips

| Chip | Memory | Bandwidth | Published speed | Notes |
|---|---|---|---|---|
| Taalas HC1 | Weights in mask ROM, SRAM for KV | Internal | 15,000 to 17,000 tok/s (Llama 3.1 8B) | N6, 815 mm², 53 billion transistors, no HBM. EE Times measured more than 15,000 tok/s |
| Cerebras WSE-3 | 44 GB SRAM | 21 PB/s | About 2,100 to 2,500 tok/s (Llama 3.3 70B) | Whole wafer |
| Groq 3 LPU | 500 MB SRAM | 150 TB/s | No comparable figure | Deterministic dataflow |
| Etched Sohu | 144 GB HBM3E | About 1.8 times an H100 | 62,500 tok/s per chip, announced | Pre-production, no public access |

The application's calibration preset gives 787 mm² and 15,234 tok/s for the HC1, within the range EE Times measured.

## What we learned and what it changes

**The per-group scale matters a lot.** MXFP4 and NVFP4 use the same 4-bit E2M1 elements; the difference is the scale: a power of two every 32 weights in MXFP4, an FP8 every 16 in NVFP4. An ICLR 2026 paper attributes a strong loss of accuracy to power-of-two scales. In hardware it is a shifter against a multiplier, so the application tells the two apart and models each.

**SRAM density.** TSMC's high-density cell has stalled since N5: 0.021 µm² in N5, 0.0199 µm² in N3, about 0.0175 µm² in N2, some 38 Mb/mm². Those figures are per bit, not for a whole macro, which loses room to decoders and sense amplifiers. Moving to a newer node barely makes weight memory, the dominant block, any cheaper.

**Multipliers without a multiplier.** Approximating weights with CSD saved 77 % of logic area in a VGG-16, at a cost of 5.6 points of accuracy. HaShiFlex reduces weights to powers of two in a 7 nm flow. Neither work tries this on a transformer: it is an experimental option, best measured with real weights.

**KV cache compression.** Cassandra compresses KV and weights with mantissa truncation and exponent compression. CXL-SpecKV gets 4 to 8 times more capacity with compression and prefetching on an FPGA. The KV cache limits long contexts, so this is a promising line.

**Speculation in hardware.** HADES builds speculative decoding into the accelerator. SwiftSpec separates drafting and verification asynchronously. It fits Qwen's MTP head and the use of idle dies.

## Still to verify

- Measure MXFP4 and NVFP4 from full floating-point weights, not from a model already quantized to 4 bits.
- Measure KV cache truncation with the real Qwen before relying on it.
- A real place-and-route flow on ASAP7 with OpenROAD to replace area estimates with measurements.

## Sources

- [Taalas HC1, product](https://taalas.com/products/)
- [EE Times on Taalas](https://www.eetimes.com/taalas-specializes-to-extremes-for-extraordinary-token-speed/)
- [CNX Software on the HC1](https://www.cnx-software.com/2026/02/22/taalas-hc1-hardwired-llama-3-1-8b-ai-accelerator-delivers-up-to-17000-tokens-s/)
- [Cerebras against the H100](https://www.spheron.network/blog/cerebras-vs-nvidia-h100-inference-2026/)
- [Groq 3 LPU](https://www.spheron.network/blog/nvidia-groq-3-lpu-explained/)
- [Etched Sohu](https://www.spheron.network/blog/etched-ai-sohu-vs-nvidia-transformer-asic-inference/)
- [MXFP4 and NVFP4, ICLR 2026](https://arxiv.org/pdf/2509.23202)
- [NVFP4 and MXFP4 compared](https://www.spheron.network/blog/nvfp4-vs-mxfp4-gpu-cloud-4bit-quantization-guide/)
- [SRAM stalling in N3](https://www.tomshardware.com/news/no-sram-scaling-implies-on-more-expensive-cpus-and-gpus)
- [SRAM improvement in N2](https://www.tomshardware.com/tech-industry/sram-scaling-isnt-dead-after-all-tsmcs-2nm-process-tech-claims-major-improvements)
- [CAxCNN](https://doaj.org/article/4f85a81037c040e5bb332cb4fa528ae3)
- [HaShiFlex](https://arxiv.org/pdf/2512.12847v1)
- [Cassandra](https://arxiv.org/pdf/2605.26558)
- [CXL-SpecKV](https://arxiv.org/abs/2512.11920v1)
- [HADES](https://arxiv.org/abs/2412.19925v1)
- [SwiftSpec](https://arxiv.org/html/2506.11309v1)
