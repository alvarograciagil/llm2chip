// The compiler's kinds of operation (`Matvec`, `AttnSum`…) as a person would name them.

const OPERATION_NAMES: Record<string, string> = {
  AttnSum: "Attention",
  DeltaStep: "DeltaNet",
  Dconv: "Convolution",
  Matvec: "Matrices",
  RmsGated: "Gated normalization",
  RmsNorm: "Normalization",
  LayerNorm: "Layer normalization",
  GroupNorm: "Group normalization",
  Rope: "RoPE",
  Silu: "SiLU",
  Gelu: "GELU",
  Sigmoid: "Sigmoid",
  SoftCap: "Soft cap",
  Moe: "Experts",
  SigmoidMoe: "Experts",
  Mla: "Latent attention",
  EmbRead: "Embedding read",
  KvAppend: "KV cache writes",
  Add: "Additions",
  AddWeight: "Biases",
  MulElem: "Element-wise products",
  Scale: "Scaling",
  SplitQueryGate: "Query and gate split",
  Copy: "Copies",
  Concat: "Concatenations",
  Slice: "Slices",
  LoadRow: "Row loads",
  LoadRows: "Row loads",
  StoreRow: "Row stores",
};

/** The name of a kind of operation, in English, for `t()`; a kind not listed keeps its own. */
export function operationName(kind: string): string {
  return OPERATION_NAMES[kind] ?? kind;
}
