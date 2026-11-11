/** The numbers that fix a model's shape, from a Hugging Face or a llama2.c configuration. */
function shape(config: string): unknown[] | null {
  let value: Record<string, unknown>;
  try {
    value = JSON.parse(config);
  } catch {
    return null;
  }
  const text = (value.text_config as Record<string, unknown> | undefined) ?? value;
  const pick = (...keys: string[]) => keys.map((k) => text[k]).find((v) => v !== undefined) ?? null;
  const numbers = [
    pick("hidden_size", "dim"),
    pick("num_hidden_layers", "n_layers"),
    pick("num_attention_heads", "n_heads"),
    pick("vocab_size"),
    pick("intermediate_size", "moe_intermediate_size", "hidden_dim"),
    pick("num_experts", "n_routed_experts"),
  ];
  return numbers[0] === null || numbers[1] === null ? null : numbers;
}

/** Whether two configurations describe the same model, whatever format its weights are in. */
export function sameModel(a: string, b: string): boolean {
  const [x, y] = [shape(a), shape(b)];
  // What one of them leaves out, such as llama2.c's MLP width, does not tell them apart.
  return x !== null && y !== null && x.every((v, i) => v === null || y[i] === null || v === y[i]);
}
