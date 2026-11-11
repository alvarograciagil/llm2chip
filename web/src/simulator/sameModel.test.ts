import { describe, expect, it } from "vitest";
import { sameModel } from "./sameModel";

describe("los pesos del modelo del chip", () => {
  it("reconoce el mismo modelo aunque el config venga en otro formato", () => {
    const llama2c = JSON.stringify({ dim: 64, n_layers: 5, n_heads: 8, vocab_size: 512, hidden_dim: 172 });
    const hf = JSON.stringify({ model_type: "llama", hidden_size: 64, num_hidden_layers: 5, num_attention_heads: 8, vocab_size: 512, intermediate_size: 172 });
    expect(sameModel(llama2c, hf)).toBe(true);
    // The catalogue's llama2.c configs do not give the MLP width.
    const catalogue = JSON.stringify({ dim: 64, n_layers: 5, n_heads: 8, vocab_size: 512, multiple_of: 4 });
    expect(sameModel(catalogue, hf)).toBe(true);
  });

  it("distingue modelos de otra forma", () => {
    const a = JSON.stringify({ hidden_size: 64, num_hidden_layers: 5, num_attention_heads: 8, vocab_size: 512 });
    const b = JSON.stringify({ hidden_size: 64, num_hidden_layers: 6, num_attention_heads: 8, vocab_size: 512 });
    expect(sameModel(a, b)).toBe(false);
    expect(sameModel("no es json", a)).toBe(false);
  });
});
