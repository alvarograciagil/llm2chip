---
title: llm2chip
emoji: 🔌
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: other
license_name: polyform-noncommercial-1.0.0
license_link: https://polyformproject.org/licenses/noncommercial/1.0.0/
short_description: A language model turned into a chip, simulated in the browser
custom_headers:
  cross-origin-opener-policy: same-origin
  cross-origin-embedder-policy: require-corp
  cross-origin-resource-policy: cross-origin
---

# llm2chip

Give llm2chip a language model and it generates the circuit of a chip built for it. This is its online demo: size a chip for a model, look at its floorplan, simulate its circuit gate by gate, and run small models on the chip's integer arithmetic in the browser.

It is the same demo as on GitHub Pages, served with the headers that let the page share memory between threads, so the simulator runs the larger models on several threads. If the Simulator says it runs on one thread, open the Space at its own address, the *.static.hf.space link.

Source, licence (PolyForm Noncommercial 1.0.0) and the full application: https://github.com/alvarograciagil/llm2chip
