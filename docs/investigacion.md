# Investigación: chips para LLM

Notas de referencia para calibrar el estimador y decidir qué merece la pena construir. Las cifras de fabricantes son afirmaciones suyas salvo que se indique lo contrario.

## Chips de referencia

| Chip | Memoria | Ancho de banda | Velocidad publicada | Notas |
|---|---|---|---|---|
| Taalas HC1 | Pesos en ROM de máscara, SRAM para KV | Interno | 15.000 a 17.000 tok/s (Llama 3.1 8B) | N6, 815 mm², 53.000 M de transistores, sin HBM. EE Times midió más de 15.000 tok/s |
| Cerebras WSE-3 | 44 GB SRAM | 21 PB/s | Unos 2.100 a 2.500 tok/s (Llama 3.3 70B) | Oblea completa |
| Groq 3 LPU | 500 MB SRAM | 150 TB/s | Sin cifra comparable | Flujo de datos determinista |
| Etched Sohu | 144 GB HBM3E | Unas 1,8 veces un H100 | 62.500 tok/s por chip, anunciado | Preproducción, sin acceso público |

La preset de calibración de la app obtiene 787 mm² y 15.234 tok/s para el HC1, dentro del rango medido por EE Times.

## Qué aprendimos y qué cambia

**La escala por grupo importa mucho.** MXFP4 y NVFP4 usan los mismos elementos E2M1 de 4 bits; la diferencia es la escala: potencia de dos cada 32 pesos en MXFP4, un FP8 cada 16 en NVFP4. Un artículo de ICLR 2026 atribuye a la escala en potencias de dos una degradación fuerte de precisión. En el hardware es un desplazador frente a un multiplicador, de modo que la app distingue ambas opciones y modela cada una.

**Densidad de SRAM.** La celda HD de TSMC se ha estancado desde N5: 0,021 µm² en N5, 0,0199 µm² en N3, unos 0,0175 µm² en N2, con unos 38 Mb/mm². Esas cifras son del bit, no de la macro completa, que pierde espacio en decodificadores y amplificadores. Bajar de nodo apenas abarata la memoria de pesos, que es el bloque dominante.

**Multiplicadores sin multiplicador.** Aproximar pesos con CSD ahorró un 77 % de área lógica en una VGG-16, con 5,6 puntos de precisión perdidos. HaShiFlex reduce pesos a potencias de dos en un flujo de 7 nm. Ninguno de los dos trabajos prueba esto en un transformer: es una opción experimental, y conviene medirla con los pesos reales.

**Compresión de la caché KV.** Cassandra comprime KV y pesos con truncado de mantisa y compresión de exponente. CXL-SpecKV logra de 4 a 8 veces más capacidad con compresión y prelectura en FPGA. La caché KV limita los contextos largos, así que es una línea prometedora.

**Especulación en hardware.** HADES integra la decodificación especulativa en el acelerador. SwiftSpec separa borrador y verificación de forma asíncrona. Encaja con la cabeza MTP de Qwen y con el uso de dies inactivos.

## Lo que falta por verificar

- Medir MXFP4 y NVFP4 con pesos en coma flotante completa, no sobre un modelo ya cuantizado a 4 bits.
- Medir el truncado de la caché KV con Qwen real antes de darlo por bueno.
- Un flujo real de colocación y rutado sobre ASAP7 con OpenROAD para sustituir las estimaciones de área por medidas.

## Fuentes

- [Taalas HC1, producto](https://taalas.com/products/)
- [EE Times sobre Taalas](https://www.eetimes.com/taalas-specializes-to-extremes-for-extraordinary-token-speed/)
- [CNX Software sobre HC1](https://www.cnx-software.com/2026/02/22/taalas-hc1-hardwired-llama-3-1-8b-ai-accelerator-delivers-up-to-17000-tokens-s/)
- [Cerebras frente a H100](https://www.spheron.network/blog/cerebras-vs-nvidia-h100-inference-2026/)
- [Groq 3 LPU](https://www.spheron.network/blog/nvidia-groq-3-lpu-explained/)
- [Etched Sohu](https://www.spheron.network/blog/etched-ai-sohu-vs-nvidia-transformer-asic-inference/)
- [MXFP4 y NVFP4, ICLR 2026](https://arxiv.org/pdf/2509.23202)
- [Comparativa NVFP4 y MXFP4](https://www.spheron.network/blog/nvfp4-vs-mxfp4-gpu-cloud-4bit-quantization-guide/)
- [Estancamiento de la SRAM en N3](https://www.tomshardware.com/news/no-sram-scaling-implies-on-more-expensive-cpus-and-gpus)
- [Mejora de SRAM en N2](https://www.tomshardware.com/tech-industry/sram-scaling-isnt-dead-after-all-tsmcs-2nm-process-tech-claims-major-improvements)
- [CAxCNN](https://doaj.org/article/4f85a81037c040e5bb332cb4fa528ae3)
- [HaShiFlex](https://arxiv.org/pdf/2512.12847v1)
- [Cassandra](https://arxiv.org/pdf/2605.26558)
- [CXL-SpecKV](https://arxiv.org/abs/2512.11920v1)
- [HADES](https://arxiv.org/abs/2412.19925v1)
- [SwiftSpec](https://arxiv.org/html/2506.11309v1)
