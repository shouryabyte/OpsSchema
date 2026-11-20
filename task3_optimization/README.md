# Task 3 — Qwen quantization
Verify official repository filenames and template before download. Run Q4_K_M and Q8_0 independently with the same harness: `python scripts/benchmark.py --llama-cli llama-cli --model PATH --quant Q4_K_M`. The harness does one warmup pass plus three recorded passes per prompt, temp=0, n=128, context=2048. It preserves raw outputs but token throughput must be parsed from llama.cpp logs or a token counter after a real run. No benchmark has run on this machine.

Quantization reduces weight bits; the KV cache stores K and V tensors for prior tokens, so its memory scales with layers, KV heads, head dimension, bytes, tokens and concurrent sequences.
