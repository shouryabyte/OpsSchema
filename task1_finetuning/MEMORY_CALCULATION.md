# Worked 7B training-memory estimate (not a measured allocation)

Assume 7.0B parameters, AdamW, bf16 compute (2 bytes), fp32 optimizer moments (8 bytes/parameter), fp32 master copy where stated (4 bytes), and a **separate 6 GB activation estimate** for microbatch 1 / sequence 2,048 with checkpointing. Activations vary strongly by architecture, implementation, batch and sequence length; they are not model-weight memory.

| Method | Weights | Trainable params + gradients | Optimizer | Master weights | Estimated subtotal incl. 6 GB activations |
|---|---:|---:|---:|---:|---:|
| Full bf16 | 7e9×2 = 14 GB | 7e9×(2+2) = 28 GB | 7e9×8 = 56 GB | 7e9×4 = 28 GB | 132 GB |
| LoRA (assume 20M adapter params) | frozen 14 GB | 20m×4 = 0.08 GB | 20m×8 = 0.16 GB | 20m×4 = 0.08 GB | 20.3 GB |
| QLoRA | 7e9×0.5 = 3.5 GB + ~0.3 GB quant metadata | 0.08 GB | 0.16 GB | 0.08 GB | 10.1 GB |

Full fine tuning’s trainable weights, gradients, optimizer and optional master copy are all included. LoRA base weights are frozen, so it has gradients/optimizer only for adapters. QLoRA’s 4-bit number is a rough packed-weight estimate; scales/double-quant metadata and kernels add overhead. Framework `memory_allocated` is a live allocator figure, whereas hardware VRAM peak is process/device telemetry; neither is this estimate.
