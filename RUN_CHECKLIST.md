# Run checklist
1. `python task1_finetuning/src/data.py`; `cd task1_finetuning; pytest tests`.
2. In GPU Colab install `requirements.txt`, implement/run reviewed PEFT loop, save configs/losses/base+adapter predictions for 75 test examples and three actual failures.
3. Download/audit NASA PDF; author 15 verified questions; evaluate both chunk settings and save results.
4. Install llama.cpp, verify Qwen GGUF Q4_K_M/Q8_0 provenance, benchmark both.
5. Run report generator; only then PDF may exist.
