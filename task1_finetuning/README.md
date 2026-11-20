# Task 1 — OpsSchema
Generate data: `python src/data.py`; validate: `pytest tests`. GPU run in Colab: install `../../requirements.txt`, then replace the explicit CPU guard in `src/train.py` only with a reviewed PEFT SFT loop. This repository deliberately records no training metrics until actual inference and training are implemented/executed.

**Labelling policy:** urgency follows business impact and breadth; action is the requested operational verb, not a keyword in the text. Dataset generator is deterministic seed-by-ID and splits use disjoint ID namespaces. It is a compact synthetic seed dataset requiring candidate review before submission.
