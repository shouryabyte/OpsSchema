from pathlib import Path
import json
root=Path(__file__).parents[1]
required=[root/'task1_finetuning/results/base_predictions.json',root/'task1_finetuning/results/finetuned_predictions.json',root/'task2_rag/results/evaluation.json',root/'task3_optimization/results/Q4_K_M_raw.json',root/'task3_optimization/results/Q8_0_raw.json']
missing=[str(p.relative_to(root)) for p in required if not p.exists()]
if missing: raise SystemExit('Refusing PDF: missing genuine evidence: '+', '.join(missing))
raise SystemExit('Evidence exists; render report.md with WeasyPrint after candidate review.')
