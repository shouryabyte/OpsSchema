from __future__ import annotations
def evidence_hit(retrieved, gold:str)->bool:
 norm=lambda s:' '.join(s.lower().split())
 return any(norm(gold) in norm(x.text) for x in retrieved)
def recall_at_5(results):return sum(results)/len(results) if results else 0.0
