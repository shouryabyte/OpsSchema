from __future__ import annotations
import json
from pathlib import Path
def parse_json(text:str):
 try:
  x=json.loads(text); required={'category','priority','action','system','summary'}
  return x if set(x)==required else None
 except json.JSONDecodeError:return None
def score(rows,predictions):
 assert len(rows)==len(predictions)
 valid=[parse_json(x) for x in predictions]
 fields=['category','priority','action']; n=len(rows)
 return {'n':n,'json_validity':sum(x is not None for x in valid)/n,**{f'{f}_accuracy':sum(v and v[f]==r['target'][f] for r,v in zip(rows,valid))/n for f in fields},'controlled_exact':sum(v and all(v[f]==r['target'][f] for f in fields) for r,v in zip(rows,valid))/n}
