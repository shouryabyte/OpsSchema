import json
from pathlib import Path
from src.data import build,validate
def test_splits_and_schema(tmp_path):
 build(tmp_path); sets=[]
 for name,n in [('train',450),('validation',75),('test',75)]:
  rows=[json.loads(x) for x in (tmp_path/f'{name}.jsonl').read_text().splitlines()];assert len(rows)==n;assert not validate(rows);sets.extend(r['id'] for r in rows)
 assert len(sets)==len(set(sets))
