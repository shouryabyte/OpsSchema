"""Reproducible NASA retrieval experiment; requires reviewed evidence, never mocks it."""
from __future__ import annotations
import json
from pathlib import Path
from evaluate import evidence_hit,recall_at_5
from rag import chunks,embed,search
ROOT=Path(__file__).parents[1]
def load_questions()->list[dict]:
 raw=json.loads((ROOT/'evaluation/questions_review_required.json').read_text())
 qs=raw.get('questions',[])
 required={'id','question','exact_gold_passage','page','source_id'}
 if raw.get('status')!='APPROVED' or len(qs)!=15 or any(not required <= set(q) for q in qs):
  raise SystemExit('Need exactly 15 candidate-approved evidence questions with exact passages/pages/source IDs.')
 return qs
def main()->None:
 pdf=ROOT/'documents/nasa_sp_2016_6105_rev2.pdf';qs=load_questions();results=[]
 for size,overlap in [(180,35),(320,60)]:
  rows=chunks(pdf,size,overlap,'NASA-SP-2016-6105-REV2');v=embed([x.text for x in rows]);hits=[]
  for q in qs:
   top=search(q['question'],rows,v,5);hit=evidence_hit(top,q['exact_gold_passage']);hits.append(hit)
   results.append({'config':f'{size}/{overlap}','question_id':q['id'],'hit_at_5':hit,'retrieved':[{'chunk_id':x.chunk_id,'page':x.page,'source_id':x.source_id} for x in top]})
  results.append({'config':f'{size}/{overlap}','summary':True,'recall_at_5':recall_at_5(hits),'matching':'normalized exact gold-passage containment in a retrieved chunk'})
 (ROOT/'results').mkdir(exist_ok=True);(ROOT/'results/evaluation.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
