"""llama.cpp benchmark harness; records raw output and timings, never invented metrics."""
from __future__ import annotations
import argparse,json,subprocess,time
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--llama-cli',required=True);p.add_argument('--model',required=True);p.add_argument('--quant',required=True);p.add_argument('--runs',type=int,default=3);a=p.parse_args()
 prompts=json.loads(Path('../prompts/prompts.json').read_text()); rows=[]
 for warm in [True,False]:
  for item in prompts:
   for run in range(a.runs):
    t=time.perf_counter();x=subprocess.run([a.llama_cli,'-m',a.model,'-p',item['prompt'],'-n','128','--temp','0','-c','2048'],capture_output=True,text=True,check=True);elapsed=time.perf_counter()-t
    if not warm:rows.append({'prompt_id':item['id'],'run':run,'quantization':a.quant,'wall_seconds':elapsed,'output':x.stdout,'stderr':x.stderr})
 Path('../results').mkdir(exist_ok=True);Path(f'../results/{a.quant}_raw.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
