from __future__ import annotations
import importlib.util,json,platform,shutil,sys
from pathlib import Path
def main():
 import os
 disk=shutil.disk_usage(Path.cwd().anchor)
 out={'timestamp_utc':__import__('datetime').datetime.utcnow().isoformat()+'Z','os':platform.platform(),'python':sys.version,'cpu_count':os.cpu_count(),'disk_free_gb':round(disk.free/1e9,2),'cuda_visible':False,'tools':{x:shutil.which(x) for x in ('ollama','llama-cli','nvidia-smi')},'packages':{x:bool(importlib.util.find_spec(x)) for x in ('torch','transformers','peft','trl','accelerate','bitsandbytes','fitz','sentence_transformers','faiss')}}
 p=Path('task1_finetuning/results');p.mkdir(exist_ok=True);(p/'environment_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
