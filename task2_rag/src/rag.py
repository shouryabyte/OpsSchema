from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass,asdict
from pathlib import Path
import fitz,numpy as np
@dataclass
class Chunk: source_id:str;page:int;chunk_id:str;text:str
def audit_pdf(path:Path)->dict:
 d=fitz.open(path);return {'path':str(path),'pages':len(d),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
def tokens(text:str)->list[str]:return re.findall(r"\S+",text)
def chunks(pdf:Path,size:int,overlap:int,source_id:str)->list[Chunk]:
 d=fitz.open(pdf);out=[]
 for p in d:
  ts=tokens(' '.join(p.get_text('text').split()))
  for i in range(0,len(ts),size-overlap):
   x=' '.join(ts[i:i+size]);
   if x:out.append(Chunk(source_id,p.number+1,f'{source_id}-p{p.number+1}-o{i}',x))
 return out
def embed(texts:list[str],model_name='sentence-transformers/all-MiniLM-L6-v2')->np.ndarray:
 from sentence_transformers import SentenceTransformer
 return SentenceTransformer(model_name).encode(texts,normalize_embeddings=True,show_progress_bar=True)
def search(query:str,items:list[Chunk],vectors:np.ndarray,k=5)->list[Chunk]:
 q=embed([query]);return [items[i] for i in np.argsort(-(vectors@q[0]))[:k]]
