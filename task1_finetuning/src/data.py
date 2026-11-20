"""Independent synthetic operational-support dataset for OpsSchema."""
from __future__ import annotations
import json, random
from pathlib import Path
from typing import Any

SCHEMA = {"category":{"access","network","hardware","software","other"},"priority":{"low","medium","high","critical"},"action":{"create","update","investigate","reset","escalate"}}
CASES = [
 ("access","high","reset","Okta","MFA challenge loop prevents payroll staff from signing in"),
 ("network","critical","investigate","VPN","Mumbai VPN gateway drops all remote sessions"),
 ("hardware","medium","create","ThinkPad T14","keyboard intermittently repeats characters"),
 ("software","low","update","Adobe Acrobat","reader needs the approved quarterly patch"),
 ("other","medium","escalate","Service desk","vendor ticket requires management review"),
 ("access","medium","create","Salesforce","new contractor needs read-only opportunity access"),
 ("network","high","investigate","Wi-Fi","warehouse scanners lose wireless connection after roaming"),
 ("hardware","critical","escalate","SAN array","controller reports predictive failure on production volume"),
 ("software","high","reset","GitLab","CI runner token is rejected after rotation"),
 ("other","low","investigate","Asset inventory","serial number is missing from inventory record"),
]
PREFIX=["Please help:","Ticket:","Heads up,", "Need assistance —", "User reports", "ops note:"]
NOISE=["", " (typo in earlier ticket)", "; affects one user", ". pls check", " — since this morning"]
def record(i:int, split:str) -> dict[str,Any]:
    c,p,a,s,summary=CASES[(i*7+(0 if split=='train' else 3 if split=='validation' else 6))%len(CASES)]
    rng=random.Random(f"ops-{split}-{i}")
    # Split-specific operational contexts prevent train/test template reuse.
    context={"train":['Pune office','finance tenant','weekday shift','pilot group'],"validation":['Bengaluru lab','HR tenant','month-end window','night shift'],"test":['Chennai branch','legal tenant','weekend maintenance','field team']}[split]
    detail=f"Impact context: {context[i%4]}; observation {['after a browser update','following maintenance','during a handoff','with error code '+str(100+i)][i%4]}."
    text=f"{rng.choice(PREFIX)} {summary}{rng.choice(NOISE)}. {detail} Ref {split[:1].upper()}{i:03d}."
    target={"category":c,"priority":p,"action":a,"system":s,"summary":summary+'; '+detail}
    return {"id":f"ops-{split}-{i:03d}","request":text,"target":target}
def build(out:Path)->None:
    counts={"train":450,"validation":75,"test":75}
    out.mkdir(parents=True,exist_ok=True)
    for split,n in counts.items():
        rows=[record(i,split) for i in range(n)]
        (out/f"{split}.jsonl").write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in rows)+"\n",encoding="utf-8")
def validate(rows:list[dict[str,Any]])->list[str]:
    errors=[]
    for r in rows:
        t=r.get("target",{})
        for key,allowed in SCHEMA.items():
            if t.get(key) not in allowed: errors.append(f"{r.get('id')}: invalid {key}")
        if not isinstance(t.get("system"),str) or not isinstance(t.get("summary"),str): errors.append(f"{r.get('id')}: strings required")
    return errors
if __name__=='__main__': build(Path(__file__).parents[1]/"dataset")
