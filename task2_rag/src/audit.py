from __future__ import annotations
import json
from pathlib import Path
from rag import audit_pdf
PDF_URL='https://www.nasa.gov/wp-content/uploads/2018/09/nasa_systems_engineering_handbook_0.pdf'
if __name__=='__main__':
 root=Path(__file__).parents[1]; x=audit_pdf(root/'documents/nasa_sp_2016_6105_rev2.pdf');x.update({'download_url':PDF_URL,'claimed_document_version':'NASA/SP-2016-6105 Rev 2','audited_at':'local execution'});(root/'results').mkdir(exist_ok=True);(root/'results/document_audit.json').write_text(json.dumps(x,indent=2));print(json.dumps(x,indent=2))
