"""Package figure-source workbooks and their exact CSV/JSON provenance."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
verification=json.loads((HERE/'Workbook_Verification.json').read_text())
assert verification['all_source_values_reconciled'] and verification['value_mismatches']==0
sources=json.loads((HERE/'Source_Hashes.json').read_text())
paths={r['path'] for rows in sources.values() for r in rows}
for rows in sources.values():
    for record in rows:
        assert hashlib.sha256((ROOT/record['path']).read_bytes()).hexdigest()==record['sha256'],('Source changed after workbook preparation',record['path'])
for relative in sorted(paths):
    source=ROOT/relative
    destination=HERE/'raw_csv'/relative
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,destination)
shutil.copyfile(ROOT/'figures'/'Source_Data_Manifest.json',HERE/'Figure_Source_Manifest.json')
covariance_sources=[]
for filename in ['conditional_noise_covariances.npz','propagation_batch_paths.npz','main_joint_covariance.json','paired_covariances.json','propagation_checks.json','Conditional_Noise_Summary.json']:
    source=ROOT/'statistics'/filename
    assert source.exists(),source
    destination=HERE/'covariance_provenance'/filename
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,destination)
    covariance_sources.append({'path':'statistics/'+filename,'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
(HERE/'Covariance_Source_Hashes.json').write_text(json.dumps(covariance_sources,indent=2)+'\n')

archive=ROOT/'NHB_Source_Data.zip'
files=[]
for p in HERE.rglob('*'):
    if not p.is_file() or '__pycache__' in p.parts or 'qa' in p.parts or p.name.endswith('.inspect.ndjson') or p.name.startswith('Partial_') or p.name=='Package_Manifest.json':continue
    files.append(p)
with ZipFile(archive,'w',ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(files):z.write(p,arcname='NHB_Source_Data/'+str(p.relative_to(HERE)))
manifest={'file':archive.name,'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
          'workbooks':verification['workbooks'],'entries':len(files),
          'source_values_reconciled':True,'new_simulations':False}
(HERE/'Package_Manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
