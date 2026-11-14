"""Validate and reclaim user-provided archives without duplicating torch checkpoints."""
from pathlib import Path
import hashlib,json,zipfile,shutil
ROOT=Path(__file__).resolve().parents[2]
def prepare(root):
 manifest=root/'manifest.json'
 report=[entry for entry in json.loads(manifest.read_text()) if 'action' in entry] if manifest.exists() else []
 for archive in sorted(root.glob('*.zip')):
  with zipfile.ZipFile(archive) as z:
   names=z.namelist()
   if z.testzip() is not None: raise ValueError(f'Corrupt archive: {archive.name}')
   if any(n.endswith('/data.pkl') for n in names):
    target=archive.with_suffix('')
    if target.exists(): raise FileExistsError(target)
    archive.rename(target)
    report.append({'source':archive.name,'action':'renamed PyTorch container without extraction','path':target.name})
    continue
   outputs=[]
   for entry in z.infolist():
    if entry.is_dir(): continue
    rel=Path(entry.filename)
    if rel.is_absolute() or '..' in rel.parts: raise ValueError('Unsafe archive path')
    dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists(): raise FileExistsError(dest)
    temp=dest.with_suffix(dest.suffix+'.partial')
    with z.open(entry) as src,temp.open('wb') as dst: shutil.copyfileobj(src,dst,1024*1024)
    if temp.stat().st_size!=entry.file_size: raise ValueError('Extraction size mismatch')
    temp.rename(dest);outputs.append(str(dest.relative_to(root)))
  archive.unlink()
  report.append({'source':archive.name,'action':'CRC verified, extracted, ZIP removed','paths':outputs})
 for p in root.rglob('*'):
  if p.suffix in ['.pt','.pth']:
   h=hashlib.sha256()
   with p.open('rb') as f:
    for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
   report.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
 (root/'manifest.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2))
if __name__=='__main__': prepare(ROOT/'weights')
