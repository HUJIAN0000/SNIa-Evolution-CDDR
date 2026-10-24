"""Restore and verify the byte-identical raw covariance from its gzip snapshot."""
from pathlib import Path
import gzip,hashlib,json,shutil,uuid

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def ensure_inputs(folder=None):
    folder=Path(folder) if folder is not None else Path(__file__).resolve().parents[1]/'data'
    metadata=json.loads((folder/'INPUTS.json').read_text(encoding='utf-8'))
    for name,info in metadata.items():
        raw=folder/name;stored=folder/info['stored_as']
        if not stored.is_file() or sha(stored)!=info['stored_sha256']:
            raise ValueError(f'Missing or changed stored input: {stored}')
        if raw.exists():
            if sha(raw)!=info['sha256']:raise ValueError(f'Existing raw input differs; refusing to replace it: {raw}')
            continue
        if info['compression']!='gzip, lossless':raise FileNotFoundError(raw)
        temp=raw.with_name(raw.name+'.partial_'+uuid.uuid4().hex[:8])
        try:
            with gzip.open(stored,'rb') as source,temp.open('xb') as target:shutil.copyfileobj(source,target)
            if sha(temp)!=info['sha256']:raise ValueError(f'Decompressed hash mismatch: {name}')
            temp.replace(raw)
        finally:
            if temp.exists():temp.unlink()
    return metadata

if __name__=='__main__':
    records=ensure_inputs();print('Verified/restored',len(records),'raw inputs; no numeric conversion applied.')
