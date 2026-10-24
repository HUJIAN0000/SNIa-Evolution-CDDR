"""Verify tracked release files using only the Python standard library."""
from pathlib import Path
import ast,gzip,hashlib,json
ROOT=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def verify(root=ROOT):
    root=Path(root)
    manifest=json.loads((root/'PACKAGE_SHA256.json').read_text(encoding='utf-8'))
    for name,entry in manifest.items():
        path=root/name
        assert path.is_file(),f'Missing release file: {name}'
        assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],f'Changed release file: {name}'
        assert path.stat().st_size<=25*1024**2,f'File exceeds GitHub browser per-file limit: {name}'
    inputs=json.loads((root/'data/INPUTS.json').read_text(encoding='utf-8'))
    for name,entry in inputs.items():
        path=root/'data'/entry['stored_as']
        if entry['compression']:
            h=hashlib.sha256()
            with gzip.open(path,'rb') as f:
                for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
            assert h.hexdigest()==entry['sha256'],name
        if (root/'data'/name).exists():assert sha(root/'data'/name)==entry['sha256'],name
    scripts=list((root/'code').rglob('*.py'))+list((root/'original').glob('*.py'))
    for p in scripts:ast.parse(p.read_text(encoding='utf-8-sig'))
    forbidden={'response','manuscript','reference','metadata'}
    assert not any(p.name in forbidden and p.is_dir() for p in root.iterdir())
    result={'tracked_files':len(manifest),'python_sources_parsed':len(scripts),
            'max_tracked_file_bytes':max(x['bytes'] for x in manifest.values()),
            'raw_inputs_verified':len(inputs),'all_tracked_files_under_25MiB':True}
    return result

if __name__=='__main__':print(json.dumps(verify(),indent=2))
