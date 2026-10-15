from pathlib import Path
import hashlib

ROOT=Path(__file__).resolve().parents[1]
expected={}
for line in (ROOT/'data'/'SOURCE_SHA256.txt').read_text().splitlines():
    h,name=line.split(None,1)
    expected[name.strip()]=h
for name,digest in expected.items():
    f=ROOT/'data'/'local_raw'/name
    if not f.exists():
        raise SystemExit(f'Missing {f}; download from original provider.')
    found=hashlib.sha256(f.read_bytes()).hexdigest()
    if found!=digest:
        raise SystemExit(f'Checksum mismatch: {name}, got {found}, expected {digest}')
    print(f'PASS {name} {digest}')
