#!/usr/bin/env python3
"""Build J1939 Physical Dataset V1 from SystemsCyber data.

Research intent
---------------
Create a *physical-state-only* feature space from J1939 messages. Cyber/network
metadata and attack identifiers are never exported as model features. Annotation
is retained only in a physically separate ground-truth file for evaluation.

This is deliberately a limited decoder for selected, independently verified
SPNs. It is NOT a replacement for the licensed SAE J1939 Digital Annex.
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

RAW_URLS = {
    "labeled_TSC1.csv": "https://raw.githubusercontent.com/SystemsCyber/J1939_Attack_Datasets/refs/heads/main/labeled_TSC1.csv",
    "normal.log": "https://raw.githubusercontent.com/SystemsCyber/J1939_Attack_Datasets/refs/heads/main/normal.log",
}

FORBIDDEN_MODEL_COLUMNS = {
    'iface','can_id','dlc','data','priority','pdu_format','pdu_specific','pdu_type',
    'pgn','pgn_label','destination','source','label','rule_name','rule_type',
    'semantics','rule_severity','rule_layer','rule_description','context_pgn','context_value'
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def download_sources(raw_dir: Path):
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, url in RAW_URLS.items():
        dst = raw_dir / name
        if dst.exists() and dst.stat().st_size > 0:
            print(f"[skip] {dst} already exists")
            continue
        print(f"[download] {url}")
        urllib.request.urlretrieve(url, dst)
        print(f"[ok] {dst} ({dst.stat().st_size:,} bytes)")


def j1939_fields(can_id: int):
    priority = (can_id >> 26) & 0x7
    edp = (can_id >> 25) & 0x1
    dp = (can_id >> 24) & 0x1
    pf = (can_id >> 16) & 0xFF
    ps = (can_id >> 8) & 0xFF
    sa = can_id & 0xFF
    if pf < 240:
        pgn = (edp << 17) | (dp << 16) | (pf << 8)
        da = ps
    else:
        pgn = (edp << 17) | (dp << 16) | (pf << 8) | ps
        da = None
    return priority, pgn, da, sa, pf, ps


def parse_candump(path: Path) -> pd.DataFrame:
    # Supports common candump forms:
    #   (123.456789) can0 18FEF200#1122334455667788
    #   (123.456789) can0 18FEF200 [8] 11 22 33 44 55 66 77 88
    pat_hash = re.compile(r'^\((?P<t>[0-9.]+)\)\s+(?P<iface>\S+)\s+(?P<id>[0-9A-Fa-f]+)#(?P<data>[0-9A-Fa-f]*)')
    pat_brkt = re.compile(r'^\((?P<t>[0-9.]+)\)\s+(?P<iface>\S+)\s+(?P<id>[0-9A-Fa-f]+)\s+\[(?P<dlc>\d+)\]\s+(?P<data>(?:[0-9A-Fa-f]{2}\s*)+)')
    rows=[]
    with path.open('r', errors='replace') as f:
        for line_no, line in enumerate(f, 1):
            line=line.strip()
            if not line: continue
            m=pat_hash.match(line)
            if m:
                data=m.group('data').upper(); cid=int(m.group('id'),16)
                rows.append((float(m.group('t')),m.group('iface'),cid,len(data)//2,data))
                continue
            m=pat_brkt.match(line)
            if m:
                data=''.join(m.group('data').split()).upper(); cid=int(m.group('id'),16)
                rows.append((float(m.group('t')),m.group('iface'),cid,int(m.group('dlc')),data))
    if not rows:
        raise ValueError(f"No candump records parsed from {path}. Inspect its format and extend parse_candump().")
    df=pd.DataFrame(rows,columns=['timestamp','iface','can_id','dlc','data'])
    meta=df['can_id'].map(j1939_fields)
    df[['priority','pgn','destination','source','pdu_format','pdu_specific']]=pd.DataFrame(meta.tolist(),index=df.index)[[0,1,2,3,4,5]]
    df['label']='normal'
    return df


def load_raw(path: Path) -> pd.DataFrame:
    if path.suffix.lower()=='.csv':
        df=pd.read_csv(path)
    else:
        df=parse_candump(path)
    df['timestamp']=pd.to_numeric(df['timestamp'],errors='coerce')
    df=df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    # Recompute ID fields and compare where the CSV already provides PGN/source.
    if 'can_id' in df:
        calc=df['can_id'].astype(int).map(j1939_fields)
        calc_df=pd.DataFrame(calc.tolist(),columns=['_priority','_pgn','_destination','_source','_pf','_ps'],index=df.index)
        if 'pgn' in df.columns:
            mism=(pd.to_numeric(df['pgn'],errors='coerce').fillna(-1).astype(int) != calc_df['_pgn']).sum()
            if mism:
                raise ValueError(f"PGN integrity check failed for {mism} rows in {path.name}")
        if 'source' in df.columns:
            mism=(pd.to_numeric(df['source'],errors='coerce').fillna(-1).astype(int) != calc_df['_source']).sum()
            if mism:
                raise ValueError(f"Source-address integrity check failed for {mism} rows in {path.name}")
        for c, cc in [('priority','_priority'),('pgn','_pgn'),('destination','_destination'),('source','_source'),('pdu_format','_pf'),('pdu_specific','_ps')]:
            if c not in df.columns:
                df[c]=calc_df[cc]
    return df


def raw_int(data_hex: str, start_byte: int, length_bytes: int, endianness: str):
    if not isinstance(data_hex,str): return None
    s=''.join(data_hex.split()).upper()
    if len(s)<2*(start_byte-1+length_bytes): return None
    b=bytes.fromhex(s)
    chunk=b[start_byte-1:start_byte-1+length_bytes]
    return int.from_bytes(chunk,byteorder=endianness,signed=False)


def decode_long(df: pd.DataFrame, rules: list[dict], source_file: str) -> pd.DataFrame:
    by_pgn={}
    for r in rules: by_pgn.setdefault(int(r['pgn']),[]).append(r)
    out=[]
    for row in df.itertuples(index=False):
        pgn=int(getattr(row,'pgn'))
        if pgn not in by_pgn: continue
        src=int(getattr(row,'source')) if not pd.isna(getattr(row,'source')) else -1
        data=str(getattr(row,'data'))
        for r in by_pgn[pgn]:
            if r.get('preferred_source') is not None and src != int(r['preferred_source']):
                continue
            rv=raw_int(data,int(r['start_byte']),int(r['length_bytes']),r['endianness'])
            if rv is None or rv >= int(r['na_raw_min']):
                continue
            value=rv*float(r['resolution'])+float(r['offset'])
            out.append({
                'timestamp':float(getattr(row,'timestamp')),
                'signal':r['name'],'value':value,'unit':r['unit'],
                'pgn':pgn,'spn':int(r['spn']),'source':src,'raw_value':rv,
                'source_file':source_file
            })
    return pd.DataFrame(out)


def ground_truth(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    gt=pd.DataFrame({'timestamp':df['timestamp'].astype(float)})
    if 'label' not in df.columns:
        gt['attack_frame']=0
    else:
        label=df['label'].fillna('normal').astype(str).str.lower().str.strip()
        gt['attack_frame']=(label!='normal').astype(int)
    # Event envelope is computed only from row-level annotations. It is not a model feature.
    abnormal=gt.loc[gt['attack_frame']==1,'timestamp']
    if len(abnormal):
        a=float(abnormal.min()); b=float(abnormal.max())
        gt['perturbation_envelope']=((gt.timestamp>=a)&(gt.timestamp<=b)).astype(int)
        gt['event_start_s']=a; gt['event_end_s']=b
    else:
        gt['perturbation_envelope']=0; gt['event_start_s']=np.nan; gt['event_end_s']=np.nan
    gt['source_file']=source_file
    return gt


def resample_physical(long_df: pd.DataFrame, rules: list[dict], period_s=0.1):
    if long_df.empty: return pd.DataFrame()
    t0=math_floor(float(long_df.timestamp.min()),period_s)
    t1=math_ceil(float(long_df.timestamp.max()),period_s)
    grid=pd.DataFrame({'timestamp':np.arange(t0,t1+period_s/2,period_s)})
    rulemap={r['name']:r for r in rules}
    result=grid.copy()
    for sig, sdf in long_df.groupby('signal'):
        sdf=sdf[['timestamp','value']].sort_values('timestamp').rename(columns={'timestamp':'sample_timestamp','value':sig})
        merged=pd.merge_asof(grid.sort_values('timestamp'),sdf.sort_values('sample_timestamp'),
                             left_on='timestamp',right_on='sample_timestamp',direction='backward',
                             tolerance=float(rulemap[sig]['stale_s']))
        result[sig]=merged[sig]
        result[f'age__{sig}_s']=result['timestamp']-merged['sample_timestamp']
        result[f'valid__{sig}']=merged[sig].notna().astype('int8')
    # Feature file must contain no cyber/network identifiers or annotation.
    forbidden=FORBIDDEN_MODEL_COLUMNS.intersection(result.columns)
    if forbidden: raise AssertionError(f"Leakage: forbidden model columns found: {sorted(forbidden)}")
    return result


def math_floor(x,q): return np.floor(x/q)*q

def math_ceil(x,q): return np.ceil(x/q)*q


def make_ground_truth_grid(raw_df: pd.DataFrame, feature_grid: pd.DataFrame, source_file: str):
    if feature_grid.empty: return pd.DataFrame()
    label=raw_df.get('label',pd.Series(['normal']*len(raw_df))).fillna('normal').astype(str).str.lower().str.strip()
    abnormal=raw_df.loc[label!='normal','timestamp']
    out=pd.DataFrame({'timestamp':feature_grid.timestamp})
    if len(abnormal):
        a=float(abnormal.min()); b=float(abnormal.max())
        out['perturbation_envelope']=((out.timestamp>=a)&(out.timestamp<=b)).astype('int8')
        # This is event-level ground truth. Row-level attack frames remain only in raw GT.
        out['event_start_s']=a; out['event_end_s']=b
    else:
        out['perturbation_envelope']=0; out['event_start_s']=np.nan; out['event_end_s']=np.nan
    out['source_file']=source_file
    return out


def quality_summary(raw_df,long_df,wide_df,source_file):
    sig=[]
    if not long_df.empty:
        for name,g in long_df.groupby('signal'):
            dt=np.diff(g.timestamp.to_numpy())
            sig.append({
                'signal':name,'decoded_samples':len(g),
                'median_interval_s':float(np.median(dt)) if len(dt) else np.nan,
                'median_rate_hz':float(1/np.median(dt)) if len(dt) and np.median(dt)>0 else np.nan,
                'min_value':float(g.value.min()),'max_value':float(g.value.max())
            })
    return pd.DataFrame(sig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--raw-dir',default='raw_source')
    ap.add_argument('--out-dir',default='derived_v1')
    ap.add_argument('--rules',default=str(Path(__file__).resolve().parents[1]/'configs'/'physical_signals.json'))
    ap.add_argument('--download',action='store_true')
    ap.add_argument('--period-ms',type=float,default=100.0)
    ap.add_argument('--inputs',nargs='*',default=['labeled_TSC1.csv','normal.log'])
    args=ap.parse_args()
    raw_dir=Path(args.raw_dir); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    if args.download: download_sources(raw_dir)
    rules=json.loads(Path(args.rules).read_text())
    manifest=[]
    for name in args.inputs:
        p=raw_dir/name
        if not p.exists():
            print(f"[warn] missing {p}; skipping",file=sys.stderr); continue
        raw=load_raw(p)
        long=decode_long(raw,rules,name)
        wide=resample_physical(long,rules,args.period_ms/1000.0)
        gt_raw=ground_truth(raw,name)
        gt_grid=make_ground_truth_grid(raw,wide,name)
        stem=Path(name).stem
        long.to_csv(out/f'{stem}__decoded_long.csv',index=False)
        wide.to_csv(out/f'{stem}__physical_features_{int(args.period_ms)}ms.csv',index=False)
        gt_raw.to_csv(out/f'{stem}__ground_truth_raw.csv',index=False)
        gt_grid.to_csv(out/f'{stem}__ground_truth_{int(args.period_ms)}ms.csv',index=False)
        q=quality_summary(raw,long,wide,name); q.to_csv(out/f'{stem}__signal_quality.csv',index=False)
        manifest.append({'source_file':name,'sha256':sha256(p),'raw_rows':len(raw),'decoded_rows':len(long),'feature_rows':len(wide)})
        print(f"[ok] {name}: raw={len(raw):,}, decoded={len(long):,}, feature_grid={len(wide):,}")
    pd.DataFrame(manifest).to_csv(out/'BUILD_MANIFEST.csv',index=False)
    print(f"[done] outputs: {out.resolve()}")

if __name__=='__main__':
    main()
