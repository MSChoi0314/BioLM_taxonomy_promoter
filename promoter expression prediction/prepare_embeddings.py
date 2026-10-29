#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np, pandas as pd
from tqdm import tqdm
from common import atomic_json_dump, elapsed

def paths(prefix: Path):
    p=Path(prefix); s=p.name; d=p.parent
    return {k:d/f'{s}_{v}' for k,v in {'emb':'embeddings.npy','exp':'expressions.npy','status':'status.npy','meta':'meta.csv','config':'run_config.json'}.items()}

def main():
    ap=argparse.ArgumentParser(description='run_* embedding outputs를 검증하고 학습용 compact memmap으로 변환')
    ap.add_argument('--input-prefix', required=True, type=Path)
    ap.add_argument('--output-dir', required=True, type=Path)
    ap.add_argument('--val-fraction', type=float, default=0.05)
    ap.add_argument('--test-fraction', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no-split', action='store_true')
    ap.add_argument('--copy-batch-size', type=int, default=32768)
    ap.add_argument('--require-all-success', action='store_true')
    a=ap.parse_args(); start=time.time(); src=paths(a.input_prefix.resolve()); out=a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    missing=[str(p) for p in [src['emb'],src['exp'],src['status'],src['meta']] if not p.exists()]
    if missing: raise FileNotFoundError('Missing embedding outputs:\n'+'\n'.join(missing))
    x=np.load(src['emb'],mmap_mode='r'); y=np.load(src['exp'],mmap_mode='r'); status=np.load(src['status'],mmap_mode='r'); meta=pd.read_csv(src['meta'])
    n=len(y)
    if x.ndim!=2 or len(x)!=n or len(status)!=n or len(meta)!=n: raise RuntimeError(f'Shape mismatch: X={x.shape}, y={y.shape}, status={status.shape}, meta={meta.shape}')
    req={'row_index','promoter_id','expression'}
    if not req.issubset(meta): raise KeyError(f'meta.csv missing {sorted(req-set(meta.columns))}')
    if not np.array_equal(meta.row_index.to_numpy(np.int64),np.arange(n,dtype=np.int64)): raise ValueError('meta row_index must be contiguous 0..N-1 and match NPY rows')
    counts={str(k):int(v) for k,v in zip(*np.unique(status,return_counts=True))}; print('[status]',counts)
    if a.require_all_success and np.any(status!=1): raise RuntimeError('Non-success rows exist; retry embedding extraction first.')
    valid=(status==1)&np.isfinite(y)
    # Row-wise finite check in chunks; avoids loading all X.
    for s in tqdm(range(0,n,a.copy_batch_size),desc='Validate finite embeddings',dynamic_ncols=True):
        e=min(n,s+a.copy_batch_size); valid[s:e] &= np.isfinite(np.asarray(x[s:e])).all(axis=1)
    meta_exp=pd.to_numeric(meta.expression,errors='coerce').to_numpy(np.float64)
    mismatch=valid & ~np.isclose(meta_exp,y.astype(np.float64),rtol=1e-5,atol=1e-5,equal_nan=False)
    if mismatch.any(): raise ValueError(f'meta expression and expressions.npy mismatch: {int(mismatch.sum()):,} rows')
    idx=np.flatnonzero(valid).astype(np.int64); print(f'[valid] {len(idx):,}/{n:,}; excluded={n-len(idx):,}')
    if not len(idx): raise RuntimeError('No valid rows.')
    Xo=np.lib.format.open_memmap(out/'X.npy',mode='w+',dtype=np.float32,shape=(len(idx),x.shape[1]))
    yo=np.lib.format.open_memmap(out/'y.npy',mode='w+',dtype=np.float32,shape=(len(idx),))
    so=np.lib.format.open_memmap(out/'source_row_index.npy',mode='w+',dtype=np.int64,shape=(len(idx),))
    for s in tqdm(range(0,len(idx),a.copy_batch_size),desc='Copy valid rows',dynamic_ncols=True):
        ii=idx[s:s+a.copy_batch_size]; Xo[s:s+len(ii)]=np.asarray(x[ii],np.float32); yo[s:s+len(ii)]=np.asarray(y[ii],np.float32); so[s:s+len(ii)]=ii
    Xo.flush(); yo.flush(); so.flush()
    mout=meta.iloc[idx].copy().reset_index(drop=True); mout.insert(0,'prepared_row_index',np.arange(len(mout))); mout.rename(columns={'row_index':'source_row_index'},inplace=True); mout.to_csv(out/'meta.csv',index=False)
    if not a.no_split:
        if min(a.val_fraction,a.test_fraction)<0 or a.val_fraction+a.test_fraction>=1: raise ValueError('Invalid split fractions')
        rng=np.random.default_rng(a.seed); order=rng.permutation(len(idx)); split=np.zeros(len(idx),np.uint8)
        nv=round(len(idx)*a.val_fraction); nt=round(len(idx)*a.test_fraction)
        split[order[:nv]]=1; split[order[nv:nv+nt]]=2; np.save(out/'split.npy',split)
        print(f'[split] train={(split==0).sum():,}, val={(split==1).sum():,}, test={(split==2).sum():,}')
    report={'input_prefix':str(a.input_prefix.resolve()),'n_source':n,'n_valid':len(idx),'n_excluded':n-len(idx),'embedding_dim':int(x.shape[1]),'source_dtype':str(x.dtype),'status_counts':counts,'elapsed':elapsed(start)}
    atomic_json_dump(report,out/'prepare_report.json'); print(json.dumps(report,indent=2)); print('[saved]',out)
if __name__=='__main__': main()
