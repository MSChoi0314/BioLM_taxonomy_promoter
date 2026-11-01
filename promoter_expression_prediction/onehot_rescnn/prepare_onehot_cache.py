#!/usr/bin/env python3
import argparse, csv, hashlib, json, os, time
from pathlib import Path
import numpy as np

LUT=np.zeros(256,dtype=np.uint8)
for ch,v in [('A',1),('C',2),('G',3),('T',4),('a',1),('c',2),('g',3),('t',4),('N',5),('n',5)]: LUT[ord(ch)]=v

def records(path, fmt, column, header):
    if fmt=='fasta':
        with open(path,encoding='utf-8') as h:
            seq=[]; found=False
            for line in h:
                line=line.strip()
                if not line: continue
                if line.startswith('>'):
                    if found: yield ''.join(seq)
                    seq=[];found=True
                else:
                    if not found: raise ValueError('FASTA has sequence before first header')
                    seq.append(line)
            if found: yield ''.join(seq)
    else:
        delim='\t' if fmt=='tsv' else ','
        with open(path,encoding='utf-8',newline='') as h:
            reader=csv.reader(h,delimiter=delim)
            columns=next(reader) if header else None
            col=columns.index(column) if columns is not None and not column.isdigit() else int(column)
            for row in reader:
                if not row: continue
                if len(row)<=col: raise ValueError('Missing sequence column')
                yield row[col].strip()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--format',choices=['fasta','tsv','csv'],required=True)
    p.add_argument('--column',default='0',help='0-based index or header column name')
    p.add_argument('--header',action='store_true')
    p.add_argument('--out-prefix',required=True,type=Path)
    p.add_argument('--train-source-index',type=Path,help='Optional embedding source_row_index.npy for bounds check')
    p.add_argument('--expected-length',type=int,default=110)
    p.add_argument('--expected-records',type=int,default=0,help='0 = do not enforce exact count')
    a=p.parse_args();a.out_prefix.parent.mkdir(parents=True,exist_ok=True)
    dest=Path(str(a.out_prefix)+'_bases.npy'); info=Path(str(a.out_prefix)+'_info.json')
    if dest.exists() or info.exists(): raise FileExistsError(f'Cache already exists: {dest} / {info}')
    tmp=Path(str(dest)+'.incomplete')
    n=0; t=time.time(); source_digest=hashlib.sha256(); first=[]
    try:
        with open(tmp,'wb',buffering=4*1024*1024) as f:
            for s in records(a.input,a.format,a.column,a.header):
                b=s.encode('ascii')
                if len(b)!=a.expected_length: raise ValueError(f'input row {n}: length {len(b)} != {a.expected_length}')
                arr=np.frombuffer(b,dtype=np.uint8)
                codes=LUT[arr]
                if np.any(codes==0):
                    invalid=sorted(set(chr(int(v)) for v in arr[codes==0]))
                    raise ValueError(f'input row {n}: unsupported characters {invalid}; only ACGTN allowed')
                f.write(codes.tobytes());source_digest.update(b)
                if n<3:first.append(s)
                n+=1
                if n%500000==0: print(f'[PREP] {n:,} rows elapsed={time.time()-t:.1f}s',flush=True)
        if a.expected_records and a.expected_records!=n: raise ValueError(f'Expected {a.expected_records:,} records, saw {n:,}')
        if a.train_source_index:
            idx=np.load(a.train_source_index,mmap_mode='r')
            good=idx[idx>=0]
            if len(good) and int(good.max())>=n: raise ValueError(f'Max source index={good.max()} but only {n} source sequences')
            print(f'[CHECK] source index count={len(idx):,}; max={good.max() if len(good) else "n/a"}; source records={n:,}')
        outtmp=Path(str(dest)+'.writing.npy')
        dest_mm=np.lib.format.open_memmap(outtmp,mode='w+',dtype=np.uint8,shape=(n,a.expected_length))
        raw=np.memmap(tmp,mode='r',dtype=np.uint8,shape=(n,a.expected_length))
        for start in range(0,n,200000): dest_mm[start:start+200000]=raw[start:start+200000]
        dest_mm.flush(); del dest_mm;del raw
        os.replace(outtmp,dest)
        report={'input':str(a.input.resolve()),'format':a.format,'column':a.column,'header':a.header,'records':n,'length':a.expected_length,'sha256_sequence_stream':source_digest.hexdigest(),'first_three_sequences':first,'codes':{'0':'padding','1':'A','2':'C','3':'G','4':'T','5':'N (uniform A/C/G/T)'}}
        info.write_text(json.dumps(report,indent=2))
        print(f'[DONE] {dest}, records={n:,}, elapsed={time.time()-t:.1f}s')
    finally:
        if tmp.exists(): tmp.unlink()
if __name__=='__main__':main()
