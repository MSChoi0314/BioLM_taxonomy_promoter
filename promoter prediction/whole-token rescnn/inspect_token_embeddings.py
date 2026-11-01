#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('prefix',type=Path);a=p.parse_args()
for kind in ['tokens','mask','expressions','status','source_row_index','token_counts']:
    q=Path(str(a.prefix)+'_'+kind+'.npy')
    if q.exists():
        x=np.load(q,mmap_mode='r')
        print(kind,'shape=',x.shape,'dtype=',x.dtype,'size_GiB=',round(q.stat().st_size/2**30,3))
        if kind=='status':
            values,counts=np.unique(x,return_counts=True)
            print('  status_counts:',dict(zip(map(int,values),map(int,counts))))
config=Path(str(a.prefix)+'_run_config.json')
if config.exists():print('run_config:',config.read_text()[:4000])
meta=Path(str(a.prefix)+'_meta.csv')
if meta.exists():
    print('meta_columns:',list(pd.read_csv(meta,nrows=2).columns))
    print('meta_head:\n',pd.read_csv(meta,nrows=3).to_string(index=False))
