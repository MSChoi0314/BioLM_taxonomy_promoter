#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd

ap=argparse.ArgumentParser()
ap.add_argument('root',type=Path)
ap.add_argument('--output',type=Path,default=None)
a=ap.parse_args()
records=[]
for p in sorted(a.root.glob('seed_*/metrics.json')):
    with p.open() as f: v=json.load(f)
    row={'seed':v['config']['seed'],'split_seed':v['config'].get('split_seed'),'run_dir':str(p.parent),'world_size':v.get('world_size'),'elapsed_h':v.get('elapsed_seconds',0)/3600}
    for k,x in v['test_all_raw_scale'].items(): row[k]=x
    records.append(row)
if not records: raise SystemExit(f'No completed runs in {a.root}')
df=pd.DataFrame(records).sort_values('seed')
out=a.output or a.root/'seed_metrics.csv'; df.to_csv(out,index=False)
print(df.to_string(index=False))
cols=['pearson_r2','pearson_r','spearman_rho','r2_cod','mae','rmse']
summary=df[[x for x in cols if x in df]].agg(['mean','std','min','max']).T
summary['n_seeds']=len(df)
summary.to_csv(out.with_name('seed_summary.csv'))
print('\nSUMMARY (sample SD)\n',summary.to_string())
