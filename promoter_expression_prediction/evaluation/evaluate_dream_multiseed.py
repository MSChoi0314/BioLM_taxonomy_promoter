#!/usr/bin/env python3
import argparse, subprocess, sys, json
from pathlib import Path
import pandas as pd
parser=argparse.ArgumentParser()
parser.add_argument('--results-root',type=Path,required=True)
parser.add_argument('--ground-truth',type=Path,required=True)
parser.add_argument('--subset-dir',type=Path,required=True)
parser.add_argument('--public-id-dir',type=Path,required=True)
parser.add_argument('--evaluator',type=Path,default=Path(__file__).with_name('evaluate_dream_v2.py'))
parser.add_argument('--include-public',action='store_true')
a=parser.parse_args()
files=sorted(a.results_root.glob('seed_*/test_predictions.csv'))
if not files: parser.error('No seed_*/test_predictions.csv files found')
allrows=[]; scores=[]
for f in files:
    seed=f.parent.name
    out=f.parent/'dream_private_eval'
    cmd=[sys.executable,str(a.evaluator.resolve()),'--predictions',str(f.resolve()),'--ground-truth',str(a.ground_truth.resolve()),'--subset-dir',str(a.subset_dir.resolve()),'--public-id-dir',str(a.public_id_dir.resolve()),'--output-dir',str(out.resolve())]
    if a.include_public: cmd.append('--include-public')
    print('[EVAL]',seed,flush=True)
    subprocess.run(cmd,check=True)
    d=pd.read_csv(out/'official_subset_metrics.csv');d.insert(0,'seed',seed.replace('seed_',''));allrows.append(d)
    sc=json.loads((out/'official_scores.json').read_text());scores.append({'seed':seed.replace('seed_',''),'pearson_score':sc['pearson_score'],'spearman_score':sc['spearman_score']})
root=a.results_root
pd.concat(allrows,ignore_index=True).to_csv(root/'dream_multiseed_subsets.csv',index=False)
summary=pd.concat(allrows,ignore_index=True).groupby('subset',sort=False).agg(n_seeds=('pearson_r2','size'),pearson_r2_mean=('pearson_r2','mean'),pearson_r2_sd=('pearson_r2','std'),spearman_mean=('spearman_rho','mean'),spearman_sd=('spearman_rho','std'))
summary.to_csv(root/'dream_multiseed_summary.csv')
pd.DataFrame(scores).to_csv(root/'dream_multiseed_scores.csv',index=False)
print('[SAVED] dream_multiseed_subsets.csv / dream_multiseed_summary.csv / dream_multiseed_scores.csv')
