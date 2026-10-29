#!/usr/bin/env python3
"""Reproduce the Random Promoter DREAM Challenge private evaluation logic.

The metric/index logic mirrors de-Boer-Lab/random-promoter-dream-challenge-2022
prixfixe/evaluation.py, with configurable paths and CSV/JSON outputs.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from collections import OrderedDict
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

WEIGHTS={'all':1.0,'high':.3,'low':.3,'yeast':.3,'random':.3,'challenging':.5,'SNVs':1.25,'motif_perturbation':.3,'motif_tiling':.4}

def load_ground_truth(path):
    with open(path) as f: rows=list(csv.reader(f,delimiter='\t'))
    return np.asarray([float(r[1]) for r in rows],dtype=np.float64)

def resolve(subdir:Path,names):
    for n in names:
        q=subdir/n
        if q.exists(): return q
    raise FileNotFoundError('None found: '+', '.join(str(subdir/n) for n in names))

def load_class(path):
    df=pd.read_csv(path)
    if 'pos' in df.columns: return np.unique(np.asarray(df['pos'],dtype=np.int64)).tolist()
    if 'alt_pos' in df.columns and 'ref_pos' in df.columns:
        # Exact official semantics: tuples are created as (alt_pos, ref_pos).
        return list(set(zip(df['alt_pos'].astype(int).tolist(),df['ref_pos'].astype(int).tolist())))
    raise ValueError(f'Unsupported subset schema: {path}')

def load_public_single(path):
    with open(path) as f: return [int(k) for k in json.load(f).keys()]

def load_public_double(path):
    with open(path) as f: return [(int(k.split(',')[0]),int(k.split(',')[1])) for k in json.load(f).keys()]

def corr(indices,pred,truth):
    gt=OrderedDict(); pdict=OrderedDict()
    for j in indices: pdict[str(j)]=float(pred[j]); gt[str(j)]=float(truth[j])
    return float(pearsonr(list(gt.values()),list(pdict.values())).statistic),float(spearmanr(list(gt.values()),list(pdict.values())).statistic)

def diffcorr(pairs,pred,truth):
    yp=[]; yt=[]
    for pair in pairs:
        # Exact official code calls pair[0] 'ref' and pair[1] 'alt', even though
        # load_class constructed (alt_pos, ref_pos). Both truth and prediction
        # receive the same sign reversal, so correlations are unchanged.
        ref,alt=pair[0],pair[1]
        yp.append(pred[alt]-pred[ref]); yt.append(truth[alt]-truth[ref])
    return float(pearsonr(yt,yp).statistic),float(spearmanr(yt,yp).statistic)

def main():
    a=argparse.ArgumentParser(description='Official DREAM private evaluator')
    a.add_argument('--predictions',required=True,type=Path,help='CSV with source_row_index and pred_expression')
    a.add_argument('--ground-truth',required=True,type=Path,help='filtered_test_data_with_MAUDE_expression.txt')
    a.add_argument('--subset-dir',required=True,type=Path); a.add_argument('--public-id-dir',required=True,type=Path); a.add_argument('--output-dir',required=True,type=Path)
    a.add_argument('--include-public',action='store_true',help='Do not discard public leaderboard IDs; default reproduces private/final evaluation')
    args=a.parse_args(); out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    truth=load_ground_truth(args.ground_truth); pdf=pd.read_csv(args.predictions)
    if not {'source_row_index','pred_expression'}.issubset(pdf.columns): raise ValueError('predictions CSV needs source_row_index,pred_expression')
    if pdf['source_row_index'].duplicated().any(): raise ValueError('duplicate source_row_index in predictions')
    pmap=dict(zip(pdf.source_row_index.astype(int),pdf.pred_expression.astype(float))); missing=[i for i in range(len(truth)) if i not in pmap]
    if missing: raise ValueError(f'Missing predictions for {len(missing):,}/{len(truth):,} test rows; first={missing[:10]}')
    pred=np.asarray([pmap[i] for i in range(len(truth))],dtype=np.float64)
    if 'true_expression' in pdf.columns:
        tmap=dict(zip(pdf.source_row_index.astype(int),pdf.true_expression.astype(float))); arr=np.asarray([tmap[i] for i in range(len(truth))]); maxerr=float(np.max(np.abs(arr-truth)))
        print(f'[CHECK] max |prediction CSV true_expression - ground truth| = {maxerr:.9g}')
        if maxerr>1e-5: raise ValueError('true_expression in prediction CSV does not match MAUDE ground truth')
    sd=args.subset_dir
    files={
      'high':resolve(sd,['high_exp_seqs.csv']), 'low':resolve(sd,['low_exp_seqs.csv']), 'yeast':resolve(sd,['yeast_seqs.csv']), 'random':resolve(sd,['all_random_seqs.csv']), 'challenging':resolve(sd,['challenging_seqs.csv']),
      'SNVs':resolve(sd,['all_SNVs_seqs.csv']), 'motif_perturbation':resolve(sd,['motif_perturbation_seqs.csv','motif_perturbation.csv']), 'motif_tiling':resolve(sd,['motif_tiling_seqs.csv'])}
    sets={k:load_class(v) for k,v in files.items()}
    if not args.include_public:
        pdid=args.public_id_dir
        ps=load_public_single(resolve(pdid,['high_exp_indices.json']))+load_public_single(resolve(pdid,['low_exp_indices.json']))+load_public_single(resolve(pdid,['yeast_exp_indices.json']))+load_public_single(resolve(pdid,['random_exp_indices.json']))+load_public_single(resolve(pdid,['challenging_exp_indices.json']))
        pdouble=load_public_double(resolve(pdid,['SNVs_exp_indices.json']))+load_public_double(resolve(pdid,['motif_perturbation_exp_indices.json']))+load_public_double(resolve(pdid,['motif_tiling_exp_indices.json']))
        public_indices=list(set(ps+[x for pair in pdouble for x in pair])); pdouble_set=set(pdouble)
        for k in ['high','low','yeast','random','challenging']: sets[k]=[x for x in sets[k] if x not in public_indices]
        for k in ['SNVs','motif_perturbation','motif_tiling']: sets[k]=[x for x in sets[k] if x not in pdouble_set]
        final_all=[i for i in range(len(truth)) if i not in public_indices]
    else: final_all=list(range(len(truth)))
    vals={}; vals['all']=corr(final_all,pred,truth)
    for k in ['high','low','yeast','random','challenging']: vals[k]=corr(sets[k],pred,truth)
    for k in ['SNVs','motif_perturbation','motif_tiling']: vals[k]=diffcorr(sets[k],pred,truth)
    pearson_score=sum(WEIGHTS[k]*(vals[k][0]**2) for k in WEIGHTS)/sum(WEIGHTS.values()); spearman_score=sum(WEIGHTS[k]*vals[k][1] for k in WEIGHTS)/sum(WEIGHTS.values())
    rows=[]
    for k in WEIGHTS:
        n=len(final_all) if k=='all' else len(sets[k]); r,rho=vals[k]; rows.append({'subset':k,'n_items':n,'weight':WEIGHTS[k],'pearson_r':r,'pearson_r2':r*r,'spearman_rho':rho,'comparison':'expression' if k in ['all','high','low','yeast','random','challenging'] else 'expression_difference'})
    pd.DataFrame(rows).to_csv(out/'official_subset_metrics.csv',index=False)
    report={'evaluation_mode':'all/public included' if args.include_public else 'private/final (~87%; public leaderboard IDs excluded)','n_ground_truth':len(truth),'pearson_score':pearson_score,'spearman_score':spearman_score,'weight_sum':sum(WEIGHTS.values()),'subset_files':{k:str(v) for k,v in files.items()}}
    with (out/'official_scores.json').open('w') as f: json.dump(report,f,indent=2)
    print('*'*60); print(f'Pearson Score: {pearson_score}'); print(f'Spearman Score: {spearman_score}'); print('*'*60)
    print(pd.DataFrame(rows).to_string(index=False)); print('[SAVED]',out)
if __name__=='__main__': main()
