#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import joblib, numpy as np, pandas as pd
from tqdm import tqdm
from common import RunningMoments, atomic_json_dump, elapsed, open_prepared, regression_metrics, seed_everything

def parse_args():
    p=argparse.ArgumentParser(description='Ridge, LightGBM, ResMLP, TabM promoter-expression regression')
    p.add_argument('--model',required=True,choices=['ridge','lightgbm','resmlp','tabm']); p.add_argument('--train-dir',required=True,type=Path); p.add_argument('--external-test-dir',type=Path); p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--batch-size',type=int,default=4096); p.add_argument('--eval-batch-size',type=int,default=16384); p.add_argument('--num-workers',type=int,default=4); p.add_argument('--max-train-samples',type=int); p.add_argument('--standardize',action=argparse.BooleanOptionalAction,default=True); p.add_argument('--resume',action='store_true')
    p.add_argument('--ridge-alpha',type=float,default=100.); p.add_argument('--ridge-stat-batch',type=int,default=32768)
    p.add_argument('--lgb-device',choices=['cpu','gpu','cuda'],default='cpu'); p.add_argument('--lgb-num-gpu',type=int,default=1); p.add_argument('--lgb-gpu-device-id',type=int,default=0); p.add_argument('--lgb-num-threads',type=int,default=max(1,(os.cpu_count() or 8)-2)); p.add_argument('--lgb-objective',choices=['regression','regression_l1','huber'],default='regression_l1'); p.add_argument('--lgb-num-leaves',type=int,default=127); p.add_argument('--lgb-max-depth',type=int,default=-1); p.add_argument('--lgb-learning-rate',type=float,default=.03); p.add_argument('--lgb-n-estimators',type=int,default=5000); p.add_argument('--lgb-min-data-in-leaf',type=int,default=200); p.add_argument('--lgb-feature-fraction',type=float,default=.8); p.add_argument('--lgb-bagging-fraction',type=float,default=.8); p.add_argument('--lgb-bagging-freq',type=int,default=1); p.add_argument('--lgb-lambda-l1',type=float,default=0.); p.add_argument('--lgb-lambda-l2',type=float,default=1.); p.add_argument('--lgb-early-stopping',type=int,default=200)
    p.add_argument('--epochs',type=int,default=100); p.add_argument('--patience',type=int,default=12); p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--weight-decay',type=float,default=1e-4); p.add_argument('--hidden-dim',type=int,default=512); p.add_argument('--num-blocks',type=int,default=4); p.add_argument('--dropout',type=float,default=.1); p.add_argument('--loss',choices=['mse','mae','huber'],default='huber'); p.add_argument('--huber-delta',type=float,default=1.); p.add_argument('--grad-clip',type=float,default=1.); p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True); p.add_argument('--compile',action=argparse.BooleanOptionalAction,default=False); p.add_argument('--tabm-k',type=int,default=16); p.add_argument('--tabm-arch-type',choices=['tabm','tabm-mini'],default='tabm-mini')
    return p.parse_args()

def select(idx,n,seed):
    if n is None or len(idx)<=n:return idx
    r=np.random.default_rng(seed); z=np.sort(r.choice(idx,n,replace=False)); return z.astype(np.int64)
def standardizer(x,idx,batch,out):
    f=out/'standardizer.npz'
    if f.exists(): z=np.load(f); return z['mean'],z['std']
    m=RunningMoments.create(x.shape[1])
    for s in tqdm(range(0,len(idx),batch),desc='Feature statistics',dynamic_ncols=True): m.update(np.asarray(x[idx[s:s+batch]],np.float32))
    mean,std=m.finalize(); np.savez(f,mean=mean,std=std); return mean,std
def predict_linear(x,idx,coef,intercept,mean,std,b):
    o=np.empty(len(idx),np.float32)
    for s in tqdm(range(0,len(idx),b),desc='Predict',dynamic_ncols=True):
        ii=idx[s:s+b]; xb=np.asarray(x[ii],np.float32); xb=(xb-mean)/std if mean is not None else xb; o[s:s+len(ii)]=xb@coef+intercept
    return o
def ridge(a,x,y,tr,va,tx,ty,ti,out):
    st=time.time(); mean=std=None
    if a.standardize: mean,std=standardizer(x,tr,a.ridge_stat_batch,out)
    ym=sum(np.asarray(y[tr[s:s+a.ridge_stat_batch]],np.float64).sum() for s in range(0,len(tr),a.ridge_stat_batch))/len(tr); d=x.shape[1]; xtx=np.zeros((d,d),np.float64); xty=np.zeros(d,np.float64)
    for s in tqdm(range(0,len(tr),a.ridge_stat_batch),desc='Ridge XtX/Xty',dynamic_ncols=True):
        ii=tr[s:s+a.ridge_stat_batch]; xb=np.asarray(x[ii],np.float32); xb=(xb-mean)/std if mean is not None else xb; xb=xb.astype(np.float64,copy=False); yb=np.asarray(y[ii],np.float64)-ym; xtx+=xb.T@xb; xty+=xb.T@yb
    xtx.flat[::d+1]+=a.ridge_alpha; coef=np.linalg.solve(xtx,xty).astype(np.float32); np.savez(out/'ridge_model.npz',coef=coef,intercept=ym,mean=np.array([]) if mean is None else mean,std=np.array([]) if std is None else std,alpha=a.ridge_alpha)
    vp=predict_linear(x,va,coef,ym,mean,std,a.eval_batch_size); tp=predict_linear(tx,ti,coef,ym,mean,std,a.eval_batch_size)
    return regression_metrics(y[va],vp),np.asarray(ty[ti],np.float32),tp,{'elapsed_train':elapsed(st)}
def materialize(x,y,idx,desc): print(f'[{desc}] {len(idx):,} x {x.shape[1]:,} float32'); return np.asarray(x[idx],np.float32),np.asarray(y[idx],np.float32)
def lightgbm(a,x,y,tr,va,tx,ty,ti,out):
    import lightgbm as lgb
    st=time.time(); xtr,ytr=materialize(x,y,tr,'LGB train'); xva,yva=materialize(x,y,va,'LGB val')
    kw=dict(objective=a.lgb_objective,metric=['l1','l2'],n_estimators=a.lgb_n_estimators,learning_rate=a.lgb_learning_rate,num_leaves=a.lgb_num_leaves,max_depth=a.lgb_max_depth,min_child_samples=a.lgb_min_data_in_leaf,colsample_bytree=a.lgb_feature_fraction,subsample=a.lgb_bagging_fraction,subsample_freq=a.lgb_bagging_freq,reg_alpha=a.lgb_lambda_l1,reg_lambda=a.lgb_lambda_l2,n_jobs=a.lgb_num_threads,random_state=a.seed,device_type=a.lgb_device)
    if a.lgb_device=='cpu': kw.update(deterministic=True,force_col_wise=True)
    elif a.lgb_device=='gpu': kw.update(gpu_device_id=a.lgb_gpu_device_id,max_bin=63)
    else: kw.update(num_gpu=a.lgb_num_gpu,max_bin=255)
    m=lgb.LGBMRegressor(**kw); m.fit(xtr,ytr,eval_set=[(xva,yva)],callbacks=[lgb.early_stopping(a.lgb_early_stopping,verbose=True),lgb.log_evaluation(25)])
    m.booster_.save_model(str(out/'lightgbm_model.txt')); joblib.dump(m,out/'lightgbm_model.joblib'); vm=regression_metrics(yva,m.predict(xva,num_iteration=m.best_iteration_)); del xtr,ytr,xva,yva
    tp=np.empty(len(ti),np.float32)
    for s in tqdm(range(0,len(ti),a.eval_batch_size),desc='LGB test',dynamic_ncols=True): ii=ti[s:s+a.eval_batch_size]; tp[s:s+len(ii)]=m.predict(np.asarray(tx[ii],np.float32),num_iteration=m.best_iteration_)
    return vm,np.asarray(ty[ti],np.float32),tp,{'best_iteration':int(m.best_iteration_),'elapsed_train':elapsed(st)}
def neural(a,x,y,tr,va,tx,ty,ti,out):
    import torch, torch.distributed as dist, torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import Dataset,DataLoader,DistributedSampler
    class DS(Dataset):
        def __init__(self,x,y,idx,mean,std):self.x=x;self.y=y;self.idx=idx;self.mean=mean;self.std=std
        def __len__(self):return len(self.idx)
        def __getitem__(self,k):
            i=int(self.idx[k]); z=np.asarray(self.x[i],np.float32); z=(z-self.mean)/self.std if self.mean is not None else z; return torch.from_numpy(np.array(z,copy=True)),torch.tensor(float(self.y[i]),dtype=torch.float32)
    class Block(nn.Module):
        def __init__(self,d,p):super().__init__();self.n=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,2*d),nn.GELU(),nn.Dropout(p),nn.Linear(2*d,d),nn.Dropout(p))
        def forward(self,z):return z+self.n(z)
    class ResMLP(nn.Module):
        def __init__(self,di,dh,n,p):super().__init__();self.f=nn.Sequential(nn.LayerNorm(di),nn.Linear(di,dh),nn.GELU(),*[Block(dh,p) for _ in range(n)],nn.LayerNorm(dh),nn.Linear(dh,1))
        def forward(self,z):return self.f(z).squeeze(-1)
    lrk=int(os.environ.get('LOCAL_RANK','0')); ws=int(os.environ.get('WORLD_SIZE','1')); ddp=ws>1
    if ddp: dist.init_process_group('nccl'); torch.cuda.set_device(lrk)
    dev=torch.device(f'cuda:{lrk}' if torch.cuda.is_available() else 'cpu'); main=not ddp or dist.get_rank()==0
    if dev.type!='cuda': a.amp=False
    mean=std=None
    if a.standardize:
        if main: mean,std=standardizer(x,tr,a.eval_batch_size,out)
        if ddp: dist.barrier(); z=np.load(out/'standardizer.npz'); mean,std=z['mean'],z['std']
    td=DS(x,y,tr,mean,std); vd=DS(x,y,va,mean,std); samp=DistributedSampler(td,shuffle=True,seed=a.seed) if ddp else None
    tl=DataLoader(td,batch_size=a.batch_size,shuffle=samp is None,sampler=samp,num_workers=a.num_workers,pin_memory=True,persistent_workers=a.num_workers>0); vl=DataLoader(vd,batch_size=a.eval_batch_size,shuffle=False,num_workers=a.num_workers,pin_memory=True,persistent_workers=a.num_workers>0)
    if a.model=='resmlp': model=ResMLP(x.shape[1],a.hidden_dim,a.num_blocks,a.dropout)
    else:
        from tabm import TabM
        model=TabM.make(n_num_features=x.shape[1],cat_cardinalities=[],d_out=1,arch_type=a.tabm_arch_type,k=a.tabm_k,d_block=a.hidden_dim,n_blocks=a.num_blocks,dropout=a.dropout)
    model.to(dev)
    if a.compile:model=torch.compile(model)
    if ddp:model=DDP(model,device_ids=[lrk])
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay); sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=3,min_lr=1e-6); scaler=torch.amp.GradScaler('cuda',enabled=a.amp)
    crit={'mse':nn.MSELoss(),'mae':nn.L1Loss(),'huber':nn.HuberLoss(delta=a.huber_delta)}[a.loss]
    def fwd(z,train):
        p=model(z)
        if a.model=='tabm': p=p.squeeze(-1); return p if train else p.mean(1)
        return p
    def evaluate(loader):
        model.eval(); yy=[];pp=[]
        with torch.inference_mode():
            for xb,yb in loader:
                xb=xb.to(dev,non_blocking=True)
                with torch.autocast(device_type=dev.type,dtype=torch.bfloat16,enabled=a.amp): pr=fwd(xb,False)
                yy.append(yb.numpy());pp.append(pr.float().cpu().numpy())
        return np.concatenate(yy),np.concatenate(pp)
    best=-np.inf; bad=0; start_epoch=1; ck=out/'checkpoint_last.pt'
    if a.resume and ck.exists():
        s=torch.load(ck,map_location=dev); (model.module if ddp else model).load_state_dict(s['model']); opt.load_state_dict(s['optimizer']); start_epoch=s['epoch']+1;best=s['best'];bad=s['bad']; print('[resume]',start_epoch,best) if main else None
    hist=[]; total=time.time()
    for ep in range(start_epoch,a.epochs+1):
        if ddp:samp.set_epoch(ep)
        model.train(); total_loss=0.;seen=0; bar=tqdm(tl,disable=not main,desc=f'Epoch {ep}/{a.epochs}',dynamic_ncols=True)
        for xb,yb in bar:
            xb=xb.to(dev,non_blocking=True);yb=yb.to(dev,non_blocking=True);opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type,dtype=torch.bfloat16,enabled=a.amp):
                pr=fwd(xb,True); loss=crit(pr,yb[:,None].expand_as(pr)) if a.model=='tabm' else crit(pr,yb)
            scaler.scale(loss).backward();scaler.unscale_(opt)
            if a.grad_clip>0:torch.nn.utils.clip_grad_norm_(model.parameters(),a.grad_clip)
            scaler.step(opt);scaler.update();total_loss+=float(loss.detach())*len(yb);seen+=len(yb)
            if main:bar.set_postfix(loss=f'{total_loss/seen:.5f}',lr=f'{opt.param_groups[0]["lr"]:.2e}')
        if ddp:dist.barrier()
        stop=False
        if main:
            yy,pp=evaluate(vl); met=regression_metrics(yy,pp); score=met['pearson_r']+met['spearman_rho']; sched.step(score); improved=score>best+1e-6; best=score if improved else best; bad=0 if improved else bad+1
            hist.append({'epoch':ep,'train_loss':total_loss/max(1,seen),**met,'selection_score_r_plus_rho':score,'lr':opt.param_groups[0]['lr']});pd.DataFrame(hist).to_csv(out/'training_history.csv',index=False)
            state={'epoch':ep,'model':(model.module if ddp else model).state_dict(),'optimizer':opt.state_dict(),'best':best,'bad':bad};torch.save(state,ck)
            if improved:torch.save(state,out/'model_best.pt')
            print(f'[epoch {ep}] r={met["pearson_r"]:.6f} r2={met["pearson_r2"]:.6f} rho={met["spearman_rho"]:.6f} R2CoD={met["r2_cod"]:.6f} bad={bad}/{a.patience}')
            stop=bad>=a.patience
        if ddp:
            control=torch.tensor([int(stop),opt.param_groups[0]['lr']],device=dev,dtype=torch.float64);dist.broadcast(control,0);stop=bool(control[0].item());
            for g in opt.param_groups:g['lr']=float(control[1].item())
        if stop:break
    if ddp:dist.barrier()
    if not main:dist.destroy_process_group();return None
    target=model.module if ddp else model; target.load_state_dict(torch.load(out/'model_best.pt',map_location=dev)['model']); yy,pp=evaluate(vl); vm=regression_metrics(yy,pp); testloader=DataLoader(DS(tx,ty,ti,mean,std),batch_size=a.eval_batch_size,shuffle=False,num_workers=a.num_workers,pin_memory=True); yt,pt=evaluate(testloader)
    if ddp:dist.destroy_process_group()
    return vm,yt,pt,{'best_selection_score_r_plus_rho':float(best),'elapsed_train':elapsed(total),'device':str(dev),'world_size':ws}
def main():
    a=parse_args();seed_everything(a.seed);out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True);atomic_json_dump(vars(a),out/'run_config.json')
    x,y,source,split,meta=open_prepared(a.train_dir)
    if split is None:raise ValueError('train-dir has no split.npy')
    tr=select(np.flatnonzero(split==0),a.max_train_samples,a.seed);va=np.flatnonzero(split==1)
    if a.external_test_dir: tx,ty,tsource,_,tmeta=open_prepared(a.external_test_dir);ti=np.arange(len(ty))
    else: tx,ty,tsource,tmeta=x,y,source,meta;ti=np.flatnonzero(split==2)
    print(f'[data] train={len(tr):,} val={len(va):,} test={len(ti):,} dim={x.shape[1]:,}')
    if not len(tr) or not len(va) or not len(ti):raise ValueError('Empty split')
    r=ridge(a,x,y,tr,va,tx,ty,ti,out) if a.model=='ridge' else lightgbm(a,x,y,tr,va,tx,ty,ti,out) if a.model=='lightgbm' else neural(a,x,y,tr,va,tx,ty,ti,out)
    if r is None:return
    vm,yt,yp,extra=r; tm=regression_metrics(yt,yp); selected=tmeta.iloc[ti].reset_index(drop=True)
    pd.DataFrame({'source_row_index':np.asarray(tsource[ti],np.int64),'promoter_id':selected.promoter_id.astype(str),'true_expression':yt,'pred_expression':yp}).to_csv(out/'test_predictions.csv',index=False)
    atomic_json_dump({'model':a.model,'validation':vm,'test_all':tm,**extra},out/'metrics.json');print(json.dumps(tm,indent=2));print('[saved]',out)
if __name__=='__main__':main()
