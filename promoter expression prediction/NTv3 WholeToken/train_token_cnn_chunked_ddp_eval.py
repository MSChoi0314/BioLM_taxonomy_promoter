#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, random, time
from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics(y,p):
    y=np.asarray(y,dtype=np.float64); p=np.asarray(p,dtype=np.float64)
    r=float(pearsonr(y,p).statistic); rho=float(spearmanr(y,p).statistic)
    return {'n':int(len(y)),'pearson_r':r,'pearson_r2':r*r,'spearman_rho':rho,'r2_cod':float(r2_score(y,p)),'mae':float(mean_absolute_error(y,p)),'rmse':float(mean_squared_error(y,p)**0.5)}

def prefix_paths(prefix:Path):
    p=prefix.parent; s=prefix.name
    return {k:p/f'{s}_{v}' for k,v in {'tokens':'tokens.npy','mask':'mask.npy','expressions':'expressions.npy','source_rows':'source_row_index.npy','status':'status.npy','meta':'meta.csv','config':'run_config.json'}.items()}

def load_token_set(prefix:Path, require_all=True):
    paths=prefix_paths(prefix)
    for k,q in paths.items():
        if not q.exists(): raise FileNotFoundError(f'{k}: {q}')
    X=np.load(paths['tokens'],mmap_mode='r'); M=np.load(paths['mask'],mmap_mode='r'); y=np.load(paths['expressions'],mmap_mode='r'); src=np.load(paths['source_rows'],mmap_mode='r'); st=np.load(paths['status'],mmap_mode='r'); meta=pd.read_csv(paths['meta'])
    if not (len(X)==len(M)==len(y)==len(src)==len(st)==len(meta)): raise ValueError('row-count mismatch')
    good=(st==1)&np.isfinite(y)&(src>=0)
    if require_all and not good.all(): raise ValueError(f'non-success rows: {(~good).sum():,}/{len(good):,}')
    idx=np.flatnonzero(good)
    return X,M,y,src,meta,idx

def main():
    ap=argparse.ArgumentParser(description='Position-aware 1D ResCNN on frozen gLM token embeddings')
    ap.add_argument('--train-prefix',required=True,type=Path); ap.add_argument('--test-prefix',required=True,type=Path); ap.add_argument('--output-dir',required=True,type=Path)
    ap.add_argument('--seed',type=int,default=42); ap.add_argument('--val-fraction',type=float,default=.05); ap.add_argument('--batch-size',type=int,default=128); ap.add_argument('--eval-batch-size',type=int,default=256); ap.add_argument('--num-workers',type=int,default=2); ap.add_argument('--chunk-size',type=int,default=65536)
    ap.add_argument('--hidden-dim',type=int,default=256); ap.add_argument('--num-blocks',type=int,default=6); ap.add_argument('--kernel-size',type=int,default=7); ap.add_argument('--dropout',type=float,default=.1); ap.add_argument('--head',choices=['flatten','meanmax'],default='flatten')
    ap.add_argument('--epochs',type=int,default=50); ap.add_argument('--patience',type=int,default=8); ap.add_argument('--lr',type=float,default=3e-4); ap.add_argument('--weight-decay',type=float,default=1e-4); ap.add_argument('--loss',choices=['mse','mae','huber'],default='huber'); ap.add_argument('--huber-delta',type=float,default=1.0); ap.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True); ap.add_argument('--grad-clip',type=float,default=1.0); ap.add_argument('--resume',action='store_true')
    a=ap.parse_args(); out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed)
    import torch, torch.nn as nn, torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import Dataset,DataLoader,Sampler
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    local_rank=int(os.environ.get('LOCAL_RANK','0')); world=int(os.environ.get('WORLD_SIZE','1')); ddp=world>1
    if ddp: dist.init_process_group('nccl', timeout=timedelta(hours=2)); torch.cuda.set_device(local_rank)
    rank=dist.get_rank() if ddp else 0; dev=torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'); main_rank=(rank==0)
    if dev.type!='cuda': a.amp=False
    X,M,y,src,meta,good=load_token_set(a.train_prefix); TX,TM,Ty,Tsrc,Tmeta,Tgood=load_token_set(a.test_prefix)
    if X.shape[1:]!=TX.shape[1:]: raise ValueError(f'train/test token shape mismatch {X.shape[1:]} vs {TX.shape[1:]}')
    npos, din=X.shape[1],X.shape[2]
    rng=np.random.default_rng(a.seed); perm=rng.permutation(good); nv=max(1,int(round(len(perm)*a.val_fraction))); va=np.sort(perm[:nv]); tr=np.sort(perm[nv:]); ti=Tgood
    if main_rank: print(f'[DATA] train={len(tr):,} val={len(va):,} test={len(ti):,} positions={npos} dim={din} world={world}'); print(f'[IO] sampler=chunk_sequential chunk_size={a.chunk_size:,} batch_size_per_gpu={a.batch_size:,}')
    class DS(Dataset):
        def __init__(self,X,M,y,idx): self.X=X;self.M=M;self.y=y;self.idx=np.asarray(idx,dtype=np.int64)
        def __len__(self): return len(self.idx)
        def __getitem__(self,k):
            i=int(self.idx[k]); x=np.array(self.X[i],dtype=np.float32,copy=True); m=np.array(self.M[i],dtype=np.float32,copy=True); yy=np.float32(self.y[i]); return torch.from_numpy(x),torch.from_numpy(m),torch.tensor(yy)
    class ResBlock(nn.Module):
        def __init__(self,d,k,dilation,p):
            super().__init__(); pad=dilation*(k-1)//2
            self.net=nn.Sequential(nn.GroupNorm(1,d),nn.GELU(),nn.Conv1d(d,d,k,padding=pad,dilation=dilation),nn.Dropout(p),nn.GroupNorm(1,d),nn.GELU(),nn.Conv1d(d,d,k,padding=pad,dilation=dilation),nn.Dropout(p))
        def forward(self,z,m): return (z+self.net(z))*m[:,None,:]
    class TokenResCNN(nn.Module):
        def __init__(self,din,d,npos,nblk,k,p,head):
            super().__init__(); self.head_type=head; self.npos=npos; self.proj=nn.Conv1d(din,d,1); self.pos=nn.Parameter(torch.zeros(1,d,npos)); nn.init.normal_(self.pos,std=.02)
            self.blocks=nn.ModuleList([ResBlock(d,k,2**(i%4),p) for i in range(nblk)])
            if head=='flatten': self.head=nn.Sequential(nn.LayerNorm(d*npos),nn.Linear(d*npos,512),nn.GELU(),nn.Dropout(p),nn.Linear(512,1))
            else: self.head=nn.Sequential(nn.LayerNorm(2*d),nn.Linear(2*d,256),nn.GELU(),nn.Dropout(p),nn.Linear(256,1))
        def forward(self,x,m):
            z=self.proj(x.transpose(1,2)); z=(z+self.pos)*m[:,None,:]
            for b in self.blocks: z=b(z,m)
            if self.head_type=='flatten': h=z.flatten(1)
            else:
                denom=m.sum(1,keepdim=True).clamp_min(1.0); mean=(z*m[:,None,:]).sum(2)/denom
                neg=torch.finfo(z.dtype).min; mx=z.masked_fill(~m[:,None,:].bool(),neg).max(2).values; h=torch.cat([mean,mx],1)
            return self.head(h).squeeze(1)

    class ChunkSequentialDistributedSampler(Sampler):
        """
        Shuffle only chunk order. Keep rows inside each chunk sequential.
        For DDP, assign neighboring local batches to neighboring ranks so
        both GPUs read nearby disk regions rather than unrelated random rows.
        """
        def __init__(self, dataset_len, chunk_size, batch_size, seed, rank=0, world_size=1):
            self.dataset_len=int(dataset_len)
            self.chunk_size=int(chunk_size)
            self.batch_size=int(batch_size)
            self.seed=int(seed)
            self.rank=int(rank)
            self.world_size=int(world_size)
            self.epoch=0
            self.global_batch=self.batch_size*self.world_size
            self.usable=(self.dataset_len//self.global_batch)*self.global_batch
            self.num_samples=self.usable//self.world_size

        def set_epoch(self, epoch):
            self.epoch=int(epoch)

        def __len__(self):
            return self.num_samples

        def __iter__(self):
            chunks=[]
            start=0
            while start<self.usable:
                end=min(start+self.chunk_size,self.usable)
                if end<self.usable:
                    span=end-start
                    span=(span//self.global_batch)*self.global_batch
                    if span==0:
                        span=self.global_batch
                    end=start+span
                chunks.append((start,end))
                start=end

            rng=np.random.default_rng(self.seed+self.epoch)
            order=rng.permutation(len(chunks))

            for ci in order:
                start,end=chunks[int(ci)]
                for gstart in range(start,end,self.global_batch):
                    local_start=gstart+self.rank*self.batch_size
                    local_end=local_start+self.batch_size
                    if local_end<=end:
                        for i in range(local_start,local_end):
                            yield i

    class DistributedSequentialEvalSampler(Sampler):
        """Split evaluation rows into disjoint contiguous ranges across ranks."""
        def __init__(self, dataset_len, rank=0, world_size=1):
            self.dataset_len=int(dataset_len)
            self.rank=int(rank)
            self.world_size=int(world_size)
            base=self.dataset_len//self.world_size
            rem=self.dataset_len%self.world_size
            self.start=self.rank*base+min(self.rank,rem)
            self.length=base+(1 if self.rank<rem else 0)
            self.end=self.start+self.length

        def __iter__(self):
            return iter(range(self.start,self.end))

        def __len__(self):
            return self.length

    train_ds=DS(X,M,y,tr); val_ds=DS(X,M,y,va); test_ds=DS(TX,TM,Ty,ti)
    sampler=ChunkSequentialDistributedSampler(len(train_ds),a.chunk_size,a.batch_size,a.seed,rank=rank,world_size=world)
    val_sampler=DistributedSequentialEvalSampler(len(val_ds),rank=rank,world_size=world)
    test_sampler=DistributedSequentialEvalSampler(len(test_ds),rank=rank,world_size=world)
    train_dl=DataLoader(train_ds,batch_size=a.batch_size,shuffle=False,sampler=sampler,num_workers=a.num_workers,pin_memory=True,persistent_workers=a.num_workers>0)
    val_dl=DataLoader(val_ds,batch_size=a.eval_batch_size,shuffle=False,sampler=val_sampler,num_workers=a.num_workers,pin_memory=True,persistent_workers=a.num_workers>0)
    test_dl=DataLoader(test_ds,batch_size=a.eval_batch_size,shuffle=False,sampler=test_sampler,num_workers=a.num_workers,pin_memory=True,persistent_workers=a.num_workers>0)
    model=TokenResCNN(din,a.hidden_dim,npos,a.num_blocks,a.kernel_size,a.dropout,a.head).to(dev); raw=model
    if ddp: model=DDP(model,device_ids=[local_rank])
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay); sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=2,min_lr=1e-6)
    crit={'mse':nn.MSELoss(),'mae':nn.L1Loss(),'huber':nn.HuberLoss(delta=a.huber_delta)}[a.loss]; scaler=torch.amp.GradScaler('cuda',enabled=a.amp)
    from tqdm import tqdm
    def eval_loader_distributed(dl):
        # Evaluate the underlying module on every rank. Using model.module avoids
        # DDP forward-time collectives while ranks may have slightly different
        # numbers of evaluation batches.
        eval_model=model.module if ddp else model
        eval_model.eval(); yy=[]; pp=[]
        with torch.inference_mode():
            for xb,mb,yb in dl:
                xb=xb.to(dev,non_blocking=True); mb=mb.to(dev,non_blocking=True)
                with torch.autocast(device_type=dev.type,dtype=torch.bfloat16,enabled=a.amp): pr=eval_model(xb,mb)
                yy.append(yb.numpy()); pp.append(pr.float().cpu().numpy())
        local_y=np.concatenate(yy) if yy else np.empty(0,dtype=np.float32)
        local_p=np.concatenate(pp) if pp else np.empty(0,dtype=np.float32)
        if not ddp:
            return local_y,local_p
        gathered=[None for _ in range(world)] if main_rank else None
        dist.gather_object((local_y,local_p), gathered, dst=0)
        if not main_rank:
            return None,None
        all_y=np.concatenate([g[0] for g in gathered])
        all_p=np.concatenate([g[1] for g in gathered])
        return all_y,all_p
    best=-np.inf; bad=0; start_ep=1; hist=[]; ck=out/'checkpoint_last.pt'
    if a.resume and ck.exists():
        s=torch.load(ck,map_location=dev); (model.module if ddp else model).load_state_dict(s['model']); opt.load_state_dict(s['optimizer']); start_ep=s['epoch']+1; best=s['best']; bad=s['bad']; hist=s.get('history',[])
    t0=time.time()
    for ep in range(start_ep,a.epochs+1):
        sampler.set_epoch(ep)
        model.train(); total=seen=0; bar=tqdm(train_dl,disable=not main_rank,desc=f'Epoch {ep}/{a.epochs}',dynamic_ncols=True)
        for xb,mb,yb in bar:
            xb=xb.to(dev,non_blocking=True); mb=mb.to(dev,non_blocking=True); yb=yb.to(dev,non_blocking=True); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type,dtype=torch.bfloat16,enabled=a.amp): pr=model(xb,mb); loss=crit(pr,yb)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            if a.grad_clip>0: torch.nn.utils.clip_grad_norm_(model.parameters(),a.grad_clip)
            scaler.step(opt); scaler.update(); total+=float(loss.detach())*len(yb); seen+=len(yb)
            if main_rank: bar.set_postfix(loss=f'{total/max(seen,1):.5f}',lr=f'{opt.param_groups[0]["lr"]:.2e}')
        if ddp: dist.barrier()
        stop=False
        yy,pp=eval_loader_distributed(val_dl)
        if main_rank:
            met=metrics(yy,pp); score=met['pearson_r']+met['spearman_rho']; sched.step(score); improved=score>best+1e-6
            if improved: best=score; bad=0
            else: bad+=1
            hist.append({'epoch':ep,'train_loss':total/max(seen,1),**met,'selection_score_r_plus_rho':score,'lr':opt.param_groups[0]['lr']}); pd.DataFrame(hist).to_csv(out/'training_history.csv',index=False)
            state={'epoch':ep,'model':(model.module if ddp else model).state_dict(),'optimizer':opt.state_dict(),'best':best,'bad':bad,'history':hist}; torch.save(state,ck)
            if improved: torch.save(state,out/'model_best.pt')
            print(f'[VAL] ep={ep} r={met["pearson_r"]:.6f} r2={met["pearson_r2"]:.6f} rho={met["spearman_rho"]:.6f} R2CoD={met["r2_cod"]:.6f} bad={bad}/{a.patience}')
            stop=bad>=a.patience
        if ddp:
            ctl=torch.tensor([int(stop),opt.param_groups[0]['lr']],device=dev,dtype=torch.float64); dist.broadcast(ctl,0); stop=bool(ctl[0].item())
            for g in opt.param_groups: g['lr']=float(ctl[1].item())
        if stop: break
    if ddp: dist.barrier()
    # Every rank loads the same best checkpoint, then validation/test are
    # evaluated in parallel and gathered to rank 0.
    best_state=torch.load(out/'model_best.pt',map_location=dev)
    (model.module if ddp else model).load_state_dict(best_state['model'])
    vy,vp=eval_loader_distributed(val_dl)
    ty,tp=eval_loader_distributed(test_dl)
    if main_rank:
        vm=metrics(vy,vp); tm=metrics(ty,tp)
        selected=Tmeta.iloc[ti].reset_index(drop=True)
        pd.DataFrame({'source_row_index':np.asarray(Tsrc[ti],dtype=np.int64),'promoter_id':selected['promoter_id'].astype(str),'true_expression':ty,'pred_expression':tp}).to_csv(out/'test_predictions.csv',index=False)
        report={'model':'token_rescnn','sampler':'chunk_sequential','chunk_size':a.chunk_size,'ddp_eval':'distributed','nccl_timeout_hours':2,'train_prefix':str(a.train_prefix),'test_prefix':str(a.test_prefix),'config':vars(a),'validation':vm,'test_all_raw_scale':tm,'best_selection_score_r_plus_rho':best,'elapsed_seconds':time.time()-t0,'world_size':world}
        with (out/'metrics.json').open('w') as f: json.dump(report,f,indent=2,default=str)
        print(json.dumps({'validation':vm,'test_all_raw_scale':tm},indent=2)); print('[SAVED]',out/'test_predictions.csv')
    if ddp: dist.destroy_process_group()
if __name__=='__main__': main()
