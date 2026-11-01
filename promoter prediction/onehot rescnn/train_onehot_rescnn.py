#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, random, time
from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def metrics(y,p):
    y=np.asarray(y,dtype=np.float64);p=np.asarray(p,dtype=np.float64)
    r=float(pearsonr(y,p).statistic);rho=float(spearmanr(y,p).statistic)
    return dict(n=len(y),pearson_r=r,pearson_r2=r*r,spearman_rho=rho,r2_cod=float(r2_score(y,p)),
                mae=float(mean_absolute_error(y,p)),rmse=float(mean_squared_error(y,p)**.5))

def load_meta(prefix):
    d=prefix.parent;s=prefix.name
    get=lambda suffix: d/f'{s}_{suffix}'
    y=np.load(get('expressions.npy'),mmap_mode='r')
    src=np.load(get('source_row_index.npy'),mmap_mode='r')
    status=np.load(get('status.npy'),mmap_mode='r')
    if not (len(y)==len(src)==len(status)):raise ValueError('Npy metadata length mismatch')
    good=np.flatnonzero((status==1)&np.isfinite(y)&(src>=0))
    if len(good)!=len(y): raise ValueError(f'Non-success training/test embedding rows: {len(y)-len(good)}. Original model requires all valid.')
    return y,src,good

class SequenceDS(Dataset):
    def __init__(self,cache,src,y,rows):self.cache=cache;self.src=src;self.y=y;self.rows=np.asarray(rows,dtype=np.int64)
    def __len__(self):return len(self.rows)
    def __getitem__(self,k):
        i=int(self.rows[k]);j=int(self.src[i]);return torch.from_numpy(np.array(self.cache[j],copy=True)),torch.tensor(float(self.y[i]),dtype=torch.float32)

class ChunkSampler(Sampler):
    def __init__(self,n,chunk,batch,seed,rank,world):
        self.n=n;self.chunk=chunk;self.batch=batch;self.seed=seed;self.rank=rank;self.world=world;self.epoch=0
        self.global_batch=batch*world;self.usable=(n//self.global_batch)*self.global_batch
    def set_epoch(self,epoch):self.epoch=epoch
    def __len__(self):return self.usable//self.world
    def __iter__(self):
        chunks=[];start=0
        while start<self.usable:
            end=min(start+self.chunk,self.usable)
            if end<self.usable:
                span=(end-start)//self.global_batch*self.global_batch
                end=start+max(span,self.global_batch)
            chunks.append((start,end));start=end
        rng=np.random.default_rng(self.seed+self.epoch)
        for ix in rng.permutation(len(chunks)):
            a,b=chunks[int(ix)]
            for g in range(a,b,self.global_batch):
                for i in range(g+self.rank*self.batch,g+(self.rank+1)*self.batch):
                    if i<b:yield i

class EvalSampler(Sampler):
    def __init__(self,n,rank,world):
        m,r=divmod(n,world);self.a=rank*m+min(rank,r);self.b=self.a+m+(rank<r)
    def __len__(self):return self.b-self.a
    def __iter__(self):return iter(range(self.a,self.b))

class ResBlock(nn.Module):
    def __init__(self,d,k,dilation,p):
        super().__init__();pad=dilation*(k-1)//2
        self.net=nn.Sequential(nn.GroupNorm(1,d),nn.GELU(),nn.Conv1d(d,d,k,padding=pad,dilation=dilation),nn.Dropout(p),
                               nn.GroupNorm(1,d),nn.GELU(),nn.Conv1d(d,d,k,padding=pad,dilation=dilation),nn.Dropout(p))
    def forward(self,z,m):return (z+self.net(z))*m[:,None,:]

class ResCNN(nn.Module):
    def __init__(self,npos,d=256,nblk=6,k=7,p=.1,head='flatten'):
        super().__init__();self.npos=npos;self.head_type=head
        self.proj=nn.Conv1d(4,d,1);self.pos=nn.Parameter(torch.zeros(1,d,npos));nn.init.normal_(self.pos,std=.02)
        self.blocks=nn.ModuleList([ResBlock(d,k,2**(i%4),p) for i in range(nblk)])
        if head=='flatten':self.head=nn.Sequential(nn.LayerNorm(d*npos),nn.Linear(d*npos,512),nn.GELU(),nn.Dropout(p),nn.Linear(512,1))
        else:self.head=nn.Sequential(nn.LayerNorm(2*d),nn.Linear(2*d,256),nn.GELU(),nn.Dropout(p),nn.Linear(256,1))
    def forward(self,codes):
        batch,seq=codes.shape
        if seq>self.npos:raise ValueError('Input exceeds configured positions')
        m=torch.zeros(batch,self.npos,device=codes.device,dtype=torch.float32);m[:,:seq]=1
        unknown=(codes==5)
        x=F.one_hot((codes.long()-1).clamp(0,3),num_classes=4).to(dtype=torch.float32)
        x=torch.where(unknown.unsqueeze(-1), torch.full_like(x,0.25), x)
        x=x.transpose(1,2)
        if self.npos>seq:x=F.pad(x,(0,self.npos-seq))
        z=(self.proj(x)+self.pos)*m[:,None,:]
        for block in self.blocks:z=block(z,m)
        if self.head_type=='flatten':h=z.flatten(1)
        else:
            mean=(z*m[:,None,:]).sum(2)/m.sum(1,keepdim=True).clamp_min(1)
            mx=z.masked_fill(~m[:,None,:].bool(),torch.finfo(z.dtype).min).max(2).values
            h=torch.cat([mean,mx],1)
        return self.head(h).squeeze(1)

def atomic_save(state,path):
    tmp=path.with_name(path.name+'.tmp');torch.save(state,tmp);os.replace(tmp,path)

def main():
    ap=argparse.ArgumentParser()
    for arg in ['train-prefix','test-prefix','train-cache','test-cache','output-dir']:ap.add_argument('--'+arg,required=True,type=Path)
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--split-seed',type=int,default=42)
    ap.add_argument('--positions',type=int,default=128,choices=[128]);ap.add_argument('--val-fraction',type=float,default=.05)
    ap.add_argument('--batch-size',type=int,default=128);ap.add_argument('--eval-batch-size',type=int,default=256)
    ap.add_argument('--num-workers',type=int,default=2);ap.add_argument('--chunk-size',type=int,default=65536)
    ap.add_argument('--hidden-dim',type=int,default=256);ap.add_argument('--num-blocks',type=int,default=6)
    ap.add_argument('--kernel-size',type=int,default=7);ap.add_argument('--head',choices=['flatten','meanmax'],default='flatten')
    ap.add_argument('--dropout',type=float,default=.1);ap.add_argument('--loss',choices=['mse','mae','huber'],default='huber')
    ap.add_argument('--huber-delta',type=float,default=1.);ap.add_argument('--lr',type=float,default=3e-4)
    ap.add_argument('--weight-decay',type=float,default=1e-4);ap.add_argument('--epochs',type=int,default=50)
    ap.add_argument('--patience',type=int,default=8);ap.add_argument('--grad-clip',type=float,default=1.)
    ap.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True);ap.add_argument('--resume',action='store_true')
    a=ap.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    if not (0<a.val_fraction<1):raise ValueError('val-fraction must be in (0,1)')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    world=int(os.environ.get('WORLD_SIZE','1'));local=int(os.environ.get('LOCAL_RANK','0'));ddp=world>1
    if ddp:
        if not torch.cuda.is_available():raise RuntimeError('DDP requires CUDA')
        torch.cuda.set_device(local)
        dist.init_process_group('nccl',timeout=timedelta(hours=2),device_id=torch.device(f'cuda:{local}'))
    rank=dist.get_rank() if ddp else 0;mainrank=rank==0
    dev=torch.device(f'cuda:{local}' if torch.cuda.is_available() else 'cpu')
    if dev.type!='cuda':a.amp=False
    Y,S,G=load_meta(a.train_prefix);TY,TS,TG=load_meta(a.test_prefix)
    X=np.load(a.train_cache,mmap_mode='r');TX=np.load(a.test_cache,mmap_mode='r')
    for cache,src,label in [(X,S,'train'),(TX,TS,'test')]:
        if cache.ndim!=2 or cache.shape[1]!=110 or cache.dtype!=np.uint8:raise ValueError(f'{label} cache must be uint8 [N,110]')
        if len(src) and (int(src.min())<0 or int(src.max())>=len(cache)):raise ValueError(f'{label} source indices out of cache bounds')
        if len(cache)>0 and (cache.min()<1 or cache.max()>5):raise ValueError(f'{label} cache invalid nucleotide codes')
    rng=np.random.default_rng(a.split_seed);perm=rng.permutation(G)
    nv=max(1,int(round(len(perm)*a.val_fraction)));va=np.sort(perm[:nv]);tr=np.sort(perm[nv:]);ti=TG
    split_digest=hashlib.sha256(va.tobytes()).hexdigest()
    if mainrank:
        print(f'[DATA] train={len(tr):,} val={len(va):,} test={len(ti):,} bases=110 positions={a.positions} world={world}',flush=True)
        print(f'[SPLIT] split_seed={a.split_seed} val_sha256={split_digest}',flush=True)
    td=SequenceDS(X,S,Y,tr);vd=SequenceDS(X,S,Y,va);zd=SequenceDS(TX,TS,TY,ti)
    sam=ChunkSampler(len(td),a.chunk_size,a.batch_size,a.seed,rank,world)
    def loader(ds,sampler,bs):return DataLoader(ds,batch_size=bs,sampler=sampler,num_workers=a.num_workers,persistent_workers=a.num_workers>0,pin_memory=dev.type=='cuda')
    train=loader(td,sam,a.batch_size);val=loader(vd,EvalSampler(len(vd),rank,world),a.eval_batch_size)
    test=loader(zd,EvalSampler(len(zd),rank,world),a.eval_batch_size)
    net=ResCNN(a.positions,a.hidden_dim,a.num_blocks,a.kernel_size,a.dropout,a.head).to(dev)
    nparam=sum(p.numel() for p in net.parameters());model=DDP(net,device_ids=[local]) if ddp else net
    if mainrank:print(f'[MODEL] parameters={nparam:,}',flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=2,min_lr=1e-6)
    criterion={'mse':nn.MSELoss(),'mae':nn.L1Loss(),'huber':nn.HuberLoss(delta=a.huber_delta)}[a.loss]
    scaler=torch.amp.GradScaler('cuda',enabled=a.amp)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k!='resume'}
    config['world_size']=world;config['validation_hash']=split_digest
    for name,path in [('train',a.train_cache),('test',a.test_cache)]:
        provenance=Path(str(path).replace('_bases.npy','_info.json'))
        if not provenance.exists():raise FileNotFoundError(f'Missing cache provenance file: {provenance}')
        config[f'{name}_cache_provenance']=json.loads(provenance.read_text())
    config_path=out/'run_config.json'
    if mainrank:
        if config_path.exists():
            old=json.loads(config_path.read_text())
            if old!=config:raise RuntimeError('Existing output has different configuration. Use a new output directory.')
        else:config_path.write_text(json.dumps(config,indent=2))
    if ddp:dist.barrier()
    def gather_eval(dl):
        raw=model.module if ddp else model;raw.eval();ys=[];ps=[]
        with torch.inference_mode():
            for xb,yb in dl:
                xb=xb.to(dev,non_blocking=True)
                with torch.autocast(device_type=dev.type,dtype=torch.bfloat16,enabled=a.amp):q=raw(xb)
                ys.append(yb.numpy());ps.append(q.float().cpu().numpy())
        yy=np.concatenate(ys) if ys else np.empty(0,dtype=np.float32)
        pp=np.concatenate(ps) if ps else np.empty(0,dtype=np.float32)
        if not ddp:return yy,pp
        gathered=[None]*world if mainrank else None
        dist.gather_object((yy,pp),gathered,dst=0)
        if not mainrank:return None,None
        return np.concatenate([x[0] for x in gathered]),np.concatenate([x[1] for x in gathered])
    ck=out/'checkpoint_last.pt';bestpath=out/'model_best.pt';best=-float('inf');bad=0;history=[];start=1
    if a.resume and ck.exists():
        state=torch.load(ck,map_location='cpu',weights_only=False)
        net.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        best=state['best'];bad=state['bad'];history=state['history'];start=state['epoch']+1
        if len(state['rng'])!=world:raise RuntimeError('world_size changed on resume')
        r=state['rng'][rank];random.setstate(r['py']);np.random.set_state(r['np']);torch.set_rng_state(r['cpu']);
        if dev.type=='cuda':torch.cuda.set_rng_state(r['cuda'],device=dev)
        if mainrank:print(f'[RESUME] epoch={state["epoch"]} next={start}',flush=True)
    from tqdm import tqdm
    total_start=time.time()
    for ep in range(start,a.epochs+1):
        t0=time.time();sam.set_epoch(ep);model.train();loss_sum=0.;seen=0
        bar=tqdm(train,disable=not mainrank,desc=f'Epoch {ep}/{a.epochs}',dynamic_ncols=True)
        for xb,yb in bar:
            xb=xb.to(dev,non_blocking=True);yb=yb.to(dev,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type,dtype=torch.bfloat16,enabled=a.amp):pred=model(xb);loss=criterion(pred,yb)
            scaler.scale(loss).backward();scaler.unscale_(opt)
            if a.grad_clip>0:torch.nn.utils.clip_grad_norm_(model.parameters(),a.grad_clip)
            scaler.step(opt);scaler.update();loss_sum+=float(loss.detach())*len(yb);seen+=len(yb)
            if mainrank:bar.set_postfix(loss=f'{loss_sum/max(seen,1):.5f}',lr=f'{opt.param_groups[0]["lr"]:.2e}')
        if ddp:dist.barrier()
        yv,pv=gather_eval(val);rng_local=dict(py=random.getstate(),np=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(dev) if dev.type=='cuda' else None)
        if ddp:
            rngs=[None]*world;dist.all_gather_object(rngs,rng_local)
        else:rngs=[rng_local]
        should_stop=False
        if mainrank:
            vm=metrics(yv,pv);score=vm['pearson_r']+vm['spearman_rho'];scheduler.step(score)
            improved=score>best+1e-6
            if improved:best=score;bad=0
            else:bad+=1
            history.append(dict(epoch=ep,train_loss=loss_sum/max(seen,1),**vm,selection_score_r_plus_rho=score,lr=opt.param_groups[0]['lr'],epoch_seconds=time.time()-t0))
            pd.DataFrame(history).to_csv(out/'training_history.csv',index=False)
            state=dict(epoch=ep,model=net.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),rng=rngs,best=best,bad=bad,history=history)
            atomic_save(state,ck)
            if improved:atomic_save(state,bestpath)
            elapsed=time.time()-t0;eta=(a.epochs-ep)*np.mean([h['epoch_seconds'] for h in history[-3:]])/3600
            print(f'[VAL] epoch={ep} r={vm["pearson_r"]:.6f} pearson_r2={vm["pearson_r2"]:.6f} rho={vm["spearman_rho"]:.6f} bad={bad}/{a.patience} epoch_h={elapsed/3600:.2f} max_epochs_eta_h={eta:.1f}',flush=True)
            should_stop=bad>=a.patience
        if ddp:
            ctl=torch.tensor([float(should_stop),opt.param_groups[0]['lr']],device=dev,dtype=torch.float64)
            dist.broadcast(ctl,0);should_stop=bool(ctl[0].item())
            for group in opt.param_groups:group['lr']=float(ctl[1].item())
        if should_stop:break
    if ddp:dist.barrier()
    if not bestpath.exists():raise RuntimeError('No best checkpoint found. Training did not complete one epoch.')
    state=torch.load(bestpath,map_location='cpu',weights_only=False);net.load_state_dict(state['model'])
    yy,pp=gather_eval(val);ty,tp=gather_eval(test)
    if mainrank:
        vm=metrics(yy,pp);tm=metrics(ty,tp)
        test_meta=pd.read_csv(a.test_prefix.parent/f'{a.test_prefix.name}_meta.csv')
        if len(test_meta)!=len(TY) or 'promoter_id' not in test_meta.columns:raise ValueError('Test metadata has missing/invalid promoter_id')
        pd.DataFrame(dict(source_row_index=np.asarray(TS[ti],dtype=np.int64),promoter_id=test_meta.iloc[ti]['promoter_id'].astype(str).to_numpy(),true_expression=ty,pred_expression=tp)).to_csv(out/'test_predictions.csv',index=False)
        report={'model':'onehot_rescnn','encoding':'ACGT_onehot','input_positions':a.positions,'nucleotides':110,'model_parameters':nparam,'config':config,'validation':vm,'test_all_raw_scale':tm,'best_selection_score_r_plus_rho':best,'elapsed_seconds':time.time()-total_start,'world_size':world}
        (out/'metrics.json').write_text(json.dumps(report,indent=2))
        print('[FINAL]',json.dumps({'validation':vm,'test_all_raw_scale':tm},indent=2),flush=True)
    if ddp:dist.destroy_process_group()
if __name__=='__main__':main()
