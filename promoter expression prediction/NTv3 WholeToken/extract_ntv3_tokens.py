#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from token_common import *
MODEL_NAME='InstaDeepAI/NTv3_650M_pre'; DIM=1536; PAD_MULTIPLE=128

def main():
    p=argparse.ArgumentParser(description='NTv3 whole-token promoter embeddings')
    p.add_argument('--input-fasta',required=True,type=Path); p.add_argument('--output-prefix',required=True,type=Path); p.add_argument('--device',default='cuda:0')
    p.add_argument('--dtype',choices=['float32','bfloat16'],default='bfloat16'); p.add_argument('--storage-dtype',choices=['float16','float32'],default='float16'); p.add_argument('--max-positions',type=int,default=128)
    p.add_argument('--sample-size',type=int); p.add_argument('--seed',type=int,default=42); p.add_argument('--flush-interval',type=int,default=500); p.add_argument('--progress-interval',type=int,default=100)
    p.add_argument('--resume',action='store_true'); p.add_argument('--retry-failed',action='store_true'); p.add_argument('--overwrite',action='store_true'); a=p.parse_args()
    fasta=a.input_fasta.resolve(); prefix=a.output_prefix.resolve(); prefix.parent.mkdir(parents=True,exist_ok=True); total=count_fasta_records(fasta); selected=choose_source_indices(total,a.sample_size,a.seed); paths=build_paths(prefix); sd=np.dtype(a.storage_dtype)
    print(f'[DATA] total={total:,} selected={len(selected):,} max_positions={a.max_positions} dim={DIM}'); print(f'[STORAGE] token array estimate={human_bytes(estimate_storage(len(selected),a.max_positions,DIM,sd))}')
    if a.overwrite and a.resume: raise ValueError('--overwrite and --resume are mutually exclusive')
    if a.overwrite:
        for q in paths.values():
            if q.exists(): q.unlink()
    if any(q.exists() for q in paths.values()) and not a.resume: raise FileExistsError('Outputs exist; use --resume or --overwrite')
    arrays=load_arrays(paths,len(selected),a.max_positions,DIM,sd) if a.resume else (write_preselected_metadata(fasta,selected,paths['meta']) or init_arrays(paths,len(selected),a.max_positions,DIM,sd))
    dev=torch.device(a.device if not a.device.startswith('cuda') or torch.cuda.is_available() else 'cpu'); tok=AutoTokenizer.from_pretrained(MODEL_NAME,trust_remote_code=True)
    kwargs={'trust_remote_code':True}
    if a.dtype=='bfloat16' and dev.type=='cuda':
        kwargs.update(stem_compute_dtype='bfloat16',down_convolution_compute_dtype='bfloat16',transformer_qkvo_compute_dtype='bfloat16',transformer_ffn_compute_dtype='bfloat16',up_convolution_compute_dtype='bfloat16')
    model=AutoModelForMaskedLM.from_pretrained(MODEL_NAME,**kwargs).to(dev).eval()
    def extractor(seq,pid):
        b=tok(seq,add_special_tokens=False,padding=True,pad_to_multiple_of=PAD_MULTIPLE,return_tensors='pt'); b={k:v.to(dev) for k,v in b.items()}
        with torch.inference_mode(): o=model(**b,output_hidden_states=True,return_dict=True)
        h=o.hidden_states[-1]; m=b.get('attention_mask')
        if m is None: m=(torch.arange(h.shape[1],device=dev)<len(seq)).unsqueeze(0)
        v=h[0][m[0].bool()].float().cpu().numpy()
        if v.shape[0]!=len(seq): raise ValueError(f'valid positions {v.shape[0]} != sequence length {len(seq)}')
        if v.shape[1]!=DIM: raise ValueError(f'hidden dim {v.shape[1]} != {DIM}')
        return v
    save_json({'model':MODEL_NAME,'dim':DIM,'total_records':total,'selected_records':len(selected),'sample_size':a.sample_size,'seed':a.seed,'max_positions':a.max_positions,'storage_dtype':a.storage_dtype,'padding_excluded':True,'special_tokens_added':False,'input_fasta':str(fasta)},paths['config'])
    stats=extraction_loop(fasta_path=fasta,selected=selected,paths=paths,arrays=arrays,extractor=extractor,clean_fn=clean_dna,flush_interval=a.flush_interval,progress_interval=a.progress_interval,resume=a.resume,retry_failed=a.retry_failed,max_positions=a.max_positions,storage_dtype=sd)
    print(stats); print('[STATUS]',{k:int(np.count_nonzero(arrays[-1]==v)) for k,v in [('pending',0),('success',1),('failed',2)]})
if __name__=='__main__': main()
