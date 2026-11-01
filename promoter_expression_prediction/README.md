# Promoter expression prediction from genomic language model embeddings

Training dataset and Test dataset have to be downloaded from [Random Promoter DREAM Challenge 2022](https://github.com/de-Boer-Lab/random-promoter-dream-challenge-2022)

Three approaches are compared:
* **A. Pooled embeddings.** DNABERT-2, Evo2 or NTv3 embeddings, mean- or max-pooled over the sequence, followed by Ridge, LightGBM, ResMLP or TabM.
* **B. Whole-token ResCNN.** A 1D residual CNN trained on the per-nucleotide NTv3 embeddings, without pooling.
* **C. One-hot ResCNN.** The same CNN trained on one-hot DNA, as a baseline that uses no language model.


## Pipeline
```
Training fasta file
   │
   ├─ A ─► run_{model}_promoter_fasta.py ─► {prefix}_embeddings.npy, _expressions.npy, _status.npy, _meta.csv
   │            └─► prepare_embeddings.py ─► prepared/{name}/X.npy, y.npy, split.npy, ...
   │                    └─► train_models.py ─────────────────────────────────┐
   │                                                                         │
   ├─ B ─► extract_ntv3_tokens.py ─► {prefix}_tokens.npy, _mask.npy, ...     │
   │            └─► inspect_token_embeddings.py                              │
                └─► train_token_cnn.py  /  run_rescnn_multiseed_v3.sh ───────┤
   │                                                                         ├─► test_predictions.csv
   └─ C ─► prepare_onehot_cache.py ─► {cache}_bases.npy                      │         │
                └─► train_onehot_rescnn.py  /  run_onehot_multiseed.sh ──────┘         ▼
                    (also reads labels and row order from B's prefix)       evaluation/evaluate_dream.py
```

Dependence of location:
* `prepare_embeddings.py` and `train_models.py` import `common.py`, so all three must stay in the same folder.
* `extract_ntv3_tokens.py` does `from token_common import *` and needs `token_common.py` next to it.
* Both shell scripts call their training script as `$BASE/<script>.py`, where `$BASE` is the folder the shell script lives in.
