# Using WholeToken embedding given by NTv3 for promoter-expression regression (DREAM 2022)

Due to nature of embedding with whole token, only ResCNN was used for prediction of yeast promoter expression. Evaluation follows the official **Random Promoter DREAM Challenge 2022** private-leaderboard logic.

---

## Pipeline

```
FASTA (>id|exp=value)
   │
   ▼  extract_ntv3_tokens.py (+ token_common.py)
NTv3-650M (frozen) → last hidden layer, 1 vector per nucleotide
   │  → intermediate:  ntv3_*_run_config.json
   ▼  train_token_cnn.py
TokenResCNN on the token matrices
   │  → intermediate: metrics_ntv3_wholetoken.json
   ▼  scripts/evaluate_dream.py
Official DREAM subset scoring (weighted Pearson r² and Spearman ρ)
      
```

`scripts/common.py` holds shared helpers (metrics, streaming feature statistics, atomic JSON writes, seeding) and is imported by `prepare_embeddings.py` and `train_models.py`. Keep all scripts in the same folder.
