# Using WholeToken embedding given by NTv3 for promoter-expression regression (DREAM 2022)

Due to nature of embedding with whole token, only ResCNN was used for prediction of yeast promoter expression. Evaluation follows the official **Random Promoter DREAM Challenge 2022** private-leaderboard logic.

---

## Pipeline

```
FASTA (>id|exp=value)
   │
   ▼  extract_ntv3_tokens.py (+ token_common.py)
NTv3-650M (frozen) → last hidden layer, 1 vector per nucleotide
   │  → Run metadata:  ntv3_*_run_config.json
   ▼  train_token_cnn.py
Perform TokenResCNN on the token matrices
   │  → Run metadata: metrics_ntv3_wholetoken.json
   ▼  evaluate_dream_official.py
Official DREAM subset scoring (weighted Pearson r² and Spearman ρ)
      
```

`token_common.py` is imported by `extract_ntv3_tokens.py`. Keep all scripts in the same folder.
