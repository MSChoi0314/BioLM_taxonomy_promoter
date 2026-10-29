## Pipeline

```
FASTA (>id|exp=value)
   │
   ▼  scripts/run_evo2_promoter_fasta.py
Evo2 layer blocks.28.mlp.l3 → per-nucleotide 4,096-d embeddings → mean or max pooling
   │   → *_run_config.json
   ▼  scripts/prepare_embeddings.py
Validate rows, keep successful/finite ones, random train/val split (default 5% val)
   │   
   ▼  scripts/train_models.py
Standardize features → train ResMLP / TabM / Ridge / LightGBM → predict external test set
   │  → metrics_*.json, run_config_*.json 
   ▼  scripts/evaluate_dream.py
Official DREAM subset scoring (weighted Pearson r² and Spearman ρ)
       outputs: official_scores.json, official_subset_metrics.csv
```

`scripts/common.py` holds shared helpers (metrics, streaming feature statistics, atomic JSON writes, seeding) and is imported by `prepare_embeddings.py` and `train_models.py`. Keep all scripts in the same folder.
