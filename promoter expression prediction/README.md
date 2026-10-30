# Genome Language embeddings for promoter-expression regression (DREAM 2022)

Predict yeast promoter expression from DNA sequence by extracting embeddings, pooling them per sequence, and training lightweight regression heads (ResMLP, TabM; Ridge and LightGBM also supported). Evaluation follows the official **Random Promoter DREAM Challenge 2022** private-leaderboard logic.
NTv3 WholeToken uses separate method of embedding and model training, therefore uses separate python files inside the directory 

---

## Pipeline

```
FASTA (>id|exp=value)
   │
   ▼  scripts/run_*_promoter_fasta.py
Genome language model provides embedding → mean or max pooling
   │  → intermediate:  *_run_config.json
   ▼  scripts/prepare_embeddings.py
Validate rows, keep successful/finite ones, random train/val split (default 5% val)
   │  
   ▼  scripts/train_models.py
Standardize features → train ResMLP / TabM / Ridge / LightGBM → predict external test set
   │  → intermediate: run_config_*.json, metrics_*.json
   ▼  scripts/evaluate_dream.py
Official DREAM subset scoring (weighted Pearson r² and Spearman ρ)
      
```

`scripts/common.py` holds shared helpers (metrics, streaming feature statistics, atomic JSON writes, seeding) and is imported by `prepare_embeddings.py` and `train_models.py`. Keep all scripts in the same folder.
