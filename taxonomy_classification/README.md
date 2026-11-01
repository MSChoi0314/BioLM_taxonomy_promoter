# Type I methanotrophs taxonomy from protein language model embeddings

Protein language model (pLM) embeddings are used to taxonomy classification of Type I methanotroph genomes. Every protein in a genome is embedded with a pLM, the protein vectors are averaged into one vector per genome.

The genome vectors are then analysed in two ways:
1. HDBSCAN builds a density hierarchy of the genomes, and the hierarchy is compared against GTDB family, genus and species labels.
2. Linear classifiers are trained to predict GTDB genus and family under nested cross-validation, with each GTDB species kept inside a single fold.

## Pipeline
```
{accession}.faa                       one protein FASTA per genome (not included)
      │
      │  01_protein_embedding/*_protein_embed.py          run once per model
      ▼
{accession}_protein_embedding.csv     one row per protein
      │
      │  02_genome_mean_pooling/mean_pooling_embedding.py
      ▼  ◄── Add family and genus information
00.DATASET/methcocca_{model}_mean_pool_embed_general.csv     one row per genome, 4 files
      │
      ├──► 03 methcocca_hdbscan_graphml.py ◄── data/GTDB_accession.xlsx
      │         │
      │         │  {model}/{subset}/condensed_tree.graphml + sample_metadata.tsv
      │         ▼
      │    03 methcocca_hdbscan_taxonomy_validation.py ◄── data/GTDB_accession.xlsx
      │
      └──► 04 methcocca_gtdb_grouped_probe.py ◄── data/GTDB_accession_species_derep.xlsx
```

## Notes
* The protein FASTA files are not included. Download them from NCBI and rename each one to `{assembly_accession}.faa` before step 1.
* The `--help` text of the validation script mentions `methcocca_hdbscan_graphml_v3.py` and `GTDB_accession_V3.xlsx`. These are the files named `methcocca_hdbscan_graphml.py` and `data/GTDB_accession.xlsx` in this repository.

Notes on `data/` :
* "unknown" in etc column of `data/GTDB_accession.xlsx` represents accession is not found in GTDB.
* Accession GCF_016925495.1 is shown as genera EFPC2 according to GTDB taxonomy.
