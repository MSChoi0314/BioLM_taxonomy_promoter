#!/usr/bin/env python3

import os
import re
import csv
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig


MODEL_NAME = "esmc_600m"
BATCH_SIZE = 1
USE_FULL_PRECISION = True
DEFAULT_FILE_EXTENSIONS = {".faa", ".fa", ".fasta"}
SPECIAL_AA_PATTERN = re.compile(r"[^ACDEFGHIKLMNPQRSTVWYBXZJUO]", re.IGNORECASE)


def setup_logging(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "protein_embedding.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return log_file


def parse_fasta(fasta_path: Path) -> List[Tuple[str, str, str]]:
    records = []
    header = None
    seq_chunks = []

    with fasta_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    sequence = "".join(seq_chunks).strip()
                    protein_id = extract_protein_id(header)
                    records.append((header, protein_id, sequence))
                header = line[1:]
                seq_chunks = []
            else:
                seq_chunks.append(line)

        if header is not None:
            sequence = "".join(seq_chunks).strip()
            protein_id = extract_protein_id(header)
            records.append((header, protein_id, sequence))

    return records


def extract_protein_id(header: str) -> str:
    return header.split()[0]


def sanitize_sequence(seq: str) -> str:
    seq = seq.upper().replace("*", "")
    seq = SPECIAL_AA_PATTERN.sub("X", seq)
    return seq


def get_model_max_residues(model) -> int:
    max_positions = getattr(model, "max_seq_len", 2048)
    if max_positions is None:
        return 2046

    effective_max = max_positions - 2
    if effective_max <= 0:
        return 2046
    return effective_max


def load_model(device: torch.device):
    logging.info(f"Loading ESMC model: {MODEL_NAME}")
    model = ESMC.from_pretrained(MODEL_NAME)

    if USE_FULL_PRECISION:
        model = model.float()

    model = model.to(device)
    model.eval()

    return model


@torch.no_grad()
def embed_sequence_mean_pool(
    sequence: str,
    model,
    device: torch.device,
    max_residues: int,
    protein_id: str,
    source_file: str,
) -> Tuple[List[float], int, bool]:
    original_length = len(sequence)
    truncated = False

    if original_length > max_residues:
        logging.warning(
            f"Truncating protein '{protein_id}' from file '{source_file}': "
            f"original_length={original_length}, truncated_to={max_residues}"
        )
        sequence = sequence[:max_residues]
        truncated = True

    protein = ESMProtein(sequence=sequence)
    protein_tensor = model.encode(protein)

    logits_output = model.logits(
        protein_tensor,
        LogitsConfig(sequence=True, return_embeddings=True),
    )
    embeddings = logits_output.embeddings

    seq_len = embeddings.shape[1]

    if seq_len <= 2:
        raise ValueError(
            f"Protein '{protein_id}' from '{source_file}' is too short after tokenization."
        )

    residue_embeddings = embeddings[:, 1:-1, :]
    pooled = residue_embeddings.mean(dim=1).squeeze(0)
    embedding = pooled.detach().cpu().tolist()

    return embedding, original_length, truncated


def write_embeddings_csv(
    output_csv: Path,
    genome_accession: str,
    rows: List[Dict],
    embedding_dim: int,
):
    fieldnames = ["protein_id", "genome_accession", "sequence_length", "truncated"] + [
        f"embedding_{i+1}" for i in range(embedding_dim)
    ]

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            out_row = {
                "protein_id": row["protein_id"],
                "genome_accession": genome_accession,
                "sequence_length": row["sequence_length"],
                "truncated": row["truncated"],
            }
            for i, val in enumerate(row["embedding"]):
                out_row[f"embedding_{i+1}"] = val
            writer.writerow(out_row)


def process_faa_file(
    faa_path: Path,
    output_dir: Path,
    model,
    device: torch.device,
    max_residues: int,
):
    genome_accession = faa_path.stem
    output_csv = output_dir / f"{genome_accession}_protein_embedding.csv"

    if output_csv.exists():
        logging.info(f"Skipping existing output: {output_csv}")
        return

    logging.info(f"Processing file: {faa_path}")
    records = parse_fasta(faa_path)

    if not records:
        logging.warning(f"No protein records found in file: {faa_path}")
        return

    rows = []
    embedding_dim = None
    success_count = 0
    fail_count = 0
    trunc_count = 0

    for header, protein_id, raw_seq in records:
        try:
            seq = sanitize_sequence(raw_seq)

            if not seq:
                raise ValueError("Empty sequence after sanitization.")

            embedding, seq_len, truncated = embed_sequence_mean_pool(
                sequence=seq,
                model=model,
                device=device,
                max_residues=max_residues,
                protein_id=protein_id,
                source_file=faa_path.name,
            )

            if embedding_dim is None:
                embedding_dim = len(embedding)

            rows.append(
                {
                    "protein_id": protein_id,
                    "sequence_length": seq_len,
                    "truncated": truncated,
                    "embedding": embedding,
                }
            )

            success_count += 1
            if truncated:
                trunc_count += 1

        except Exception as e:
            fail_count += 1
            logging.error(
                f"Failed protein '{protein_id}' in file '{faa_path.name}': {e}"
            )
            continue

    if not rows:
        logging.warning(f"No successful embeddings generated for file: {faa_path}")
        return

    write_embeddings_csv(
        output_csv=output_csv,
        genome_accession=genome_accession,
        rows=rows,
        embedding_dim=embedding_dim,
    )

    logging.info(
        f"Finished file: {faa_path.name} | "
        f"success={success_count}, failed={fail_count}, truncated={trunc_count}, "
        f"output={output_csv}"
    )


def collect_faa_files(input_dir: Path) -> List[Path]:
    files = []
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in DEFAULT_FILE_EXTENSIONS:
            files.append(p)
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-protein ESMC 600M embeddings from multi-protein .faa files."
    )
    parser.add_argument(
        "-i", "--input_dir", required=True,
        help="Directory containing input .faa files"
    )
    parser.add_argument(
        "-o", "--output_dir", required=True,
        help="Directory to write output CSV files"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist or is not a directory: {input_dir}")

    log_file = setup_logging(output_dir)
    logging.info("Starting protein embedding job")
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Model: {MODEL_NAME}")
    logging.info(f"Batch size: {BATCH_SIZE}")
    logging.info(f"Precision: float32 (full precision)")
    logging.info(f"Log file: {log_file}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available. This script is configured to use GPU.")

    device = torch.device("cuda")
    logging.info(f"Using device: {device}")
    logging.info(f"GPU name: {torch.cuda.get_device_name(0)}")

    model = load_model(device=device)
    max_residues = get_model_max_residues(model)
    logging.info(f"Maximum supported residues before truncation: {max_residues}")

    faa_files = collect_faa_files(input_dir)
    if not faa_files:
        logging.warning(f"No FASTA files found in: {input_dir}")
        return

    logging.info(f"Found {len(faa_files)} input FASTA files")

    for faa_path in faa_files:
        try:
            process_faa_file(
                faa_path=faa_path,
                output_dir=output_dir,
                model=model,
                device=device,
                max_residues=max_residues,
            )
            torch.cuda.empty_cache()
        except Exception as e:
            logging.error(f"Failed file '{faa_path.name}': {e}")
            continue

    logging.info("All jobs finished.")


if __name__ == "__main__":
    main()

