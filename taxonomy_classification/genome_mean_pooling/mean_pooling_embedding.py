import argparse
import glob
import os
import re
import pandas as pd


def process_embedding_files(input_dir: str, output_csv: str) -> None:
    search_pattern = os.path.join(input_dir, "*_protein_embedding.csv")
    csv_files = sorted(glob.glob(search_pattern))

    if not csv_files:
        print(f"No files matching '*_protein_embedding.csv' found in: {input_dir}")
        return

    print(f"Found {len(csv_files)} files to process...")

    results = []

    for i, filepath in enumerate(csv_files, 1):
        filename = os.path.basename(filepath)

        accession_match = re.search(r"^(.*)_protein_embedding\.csv$", filename)
        accession = accession_match.group(1) if accession_match else filename

        try:
            df = pd.read_csv(filepath)

            if df.empty:
                print(f"[{i}/{len(csv_files)}] Warning: {filename} is empty. Skipping.")
                continue

            protein_count = len(df)

            truncated_col = df.iloc[:, 3]
            truncated_count = (
                truncated_col.astype(str).str.strip().str.upper() == "TRUE"
            ).sum()

            embedding_data = df.iloc[:, 4:].apply(pd.to_numeric, errors="coerce")
            mean_embeddings = embedding_data.mean(axis=0)

            row = {
                "genome_accession": accession,
                "protein_count": protein_count,
                "truncated_protein": truncated_count,
            }

            for col_name, mean_val in mean_embeddings.items():
                row[col_name] = mean_val

            results.append(row)

            if i % 100 == 0 or i == len(csv_files):
                print(f"[{i}/{len(csv_files)}] Processed: {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if results:
        out_df = pd.DataFrame(results)

        output_dir = os.path.dirname(output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        out_df.to_csv(output_csv, index=False)
        print(f"\nProcessing complete! Saved aggregated summary to: {output_csv}")
        print(f"Total genome accessions processed: {len(out_df)}")
        print(f"Output table shape: {out_df.shape[0]} rows x {out_df.shape[1]} columns")
    else:
        print("No records were successfully processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mean-pool protein PLM embeddings by genome accession."
    )
    parser.add_argument(
        "--input_dir",
        "-i",
        type=str,
        required=True,
        help="Path to directory containing (genomic_accession)_protein_embedding.csv files",
    )
    parser.add_argument(
        "--output_csv",
        "-o",
        type=str,
        default="genome_mean_pooled_embeddings.csv",
        help="Path to the output aggregated CSV file (default: genome_mean_pooled_embeddings.csv)",
    )

    args = parser.parse_args()
    process_embedding_files(args.input_dir, args.output_csv)

