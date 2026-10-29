#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch
from evo2 import Evo2


# =============================================================================
# Constants
# =============================================================================

IUPAC_DNA = set("ACGTRYSWKMBDHVN")

EXPRESSION_HEADER_RE = re.compile(
    r"^(?P<promoter_id>[^|\s]+)\|exp="
    r"(?P<expression>[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?)"
    r"(?:\s.*)?$"
)

MODEL_NAME = "evo2_7b"
LAYER_NAME = "blocks.28.mlp.l3"
EXPECTED_DIM = 4096

STATUS_PENDING = np.uint8(0)
STATUS_SUCCESS = np.uint8(1)
STATUS_FAILED = np.uint8(2)


# =============================================================================
# FASTA parsing
# =============================================================================

def parse_fasta_header(
    header: str,
    line_number: int,
    fasta_path: Path,
) -> Tuple[str, float]:
    """
    FASTA header 형식:

        >promoter_id|exp=<numeric_expression>
    """
    match = EXPRESSION_HEADER_RE.fullmatch(header.strip())

    if match is None:
        raise ValueError(
            f"Invalid FASTA header at line {line_number}: {header!r}\n"
            "Expected format: >promoter_id|exp=<numeric_expression>\n"
            f"FASTA: {fasta_path}"
        )

    promoter_id = match.group("promoter_id")
    expression_text = match.group("expression")
    expression = float(expression_text)

    if not np.isfinite(expression):
        raise ValueError(
            f"Expression must be finite at line {line_number}: "
            f"{expression_text!r}"
        )

    return promoter_id, expression


def iterate_fasta(
    fasta_path: Path,
) -> Iterator[Tuple[str, float, str]]:
    """
    FASTA 전체를 RAM에 올리지 않고 한 레코드씩 반환합니다.

    Yields
    ------
    promoter_id, expression, raw_sequence
    """
    promoter_id: str | None = None
    expression: float | None = None
    sequence_chunks: list[str] = []

    with fasta_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if promoter_id is not None:
                    yield (
                        promoter_id,
                        float(expression),
                        "".join(sequence_chunks),
                    )

                header = line[1:].strip()

                if not header:
                    raise ValueError(
                        f"Empty FASTA header at line {line_number}: "
                        f"{fasta_path}"
                    )

                promoter_id, expression = parse_fasta_header(
                    header=header,
                    line_number=line_number,
                    fasta_path=fasta_path,
                )

                sequence_chunks = []

            else:
                if promoter_id is None:
                    raise ValueError(
                        "Sequence data was found before the first FASTA "
                        f"header at line {line_number}: {fasta_path}"
                    )

                sequence_chunks.append(line)

    if promoter_id is not None:
        yield (
            promoter_id,
            float(expression),
            "".join(sequence_chunks),
        )


def count_fasta_records(fasta_path: Path) -> int:
    """
    FASTA header 수를 세어 전체 레코드 수를 확인합니다.
    """
    count = 0

    with fasta_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if raw_line.lstrip().startswith(">"):
                count += 1

    return count


# =============================================================================
# Sequence preprocessing
# =============================================================================

def write_warning(warning_handle, message: str) -> None:
    print(message)
    warning_handle.write(message + "\n")
    warning_handle.flush()


def clean_dna_sequence(
    raw_sequence: str,
    promoter_id: str,
    warning_handle,
) -> str:
    """
    DNA sequence 전처리:

    - 공백 제거
    - 대문자 변환
    - U -> T
    - '-' 및 '.' 제거
    - IUPAC DNA 문자 검사
    """
    sequence = re.sub(r"\s+", "", raw_sequence).upper()

    if "U" in sequence:
        write_warning(
            warning_handle,
            f"[WARN] U converted to T: {promoter_id}",
        )
        sequence = sequence.replace("U", "T")

    if "-" in sequence or "." in sequence:
        write_warning(
            warning_handle,
            f"[WARN] gap characters removed: {promoter_id}",
        )
        sequence = sequence.replace("-", "").replace(".", "")

    invalid_characters = sorted(set(sequence) - IUPAC_DNA)

    if invalid_characters:
        raise ValueError(
            f"Invalid DNA character(s): {invalid_characters}. "
            "Allowed IUPAC symbols are ACGTRYSWKMBDHVN."
        )

    return sequence


# =============================================================================
# Device and pooling
# =============================================================================

def choose_device(requested_device: str) -> torch.device:
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("[INFO] CUDA is unavailable. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device(requested_device)


def pool_embeddings(
    valid_embeddings: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    """
    [positions, hidden_dim] embedding을 pooling합니다.
    """
    if valid_embeddings.ndim != 2:
        raise ValueError(
            "Expected 2D embeddings, observed shape="
            f"{tuple(valid_embeddings.shape)}"
        )

    if valid_embeddings.shape[0] == 0:
        raise ValueError(
            "No valid nucleotide position remained for pooling."
        )

    # pooling 계산은 float32로 수행
    valid_embeddings = valid_embeddings.float()

    if pooling == "mean":
        return valid_embeddings.mean(dim=0)

    if pooling == "max":
        return valid_embeddings.max(dim=0).values

    raise ValueError(
        f"Unsupported pooling method: {pooling!r}"
    )


# =============================================================================
# Evo2 embedding extraction
# =============================================================================

def normalize_evo2_embedding_tensor(
    tensor,
    sequence_length: int,
) -> torch.Tensor:
    """
    Evo2 embedding 반환 형식을 [sequence_length, 4096]으로 정규화합니다.
    """
    if isinstance(tensor, (tuple, list)):
        if len(tensor) != 1:
            raise ValueError(
                "Unexpected tuple/list embedding output: "
                f"length={len(tensor)}"
            )

        tensor = tensor[0]

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            "Evo2 embedding output is not a torch.Tensor: "
            f"{type(tensor).__name__}"
        )

    if tensor.ndim == 3:
        if tensor.shape[0] != 1:
            raise ValueError(
                "Expected batch size 1, observed shape="
                f"{tuple(tensor.shape)}"
            )

        tensor = tensor.squeeze(0)

    if tensor.ndim != 2:
        raise ValueError(
            "Expected 2D or batch-1 3D tensor, observed shape="
            f"{tuple(tensor.shape)}"
        )

    if tensor.shape[-1] == EXPECTED_DIM:
        position_embeddings = tensor

    elif tensor.shape[0] == EXPECTED_DIM:
        position_embeddings = tensor.transpose(0, 1)

    else:
        raise ValueError(
            "Cannot identify 4096-dimensional embedding axis: "
            f"shape={tuple(tensor.shape)}"
        )

    if position_embeddings.shape[0] != sequence_length:
        raise ValueError(
            "Evo2 token/sequence length mismatch: "
            f"sequence={sequence_length}, "
            f"embedding_positions={position_embeddings.shape[0]}"
        )

    return position_embeddings


def extract_evo2_embedding(
    sequence: str,
    promoter_id: str,
    model,
    device: torch.device,
    pooling: str,
) -> np.ndarray:
    """
    Evo2 지정 레이어에서 nucleotide-level embedding을 추출하고
    mean 또는 max pooling합니다.
    """
    tokenized = model.tokenizer.tokenize(sequence)

    token_ids = torch.tensor(
        tokenized,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    if token_ids.shape[1] != len(sequence):
        raise ValueError(
            f"Evo2 tokenizer length mismatch for {promoter_id}: "
            f"sequence={len(sequence)}, "
            f"tokens={token_ids.shape[1]}"
        )

    with torch.inference_mode():
        _, embeddings = model(
            token_ids,
            return_embeddings=True,
            layer_names=[LAYER_NAME],
        )

    if not isinstance(embeddings, dict):
        raise TypeError(
            "Evo2 embeddings output must be a dictionary, observed "
            f"{type(embeddings).__name__}"
        )

    if LAYER_NAME not in embeddings:
        raise KeyError(
            f"Requested layer was not returned: {LAYER_NAME}. "
            f"Available keys={list(embeddings.keys())}"
        )

    position_embeddings = normalize_evo2_embedding_tensor(
        tensor=embeddings[LAYER_NAME],
        sequence_length=len(sequence),
    )

    pooled = pool_embeddings(
        valid_embeddings=position_embeddings,
        pooling=pooling,
    )

    vector = pooled.detach().cpu().numpy()

    if vector.ndim != 1:
        raise ValueError(
            "Expected a 1D pooled embedding, observed "
            f"shape={vector.shape}"
        )

    if vector.shape[0] != EXPECTED_DIM:
        raise ValueError(
            "Unexpected pooled embedding dimension: "
            f"{vector.shape[0]} "
            f"(expected {EXPECTED_DIM})"
        )

    return vector


# =============================================================================
# Output paths
# =============================================================================

def build_output_paths(output_prefix: Path) -> dict[str, Path]:
    parent = output_prefix.parent
    stem = output_prefix.name

    return {
        "embeddings": parent / f"{stem}_embeddings.npy",
        "expressions": parent / f"{stem}_expressions.npy",
        "status": parent / f"{stem}_status.npy",
        "meta": parent / f"{stem}_meta.csv",
        "warnings": parent / f"{stem}_warnings.log",
        "config": parent / f"{stem}_run_config.json",
    }


# =============================================================================
# Metadata and configuration
# =============================================================================

def write_metadata_csv(
    fasta_path: Path,
    metadata_path: Path,
    total_records: int,
) -> None:
    """
    NPY의 row_index와 정확히 대응하는 metadata CSV를 생성합니다.
    """
    print(f"[META] Creating metadata CSV: {metadata_path}")

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    written_records = 0

    with metadata_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow(
            [
                "row_index",
                "promoter_id",
                "expression",
                "raw_sequence_length",
            ]
        )

        for row_index, (
            promoter_id,
            expression,
            raw_sequence,
        ) in enumerate(iterate_fasta(fasta_path)):

            raw_sequence_length = len(
                re.sub(r"\s+", "", raw_sequence)
            )

            writer.writerow(
                [
                    row_index,
                    promoter_id,
                    expression,
                    raw_sequence_length,
                ]
            )

            written_records += 1

            if written_records % 100000 == 0:
                print(
                    f"[META] "
                    f"{written_records:,}/{total_records:,}"
                )

    if written_records != total_records:
        raise RuntimeError(
            "Metadata record count mismatch: "
            f"expected={total_records:,}, "
            f"written={written_records:,}"
        )

    print(
        f"[META] Saved: {metadata_path} | "
        f"n_records={written_records:,}"
    )


def write_run_config(
    config_path: Path,
    args,
    paths: dict[str, Path],
    total_records: int,
    storage_dtype: np.dtype,
) -> None:
    """
    실행 조건과 결과 파일 경로를 JSON으로 저장합니다.
    """
    config = {
        "input_fasta": str(
            Path(args.input_fasta).resolve()
        ),
        "output_prefix": str(
            Path(args.output_prefix).resolve()
        ),
        "model_name": MODEL_NAME,
        "layer_name": LAYER_NAME,
        "embedding_dimension": EXPECTED_DIM,
        "number_of_records": total_records,
        "device": args.device,
        "storage_dtype": str(storage_dtype),
        "pooling": args.pooling,
        "flush_interval": args.flush_interval,
        "progress_interval": args.progress_interval,
        "empty_cache_interval": args.empty_cache_interval,
        "retry_failed": args.retry_failed,
        "files": {
            key: str(value.resolve())
            for key, value in paths.items()
        },
        "status_codes": {
            "0": "pending",
            "1": "success",
            "2": "failed",
        },
        "representation": LAYER_NAME,
    }

    config_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            config,
            handle,
            indent=2,
            ensure_ascii=False,
        )


# =============================================================================
# NPY initialization and resume
# =============================================================================

def initialize_output_arrays(
    paths: dict[str, Path],
    total_records: int,
    storage_dtype: np.dtype,
) -> tuple[np.memmap, np.memmap, np.memmap]:
    """
    새로운 NPY memmap 파일을 생성합니다.
    """
    print("[INIT] Creating NPY memmap files")

    embeddings = np.lib.format.open_memmap(
        paths["embeddings"],
        mode="w+",
        dtype=storage_dtype,
        shape=(total_records, EXPECTED_DIM),
    )

    expressions = np.lib.format.open_memmap(
        paths["expressions"],
        mode="w+",
        dtype=np.float32,
        shape=(total_records,),
    )

    status = np.lib.format.open_memmap(
        paths["status"],
        mode="w+",
        dtype=np.uint8,
        shape=(total_records,),
    )

    embeddings[:] = np.nan
    expressions[:] = np.nan
    status[:] = STATUS_PENDING

    embeddings.flush()
    expressions.flush()
    status.flush()

    return embeddings, expressions, status


def load_existing_output_arrays(
    paths: dict[str, Path],
    total_records: int,
    storage_dtype: np.dtype,
) -> tuple[np.memmap, np.memmap, np.memmap]:
    """
    기존 NPY를 읽기/쓰기 모드로 열어 이어서 실행합니다.
    """
    required_paths = [
        paths["embeddings"],
        paths["expressions"],
        paths["status"],
        paths["meta"],
    ]

    missing_paths = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        missing_text = "\n".join(
            f"  - {path}"
            for path in missing_paths
        )

        raise FileNotFoundError(
            "Cannot resume because required files are missing:\n"
            f"{missing_text}"
        )

    embeddings = np.lib.format.open_memmap(
        paths["embeddings"],
        mode="r+",
    )

    expressions = np.lib.format.open_memmap(
        paths["expressions"],
        mode="r+",
    )

    status = np.lib.format.open_memmap(
        paths["status"],
        mode="r+",
    )

    expected_embedding_shape = (
        total_records,
        EXPECTED_DIM,
    )

    if embeddings.shape != expected_embedding_shape:
        raise RuntimeError(
            "Existing embedding shape mismatch: "
            f"expected={expected_embedding_shape}, "
            f"observed={embeddings.shape}"
        )

    if embeddings.dtype != storage_dtype:
        raise RuntimeError(
            "Existing embedding dtype mismatch: "
            f"expected={storage_dtype}, "
            f"observed={embeddings.dtype}"
        )

    if expressions.shape != (total_records,):
        raise RuntimeError(
            "Existing expression shape mismatch: "
            f"expected={(total_records,)}, "
            f"observed={expressions.shape}"
        )

    if status.shape != (total_records,):
        raise RuntimeError(
            "Existing status shape mismatch: "
            f"expected={(total_records,)}, "
            f"observed={status.shape}"
        )

    return embeddings, expressions, status


def check_output_conflicts(
    paths: dict[str, Path],
    overwrite: bool,
    resume: bool,
) -> None:
    if overwrite and resume:
        raise ValueError(
            "--overwrite and --resume cannot be used together."
        )

    existing_paths = [
        path
        for path in paths.values()
        if path.exists()
    ]

    if existing_paths and not overwrite and not resume:
        existing_text = "\n".join(
            f"  - {path}"
            for path in existing_paths
        )

        raise FileExistsError(
            "Output files already exist:\n"
            f"{existing_text}\n"
            "Use --resume to continue or --overwrite to recreate them."
        )

    if overwrite:
        for path in existing_paths:
            print(f"[OVERWRITE] Removing: {path}")
            path.unlink()


# =============================================================================
# Progress helpers
# =============================================================================

def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))

    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    if days > 0:
        return (
            f"{days}d {hours:02d}h "
            f"{minutes:02d}m {seconds:02d}s"
        )

    return (
        f"{hours:02d}h "
        f"{minutes:02d}m "
        f"{seconds:02d}s"
    )


def get_status_counts(
    status: np.ndarray,
) -> tuple[int, int, int]:
    pending = int(
        np.count_nonzero(status == STATUS_PENDING)
    )

    success = int(
        np.count_nonzero(status == STATUS_SUCCESS)
    )

    failed = int(
        np.count_nonzero(status == STATUS_FAILED)
    )

    return pending, success, failed


def print_status_summary(status: np.ndarray) -> None:
    pending, success, failed = get_status_counts(status)

    print("=" * 80)
    print("[STATUS]")
    print(f"Pending: {pending:,}")
    print(f"Success: {success:,}")
    print(f"Failed : {failed:,}")
    print(f"Total  : {len(status):,}")
    print("=" * 80)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Evo2 promoter embeddings and save them incrementally "
            "as NPY memmap files with metadata CSV."
        )
    )

    parser.add_argument(
        "--input_fasta",
        required=True,
        help=(
            "Input FASTA. Header format: "
            ">promoter_id|exp=<numeric_expression>"
        ),
    )

    parser.add_argument(
        "--output_prefix",
        required=True,
        help=(
            "Output file prefix. Example: "
            "/path/evo2_promoter_mean"
        ),
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
        help="PyTorch device. Default: cuda:0",
    )

    parser.add_argument(
        "--pooling",
        choices=["mean", "max"],
        default="mean",
    )

    parser.add_argument(
        "--storage_dtype",
        choices=["float32", "float16"],
        default="float32",
        help=(
            "NPY embedding storage dtype. "
            "float32 preserves more precision; float16 uses half the disk."
        ),
    )

    parser.add_argument(
        "--flush_interval",
        type=int,
        default=5000,
        help=(
            "Flush NPY files to disk every N newly processed rows. "
            "Default: 5000"
        ),
    )

    parser.add_argument(
        "--progress_interval",
        type=int,
        default=1000,
        help=(
            "Print progress every N newly processed rows. "
            "Default: 1000"
        ),
    )

    parser.add_argument(
        "--empty_cache_interval",
        type=int,
        default=0,
        help=(
            "Call torch.cuda.empty_cache() every N newly processed rows. "
            "0 disables it. Default: 0"
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from existing output files. "
            "Rows with status=1 are skipped."
        ),
    )

    parser.add_argument(
        "--retry_failed",
        action="store_true",
        help=(
            "With --resume, retry rows with status=2. "
            "Otherwise failed rows are skipped."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output files and start over.",
    )

    args = parser.parse_args()

    if args.flush_interval <= 0:
        raise ValueError(
            "--flush_interval must be greater than 0."
        )

    if args.progress_interval <= 0:
        raise ValueError(
            "--progress_interval must be greater than 0."
        )

    if args.empty_cache_interval < 0:
        raise ValueError(
            "--empty_cache_interval must be 0 or greater."
        )

    input_fasta = Path(args.input_fasta).resolve()
    output_prefix = Path(args.output_prefix).resolve()

    if not input_fasta.is_file():
        raise FileNotFoundError(
            f"Input FASTA not found: {input_fasta}"
        )

    output_prefix.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = build_output_paths(output_prefix)
    storage_dtype = np.dtype(args.storage_dtype)

    check_output_conflicts(
        paths=paths,
        overwrite=args.overwrite,
        resume=args.resume,
    )

    print(f"[COUNT] FASTA records: {input_fasta}")

    total_records = count_fasta_records(input_fasta)

    if total_records == 0:
        raise ValueError(
            f"No FASTA records found: {input_fasta}"
        )

    print(f"[COUNT] Total records: {total_records:,}")

    if args.resume:
        embeddings, expressions, status = (
            load_existing_output_arrays(
                paths=paths,
                total_records=total_records,
                storage_dtype=storage_dtype,
            )
        )

        print("[RESUME] Existing output files loaded.")
        print_status_summary(status)

    else:
        write_metadata_csv(
            fasta_path=input_fasta,
            metadata_path=paths["meta"],
            total_records=total_records,
        )

        embeddings, expressions, status = (
            initialize_output_arrays(
                paths=paths,
                total_records=total_records,
                storage_dtype=storage_dtype,
            )
        )

    write_run_config(
        config_path=paths["config"],
        args=args,
        paths=paths,
        total_records=total_records,
        storage_dtype=storage_dtype,
    )

    device = choose_device(args.device)

    if device.type != "cuda":
        print(
            "[WARN] Evo2 evo2_7b inference is intended for "
            "a supported CUDA GPU."
        )

    print(
        f"[LOAD] model={MODEL_NAME} "
        f"device={device} "
        f"pooling={args.pooling} "
        f"storage_dtype={storage_dtype} "
        f"layer={LAYER_NAME}"
    )

    # 원본 코드와 동일한 Evo2 모델 로딩 방식
    model = Evo2(MODEL_NAME)

    start_time = time.time()

    newly_processed = 0
    newly_successful = 0
    newly_failed = 0
    skipped = 0

    warning_mode = "a" if args.resume else "w"

    try:
        with paths["warnings"].open(
            warning_mode,
            encoding="utf-8",
            buffering=1,
        ) as warning_handle:

            for row_index, (
                promoter_id,
                expression,
                raw_sequence,
            ) in enumerate(iterate_fasta(input_fasta)):

                current_status = status[row_index]

                if current_status == STATUS_SUCCESS:
                    skipped += 1
                    continue

                if (
                    current_status == STATUS_FAILED
                    and not args.retry_failed
                ):
                    skipped += 1
                    continue

                display_index = row_index + 1

                print(
                    f"[{display_index}/{total_records}] "
                    f"{promoter_id} "
                    f"expression={expression}"
                )

                # 임베딩 성공 여부와 관계없이 expression 저장
                expressions[row_index] = np.float32(
                    expression
                )

                try:
                    sequence = clean_dna_sequence(
                        raw_sequence=raw_sequence,
                        promoter_id=promoter_id,
                        warning_handle=warning_handle,
                    )

                    if not sequence:
                        raise ValueError(
                            "Empty sequence after cleaning."
                        )

                    vector = extract_evo2_embedding(
                        sequence=sequence,
                        promoter_id=promoter_id,
                        model=model,
                        device=device,
                        pooling=args.pooling,
                    )

                    embeddings[row_index, :] = vector.astype(
                        storage_dtype,
                        copy=False,
                    )

                    # embedding 기록 이후 성공 상태로 변경
                    status[row_index] = STATUS_SUCCESS
                    newly_successful += 1

                except Exception as error:
                    embeddings[row_index, :] = np.nan
                    status[row_index] = STATUS_FAILED
                    newly_failed += 1

                    message = (
                        f"[WARN] failed: "
                        f"row_index={row_index} "
                        f"promoter_id={promoter_id} -> "
                        f"{type(error).__name__}: {error}"
                    )

                    write_warning(
                        warning_handle,
                        message,
                    )

                newly_processed += 1

                if (
                    device.type == "cuda"
                    and args.empty_cache_interval > 0
                    and newly_processed
                    % args.empty_cache_interval
                    == 0
                ):
                    torch.cuda.empty_cache()

                if (
                    newly_processed
                    % args.flush_interval
                    == 0
                ):
                    embeddings.flush()
                    expressions.flush()
                    status.flush()

                    print(
                        "[FLUSH] "
                        f"newly_processed={newly_processed:,}"
                    )

                if (
                    newly_processed
                    % args.progress_interval
                    == 0
                ):
                    elapsed = time.time() - start_time

                    average_seconds = (
                        elapsed / newly_processed
                        if newly_processed > 0
                        else 0.0
                    )

                    remaining_candidates = (
                        total_records
                        - skipped
                        - newly_processed
                    )

                    estimated_remaining = (
                        average_seconds
                        * max(0, remaining_candidates)
                    )

                    print(
                        "[PROGRESS] "
                        f"new={newly_processed:,} | "
                        f"success={newly_successful:,} | "
                        f"failed={newly_failed:,} | "
                        f"skipped={skipped:,} | "
                        f"elapsed={format_duration(elapsed)} | "
                        f"estimated_remaining="
                        f"{format_duration(estimated_remaining)}"
                    )

    except KeyboardInterrupt:
        print(
            "\n[INTERRUPTED] Keyboard interrupt received."
        )
        print(
            "[SAVE] Flushing current NPY files before exit."
        )

        embeddings.flush()
        expressions.flush()
        status.flush()

        print_status_summary(status)
        sys.exit(130)

    finally:
        # Python 정상 종료 또는 예외 종료 시 현재 결과를 반영
        embeddings.flush()
        expressions.flush()
        status.flush()

    elapsed = time.time() - start_time

    final_pending, final_success, final_failed = (
        get_status_counts(status)
    )

    print_status_summary(status)

    print("=" * 80)
    print("[COMPLETED]")
    print(f"Elapsed time : {format_duration(elapsed)}")
    print(f"Embeddings   : {paths['embeddings']}")
    print(f"Expressions  : {paths['expressions']}")
    print(f"Status       : {paths['status']}")
    print(f"Metadata     : {paths['meta']}")
    print(f"Warnings     : {paths['warnings']}")
    print(f"Run config   : {paths['config']}")
    print(f"Success      : {final_success:,}")
    print(f"Failed       : {final_failed:,}")
    print(f"Pending      : {final_pending:,}")
    print(f"Embedding dim: {EXPECTED_DIM}")
    print(f"Pooling      : {args.pooling}")
    print(f"Layer        : {LAYER_NAME}")
    print(f"Storage dtype: {storage_dtype}")
    print("=" * 80)

    if final_pending > 0:
        print(
            "[NOTICE] Pending rows remain. "
            "Run again with --resume."
        )

    if final_failed > 0:
        print(
            "[NOTICE] Failed rows remain. "
            "Run again with --resume --retry_failed "
            "to retry them."
        )


if __name__ == "__main__":
    main()