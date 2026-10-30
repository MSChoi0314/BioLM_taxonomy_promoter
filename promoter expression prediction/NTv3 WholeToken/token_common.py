#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
import time
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np

IUPAC_DNA = set("ACGTRYSWKMBDHVN")
EXPRESSION_HEADER_RE = re.compile(
    r"^(?P<promoter_id>[^|\s]+)\|exp="
    r"(?P<expression>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?:\s.*)?$"
)
STATUS_PENDING = np.uint8(0)
STATUS_SUCCESS = np.uint8(1)
STATUS_FAILED = np.uint8(2)


def parse_fasta_header(header: str, line_number: int, fasta_path: Path) -> Tuple[str, float]:
    m = EXPRESSION_HEADER_RE.fullmatch(header.strip())
    if m is None:
        raise ValueError(
            f"Invalid FASTA header at line {line_number}: {header!r}\n"
            "Expected: >promoter_id|exp=<numeric_expression>"
        )
    exp = float(m.group("expression"))
    if not np.isfinite(exp):
        raise ValueError(f"Non-finite expression at line {line_number}")
    return m.group("promoter_id"), exp


def iterate_fasta(fasta_path: Path) -> Iterator[Tuple[int, str, float, str]]:
    promoter_id = None
    expression = None
    chunks: list[str] = []
    source_row_index = -1
    with fasta_path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if promoter_id is not None:
                    yield source_row_index, promoter_id, float(expression), "".join(chunks)
                source_row_index += 1
                promoter_id, expression = parse_fasta_header(line[1:].strip(), line_number, fasta_path)
                chunks = []
            else:
                if promoter_id is None:
                    raise ValueError(f"Sequence before FASTA header at line {line_number}")
                chunks.append(line)
    if promoter_id is not None:
        yield source_row_index, promoter_id, float(expression), "".join(chunks)


def count_fasta_records(fasta_path: Path) -> int:
    n = 0
    with fasta_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.lstrip().startswith(">"):
                n += 1
    return n


def clean_dna(raw_sequence: str) -> str:
    seq = re.sub(r"\s+", "", raw_sequence).upper().replace("U", "T")
    seq = seq.replace("-", "").replace(".", "")
    bad = sorted(set(seq) - IUPAC_DNA)
    if bad:
        raise ValueError(f"Invalid DNA symbols: {bad}")
    if not seq:
        raise ValueError("Empty sequence after cleaning")
    return seq


def choose_source_indices(total: int, sample_size: int | None, seed: int) -> np.ndarray:
    if sample_size is None or sample_size >= total:
        return np.arange(total, dtype=np.int64)
    if sample_size <= 0:
        raise ValueError("--sample-size must be >0")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=sample_size, replace=False).astype(np.int64))


def human_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024
    return f"{x:.2f} PiB"


def fmt_time(sec: float) -> str:
    sec = max(0, int(sec))
    d, rem = divmod(sec, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    return f"{d}d {h:02d}:{m:02d}:{s:02d}" if d else f"{h:02d}:{m:02d}:{s:02d}"


def build_paths(prefix: Path) -> dict[str, Path]:
    p = prefix.parent
    s = prefix.name
    return {
        "tokens": p / f"{s}_tokens.npy",
        "mask": p / f"{s}_mask.npy",
        "expressions": p / f"{s}_expressions.npy",
        "source_rows": p / f"{s}_source_row_index.npy",
        "token_counts": p / f"{s}_token_counts.npy",
        "status": p / f"{s}_status.npy",
        "meta": p / f"{s}_meta.csv",
        "config": p / f"{s}_run_config.json",
        "warnings": p / f"{s}_warnings.log",
    }


def init_arrays(paths: dict[str, Path], n: int, max_positions: int, dim: int, dtype: np.dtype):
    tokens = np.lib.format.open_memmap(paths["tokens"], mode="w+", dtype=dtype, shape=(n, max_positions, dim))
    mask = np.lib.format.open_memmap(paths["mask"], mode="w+", dtype=np.uint8, shape=(n, max_positions))
    expressions = np.lib.format.open_memmap(paths["expressions"], mode="w+", dtype=np.float32, shape=(n,))
    source_rows = np.lib.format.open_memmap(paths["source_rows"], mode="w+", dtype=np.int64, shape=(n,))
    token_counts = np.lib.format.open_memmap(paths["token_counts"], mode="w+", dtype=np.int16, shape=(n,))
    status = np.lib.format.open_memmap(paths["status"], mode="w+", dtype=np.uint8, shape=(n,))
    tokens[:] = 0
    mask[:] = 0
    expressions[:] = np.nan
    source_rows[:] = -1
    token_counts[:] = 0
    status[:] = STATUS_PENDING
    for arr in (tokens, mask, expressions, source_rows, token_counts, status):
        arr.flush()
    return tokens, mask, expressions, source_rows, token_counts, status


def load_arrays(paths: dict[str, Path], n: int, max_positions: int, dim: int, dtype: np.dtype):
    required = ["tokens", "mask", "expressions", "source_rows", "token_counts", "status", "meta"]
    missing = [str(paths[k]) for k in required if not paths[k].exists()]
    if missing:
        raise FileNotFoundError("Missing resume files:\n" + "\n".join(missing))
    tokens = np.lib.format.open_memmap(paths["tokens"], mode="r+")
    mask = np.lib.format.open_memmap(paths["mask"], mode="r+")
    expressions = np.lib.format.open_memmap(paths["expressions"], mode="r+")
    source_rows = np.lib.format.open_memmap(paths["source_rows"], mode="r+")
    token_counts = np.lib.format.open_memmap(paths["token_counts"], mode="r+")
    status = np.lib.format.open_memmap(paths["status"], mode="r+")
    if tokens.shape != (n, max_positions, dim):
        raise ValueError(f"Token shape mismatch: {tokens.shape} != {(n,max_positions,dim)}")
    if tokens.dtype != dtype:
        raise ValueError(f"Token dtype mismatch: {tokens.dtype} != {dtype}")
    if mask.shape != (n, max_positions) or expressions.shape != (n,) or source_rows.shape != (n,) or token_counts.shape != (n,) or status.shape != (n,):
        raise ValueError("Resume array shape mismatch")
    return tokens, mask, expressions, source_rows, token_counts, status


def write_preselected_metadata(fasta_path: Path, selected: np.ndarray, metadata_path: Path):
    selected_set = set(map(int, selected))
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with metadata_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["row_index", "source_row_index", "promoter_id", "expression", "raw_sequence_length"])
        for source_idx, promoter_id, exp, raw_seq in iterate_fasta(fasta_path):
            if source_idx not in selected_set:
                continue
            w.writerow([written, source_idx, promoter_id, exp, len(re.sub(r"\s+", "", raw_seq))])
            written += 1
    if written != len(selected):
        raise RuntimeError(f"Metadata selected count mismatch: {written} != {len(selected)}")


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)
    tmp.replace(path)


def estimate_storage(n: int, max_positions: int, dim: int, dtype: np.dtype) -> int:
    return n * max_positions * dim * dtype.itemsize


def extraction_loop(
    *, fasta_path: Path, selected: np.ndarray, paths: dict[str, Path], arrays,
    extractor, clean_fn, flush_interval: int, progress_interval: int, resume: bool,
    retry_failed: bool, max_positions: int, storage_dtype: np.dtype,
):
    tokens, mask, expressions, source_rows, token_counts, status = arrays
    selected_to_output = {int(src): i for i, src in enumerate(selected)}
    start = time.time(); new = succ = fail = skip = 0
    warning_mode = "a" if resume else "w"
    with paths["warnings"].open(warning_mode, encoding="utf-8", buffering=1) as wh:
        for source_idx, promoter_id, exp, raw_seq in iterate_fasta(fasta_path):
            out_idx = selected_to_output.get(source_idx)
            if out_idx is None:
                continue
            cur = status[out_idx]
            if cur == STATUS_SUCCESS or (cur == STATUS_FAILED and not retry_failed):
                skip += 1
                continue
            expressions[out_idx] = np.float32(exp)
            source_rows[out_idx] = np.int64(source_idx)
            try:
                seq = clean_fn(raw_seq)
                h = extractor(seq, promoter_id)
                if h.ndim != 2:
                    raise ValueError(f"Expected [positions, dim], got {h.shape}")
                L = int(h.shape[0])
                if L <= 0 or L > max_positions:
                    raise ValueError(f"token_count={L} exceeds max_positions={max_positions}")
                if not np.isfinite(h).all():
                    raise ValueError("Non-finite token embedding")
                tokens[out_idx, :, :] = 0
                tokens[out_idx, :L, :] = h.astype(storage_dtype, copy=False)
                mask[out_idx, :] = 0
                mask[out_idx, :L] = 1
                token_counts[out_idx] = np.int16(L)
                status[out_idx] = STATUS_SUCCESS
                succ += 1
            except Exception as e:
                tokens[out_idx, :, :] = 0
                mask[out_idx, :] = 0
                token_counts[out_idx] = 0
                status[out_idx] = STATUS_FAILED
                fail += 1
                msg = f"[WARN] failed out_row={out_idx} source_row={source_idx} promoter_id={promoter_id}: {type(e).__name__}: {e}"
                print(msg); wh.write(msg + "\n")
            new += 1
            if new % flush_interval == 0:
                for arr in arrays: arr.flush()
                print(f"[FLUSH] processed={new:,}")
            if new % progress_interval == 0:
                elapsed_s = time.time() - start
                rate = new / max(elapsed_s, 1e-9)
                rem = max(0, len(selected) - skip - new)
                print(f"[PROGRESS] new={new:,}/{len(selected):,} success={succ:,} failed={fail:,} skipped={skip:,} rate={rate:.2f}/s elapsed={fmt_time(elapsed_s)} ETA={fmt_time(rem/max(rate,1e-9))}")
    for arr in arrays: arr.flush()
    return {"new_processed": new, "new_success": succ, "new_failed": fail, "skipped": skip, "elapsed_seconds": time.time()-start}
