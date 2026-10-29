#!/usr/bin/env python3
from __future__ import annotations
import json, os, random, tempfile, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def atomic_json_dump(obj, path: Path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as h:
            json.dump(obj, h, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def elapsed(start: float) -> str:
    sec = int(time.time() - start); d, sec = divmod(sec, 86400); h, sec = divmod(sec, 3600); m, s = divmod(sec, 60)
    return (f'{d}d ' if d else '') + f'{h:02d}:{m:02d}:{s:02d}'


def regression_metrics(y_true, y_pred):
    y = np.asarray(y_true, dtype=np.float64); p = np.asarray(y_pred, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(p); y, p = y[ok], p[ok]
    if len(y) < 2 or np.std(y) == 0 or np.std(p) == 0:
        pr = sr = float('nan')
    else:
        pr = float(pearsonr(y, p).statistic); sr = float(spearmanr(y, p).statistic)
    return {
        'n': int(len(y)), 'pearson_r': pr,
        'pearson_r2': float(pr * pr) if np.isfinite(pr) else float('nan'),
        'spearman_rho': sr,
        'r2_cod': float(r2_score(y, p)) if len(y) >= 2 else float('nan'),
        'mae': float(mean_absolute_error(y, p)) if len(y) else float('nan'),
        'rmse': float(np.sqrt(mean_squared_error(y, p))) if len(y) else float('nan'),
    }

@dataclass
class RunningMoments:
    n: int; mean: np.ndarray; m2: np.ndarray
    @classmethod
    def create(cls, d): return cls(0, np.zeros(d, np.float64), np.zeros(d, np.float64))
    def update(self, x):
        x = np.asarray(x, np.float64)
        if not len(x): return
        bn = len(x); bm = x.mean(0); bm2 = ((x - bm) ** 2).sum(0)
        if self.n == 0: self.n, self.mean, self.m2 = bn, bm, bm2; return
        delta = bm - self.mean; total = self.n + bn
        self.m2 += bm2 + delta * delta * self.n * bn / total
        self.mean += delta * bn / total; self.n = total
    def finalize(self):
        var = self.m2 / max(1, self.n - 1); std = np.sqrt(np.maximum(var, 0)); std[std < 1e-8] = 1.0
        return self.mean.astype(np.float32), std.astype(np.float32)


def open_prepared(path: Path):
    path = Path(path)
    x = np.load(path/'X.npy', mmap_mode='r'); y = np.load(path/'y.npy', mmap_mode='r')
    source = np.load(path/'source_row_index.npy', mmap_mode='r')
    split = np.load(path/'split.npy', mmap_mode='r') if (path/'split.npy').exists() else None
    meta = __import__('pandas').read_csv(path/'meta.csv')
    if len(x) != len(y) or len(x) != len(source) or len(x) != len(meta): raise RuntimeError(f'Prepared row mismatch: {path}')
    return x, y, source, split, meta
