#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_recall_fscore_support)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

DESCRIPTION = """GTDB species-representative nested CV for frozen proteome embeddings.

No PCA/UMAP, no family merging, no automatic ungrouped fallback, no fold dropping.
The input metadata is an allow-list; embedded NCBI labels are never targets.
Exactly one preselected genome per GTDB species is required. No ANI is computed.
Quality-based representative selection must be completed before using this script.
"""

VERSION = "2.0.0-species-representative"
DEFAULT_FILES = {
    "ProtT5": "methcocca_prott5_mean_pool_embed_general.csv",
    "ESMC": "methcocca_esmc_mean_pool_embed_general.csv",
    "ProtBERT": "methcocca_protbert_mean_pool_embed_general.csv",
    "ESM2": "methcocca_esm2_mean_pool_embed_general.csv",
}
KEYS = ["task", "model", "probe"]


def seed_for(seed, *parts):
    digest = hashlib.sha256("|".join(map(str, (seed,) + parts)).encode()).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_tsv(data, path):
    pd.DataFrame(data).to_csv(path, sep="\t", index=False)


def accession(value):
    text = str(value).strip()
    text = re.sub(r"^(RS_|GB_)", "", text)
    if not re.fullmatch(r"GC[AF]_\d+\.\d+", text):
        raise ValueError(f"Invalid versioned assembly accession: {value!r}")
    return text


def clean_label(series, prefix):
    s = series.astype("string").str.strip().str.replace(rf"^{prefix}__", "", regex=True)
    invalid = s.str.lower().isin(["", "nan", "none", "na", "unclassified", "unknown", "other"])
    return s.mask(invalid)


def read_table(path, sheet=0):
    if str(path).lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path, sheet_name=sheet)
    return pd.read_csv(path, sep="\t" if str(path).lower().endswith(".tsv") else ",")


def load_metadata(path, sheet=0, group_map=None):
    d = read_table(path, sheet)
    aliases = {"Assembly Accession": "genome_accession", "Family": "family",
               "Genus": "genus", "Species": "species"}
    d = d.rename(columns=aliases)
    required = ["genome_accession", "family", "genus", "species"]
    if not set(required).issubset(d):
        raise ValueError(f"Metadata requires {required}; found {d.columns.tolist()}")
    d["genome_accession"] = d.genome_accession.map(accession)
    if d.genome_accession.duplicated().any():
        raise ValueError("Duplicate metadata accession; do not silently deduplicate.")
    for name, prefix in [("family", "f"), ("genus", "g"), ("species", "s")]:
        d[name] = clean_label(d[name], prefix)
    if group_map:
        mapping = read_table(group_map)
        if not {"genome_accession", "group"}.issubset(mapping):
            raise ValueError("Group map requires genome_accession and group columns.")
        mapping["genome_accession"] = mapping.genome_accession.map(accession)
        if mapping.genome_accession.duplicated().any():
            raise ValueError("Group map contains duplicate accessions.")
        mapping["group"] = clean_label(mapping.group, "unused")
        d = d.merge(mapping[["genome_accession", "group"]], on="genome_accession", how="left", validate="1:1")
    else:
        d["group"] = d.species
    d["group"] = d.group.astype("string")
    d = d.sort_values("genome_accession").reset_index(drop=True)
    for child, parent in [("genus", "family"), ("species", "genus"), ("species", "family")]:
        pairs = d.dropna(subset=[child, parent]).groupby(child)[parent].nunique()
        if (pairs > 1).any():
            raise ValueError(f"Inconsistent GTDB hierarchy: {child} maps to multiple {parent} labels.")
    return d


def check_pure_groups(y, groups):
    table = pd.DataFrame({"y": y, "group": groups})
    bad = table.groupby("group").y.nunique()
    if (bad > 1).any():
        raise ValueError("Groups cross target labels: " + ", ".join(bad[bad > 1].index.astype(str)) +
                         ". Resolve taxonomy/group conflicts; do not split these groups by target.")


def validate_representatives(meta):
    columns = ["family", "genus", "species", "group"]
    missing = meta[columns].isna().any(axis=1)
    if missing.any():
        ids = meta.loc[missing, "genome_accession"].tolist()
        raise ValueError(f"Species-representative metadata contains unresolved taxonomy: {ids}")
    duplicated = meta.species.duplicated(keep=False)
    if duplicated.any():
        detail = meta.loc[duplicated, ["genome_accession", "species"]].to_dict("records")
        raise ValueError(f"Exactly one genome per GTDB species is required. Duplicates: {detail}")
    if not meta.group.eq(meta.species).all():
        raise ValueError("This analysis uses GTDB species IDs as singleton groups.")


def partition_groups(y, groups, n_splits, seed):
    y, groups = np.asarray(y, str), np.asarray(groups, str)
    check_pure_groups(y, groups)
    rng = np.random.default_rng(seed)
    fold = np.full(len(y), -1, dtype=int)
    for label in sorted(set(y)):
        ids = sorted(set(groups[y == label]))
        if len(ids) < n_splits:
            raise ValueError(f"{label}: {len(ids)} groups cannot populate {n_splits} test folds.")
        rng.shuffle(ids)
        ids.sort(key=lambda g: -int(np.sum(groups == g)))
        ng = np.zeros(n_splits, dtype=int)
        ns = np.zeros(n_splits, dtype=int)
        for group in ids:
            candidates = np.flatnonzero(ng == ng.min())
            candidates = candidates[ns[candidates] == ns[candidates].min()]
            f = int(rng.choice(candidates))
            idx = np.flatnonzero(groups == group)
            fold[idx] = f
            ng[f] += 1
            ns[f] += len(idx)
    result = []
    for f in range(n_splits):
        tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        if set(groups[tr]) & set(groups[te]):
            raise RuntimeError("Group leakage detected.")
        if set(y[tr]) != set(y) or set(y[te]) != set(y):
            raise RuntimeError("Incomplete class coverage detected.")
        result.append((tr, te))
    if not np.array_equal(np.sort(np.concatenate([te for _, te in result])), np.arange(len(y))):
        raise RuntimeError("Incomplete or duplicate test coverage.")
    return result


def group_counts(y, groups):
    return pd.DataFrame({"y": y, "group": groups}).groupby("y").group.nunique()


def make_task(meta, task, args):
    rank = task
    reasons = pd.Series("", index=meta.index, dtype=object)
    reasons.loc[meta[rank].isna()] = f"missing_{rank}_label"
    reasons.loc[meta.group.isna()] = reasons.loc[meta.group.isna()].map(
        lambda r: (r + ";" if r else "") + "missing_group")
    candidate = meta.loc[reasons.eq("")].copy()
    candidate["target"] = candidate[rank].astype(str)
    check_pure_groups(candidate.target.to_numpy(), candidate.group.to_numpy())
    counts = candidate.groupby("target").agg(n_genomes=("genome_accession", "size"),
                                            n_species=("species", "nunique"), n_groups=("group", "nunique"))
    minimum = args.min_genus_genomes if task == "genus" else args.min_family_genomes
    counts["included"] = (counts.n_genomes >= minimum) & (counts.n_groups >= args.min_groups)
    counts["reason"] = ["included" if ok else ";".join(
        ([f"n_genomes<{minimum}"] if n < minimum else []) +
        ([f"n_groups<{args.min_groups}"] if g < args.min_groups else []))
        for n, g, ok in zip(counts.n_genomes, counts.n_groups, counts.included)]
    excluded = counts.loc[~counts.included, "reason"].to_dict()
    for idx in candidate.index:
        if candidate.at[idx, "target"] in excluded:
            reasons.at[idx] = excluded[candidate.at[idx, "target"]]
    audit = meta.copy()
    audit["included"] = reasons.eq("")
    audit["exclusion_reason"] = reasons
    task_df = candidate.loc[~candidate.target.isin(excluded)].reset_index(drop=True)
    return task_df, counts.reset_index(), audit


def plan_task(d, args, task):
    if d.target.nunique() < 2:
        raise ValueError(f"{task}: fewer than two eligible classes; inspect eligibility.tsv.")
    y, g = d.target.to_numpy(str), d.group.to_numpy(str)
    k = min(args.max_outer_splits, int(group_counts(y, g).min()))
    plans, audits = [], []
    for repeat in range(args.outer_repeats):
        outer = partition_groups(y, g, k, seed_for(args.seed, task, "outer", repeat))
        for f, (tr, te) in enumerate(outer):
            inner_k = min(args.max_inner_splits, int(group_counts(y[tr], g[tr]).min()))
            if inner_k < 2:
                raise ValueError(f"{task}: inner grouped CV infeasible; no untuned fallback is allowed.")
            inner = partition_groups(y[tr], g[tr], inner_k, seed_for(args.seed, task, "inner", repeat, f))
            plans.append((repeat, f, tr, te, inner))
            audits.append({"repeat": repeat, "outer_fold": f, "n_train": len(tr), "n_test": len(te),
                           "n_train_groups": len(set(g[tr])), "n_test_groups": len(set(g[te])),
                           "inner_splits": inner_k, "group_overlap": 0})
    return plans, pd.DataFrame(audits), k


def inverse_group_size(groups):
    groups = np.asarray(groups, str)
    counts = pd.Series(groups).value_counts()
    return np.array([1.0 / counts[g] for g in groups])


def metrics(y, pred, groups):
    w = inverse_group_size(groups)
    return {"accuracy": accuracy_score(y, pred),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
            "mcc": matthews_corrcoef(y, pred),
            "group_balanced_accuracy": balanced_accuracy_score(y, pred, sample_weight=w),
            "group_weighted_accuracy": accuracy_score(y, pred, sample_weight=w)}


def build_model(probe, n_classes, c, seed, train_weighting):
    cw = "balanced" if train_weighting == "class" else None
    if probe == "logreg":
        clf = LogisticRegression(C=c, solver="lbfgs", class_weight=cw, max_iter=10000, random_state=seed)
    elif probe == "linear_svm":
        clf = LinearSVC(C=c, dual="auto", class_weight=cw, max_iter=50000, random_state=seed)
    elif probe == "shrinkage_lda":
        if train_weighting != "class":
            raise ValueError("LDA does not support class_group sample weighting; choose LR/SVM only.")
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=np.ones(n_classes) / n_classes)
    else:
        raise ValueError(probe)
    return make_pipeline(StandardScaler(), clf)


def fit_model(model, x, y, groups, train_weighting):
    kwargs = {}
    if train_weighting == "class_group":
        ng = group_counts(y, groups)
        w = inverse_group_size(groups) / np.array([ng[c] for c in y])
        w *= len(w) / w.sum()
        kwargs[model.steps[-1][0] + "__sample_weight"] = w
        kwargs["standardscaler__sample_weight"] = w
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        model.fit(x, y, **kwargs)
    messages = [f"{v.category.__name__}: {v.message}" for v in seen]
    return model, messages


def evaluate_candidate(c, probe, x, y, groups, inner, n_classes, seed, args):
    pred = np.empty(len(y), dtype=object)
    messages = []
    for f, (tr, va) in enumerate(inner):
        model = build_model(probe, n_classes, c, seed_for(seed, f), args.train_weighting)
        model, caught = fit_model(model, x[tr], y[tr], groups[tr], args.train_weighting)
        pred[va] = model.predict(x[va])
        messages.extend(caught)
    score = metrics(y, pred.astype(str), groups)[args.tune_metric]
    if not np.isfinite(score):
        raise RuntimeError("Non-finite inner-CV score.")
    return float(c), float(score), messages


def model_scores(model, x, classes):
    own = list(model.classes_)
    if hasattr(model, "predict_proba"):
        raw, kind = model.predict_proba(x), "probability"
    else:
        raw, kind = model.decision_function(x), "decision_score"
        if raw.ndim == 1:
            raw = np.column_stack([-raw, raw])
    aligned = raw[:, [own.index(c) for c in classes]]
    if not np.isfinite(aligned).all():
        raise RuntimeError("Non-finite prediction scores.")
    return aligned, kind


def load_features(args, meta, out):
    aliases = {}
    if args.accession_map:
        mapping = read_table(args.accession_map)
        if not {"embedding_accession", "metadata_accession"}.issubset(mapping):
            raise ValueError("Accession map requires embedding_accession, metadata_accession.")
        for _, row in mapping.iterrows():
            a, b = accession(row.embedding_accession), accession(row.metadata_accession)
            if a in aliases and aliases[a] != b:
                raise ValueError(f"Conflicting accession map: {a}")
            aliases[a] = b
    requested = set(meta.genome_accession)
    loaded, audits, provenance = {}, [], []
    common = set(requested)
    for name in args.models:
        path = Path(args.input_dir) / DEFAULT_FILES[name]
        if args.files:
            overrides = dict(item.split("=", 1) for item in args.files)
            path = Path(overrides.get(name, str(path)))
        df = pd.read_csv(path)
        if "genome_accession" not in df:
            raise ValueError(f"{path}: missing genome_accession")
        original = df.genome_accession.map(accession)
        df["genome_accession"] = original.map(lambda a: aliases.get(a, a))
        cols = sorted([c for c in df if re.fullmatch(r"embedding_\d+", str(c))],
                      key=lambda c: int(c.split("_")[-1]))
        if not cols:
            raise ValueError(f"{path}: no embedding_<integer> columns")
        df = df.loc[df.genome_accession.isin(requested)].copy()
        if df.genome_accession.duplicated().any():
            raise ValueError(f"{name}: duplicate accession after mapping; resolve explicitly.")
        x = df[cols].apply(pd.to_numeric, errors="raise").to_numpy(float)
        if not np.isfinite(x).all():
            raise ValueError(f"{name}: non-finite embeddings; do not silently drop samples.")
        available = set(df.genome_accession)
        for a in sorted(requested):
            audits.append({"model": name, "genome_accession": a, "available": a in available})
        common &= available
        loaded[name] = pd.DataFrame(x, index=df.genome_accession, columns=cols)
        provenance.append({"model": name, "path": str(path.resolve()), "sha256": sha256(path),
                           "n_features": len(cols), "n_metadata_matched": len(df)})
    write_tsv(audits, out / "embedding_match_audit.tsv")
    write_tsv(provenance, out / "embedding_files.tsv")
    if common != requested and not args.allow_missing_embeddings:
        missing = sorted(requested - common)
        raise ValueError(f"{len(missing)} metadata genomes missing from at least one embedding file. "
                         f"Examples: {missing[:5]}. Inspect embedding_match_audit.tsv. "
                         "GCA/GCF are NOT interchangeable automatically. Supply --accession-map if verified.")
    if not common:
        raise ValueError("No matched genomes.")
    matched = meta.loc[meta.genome_accession.isin(common)].reset_index(drop=True)
    write_tsv(meta.loc[~meta.genome_accession.isin(common)], out / "missing_embedding_genomes.tsv")
    return loaded, matched


def write_profile(meta, out, prefix):
    write_tsv(meta, out / f"{prefix}_metadata.tsv")
    write_tsv(meta.groupby("family", dropna=False).agg(
        n_genomes=("genome_accession", "size"), n_genera=("genus", "nunique"),
        n_species=("species", "nunique"), n_groups=("group", "nunique")).reset_index(),
        out / f"{prefix}_family_counts.tsv")
    write_tsv(meta.groupby(["family", "genus"], dropna=False).agg(
        n_genomes=("genome_accession", "size"), n_species=("species", "nunique"),
        n_groups=("group", "nunique")).reset_index(), out / f"{prefix}_genus_counts.tsv")
    write_tsv(meta.groupby(["family", "genus", "species"], dropna=False).size().rename(
        "n_genomes").reset_index(), out / f"{prefix}_species_counts.tsv")


def save_plan(d, plans, out):
    outer_rows, inner_rows = [], []
    for repeat, fold, tr, te, inner in plans:
        for role, idx in [("train", tr), ("test", te)]:
            for i in idx:
                outer_rows.append({"repeat": repeat, "outer_fold": fold, "role": role,
                                   "genome_accession": d.at[i, "genome_accession"],
                                   "target": d.at[i, "target"], "group": d.at[i, "group"]})
        for inf, (itr, iva) in enumerate(inner):
            for role, idx in [("train", itr), ("validation", iva)]:
                for j in idx:
                    i = tr[j]
                    inner_rows.append({"repeat": repeat, "outer_fold": fold, "inner_fold": inf,
                                       "role": role, "genome_accession": d.at[i, "genome_accession"],
                                       "target": d.at[i, "target"], "group": d.at[i, "group"]})
    write_tsv(outer_rows, out / "outer_assignments.tsv")
    write_tsv(inner_rows, out / "inner_assignments.tsv")


def summarize(pred, out, expected_repeats):
    repeat_rows, class_rows, cm_rows = [], [], []
    for key, sub in pred.groupby(KEYS + ["repeat"], sort=True):
        if sub.genome_accession.duplicated().any():
            raise RuntimeError("Duplicate OOF predictions within a repeat.")
        ids = dict(zip(KEYS + ["repeat"], key))
        y, p, g = sub.true_label.to_numpy(str), sub.predicted_label.to_numpy(str), sub.group.to_numpy(str)
        labels = sorted(set(y))
        repeat_rows.append({**ids, "n_genomes": len(y), "n_groups": len(set(g)), **metrics(y, p, g)})
        precision, recall, f1, support = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
        w = inverse_group_size(g)
        group_cm = confusion_matrix(y, p, labels=labels, sample_weight=w)
        cm = confusion_matrix(y, p, labels=labels)
        for i, c in enumerate(labels):
            class_rows.append({**ids, "class": c, "n_genomes": int(support[i]),
                               "n_groups": len(set(g[y == c])), "precision": precision[i],
                               "recall": recall[i], "f1": f1[i],
                               "group_balanced_recall": group_cm[i, i] / group_cm[i].sum()})
            for j, pc in enumerate(labels):
                cm_rows.append({**ids, "true_label": c, "predicted_label": pc, "count": int(cm[i, j]),
                                "row_fraction": cm[i, j] / cm[i].sum(),
                                "group_weighted_count": group_cm[i, j],
                                "group_weighted_row_fraction": group_cm[i, j] / group_cm[i].sum()})
    r = pd.DataFrame(repeat_rows)
    write_tsv(r, out / "repeat_metrics.tsv")
    names = list(metrics(np.array(["a", "b"]), np.array(["a", "b"]), np.array(["x", "z"])))
    summaries = []
    for key, sub in r.groupby(KEYS):
        if sub.repeat.nunique() != expected_repeats:
            raise RuntimeError("Missing complete repeats.")
        row = dict(zip(KEYS, key))
        row.update(n_repeats=len(sub), n_genomes=int(sub.n_genomes.iloc[0]), n_groups=int(sub.n_groups.iloc[0]))
        for m in names:
            row[m + "_mean"] = sub[m].mean()
            row[m + "_sd"] = sub[m].std(ddof=1)
        summaries.append(row)
    write_tsv(summaries, out / "comparison_summary.tsv")
    pc, cm = pd.DataFrame(class_rows), pd.DataFrame(cm_rows)
    write_tsv(pc, out / "per_class_by_repeat.tsv")
    write_tsv(pc.groupby(KEYS + ["class"])[["n_genomes", "n_groups", "precision", "recall", "f1",
                 "group_balanced_recall"]].agg(["mean", "std"]).pipe(flat_columns).reset_index(),
              out / "per_class_summary.tsv")
    write_tsv(cm, out / "confusion_by_repeat.tsv")
    write_tsv(cm.groupby(KEYS + ["true_label", "predicted_label"])[["count", "row_fraction",
        "group_weighted_count", "group_weighted_row_fraction"]].mean().add_suffix("_mean_per_repeat").reset_index(),
        out / "confusion_mean.tsv")
    instability = pred.assign(correct=pred.true_label == pred.predicted_label).groupby(
        KEYS + ["genome_accession", "true_label", "group"]).agg(n_predictions=("correct", "size"),
                                                                  fraction_correct=("correct", "mean")).reset_index()
    if not instability.n_predictions.eq(expected_repeats).all():
        raise RuntimeError("Unequal repeated OOF coverage.")
    write_tsv(instability, out / "genome_prediction_stability.tsv")


def flat_columns(df):
    df.columns = ["_".join(c) if isinstance(c, tuple) else c for c in df.columns]
    return df


def run_task(d, plans, features, args, task, out):
    y, groups = d.target.to_numpy(str), d.group.to_numpy(str)
    classes = sorted(set(y))
    predictions, fold_rows, tuning_rows, chosen_rows, warning_rows = [], [], [], [], []
    total = len(plans) * len(features) * len(args.probes)
    done, start = 0, time.time()
    for repeat, fold, tr, te, inner in plans:
        for name, frame in features.items():
            x = frame.loc[d.genome_accession].to_numpy(float)
            for probe in args.probes:
                context = {"task": task, "model": name, "probe": probe,
                           "repeat": repeat, "outer_fold": fold}
                fit_seed = seed_for(args.seed, task, repeat, fold)
                if probe == "shrinkage_lda":
                    selected, best_score = 1.0, None
                else:
                    candidates = Parallel(n_jobs=args.n_jobs)(delayed(evaluate_candidate)(
                        c, probe, x[tr], y[tr], groups[tr], inner, len(classes), fit_seed, args)
                        for c in sorted(set(args.c_grid)))
                    for c, score, caught in candidates:
                        tuning_rows.append({**context, "C": c, "inner_oof_score": score,
                                            "metric": args.tune_metric, "n_inner_splits": len(inner)})
                        warning_rows.extend({**context, "stage": "inner", "C": c, "warning": w} for w in caught)
                    selected, best_score, _ = max(candidates, key=lambda v: v[1])
                model = build_model(probe, len(classes), selected, fit_seed, args.train_weighting)
                fitted, caught = fit_model(model, x[tr], y[tr], groups[tr], args.train_weighting)
                warning_rows.extend({**context, "stage": "outer_fit", "C": selected, "warning": w} for w in caught)
                p = fitted.predict(x[te])
                scores, score_type = model_scores(fitted, x[te], classes)
                fold_rows.append({**context, "n_train": len(tr), "n_test": len(te), **metrics(y[te], p, groups[te])})
                chosen_rows.append({**context, "C": selected if probe != "shrinkage_lda" else np.nan,
                                    "inner_oof_score": best_score, "metric": args.tune_metric})
                for j, idx in enumerate(te):
                    row = {**context, "genome_accession": d.at[idx, "genome_accession"],
                           "family": d.at[idx, "family"], "genus": d.at[idx, "genus"],
                           "species": d.at[idx, "species"], "group": groups[idx], "true_label": y[idx],
                           "predicted_label": p[j], "score_type": score_type}
                    row.update({"score__" + c: scores[j, k] for k, c in enumerate(classes)})
                    predictions.append(row)
                done += 1
                if done == 1 or done % args.progress_every == 0 or done == total:
                    elapsed = time.time() - start
                    print(f"[PROGRESS] {task}: {done}/{total}; elapsed={elapsed/60:.1f} min; "
                          f"ETA={(total-done)*elapsed/done/60:.1f} min", flush=True)
        write_tsv(predictions, out / "oof_predictions.tsv")
        write_tsv(fold_rows, out / "fold_metrics.tsv")
        write_tsv(tuning_rows, out / "inner_search.tsv")
        write_tsv(chosen_rows, out / "selected_hyperparameters.tsv")
        write_tsv(warning_rows, out / "fit_warnings.tsv")
    summarize(pd.DataFrame(predictions), out, args.outer_repeats)
    if warning_rows:
        print(f"[WARN] {len(warning_rows)} captured fit warnings; inspect {out/'fit_warnings.tsv'}", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--metadata", required=True, help="One-genome-per-GTDB-species allow-list XLSX/TSV/CSV")
    p.add_argument("--sheet", default="0", help="Worksheet name, or zero-based numeric index")
    p.add_argument("--input-dir", default="00.DATASET")
    p.add_argument("--outdir", default="methcocca_gtdb_species_representative")
    p.add_argument("--models", nargs="+", choices=list(DEFAULT_FILES), default=list(DEFAULT_FILES))
    p.add_argument("--files", nargs="+", help="Override files using Model=/path/file.csv")
    p.add_argument("--probes", nargs="+", choices=["logreg", "linear_svm", "shrinkage_lda"],
                   default=["logreg", "linear_svm", "shrinkage_lda"])
    p.add_argument("--group-map", help="Optional external genome_accession/group map, e.g. ANI clusters. Default: GTDB species")
    p.add_argument("--accession-map", help="Verified embedding_accession/metadata_accession alias table")
    p.add_argument("--allow-missing-embeddings", action="store_true", help="Explicitly allow matched intersection; counts/eligibility are recomputed")
    p.add_argument("--min-genus-genomes", type=int, default=4, help="Minimum species representatives per genus")
    p.add_argument("--min-family-genomes", type=int, default=3, help="Minimum species representatives per family")
    p.add_argument("--min-groups", type=int, default=3, help="Minimum groups per class for nested grouped CV; must be >=3")
    p.add_argument("--tasks", nargs="+", choices=["genus", "family"], default=["genus", "family"])
    p.add_argument("--outer-repeats", type=int, default=25)
    p.add_argument("--max-outer-splits", type=int, default=3)
    p.add_argument("--max-inner-splits", type=int, default=2)
    p.add_argument("--seed", type=int, default=20261029)
    p.add_argument("--c-grid", type=float, nargs="+", default=[.001, .01, .1, 1, 10, 100, 1000])
    p.add_argument("--tune-metric", choices=["group_balanced_accuracy", "balanced_accuracy"], default="balanced_accuracy",
                   help="Both choices are equal when there is exactly one genome per species")
    p.add_argument("--train-weighting", choices=["class", "class_group"], default="class",
                   help="class preserves original probes; class_group gives each class/group equal mass (LR/SVM only)")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--plan-only", action="store_true", help="Inspect metadata and generate audited nested splits without loading embeddings")
    args = p.parse_args(argv)
    if args.group_map:
        p.error("This species-representative version fixes groups to GTDB species. Use the original version for ANI group maps.")
    if args.train_weighting != "class":
        p.error("Use class weighting for singleton species representatives; class_group weighting is unnecessary here.")
    if args.min_groups < 3 or min(args.min_genus_genomes, args.min_family_genomes) < 1:
        p.error("min-groups must be >=3 and minimum genome counts must be positive.")
    if min(args.max_outer_splits, args.max_inner_splits) < 2 or args.outer_repeats < 1:
        p.error("CV splits must be >=2 and repeats >=1.")
    if args.progress_every < 1 or args.n_jobs == 0 or any(not np.isfinite(c) or c <= 0 for c in args.c_grid):
        p.error("Invalid progress interval, n-jobs, or C grid.")
    if args.train_weighting == "class_group" and "shrinkage_lda" in args.probes:
        p.error("class_group weighting requires --probes logreg linear_svm (LDA lacks sample_weight).")
    if len(set(args.models)) != len(args.models) or len(set(args.probes)) != len(args.probes):
        p.error("Duplicate models/probes are not allowed.")
    if len(set(args.tasks)) != len(args.tasks):
        p.error("Duplicate tasks are not allowed.")
    if args.files:
        try:
            pairs = [item.split("=", 1) for item in args.files]
            overrides = dict(pairs)
        except ValueError:
            p.error("Each --files override must have the form Model=/path/file.csv.")
        if len(overrides) != len(pairs) or not set(overrides).issubset(args.models):
            p.error("File overrides must name distinct selected models.")
    return args


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("*/oof_predictions.tsv")) or (out / "run_manifest.json").exists():
        raise ValueError("Output directory contains an existing run. Choose a new --outdir.")
    sheet = int(args.sheet) if args.sheet.isdigit() else args.sheet
    meta = load_metadata(args.metadata, sheet, args.group_map)
    validate_representatives(meta)
    write_profile(meta, out, "source")
    features = None
    if not args.plan_only:
        features, meta = load_features(args, meta, out)
    write_profile(meta, out, "matched")
    manifest = {"version": VERSION, "arguments": vars(args), "metadata_sha256": sha256(args.metadata),
                "script_sha256": sha256(__file__), "python": platform.python_version(),
                "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
                "joblib": joblib.__version__, "group_source": args.group_map or "GTDB species",
                "no_pca": True, "no_family_merge": True, "status": "planning", "tasks": {}}
    manifest.update({"analysis_unit": "one preselected genome per GTDB species",
                     "one_genome_per_species": True, "ani_filter_performed_by_script": False,
                     "representative_quality_verified_by_script": False,
                     "ba_equals_group_ba": True})
    for name, path in [("group_map", args.group_map), ("accession_map", args.accession_map)]:
        if path:
            manifest[name + "_sha256"] = sha256(path)
    manifest_path = out / "run_manifest.json"
    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    save_manifest()
    task_plans = {}
    for task in args.tasks:
        dest = out / task
        dest.mkdir(exist_ok=True)
        d, counts, audit = make_task(meta, task, args)
        write_tsv(counts, dest / "eligibility.tsv")
        write_tsv(audit, dest / "sample_inclusion.tsv")
        write_tsv(d, dest / "modeling_metadata.tsv")
        plans, split_audit, k = plan_task(d, args, task)
        save_plan(d, plans, dest)
        write_tsv(split_audit, dest / "split_audit.tsv")
        signatures = []
        for r in range(args.outer_repeats):
            partitions = [tuple(sorted(d.iloc[te].genome_accession)) for rr, _, _, te, _ in plans if rr == r]
            signatures.append(tuple(sorted(partitions)))
        info = {"n_genomes": len(d), "n_classes": d.target.nunique(), "n_species": d.species.nunique(),
                "n_groups": d.group.nunique(), "outer_splits": k, "n_outer_fits_per_configuration": len(plans),
                "unique_outer_partitions": len(set(signatures)), "classes": sorted(d.target.unique())}
        manifest["tasks"][task] = info
        task_plans[task] = (d, plans, dest)
        print(f"[PLAN] {task}: {json.dumps(info)}", flush=True)
    manifest["status"] = "plan_complete" if args.plan_only else "running"
    save_manifest()
    if args.plan_only:
        print("[DONE] Metadata-only plan. No embedding matching or model performance was evaluated.")
        return
    for task, (d, plans, dest) in task_plans.items():
        run_task(d, plans, features, args, task, dest)
        manifest["tasks"][task]["complete"] = True
        save_manifest()
    manifest["status"] = "complete"
    save_manifest()
    print("[DONE] Complete grouped nested CV. Report repeat-level mean/SD, not averaged-score ensembles.")


if __name__ == "__main__":
    main()
