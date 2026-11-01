#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import logging
import math
import hashlib
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, homogeneity_completeness_v_measure

DESCRIPTION = """Validate GTDB taxonomy from a NEW workbook on unchanged HDBSCAN trees.

For each taxonomic rank, the original tree and internal nodes stay unchanged,
but only rank-annotated descendant leaves contribute to subtree sizes, purity,
F1, exact recovery, and permutation tests. No refitting or pruning is done.
ARI/AMI already exclude unannotated leaves. This is a GTDB-annotated-only
assessment, NOT evolutionary monophyly or full-tree subtree purity.
"""

LOG = logging.getLogger('hdbscan_taxonomy_validation')
RANKS = ('family', 'genus', 'species')
MISSING = {'', 'nan', 'none', 'null', 'na', 'n/a', 'unknown', 'unclassified'}


def normalize_accession(value):
    match = re.search(r'(GC[AF]_\d+\.\d+)', str(value))
    return match.group(1) if match else ''


def load_gtdb_workbook(path):
    raw = pd.read_excel(path, dtype=str, keep_default_na=False)
    lookup = {str(col).strip().lower().replace('_', ' '): col for col in raw.columns}
    required = {'assembly accession', 'family', 'genus', 'species'}
    if not required.issubset(lookup):
        raise ValueError(f'{path}: missing columns {sorted(required - set(lookup))}')
    meta = pd.DataFrame({
        'accession': raw[lookup['assembly accession']].map(normalize_accession),
        **{rank: raw[lookup[rank]].map(clean_label) for rank in RANKS},
    })
    if meta.accession.eq('').any():
        raise ValueError(f'{path}: workbook contains invalid/blank assembly accessions')
    if meta.accession.duplicated().any():
        raise ValueError(f'{path}: duplicate assembly accessions: '
                         f"{meta.loc[meta.accession.duplicated(), 'accession'].tolist()[:10]}")
    return meta.set_index('accession'), hashlib.sha256(path.read_bytes()).hexdigest()


def replace_taxonomy(samples, workbook, *, strict=True):
    samples = samples.copy()
    missing = ~samples.accession.isin(workbook.index)
    if missing.any() and strict:
        raise ValueError('Tree accessions missing from taxonomy workbook: '
                         + ', '.join(samples.loc[missing, 'accession'].tolist()[:12]))
    audit = []
    for rank in RANKS:
        old = samples[rank].map(clean_label)
        new = samples['accession'].map(workbook[rank])
        if not strict:
            new = new.where(~missing, '')
        new = new.fillna('').map(clean_label)
        changed = old.ne(new)
        for idx in np.flatnonzero(changed.to_numpy()):
            audit.append({'accession': samples.iloc[idx]['accession'], 'rank': rank,
                          'previous_label': old.iloc[idx], 'updated_label': new.iloc[idx]})
        samples[rank] = new
    return samples, pd.DataFrame(audit, columns=['accession','rank','previous_label','updated_label'])


def clean_label(value):
    if value is None or pd.isna(value):
        return ''
    text = str(value).strip()
    return '' if text.casefold() in MISSING else text


def load_run(folder):
    graph_path = folder / 'condensed_tree.graphml'
    meta_path = folder / 'sample_metadata.tsv'
    if not graph_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f'Missing graph or metadata in {folder}')
    graph = nx.read_graphml(graph_path)
    if not graph.is_directed() or not nx.is_arborescence(graph):
        raise ValueError(f'{graph_path}: expected one rooted directed tree')
    samples = pd.read_csv(meta_path, sep='\t', dtype={'accession': str}, keep_default_na=False)
    required = {'accession', 'graph_node_id', 'family', 'genus', 'species', 'hdbscan_cluster'}
    missing = required - set(samples.columns)
    if missing:
        raise ValueError(f'{meta_path}: missing columns {sorted(missing)}')
    if samples['accession'].duplicated().any() or samples['graph_node_id'].duplicated().any():
        raise ValueError(f'{meta_path}: duplicate accession/node identifiers')
    tips = samples['graph_node_id'].astype(str).tolist()
    if len(tips) < 3:
        raise ValueError(f'{folder}: too few samples')
    if any(tip not in graph for tip in tips):
        raise ValueError(f'{folder}: sample node not found in GraphML')
    graph_tips = {node for node, data in graph.nodes(data=True) if data.get('node_type') == 'sample'}
    if graph_tips != set(tips):
        raise ValueError(f'{folder}: sample node set differs between GraphML and sample_metadata.tsv')
    if any(graph.out_degree(tip) != 0 for tip in tips):
        raise ValueError(f'{folder}: sample node must be a terminal leaf')
    for tip, accession in zip(tips, samples['accession']):
        if str(graph.nodes[tip].get('accession', '')) != accession:
            raise ValueError(f'{folder}: accession mismatch for {tip}')
    if any(graph.out_degree(node) == 0 for node in graph if node not in graph_tips):
        raise ValueError(f'{folder}: unexpected terminal internal node')
    sample_to_index = {tip: i for i, tip in enumerate(tips)}
    masks = {}
    sizes = {}
    for node in reversed(list(nx.topological_sort(graph))):
        if node in sample_to_index:
            mask = 1 << sample_to_index[node]
        else:
            mask = 0
            for child in graph.successors(node):
                mask |= masks[child]
        masks[node] = mask
        sizes[node] = mask.bit_count()
    root = next(node for node, degree in graph.in_degree() if degree == 0)
    expected = (1 << len(tips)) - 1
    if masks[root] != expected:
        raise ValueError(f'{folder}: sample descendants do not span exactly all samples')
    return graph, samples, masks, sizes, root


def annotated_view(labels, full_masks):
    annotated = 0
    for i, value in enumerate(labels):
        if clean_label(value):
            annotated |= 1 << i
    masks = {node: mask & annotated for node, mask in full_masks.items()}
    sizes = {node: mask.bit_count() for node, mask in masks.items()}
    return masks, sizes, annotated.bit_count()


def label_masks(labels):
    groups = defaultdict(int)
    for i, label in enumerate(labels):
        value = clean_label(label)
        if value:
            groups[value] |= 1 << i
    return dict(groups)


def taxon_stats(taxon, target, masks, sizes, root, n_samples, full_sizes=None):
    n = target.bit_count()
    containers = [node for node, mask in masks.items() if mask & target == target]
    lca = min(containers, key=lambda node: (sizes[node], (full_sizes or sizes)[node]))
    lca_size = sizes[lca]
    exact = int(n >= 2 and lca_size == n)
    best = None
    for node, mask in masks.items():
        size = sizes[node]
        if size < 2:
            continue
        tp = (mask & target).bit_count()
        if tp == 0:
            continue
        f1 = 2 * tp / (n + size)
        purity = tp / size
        recall = tp / n
        choice = (f1, tp, -size, str(node))
        if best is None or choice > best[0]:
            best = (choice, node, size, tp, purity, recall)
    if best is None:
        raise AssertionError('No candidate internal subtree')
    _, best_node, best_size, tp, best_purity, best_recall = best
    return {
        'taxon': taxon, 'n_taxon': n, 'fraction_of_all_samples': n / n_samples,
        'exact_subtree': exact, 'lca_node': lca, 'lca_subtree_n': lca_size,
        'lca_extra_samples': lca_size - n, 'lca_purity': n / lca_size,
        'best_subtree_node': best_node, 'best_subtree_n': best_size,
        'best_subtree_tp': tp, 'best_subtree_precision': best_purity,
        'best_subtree_recall': best_recall, 'best_subtree_f1': 2 * tp / (n + best_size),
    }


def score_rank(labels, masks, sizes, root, min_taxon_n, n_samples, full_sizes=None):
    groups = label_masks(labels)
    eligible = {label: mask for label, mask in groups.items() if mask.bit_count() >= min_taxon_n}
    rows = [taxon_stats(label, mask, masks, sizes, root, n_samples, full_sizes)
            for label, mask in sorted(eligible.items())]
    valid_comparison = len(groups) >= 2 and len(eligible) >= 1
    if not valid_comparison:
        return rows, None
    exact = np.array([row['exact_subtree'] for row in rows], dtype=float)
    purity = np.array([row['lca_purity'] for row in rows], dtype=float)
    f1 = np.array([row['best_subtree_f1'] for row in rows], dtype=float)
    sizes_taxa = np.array([row['n_taxon'] for row in rows], dtype=float)
    summary = {
        'n_taxa_total': len(groups), 'n_taxa_eligible': len(rows),
        'n_taxa_exact': int(exact.sum()),
        'fraction_exact_macro': float(exact.mean()),
        'lca_purity_macro': float(purity.mean()),
        'best_subtree_f1_macro': float(f1.mean()),
        'fraction_exact_weighted': float(np.average(exact, weights=sizes_taxa)),
        'lca_purity_weighted': float(np.average(purity, weights=sizes_taxa)),
        'best_subtree_f1_weighted': float(np.average(f1, weights=sizes_taxa)),
        'n_genomes_in_eligible_taxa': int(sizes_taxa.sum()),
    }
    return rows, summary


def cluster_metrics(labels, clusters):
    y = np.array([clean_label(z) for z in labels], dtype=object)
    assigned = np.array([int(v) for v in clusters], dtype=int)
    valid = y != ''
    out = {'n_annotated': int(valid.sum()), 'n_unannotated': int((~valid).sum()),
           'n_annotated_nonnoise': int((valid & (assigned >= 0)).sum()),
           'nonnoise_coverage': float((valid & (assigned >= 0)).sum() / valid.sum()) if valid.any() else math.nan,
           'n_predicted_clusters': int(len(set(assigned[valid]) - {-1}))}
    if valid.sum() < 3 or len(set(y[valid])) < 2:
        return out
    for suffix, idx in [('all_including_noise_as_label', valid),
                        ('nonnoise_only', valid & (assigned >= 0))]:
        if idx.sum() < 3 or len(set(y[idx])) < 2 or len(set(assigned[idx])) < 2:
            out.update({f'ari_{suffix}': math.nan, f'ami_{suffix}': math.nan,
                        f'homogeneity_{suffix}': math.nan,
                        f'completeness_{suffix}': math.nan, f'v_measure_{suffix}': math.nan})
            continue
        truth, predicted = y[idx].tolist(), assigned[idx].tolist()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message=r'The number of unique classes is greater than 50% of the number of samples\.', category=UserWarning)
            h, c, v = homogeneity_completeness_v_measure(truth, predicted)
            ari = adjusted_rand_score(truth, predicted)
            ami = adjusted_mutual_info_score(truth, predicted)
        out.update({f'ari_{suffix}': float(ari),
                    f'ami_{suffix}': float(ami),
                    f'homogeneity_{suffix}': float(h), f'completeness_{suffix}': float(c),
                    f'v_measure_{suffix}': float(v)})
    return out


def permutation_test(labels, masks, sizes, root, min_taxon_n, n_samples,
                     observed, repeats, rng, full_sizes=None):
    if observed is None or repeats < 1:
        return None
    labels = np.array([clean_label(x) for x in labels], dtype=object)
    annotated_idx = np.flatnonzero(labels != '')
    annotated_labels = labels[annotated_idx].copy()
    null = {key: [] for key in ('fraction_exact_macro', 'lca_purity_macro', 'best_subtree_f1_macro')}
    shuffled = labels.copy()
    for _ in range(repeats):
        shuffled[annotated_idx] = rng.permutation(annotated_labels)
        _, stat = score_rank(shuffled, masks, sizes, root, min_taxon_n, n_samples, full_sizes)
        if stat is None:
            continue
        for key in null:
            null[key].append(stat[key])
    out = {}
    for key, vals in null.items():
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        val = observed[key]
        out[f'{key}_null_mean'] = float(arr.mean())
        out[f'{key}_null_q025'] = float(np.quantile(arr, .025))
        out[f'{key}_null_q975'] = float(np.quantile(arr, .975))
        out[f'{key}_permutation_p_greater'] = float((1 + np.count_nonzero(arr >= val - 1e-12)) / (len(arr) + 1))
        out[f'{key}_observed_minus_null_mean'] = float(val - arr.mean())
    out['permutations'] = repeats
    return out


def per_taxon_permutations(rows, labels, masks, sizes, root, n_samples, repeats, rng, full_sizes=None):
    annotated = np.array([i for i, label in enumerate(labels) if clean_label(label)], dtype=int)
    if len(annotated) < 2 or repeats < 1:
        return
    by_size = defaultdict(list)
    for row in rows:
        by_size[row['n_taxon']].append(row)
    for group_size, target_rows in by_size.items():
        null_purity, null_f1 = [], []
        for _ in range(repeats):
            indices = rng.choice(annotated, size=group_size, replace=False)
            mask = 0
            for i in indices:
                mask |= 1 << int(i)
            s = taxon_stats('permuted', mask, masks, sizes, root, n_samples, full_sizes)
            null_purity.append(s['lca_purity'])
            null_f1.append(s['best_subtree_f1'])
        purity = np.asarray(null_purity)
        f1 = np.asarray(null_f1)
        for row in target_rows:
            row['lca_purity_size_matched_p_greater'] = float((1 + np.count_nonzero(purity >= row['lca_purity'] - 1e-12)) / (repeats + 1))
            row['best_subtree_f1_size_matched_p_greater'] = float((1 + np.count_nonzero(f1 >= row['best_subtree_f1'] - 1e-12)) / (repeats + 1))
            row['lca_purity_size_matched_null_mean'] = float(purity.mean())
            row['best_subtree_f1_size_matched_null_mean'] = float(f1.mean())


def add_bh_qvalues(rows, p_column, q_column):
    idx = [i for i, row in enumerate(rows) if p_column in row and np.isfinite(row[p_column])]
    if not idx:
        return
    ordered = sorted(idx, key=lambda i: rows[i][p_column])
    m = len(ordered)
    running = 1.0
    for rank in range(m, 0, -1):
        i = ordered[rank - 1]
        running = min(running, rows[i][p_column] * m / rank)
        rows[i][q_column] = float(running)


def analyze(folder, out, ranks, min_taxon_n, permutations, seed, taxon_permutations, workbook):
    graph, samples, masks, sizes, root = load_run(folder)
    samples, annotation_changes = replace_taxonomy(samples, workbook)
    out.mkdir(parents=True, exist_ok=True)
    annotation_changes.to_csv(out / 'taxonomy_changes.tsv', sep='\t', index=False)
    samples[['accession', 'graph_node_id', *RANKS]].to_csv(
        out / 'taxonomy_labels_used.tsv', sep='\t', index=False)
    LOG.info('%s: %d corrected taxonomy labels', folder, len(annotation_changes))
    model, subset = folder.parent.name, folder.name
    n_samples = len(samples)
    out.mkdir(parents=True, exist_ok=True)
    total = []
    all_taxa = []
    for rank in ranks:
        labels = samples[rank].tolist()
        rank_masks, rank_sizes, n_rank_annotated = annotated_view(labels, masks)
        taxa, stat = score_rank(labels, rank_masks, rank_sizes, root, min_taxon_n, n_rank_annotated, sizes)
        counts = Counter(clean_label(z) for z in labels)
        n_annotated = sum(count for key, count in counts.items() if key)
        summary = {'model': model, 'subset': subset, 'rank': rank, 'n_samples': n_samples,
                   'n_annotated': n_annotated, 'n_unclassified': n_samples - n_annotated,
                   'subtree_denominator': 'GTDB_annotated_only',
                   'n_taxa_total': len(counts) - int('' in counts),
                   'min_taxon_n': min_taxon_n,
                   'n_internal_nodes': graph.number_of_nodes() - n_samples,
                   'topology_informative': int(stat is not None)}
        if stat is not None:
            summary.update(stat)
            cluster = cluster_metrics(labels, samples['hdbscan_cluster'].tolist())
            summary.update(cluster)
            rank_seed = (seed + sum(ord(c) for c in (model + subset + rank))) % (2**32)
            rng = np.random.default_rng(rank_seed)
            null = permutation_test(labels, rank_masks, rank_sizes, root, min_taxon_n, n_rank_annotated, stat,
                                    permutations, rng, sizes)
            if null:
                summary.update(null)
            if taxon_permutations:
                per_taxon_permutations(taxa, labels, rank_masks, rank_sizes, root, n_rank_annotated,
                                       taxon_permutations, rng, sizes)
                add_bh_qvalues(taxa, 'lca_purity_size_matched_p_greater',
                               'lca_purity_size_matched_q_bh')
                add_bh_qvalues(taxa, 'best_subtree_f1_size_matched_p_greater',
                               'best_subtree_f1_size_matched_q_bh')
        else:
            LOG.info('Not informative: %s / %s / %s (fewer than 2 annotated taxa or no eligible groups)',
                     model, subset, rank)
        total.append(summary)
        for row in taxa:
            row.update({'model': model, 'subset': subset, 'rank': rank,
                        'n_annotated': n_annotated, 'n_samples': n_samples,
                        'subtree_denominator': 'GTDB_annotated_only'})
            all_taxa.append(row)
    pd.DataFrame(total).to_csv(out/'rank_summary.tsv', sep='\t', index=False)
    pd.DataFrame(all_taxa).to_csv(out/'taxon_details.tsv', sep='\t', index=False)
    return total, all_taxa


def main():
    p = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--metadata', type=Path, required=True,
                   help='Current GTDB workbook, e.g. GTDB_accession_V3.xlsx; overrides old TSV labels')
    p.add_argument('--input-dir', type=Path, required=True,
                   help='Output directory of methcocca_hdbscan_graphml_v3.py')
    p.add_argument('--outdir', type=Path, default=Path('methcocca_hdbscan_taxonomy_validation_v2'))
    p.add_argument('--ranks', nargs='+', choices=RANKS, default=['family', 'genus', 'species'])
    p.add_argument('--min-taxon-n', type=int, default=2,
                   help='Exclude singleton taxa from subtree recovery metrics')
    p.add_argument('--permutations', type=int, default=9999,
                   help='Permute GTDB labels among annotated leaves; 0 disables')
    p.add_argument('--taxon-permutations', type=int, default=9999,
                   help='Size-matched null for individual taxa; 0 disables')
    p.add_argument('--seed', type=int, default=20261028)
    p.add_argument('--all-subsets', action='store_true',
                   help='Also evaluate individual and combined genus outputs; default is all_genomes only')
    p.add_argument('--models', nargs='+', default=None)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    if args.min_taxon_n < 2 or args.permutations < 0 or args.taxon_permutations < 0:
        p.error('Require min-taxon-n>=2 and nonnegative permutation counts')
    if args.outdir.resolve() == args.input_dir.resolve():
        p.error('Use a separate output directory')
    candidates = sorted(args.input_dir.glob('*/**/condensed_tree.graphml'))
    candidates = [g.parent for g in candidates
                  if (args.all_subsets or g.parent.name == 'all_genomes')
                  and (args.models is None or g.parent.parent.name in args.models)]
    if not candidates:
        p.error(f'No matching v3 outputs found below {args.input_dir}')
    workbook, workbook_sha256 = load_gtdb_workbook(args.metadata)
    LOG.info('Loaded fresh taxonomy from %s (%d accessions)', args.metadata, len(workbook))
    global_summary, global_taxa = [], []
    manifest = []
    for folder in candidates:
        name = f'{folder.parent.name}/{folder.name}'
        LOG.info('Evaluating %s', name)
        target = args.outdir / folder.parent.name / folder.name
        try:
            rows, taxa = analyze(folder, target, args.ranks, args.min_taxon_n,
                                 args.permutations, args.seed, args.taxon_permutations, workbook)
        except Exception as exc:
            LOG.exception('Failed %s', name)
            manifest.append({'run': name, 'status': 'failed', 'error': str(exc)})
            continue
        global_summary.extend(rows)
        global_taxa.extend(taxa)
        manifest.append({'run': name, 'status': 'ok', 'error': ''})
    args.outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(global_summary).to_csv(args.outdir/'all_rank_summary.tsv', sep='\t', index=False)
    pd.DataFrame(global_taxa).to_csv(args.outdir/'all_taxon_details.tsv', sep='\t', index=False)
    pd.DataFrame(manifest).to_csv(args.outdir/'validation_manifest.tsv', sep='\t', index=False)
    config = {'metadata_workbook': str(args.metadata.resolve()),
              'metadata_sha256': workbook_sha256,
              'labels_from': 'current workbook matched by exact NCBI accession',
              'retrain_hdbscan': False, 'description': DESCRIPTION, 'ranks': args.ranks, 'min_taxon_n': args.min_taxon_n,
              'permutations': args.permutations, 'taxon_permutations': args.taxon_permutations,
              'seed': args.seed, 'all_subsets': args.all_subsets,
              'interpretation': 'Topology-based GTDB taxonomic coherence among rank-annotated leaves only; not evolutionary monophyly',
              'subtree_denominator': 'Only leaves with nonmissing GTDB label at each rank, original HDBSCAN tree unchanged',
              'noise_ari_caution': 'Noise as a shared class can mislead; always report nonnoise coverage'}
    (args.outdir/'validation_parameters.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    n_failed = sum(row['status'] == 'failed' for row in manifest)
    LOG.info('Completed %d runs (%d failed). Results: %s', len(manifest), n_failed, args.outdir)
    if n_failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
