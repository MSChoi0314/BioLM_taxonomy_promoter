#!/usr/bin/env python3
from __future__ import annotations
import argparse
import colorsys
import hashlib
import json
import logging
import re
import warnings
from pathlib import Path

import hdbscan
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

DESCRIPTION = """HDBSCAN condensed-tree GraphML for methanotroph pLM embeddings.

Default: direct HDBSCAN on full-dimensional, L2-normalized embeddings.
GTDB workbook is an accession-to-taxonomy annotation table (not a GTDB accession list).
Default: include every accession in metadata, including etc=unknown and blank genus.
Optional --exclude-accession removes exact accessions from all analyses.

GraphML encodes a density-based condensed hierarchy, NOT an evolutionary tree.
Requires numpy, pandas, scipy, scikit-learn, hdbscan, networkx, openpyxl.
"""

LOG = logging.getLogger('methcocca_hdbscan')
FILES = {
    'ProtT5': 'methcocca_prott5_mean_pool_embed_general.csv',
    'ESMC': 'methcocca_esmc_mean_pool_embed_general.csv',
    'ProtBERT': 'methcocca_protbert_mean_pool_embed_general.csv',
    'ESM2': 'methcocca_esm2_mean_pool_embed_general.csv',
}
DEFAULT_GENERA = ['Methylobacter', 'Methylobacter_A', 'Methylococcus',
                  'Methylomonas', 'Methylomicrobium', 'Methylotuvimicrobium',
                  'Methylosarcina']
DEFAULT_COMBINED = [
    'Methylomicrobium,Methylotuvimicrobium',
    'Methylomicrobium,Methylosarcina,Methylotuvimicrobium',
]


def accession(value):
    hit = re.search(r'(GC[AF]_\d+\.\d+)', str(value))
    return hit.group(1) if hit else ''


def clean(value):
    return '' if pd.isna(value) else str(value).strip()


def safe(value):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value)).strip('_')


PALETTES = {
    'vivid': ['#0072B2','#E7298A','#00A087','#E69F00','#7B43A8','#E64B35','#00BFC4','#679F00'],
    'bright': ['#2563EB','#E11D48','#0D9488','#F59E0B','#9333EA','#EA580C','#0891B2','#65A30D'],
    'contrast': ['#005BBB','#D81B60','#009E73','#F0A202','#6F42C1','#F15A3C','#009FB7','#84A20B'],
}
UNCLASSIFIED_COLOR = '#A9ADB5'
OTHER_COLOR = '#DDE5F0'
INTERNAL_COLOR = '#525866'


def read_metadata(path, exclusions=()):
    raw = pd.read_excel(path, dtype=str).fillna('')
    cols = {str(c).strip().lower().replace('_', ' '): c for c in raw.columns}
    required = ['assembly accession', 'family', 'genus', 'species', 'etc']
    missing = [k for k in required if k not in cols]
    if missing:
        raise ValueError(f'Missing Excel columns: {missing}')
    meta = pd.DataFrame({
        'accession': raw[cols['assembly accession']].map(accession),
        **{k: raw[cols[k]].map(clean) for k in ['family', 'genus', 'species', 'etc']},
    })
    meta['no_taxonomy'] = meta[['family', 'genus', 'species']].eq('').all(axis=1)
    meta['etc_unknown'] = meta['etc'].str.casefold().eq('unknown')
    meta['unclassified_genus'] = meta.genus.str.strip().eq('')
    meta['filter_reason'] = ''
    meta.loc[meta.accession.eq(''), 'filter_reason'] = 'invalid_accession'
    meta.loc[meta.accession.isin(exclusions) & meta.filter_reason.eq(''), 'filter_reason'] = 'user_excluded'
    ids = meta.loc[meta.accession.ne(''), 'accession']
    if ids.duplicated().any():
        raise ValueError('Duplicated accessions in Excel: ' + ', '.join(ids[ids.duplicated()].tolist()[:10]))
    return meta


def make_colors(metadata, threshold):
    meta = metadata.copy()
    counts = meta.loc[meta.genus.ne(''), 'genus'].value_counts()
    major = sorted(counts[counts >= threshold].index, key=lambda g: (-int(counts[g]), g))
    if len(major) > 8:
        raise ValueError(f'{len(major)} major genera exceed 8 distinct color slots. '
                         'Increase --major-genus-min-n to avoid color reuse.')
    major_index = {g:i for i,g in enumerate(major)}
    meta['genus_group'] = meta.genus.map(
        lambda g: 'Unclassified' if not g else (g if g in major_index else 'Other'))
    meta['species_group'] = meta.apply(
        lambda r: r.species if r.species else (f'{r.genus} | unidentified species' if r.genus else 'Unclassified'), axis=1)
    meta['taxon_species_key'] = meta.apply(
        lambda r: (r.genus + ' | ' + r.species) if r.genus and r.species
        else ((r.genus + ' | unidentified species') if r.genus else 'Unclassified'), axis=1)
    ordered = {g: sorted(meta.loc[meta.genus.eq(g), 'taxon_species_key'].unique()) for g in major}
    for palname, base in PALETTES.items():
        column = 'color_' + palname
        genus_col = column + '_genus'
        species_col = column + '_species'
        meta[genus_col] = meta.genus_group.map(
            lambda group: UNCLASSIFIED_COLOR if group == 'Unclassified'
            else OTHER_COLOR if group == 'Other' else base[major_index[group]])
        color_by_species = {}
        for g, keys in ordered.items():
            root = base[major_index[g]]
            for j, key in enumerate(keys):
                color_by_species[key] = vary_color(root, j, len(keys))
        meta[species_col] = meta.apply(
            lambda r: (UNCLASSIFIED_COLOR if r.genus_group == 'Unclassified' else
                       color_by_species[r.taxon_species_key] if r.genus_group != 'Other'
                       else OTHER_COLOR), axis=1)
        meta[column + '_species_all'] = meta[species_col]
        meta[column] = meta[genus_col]
    return meta, major, counts


def vary_color(hex_color, index, size):
    if size <= 1:
        return hex_color
    rgb = [int(hex_color[k:k+2], 16)/255 for k in (1,3,5)]
    h, l, sat = colorsys.rgb_to_hls(*rgb)
    position = index / (size - 1)
    l = 0.29 + 0.43 * position
    sat = 0.83 if index % 2 == 0 else 0.66
    h = (h + (index % 3 - 1) * 0.014) % 1.0
    return '#' + ''.join(f'{round(255*z):02X}' for z in colorsys.hls_to_rgb(h,l,sat))


def recolor_subset(meta, subset):
    if not subset.startswith(('genus_', 'combined_')):
        return meta
    meta = meta.copy()
    genera = sorted(g for g in meta.genus.unique() if g)
    for palette_name, swatches in PALETTES.items():
        scol = 'color_' + palette_name + '_species'
        allcol = 'color_' + palette_name + '_species_all'
        minor_base = {'vivid':'#F43F5E', 'bright':'#BE185D', 'contrast':'#C026D3'}[palette_name]
        for g in genera:
            keys = sorted(meta.loc[meta.genus.eq(g), 'taxon_species_key'].unique())
            genus_base = meta.loc[meta.genus.eq(g), 'color_'+palette_name+'_genus'].iloc[0]
            if genus_base == OTHER_COLOR:
                genus_base = minor_base
            for j, key in enumerate(keys):
                value = vary_color(genus_base, j, len(keys))
                mask = meta.genus.eq(g) & meta.taxon_species_key.eq(key)
                meta.loc[mask, scol] = value
                meta.loc[mask, allcol] = value
    return meta


def write_legend(metadata, folder, subset):
    rows = []
    for col in [c for c in metadata.columns if c.startswith('color_')]:
        labelcol = 'genus_group' if col.endswith('_genus') or col in ('color_vivid','color_bright','color_contrast') else 'taxon_species_key'
        display = metadata[list(dict.fromkeys([labelcol, col, 'genus_group']))].copy()
        if labelcol == 'taxon_species_key' and subset == 'all_genomes':
            display.loc[display.genus_group.isin(['Other','Unclassified']), labelcol] = display.genus_group
        unique = display[[labelcol, col]].drop_duplicates().sort_values([labelcol,col])
        for _, row in unique.iterrows():
            rows.append({'subset':subset,'color_column':col,'taxon_label':row[labelcol], 'hex_color':row[col]})
    pd.DataFrame(rows).to_csv(folder/'color_legend.tsv', sep='\t', index=False)


def read_embedding(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    headers = pd.read_csv(path, nrows=0).columns.tolist()
    if 'genome_accession' not in headers:
        raise ValueError(f'{path}: genome_accession missing')
    columns = sorted([c for c in headers if re.fullmatch(r'embedding_\d+', str(c))],
                     key=lambda s: int(s.rsplit('_', 1)[1]))
    if not columns:
        raise ValueError(f'{path}: embedding_N columns missing')
    data = pd.read_csv(path, usecols=['genome_accession'] + columns, low_memory=False)
    ids = data['genome_accession'].map(accession)
    X = data[columns].to_numpy(dtype=np.float64, copy=True)
    norm = np.linalg.norm(X, axis=1)
    good = ids.ne('').to_numpy() & np.isfinite(X).all(axis=1) & np.isfinite(norm) & (norm > 0)
    if (~good).sum():
        LOG.warning('%s: removed %d invalid accessions/embeddings', path.name, int((~good).sum()))
    ids = ids.loc[good].reset_index(drop=True)
    if ids.duplicated().any():
        raise ValueError(f'{path}: duplicate genome accessions: {ids[ids.duplicated()].tolist()[:8]}')
    X = X[good] / norm[good, None]
    return pd.DataFrame({'accession': ids}), np.ascontiguousarray(X)


def identify_duplicates(X, meta):
    groups = {}
    for i, row in enumerate(X):
        groups.setdefault(row.tobytes(), []).append(i)
    dup = [v for v in groups.values() if len(v) > 1]
    rows = []
    for j, group in enumerate(dup, 1):
        for i in group:
            rows.append({'duplicate_group': j, 'accession': meta.iloc[i]['accession'],
                         'sample_index': i, 'group_size': len(group)})
    return pd.DataFrame(rows, columns=['duplicate_group', 'accession', 'sample_index', 'group_size'])


def build_graph(clusterer, meta, transformed, mode):
    edges = clusterer.condensed_tree_.to_pandas()
    n = len(meta)
    G = nx.DiGraph()
    for i, row in meta.reset_index(drop=True).iterrows():
        G.add_node(f'leaf_{i}', name=str(row['accession']),
                   label=str(row['accession']), node_type='sample', sample_index=int(i),
                   accession=str(row['accession']), family=str(row['family']),
                   genus=str(row['genus']), species=str(row['species']), etc=str(row['etc']),
                   hdbscan_cluster=int(clusterer.labels_[i]),
                   hdbscan_probability=float(clusterer.probabilities_[i]),
                   hdbscan_outlier_score=float(clusterer.outlier_scores_[i])
                   if np.isfinite(clusterer.outlier_scores_[i]) else -1.0,
                   is_noise=int(clusterer.labels_[i] == -1),
                   genus_group=str(row['genus_group']), species_group=str(row['species_group']),
                   taxon_species_key=str(row['taxon_species_key']),
                   **{col: str(row[col]) for col in meta.columns if col.startswith('color_')})
    for edge in edges.itertuples(index=False):
        parent_num, child_num = int(edge.parent), int(edge.child)
        p = f'internal_{parent_num}'
        c = f'leaf_{child_num}' if child_num < n else f'internal_{child_num}'
        if p not in G:
            G.add_node(p, name=p, label='', node_type='internal', hdbscan_id=parent_num,
                       **{col: INTERNAL_COLOR for col in meta.columns if col.startswith('color_')})
        if child_num >= n and c not in G:
            G.add_node(c, name=c, label='', node_type='internal', hdbscan_id=child_num,
                       **{col: INTERNAL_COLOR for col in meta.columns if col.startswith('color_')})
        lam = float(edge.lambda_val)
        G.add_edge(p, c, lambda_val=float(lam) if np.isfinite(lam) else 'inf',
                   lambda_is_infinite=int(not np.isfinite(lam)),
                   child_size=int(edge.child_size), relation='condensed_hierarchy',
                   inverse_lambda_distance=float(1.0 / lam) if np.isfinite(lam) and lam > 0 else 0.0)
    if not nx.is_arborescence(G):
        raise ValueError('The generated condensed hierarchy is not a single rooted directed tree.')
    roots = [node for node, deg in G.in_degree() if deg == 0]
    G.graph.update(description='HDBSCAN condensed hierarchy; not a phylogenetic tree',
                   root_node=roots[0], distance='euclidean', normalization='L2',
                   reduction=mode, n_input_dimensions=int(transformed.shape[1]))
    return G, edges


def fit_subset(model, name, indices, X, meta, outdir, args):
    folder = outdir / model / safe(name)
    folder.mkdir(parents=True, exist_ok=True)
    m = recolor_subset(meta.iloc[indices].reset_index(drop=True), name)
    full_X = np.ascontiguousarray(X[indices], dtype=np.float64)
    n, d = full_X.shape
    summary = {'model': model, 'subset': name, 'n_samples': n, 'embedding_dimensions': d,
               'mode': args.mode, 'status': 'skipped', 'n_clusters': 0, 'n_noise': 0}
    if n < max(3, args.min_cluster_size):
        LOG.warning('[SKIP] %s / %s: n=%d', model, name, n)
        summary['reason'] = 'insufficient_samples'
        return summary
    if args.mode == 'pca':
        maximum = min(n - 1, d)
        if args.pca_components > maximum:
            LOG.warning('[SKIP] %s / %s: PCA components %d > allowable %d',
                        model, name, args.pca_components, maximum)
            summary['reason'] = 'pca_components_exceed_subset_rank_bound'
            return summary
        pca = PCA(n_components=args.pca_components, svd_solver='full')
        Z = np.ascontiguousarray(pca.fit_transform(full_X))
        summary['pca_explained_variance_ratio'] = float(pca.explained_variance_ratio_.sum())
        summary['dimensions_used'] = args.pca_components
    else:
        Z = full_X
        summary['dimensions_used'] = d
    duplicates = identify_duplicates(Z, m)
    duplicates.to_csv(folder / 'identical_embeddings.tsv', sep='\t', index=False)
    if len(duplicates):
        LOG.info('[DUPLICATES] %s / %s: %d genomes in exact duplicate-vector groups; vectors unchanged',
                 model, name, len(duplicates))
    LOG.info('[RUN] %s / %s: n=%d, dimensions=%d, mode=%s', model, name, n, Z.shape[1], args.mode)
    estimator = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric='euclidean', cluster_selection_method='eom',
        allow_single_cluster=args.allow_single_cluster,
        core_dist_n_jobs=args.jobs, approx_min_span_tree=False,
        match_reference_implementation=False,
    )
    labels = estimator.fit_predict(Z)
    G, edges = build_graph(estimator, m, Z, args.mode)
    infinity_count = int((~np.isfinite(edges['lambda_val'].to_numpy(float))).sum())
    summary['infinite_lambda_edges'] = infinity_count
    summary['exact_duplicate_genomes'] = len(duplicates)
    if infinity_count:
        LOG.info('[INFINITE LAMBDA] %s / %s: %d edges; retained in source hierarchy, marked in GraphML',
                 model, name, infinity_count)
    samples = m.copy()
    samples['sample_index'] = np.arange(n)
    samples['graph_node_id'] = [f'leaf_{i}' for i in range(n)]
    samples['hdbscan_cluster'] = labels
    samples['hdbscan_probability'] = estimator.probabilities_
    samples['hdbscan_outlier_score'] = estimator.outlier_scores_
    samples.to_csv(folder / 'sample_metadata.tsv', sep='\t', index=False)
    write_legend(samples, folder, name)
    edges.to_csv(folder / 'condensed_tree_edges.tsv', sep='\t', index=False)
    nx.write_graphml(G, folder / 'condensed_tree.graphml')
    reread = nx.read_graphml(folder / 'condensed_tree.graphml')
    if (reread.number_of_nodes() != G.number_of_nodes()
            or reread.number_of_edges() != G.number_of_edges()
            or not nx.is_arborescence(reread)):
        raise RuntimeError(f'{folder}: GraphML round-trip validation failed')
    cluster_summary = (samples.loc[samples.hdbscan_cluster.ge(0)]
                       .groupby('hdbscan_cluster').agg(
                           n=('accession', 'size'),
                           genera=('genus', lambda s: ';'.join(sorted(set(s) - {''}))),
                           species=('species', lambda s: ';'.join(sorted(set(s) - {''})))
                       ).reset_index())
    cluster_summary.to_csv(folder / 'cluster_summary.tsv', sep='\t', index=False)
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(11, 6))
            with warnings.catch_warnings():
                if infinity_count:
                    warnings.filterwarnings('ignore', message='Infinite lambda values encountered in chosen clusters.*', category=UserWarning)
                estimator.condensed_tree_.plot(axis=ax, select_clusters=(infinity_count == 0))
            ax.set_title(f'{model} | {name} | HDBSCAN condensed hierarchy (n={n})')
            fig.tight_layout()
            fig.savefig(folder / 'condensed_tree.png', dpi=220)
            plt.close(fig)
        except (ImportError, ValueError, RuntimeError, OverflowError) as exc:
            LOG.warning('[PLOT] %s / %s: image plot unavailable: %s', model, name, exc)
    summary.update(status='ok', n_clusters=len(set(labels) - {-1}),
                   n_noise=int(np.sum(labels == -1)), n_graph_nodes=G.number_of_nodes(),
                   n_graph_edges=G.number_of_edges(), n_internal_nodes=G.number_of_nodes() - n)
    (folder / 'run_details.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main():
    p = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input-dir', type=Path, default=Path('00.DATASET'))
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--outdir', type=Path, default=Path('methcocca_hdbscan_graphml_v2'))
    p.add_argument('--min-cluster-size', type=int, default=2)
    p.add_argument('--min-samples', type=int, default=2)
    p.add_argument('--jobs', type=int, default=1)
    p.add_argument('--mode', choices=['direct', 'pca'], default='direct',
                   help='Direct full-dimensional embeddings (default), or optional subset-wise PCA.')
    p.add_argument('--pca-components', type=int, default=5)
    p.add_argument('--genera', nargs='+', default=DEFAULT_GENERA)
    p.add_argument('--min-genus-n', type=int, default=3)
    p.add_argument('--combined-group', action='append', default=[],
                   help='Comma-separated genus names; repeat to add more groups.')
    p.add_argument('--no-default-combined', action='store_true')
    p.add_argument('--exclude-accession', action='append', default=[],
                   help='Accession to exclude across ALL subsets. Repeat for multiple. Also accepts comma-separated IDs.')
    p.add_argument('--exclude-file', type=Path, default=None,
                   help='Optional text file with accessions, one per line (also supports # comments).')
    p.add_argument('--major-genus-min-n', type=int, default=5,
                   help='Minimum number of matched genomes to give a genus its own color.')
    p.add_argument('--no-common-intersection', action='store_true',
                   help='Not supported for stable cross-model color mapping; omit this flag.')
    p.add_argument('--allow-single-cluster', action='store_true')
    p.add_argument('--no-plot', action='store_true')
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    if a.min_cluster_size < 2 or a.min_samples < 1 or a.min_genus_n < 2 or a.pca_components < 1:
        p.error('Invalid min_cluster_size, min_samples, min_genus_n, or pca_components')
    a.outdir.mkdir(parents=True, exist_ok=True)
    excluded = {z.strip() for item in a.exclude_accession for z in item.split(',') if z.strip()}
    if a.exclude_file is not None:
        for line in a.exclude_file.read_text(encoding='utf-8').splitlines():
            line = line.split('#', 1)[0].strip()
            if line:
                excluded.add(line)
    invalid_excludes = sorted(z for z in excluded if accession(z) != z)
    if invalid_excludes:
        p.error('Invalid excluded accession(s): ' + ', '.join(invalid_excludes))
    if a.major_genus_min_n < 2:
        p.error('--major-genus-min-n must be at least 2')
    if a.no_common_intersection:
        p.error('--no-common-intersection is disabled for consistent palette identities across the four models')
    excel = read_metadata(a.metadata, excluded)
    excel.to_csv(a.outdir / 'metadata_filter_audit.tsv', sep='\t', index=False)
    selected = excel.loc[excel.filter_reason.eq(''), ['accession', 'family', 'genus', 'species', 'etc']].copy()
    LOG.info('Excel: %d total; etc=unknown %d; blank genus %d; user excluded %d; retained %d',
             len(excel), int(excel.etc_unknown.sum()), int(excel.unclassified_genus.sum()),
             int((excel.filter_reason == 'user_excluded').sum()), len(selected))
    if selected.empty:
        raise ValueError('No Excel rows remain after filtering')
    loaded, audit = {}, []
    for model, filename in FILES.items():
        recs, matrix = read_embedding(a.input_dir / filename)
        merged = selected.merge(recs.assign(embedding_index=np.arange(len(recs))),
                                on='accession', how='inner', validate='one_to_one')
        loaded[model] = (merged, matrix)
        audit.append({'model': model, 'valid_embedding_rows': len(recs),
                      'selected_excel_rows': len(selected), 'matched_rows': len(merged),
                      'excel_only': len(selected) - len(merged),
                      'embedding_only': len(recs) - len(merged)})
    pd.DataFrame(audit).to_csv(a.outdir / 'matching_summary.tsv', sep='\t', index=False)
    common = set.intersection(*(set(m.accession) for m, _ in loaded.values()))
    selected.assign(in_all_models=selected.accession.isin(common)).to_csv(
        a.outdir / 'gtdb_matching_audit.tsv', sep='\t', index=False)
    LOG.info('Retained Excel accessions present in all four embedding models: %d', len(common))
    color_meta = selected.loc[selected.accession.isin(common)].copy()
    color_meta, major, genus_counts = make_colors(color_meta, a.major_genus_min_n)
    LOG.info('Major genera (n >= %d): %s', a.major_genus_min_n,
             ', '.join(f'{g}={genus_counts[g]}' for g in major))
    color_meta.to_csv(a.outdir/'color_assignment_global.tsv', sep='\t', index=False)
    color_meta[['genus','genus_group']].drop_duplicates().to_csv(
        a.outdir/'genus_group_audit.tsv', sep='\t', index=False)
    results = []
    combined_groups = ([] if a.no_default_combined else list(DEFAULT_COMBINED)) + a.combined_group
    seen = set()
    for model, (merged, matrix) in loaded.items():
        if not a.no_common_intersection:
            merged = merged.loc[merged.accession.isin(common)]
        merged = merged.sort_values('accession').reset_index(drop=True)
        m = merged[['accession', 'family', 'genus', 'species', 'etc']].copy()
        m = m.merge(color_meta.drop(columns=['family','genus','species','etc']),
                    on='accession', how='left', validate='one_to_one')
        if m['genus_group'].isna().any():
            raise ValueError('Some model accessions lack a global color assignment. Use common intersection.')
        x = matrix[merged.embedding_index.to_numpy(dtype=int)]
        if len(m) < 3:
            raise ValueError(f'{model}: fewer than 3 matched genomes')
        results.append(fit_subset(model, 'all_genomes', np.arange(len(m)), x, m, a.outdir, a))
        for genus in a.genera:
            idx = np.flatnonzero(m.genus.eq(genus).to_numpy())
            results.append(fit_subset(model, 'genus_' + genus, idx, x, m, a.outdir, a))
        for text in combined_groups:
            groups = list(dict.fromkeys(z.strip() for z in text.split(',') if z.strip()))
            if len(groups) < 2:
                raise ValueError(f'Invalid --combined-group: {text}')
            key = tuple(sorted(groups))
            if key in seen:
                continue
            idx = np.flatnonzero(m.genus.isin(groups).to_numpy())
            results.append(fit_subset(model, 'combined_' + '__'.join(groups), idx, x, m, a.outdir, a))
        seen.clear()
    pd.DataFrame(results).to_csv(a.outdir / 'run_summary.tsv', sep='\t', index=False)
    params = {'mode': a.mode, 'l2_normalization': True, 'metric': 'euclidean',
              'pca_components': a.pca_components if a.mode == 'pca' else None,
              'min_cluster_size': a.min_cluster_size, 'min_samples': a.min_samples,
              'included_unclassified_and_unknown': True,
              'excluded_accessions': sorted(excluded),
              'major_genus_min_n': a.major_genus_min_n,
              'major_genera': major,
              'palette_names': list(PALETTES),
              'common_intersection': not a.no_common_intersection,
              'default_combined': not a.no_default_combined,
              'graph_semantics': 'density-based HDBSCAN condensed tree; not a phylogenetic tree'}
    (a.outdir / 'parameters.json').write_text(json.dumps(params, indent=2), encoding='utf-8')
    LOG.info('Done: %s', a.outdir)


if __name__ == '__main__':
    main()
