#!/usr/bin/env python3
"""전체 460k pose의 feature를 하나로 통합하고 sc_rmsd 라벨을 붙입니다(Process A 마무리, ④).

입력은 세 갈래의 전체 추출 결과와 라벨입니다.
  - A6c(21 merge) combined_features_all.parquet : cat1 접촉, cat2 상호작용, cat3 표면,
                                                   cat4 리간드 기하, cat6/7 물·금속, cat8 descriptor
  - P2Rank(22 merge) pocket_features_all.parquet : cat5 포켓 (아직 없으면 생략 가능)
  - gnina(추후) docking score                     : cat2 보강 (아직 없으면 생략 가능)
  - 24 labels.parquet                             : sc_rmsd 정답 라벨
  - 18 system_split.csv                           : dev/test 분할 (있으면 결합)

A6c 결과를 기준 pose 집합으로 삼아 (system_id, model, seed, sample)로 left join 하며,
각 입력의 pose 키 집합이 기준과 같은지 검사합니다. feature를 0으로 채우지 않고 널을 그대로
보존하며, 각 추출원의 status·사유 열을 접두사와 함께 남겨 왜 비었는지 추적할 수 있게 합니다.
feature 이름은 추출기가 쓴 이름을 그대로 두고, 어떤 feature가 어느 후보군인지는 별도
feature_catalog.csv로 제공합니다(열 이름을 바꾸지 않아 하류에서 혼동이 없게 합니다).

출력: out/feature_matrix_all/26_run_integrate_all[_TAG]/ 아래 고정 폴더.
"""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import Counter
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import time

_support_path = Path(__file__).with_name('05_audit_input_roles.py')
_spec = importlib.util.spec_from_file_location('roles_support', _support_path)
roles = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(roles)
audit = roles.audit
helper = roles.helper

VERSION = 'feature_matrix_all_0.1.0'
OUT_KIND = 'feature_matrix_all'
OUT_BASE = '26_run_integrate_all'
KEYS = ['system_id', 'model', 'seed', 'sample']
BASE = helper.ALLOWED_ROOT / 'Scoring_Function' / 'out'

DEFAULT_COMBINED = BASE / 'combined_all' / '21_run_combined_shard' / 'combined_features_all.parquet'
DEFAULT_POCKET = BASE / 'pocket_all' / '22_run_pocket_shard' / 'pocket_features_all.parquet'
DEFAULT_LABELS = BASE / 'labels' / '24_run_build_labels' / 'labels.parquet'
DEFAULT_SPLIT = BASE / 'data_split' / '18_run_make_split' / 'system_split.csv'

# 추출기가 공통으로 붙이는 메타 열. 이 목록을 뺀 나머지를 feature로 간주합니다.
META_COLUMNS = {'system_id', 'model', 'seed', 'sample', 'pose_id', 'schema_version', 'cif_sha256',
                'status', 'skip_reasons_json', 'error_message',
                'ligand_typing', 'cat4_available', 'ligand_mapping_status'}
# 모드별 상태 열(gnina의 mode_status_*)은 feature가 아니라 널의 원인을 설명하는 메타입니다.
META_PREFIXES = ('mode_status_',)
# 각 추출원에서 접두사를 붙여 보존할 메타 열(겹치는 이름 충돌 방지).
PREFIXED_META = ('status', 'skip_reasons_json', 'error_message', 'schema_version', 'cif_sha256')

# feature -> 후보군. 열 이름은 바꾸지 않고 분류만 제공합니다(선별 단계에서 군별로 다루기 위함).
CATEGORY_OF = {}
for _cat, _names in [
    ('cat1_contact', ['pl_min_distance_A'] + [f'{p}_lt_{c}A' for c in (3, 4, 5)
     for p in ('pl_pairs', 'ligand_atoms', 'protein_atoms', 'protein_residues')]),
    ('cat2_interaction', ['potential_hbond_pairs', 'hydrophobic_pairs', 'receptor_clash_pairs',
                          'salt_bridge_pairs', 'aromatic_contact_pairs']),
    ('cat3_surface', ['ligand_sasa_isolated_A2', 'ligand_sasa_complex_A2', 'ligand_buried_sasa_A2',
                      'ligand_buried_fraction', 'ligand_sasa_isolated_polar_A2',
                      'ligand_sasa_isolated_nonpolar_A2', 'ligand_buried_polar_A2']),
    ('cat4_geometry', ['lig_bond_count', 'lig_bond_len_min_A', 'lig_bond_len_max_A',
                       'lig_bond_len_mean_A', 'lig_long_bonds', 'lig_short_bonds',
                       'lig_internal_clashes', 'lig_min_nonbonded_dist_A']),
    ('cat5_pocket', ['num_predicted_pockets', 'top_pocket_score', 'top_pocket_probability',
                     'ligand_pocket_rank', 'ligand_pocket_score', 'ligand_pocket_probability',
                     'ligand_to_pocket_center_dist_A']),
    ('cat67_water_metal', ['water_observed_count', 'water_bridge_count', 'metal_observed_count',
                           'metal_in_ligand_count', 'metal_in_receptor_count',
                           'metal_independent_count', 'min_metal_ligand_distance_A',
                           'metal_ligand_receptor_bridge']),
    ('cat8_ligand', ['lig_mol_weight', 'lig_heavy_atoms', 'lig_rotatable_bonds', 'lig_hbd',
                     'lig_hba', 'lig_tpsa', 'lig_logp', 'lig_ring_count', 'lig_aromatic_rings',
                     'lig_fraction_csp3', 'lig_num_heteroatoms', 'lig_formal_charge']),
]:
    for _n in _names:
        CATEGORY_OF[_n] = _cat

LABEL_COLUMNS = ['sc_rmsd', 'ligand_rmsd', 'label_known', 'ligand_rmsd_known',
                 'near_native', 'near_native_fallback', 'eval_status', 'eval_mapping_status']


def normalize_keys(df):
    """조인 전에 키 열의 dtype을 맞춥니다."""
    df['system_id'] = df['system_id'].astype(str)
    df['model'] = df['model'].astype(str)
    for k in ('seed', 'sample'):
        df[k] = df[k].astype(int)
    return df


def load_source(path, prefix, keep_extra=()):
    """추출 결과를 읽어 키 + feature + 접두사 붙인 메타 열만 남깁니다."""
    import pandas as pd
    df = normalize_keys(pd.read_parquet(path))
    if df.duplicated(KEYS).any():
        raise ValueError(f'{prefix}: pose 키 중복이 있습니다: {path}')
    features = [c for c in df.columns
                if c not in META_COLUMNS and not c.startswith(META_PREFIXES)]
    status_extra = [c for c in df.columns if c.startswith(META_PREFIXES)]
    out = df[KEYS + features + status_extra].copy()
    for col in PREFIXED_META:
        if col in df.columns:
            out[f'{prefix}_{col}'] = df[col].values
    for col in keep_extra:
        if col in df.columns:
            out[col] = df[col].values
    keys = set(map(tuple, df[KEYS].to_numpy().tolist()))
    return out, features, keys


def run(args):
    """세 추출원과 라벨을 pose 키로 통합해 학습용 행렬을 만듭니다."""
    started = time.perf_counter()
    out_name = OUT_BASE + (f'_{args.tag}' if args.tag else '')
    parent = helper.safe_mkdir(BASE / OUT_KIND)
    out = helper.checked_path(parent / out_name)
    if out.exists():
        raise FileExistsError(f'기존 결과를 덮어쓰지 않습니다(--tag로 분리): {out}')
    out.mkdir()
    for name in ('tmp', 'cache'):
        helper.safe_mkdir(out / name)
    for key in ('TMPDIR', 'TMP', 'TEMP'):
        os.environ[key] = str(out / 'tmp')
    os.environ['XDG_CACHE_HOME'] = str(out / 'cache')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    print('OUTPUT_DIR:', out, flush=True)

    import pandas as pd
    import pyarrow as pa
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)

    sources, source_info = [], []

    # 1) A6c(21) 결과를 기준 pose 집합으로 삼습니다.
    if not args.combined.exists():
        raise FileNotFoundError(f'A6c 통합 결과가 없습니다: {args.combined}')
    sources.append(helper.fingerprint(args.combined))
    # 리간드 대응 상태는 왜 cat2·cat4가 비는지 설명하므로 그대로 보존합니다.
    base, feats_combined, base_keys = load_source(
        args.combined, 'combined', keep_extra=('ligand_mapping_status', 'cat4_available', 'ligand_typing'))
    print('COMBINED poses:', len(base), '| features:', len(feats_combined), flush=True)
    source_info.append({'source': 'combined(A6c)', 'path': str(args.combined),
                        'poses': int(len(base)), 'features': len(feats_combined)})

    merged = base
    attached = {'cat5_pocket': False, 'gnina': False}

    # 2) P2Rank(22) 결과 - 아직 없으면 생략합니다.
    key_mismatch = {}
    if args.pocket is not None and args.pocket.exists():
        sources.append(helper.fingerprint(args.pocket))
        pocket, feats_pocket, pkeys = load_source(args.pocket, 'pocket')
        key_mismatch['pocket'] = {'missing_in_source': len(base_keys - pkeys),
                                  'extra_in_source': len(pkeys - base_keys)}
        merged = merged.merge(pocket, on=KEYS, how='left')
        attached['cat5_pocket'] = True
        print('POCKET poses:', len(pocket), '| features:', len(feats_pocket), flush=True)
        source_info.append({'source': 'pocket(P2Rank)', 'path': str(args.pocket),
                            'poses': int(len(pocket)), 'features': len(feats_pocket)})
    else:
        print('POCKET: 결과가 없어 생략합니다(cat5 미포함).', flush=True)

    # 3) gnina 결과 - 아직 없으면 생략합니다.
    if args.gnina is not None and args.gnina.exists():
        sources.append(helper.fingerprint(args.gnina))
        gn, feats_gnina, gkeys = load_source(args.gnina, 'gnina')
        key_mismatch['gnina'] = {'missing_in_source': len(base_keys - gkeys),
                                 'extra_in_source': len(gkeys - base_keys)}
        merged = merged.merge(gn, on=KEYS, how='left')
        attached['gnina'] = True
        for name in feats_gnina:
            CATEGORY_OF.setdefault(name, 'cat2_gnina')
        print('GNINA poses:', len(gn), '| features:', len(feats_gnina), flush=True)
        source_info.append({'source': 'gnina', 'path': str(args.gnina),
                            'poses': int(len(gn)), 'features': len(feats_gnina)})
    else:
        print('GNINA: 결과가 없어 생략합니다(docking score 미포함).', flush=True)

    # 4) 라벨(24) 결합.
    if not args.labels.exists():
        raise FileNotFoundError(f'라벨 테이블이 없습니다: {args.labels}')
    sources.append(helper.fingerprint(args.labels))
    lab = normalize_keys(pd.read_parquet(args.labels))
    lkeys = set(map(tuple, lab[KEYS].to_numpy().tolist()))
    key_mismatch['labels'] = {'missing_in_source': len(base_keys - lkeys),
                              'extra_in_source': len(lkeys - base_keys)}
    merged = merged.merge(lab[KEYS + LABEL_COLUMNS], on=KEYS, how='left')

    # 5) 분할(18) 결합 - 있으면.
    split_attached = False
    if args.split is not None and Path(args.split).exists():
        sources.append(helper.fingerprint(args.split))
        sp = pd.read_csv(args.split, dtype={'system_id': str})
        cols = [c for c in ('system_id', 'split', 'cv_fold', 'pocket_cluster', 'is_diagnostic')
                if c in sp.columns]
        merged = merged.merge(sp[cols].drop_duplicates('system_id'), on='system_id', how='left')
        split_attached = True

    merged.insert(4, 'pose_id', merged['system_id'] + '|' + merged['model'] + '|'
                  + merged['seed'].astype(str) + '|' + merged['sample'].astype(str))
    if merged.duplicated(KEYS).any() or len(merged) != len(base_keys):
        raise RuntimeError('통합 과정에서 행 수·중복이 맞지 않습니다.')

    # 6) feature 목록과 분류(열 이름은 바꾸지 않습니다).
    feature_columns = [c for c in merged.columns if c in CATEGORY_OF]
    uncatalogued = [c for c in merged.columns
                    if c not in CATEGORY_OF and c not in META_COLUMNS
                    and c not in LABEL_COLUMNS
                    and not c.startswith(META_PREFIXES)
                    and not c.endswith(('_status', '_skip_reasons_json', '_error_message',
                                        '_schema_version', '_cif_sha256'))
                    and c not in ('split', 'cv_fold', 'pocket_cluster', 'is_diagnostic')]

    catalog_rows = []
    for col in feature_columns:
        nn = int(merged[col].notna().sum())
        catalog_rows.append({'feature': col, 'category': CATEGORY_OF[col],
                             'non_null': nn, 'null': int(len(merged) - nn),
                             'coverage': round(nn / len(merged), 4)})
    catalog_rows.sort(key=lambda r: (r['category'], r['feature']))
    audit.save_csv(out / 'feature_catalog.csv', catalog_rows,
                   ('feature', 'category', 'non_null', 'null', 'coverage'))

    # 7) 후보군별 커버리지 요약.
    cat_rows = []
    for cat in sorted({r['category'] for r in catalog_rows}):
        cols = [r['feature'] for r in catalog_rows if r['category'] == cat]
        full = int(merged[cols].notna().all(axis=1).sum())
        cat_rows.append({'category': cat, 'features': len(cols), 'poses_all_features': full,
                         'coverage': round(full / len(merged), 4)})
    audit.save_csv(out / 'category_coverage.csv', cat_rows,
                   ('category', 'features', 'poses_all_features', 'coverage'))

    # 8) 추출 상태 분포(왜 비었는지 추적용).
    status_rows = []
    for col in [c for c in merged.columns if c.endswith('_status') or c == 'ligand_mapping_status']:
        for value, n in Counter(merged[col].fillna('(null)')).most_common():
            status_rows.append({'column': col, 'value': str(value), 'count': int(n)})
    audit.save_csv(out / 'status_breakdown.csv', status_rows, ('column', 'value', 'count'))

    # 9) 라벨·분할 요약.
    n_label = int(merged['label_known'].fillna(False).sum())
    n_nn = int(merged['near_native'].fillna(False).sum())
    sys_total = int(merged['system_id'].nunique())
    elig = merged.groupby('system_id')['near_native'].any()
    n_elig = int(elig.sum())
    split_counts = (merged.drop_duplicates('system_id')['split'].value_counts().to_dict()
                    if split_attached else None)

    # 10) 주 산출물 저장.
    buf = io.BytesIO()
    merged.to_parquet(buf, engine='pyarrow', index=False, compression='snappy')
    with helper.checked_path(out / 'unified_all.parquet').open('xb') as handle:
        handle.write(buf.getvalue())

    changed = []
    for source in sources:
        try:
            info = Path(source['path']).stat()
            if (info.st_size, info.st_mtime_ns) != (source['size_bytes'], source['mtime_ns']):
                changed.append(source['path'])
        except OSError:
            changed.append(source['path'])

    keys_ok = all(v['missing_in_source'] == 0 and v['extra_in_source'] == 0
                  for v in key_mismatch.values())
    complete = attached['cat5_pocket'] and attached['gnina']
    status = 'INTEGRATION_COMPLETE' if (keys_ok and not changed and not uncatalogued) else 'CHECK_REQUIRED'
    report = {
        'version': VERSION, 'status': status, 'output_dir': str(out),
        'inputs': source_info, 'sources': sources,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'support_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (_support_path, roles._support_path, audit._support_path)},
        'python': sys.executable, 'pyarrow_version': pa.__version__,
        'poses': int(len(merged)), 'systems': sys_total,
        'feature_count': len(feature_columns), 'categories_attached': attached,
        'all_feature_sources_present': complete,
        'key_mismatch': key_mismatch, 'key_sets_match': keys_ok,
        'uncatalogued_columns': uncatalogued,
        'category_coverage': cat_rows,
        'labels': {'poses_label_known': n_label,
                   'poses_label_missing': int(len(merged) - n_label),
                   'poses_near_native': n_nn,
                   'systems': sys_total, 'eligible_systems': n_elig,
                   'oracle_rate': round(n_elig / sys_total, 4) if sys_total else None},
        'split_attached': split_attached, 'split_systems': split_counts,
        'changed_sources': changed, 'elapsed_seconds': time.perf_counter() - started,
        'policy': ['feature를 0으로 채우지 않고 널을 보존합니다.',
                   'feature 열 이름을 바꾸지 않으며 후보군 분류는 feature_catalog.csv로 제공합니다.',
                   '각 추출원의 status·사유 열을 접두사와 함께 남겨 널의 원인을 추적할 수 있게 합니다.',
                   'A6c 결과를 기준 pose 집합으로 삼고 나머지는 left join 후 키 집합 일치를 검사합니다.'],
        'notes': ['cat5(P2Rank) 또는 gnina 결과가 없으면 해당 열 없이 생성되며, '
                  '나중에 --tag를 바꿔 다시 통합하면 됩니다.',
                  'all_feature_sources_present=False면 아직 일부 추출원이 빠진 중간 산출물입니다.']}
    helper.write_json(out / 'integration_report.json', report)
    helper.write_json(out / 'RUN_COMPLETE.json', {'outputs_written': True, 'integration_finished': True,
        'status': status, 'poses': int(len(merged)), 'feature_count': len(feature_columns),
        'all_feature_sources_present': complete, 'key_sets_match': keys_ok,
        'sources_unchanged': not changed})

    print('STATUS:', status, flush=True)
    print('POSES:', len(merged), '| SYSTEMS:', sys_total, '| FEATURES:', len(feature_columns), flush=True)
    print('ATTACHED:', roles.j(attached), '| ALL_SOURCES:', complete, flush=True)
    print('KEY_SETS_MATCH:', keys_ok, roles.j(key_mismatch), flush=True)
    print('LABELS known/missing/near_native:', n_label, len(merged) - n_label, n_nn, flush=True)
    print('ELIGIBLE_SYSTEMS:', n_elig, '/', sys_total, flush=True)
    for r in cat_rows:
        print(f"  {r['category']:<20} features={r['features']:<3} 전부있는pose={r['poses_all_features']:>7} "
              f"({r['coverage']:.1%})", flush=True)
    if uncatalogued:
        print('UNCATALOGUED_COLUMNS:', uncatalogued, flush=True)
    return 0 if status == 'INTEGRATION_COMPLETE' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--combined', type=Path, default=DEFAULT_COMBINED)
    parser.add_argument('--pocket', type=Path, default=DEFAULT_POCKET, help='없으면 cat5 없이 통합합니다.')
    parser.add_argument('--gnina', type=Path, default=None, help='gnina 전체 결과 parquet(준비되면 지정).')
    parser.add_argument('--labels', type=Path, default=DEFAULT_LABELS)
    parser.add_argument('--split', type=Path, default=DEFAULT_SPLIT)
    parser.add_argument('--tag', type=str, default=None, help='출력 폴더 접미사(중간 통합과 최종 통합 분리용).')
    args = parser.parse_args()
    for key in ('combined', 'pocket', 'gnina', 'labels', 'split'):
        v = getattr(args, key)
        if v is not None:
            setattr(args, key, v.absolute())
    return run(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
