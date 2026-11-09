#!/usr/bin/env python3
"""8개 후보군의 표본 feature 표를 pose_id로 통합합니다(로드맵 C단계 마무리).

각 후보군 결과 parquet에서 feature 열과 계산 상태만 골라 (system_id, model, seed, sample)로
join하여 하나의 표본 feature 행렬을 만듭니다. feature를 0으로 채우지 않으며, 후보군별 계산 상태와
널(계산 불가/부재)을 그대로 보존합니다. 후보군별 커버리지와 pose별 완성도(몇 개 후보군이 계산됐는지)를
함께 기록합니다. GT·좌표를 다루지 않는 순수 통합 단계입니다. 05→04→03 함수에서 저장·경로 보호를
재사용합니다.

출력: out/feature_matrix/17_run_integrate_features/ 아래 고정 폴더.
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

VERSION = 'feature_matrix_0.1.0'
OUT_KIND = 'feature_matrix'
OUT_NAME = '17_run_integrate_features'
KEYS = ['system_id', 'model', 'seed', 'sample']
BASE = helper.ALLOWED_ROOT / 'Scoring_Function' / 'out'

# (후보군 키, parquet 경로, feature 열 목록). cat1은 표본(07) 결과를 사용합니다.
TABLES = [
    ('cat1_contact', BASE / 'contact_features/07_run_contact_features/contact_features.parquet',
     ['pl_min_distance_A', 'pl_pairs_lt_3A', 'ligand_atoms_lt_3A', 'protein_atoms_lt_3A', 'protein_residues_lt_3A',
      'pl_pairs_lt_4A', 'ligand_atoms_lt_4A', 'protein_atoms_lt_4A', 'protein_residues_lt_4A',
      'pl_pairs_lt_5A', 'ligand_atoms_lt_5A', 'protein_atoms_lt_5A', 'protein_residues_lt_5A']),
    ('cat2_interaction', BASE / 'interaction_features/16_run_interaction_features/interaction_features.parquet',
     ['potential_hbond_pairs', 'hydrophobic_pairs', 'receptor_clash_pairs', 'salt_bridge_pairs', 'aromatic_contact_pairs']),
    ('cat3_surface', BASE / 'surface_features/11_run_surface_features/surface_features.parquet',
     ['ligand_sasa_isolated_A2', 'ligand_sasa_complex_A2', 'ligand_buried_sasa_A2', 'ligand_buried_fraction',
      'ligand_sasa_isolated_polar_A2', 'ligand_sasa_isolated_nonpolar_A2', 'ligand_buried_polar_A2']),
    ('cat4_geometry', BASE / 'ligand_geometry/15_run_ligand_geometry/ligand_geometry.parquet',
     ['lig_bond_count', 'lig_bond_len_min_A', 'lig_bond_len_max_A', 'lig_bond_len_mean_A',
      'lig_long_bonds', 'lig_short_bonds', 'lig_internal_clashes', 'lig_min_nonbonded_dist_A']),
    ('cat5_pocket', BASE / 'pocket_features/14_run_pocket_features/pocket_features.parquet',
     ['num_predicted_pockets', 'top_pocket_score', 'top_pocket_probability', 'ligand_pocket_rank',
      'ligand_pocket_score', 'ligand_pocket_probability', 'ligand_to_pocket_center_dist_A']),
    ('cat67_water_metal', BASE / 'water_metal_features/13_run_water_metal_features/water_metal_features.parquet',
     ['water_observed_count', 'water_bridge_count', 'metal_observed_count', 'metal_in_ligand_count',
      'metal_in_receptor_count', 'metal_independent_count', 'min_metal_ligand_distance_A', 'metal_ligand_receptor_bridge']),
    ('cat8_ligand', BASE / 'ligand_descriptors/12_run_ligand_descriptors/ligand_descriptors.parquet',
     ['lig_mol_weight', 'lig_heavy_atoms', 'lig_rotatable_bonds', 'lig_hbd', 'lig_hba', 'lig_tpsa',
      'lig_logp', 'lig_ring_count', 'lig_aromatic_rings', 'lig_fraction_csp3', 'lig_num_heteroatoms', 'lig_formal_charge']),
]


def run(args):
    """후보군 표들을 읽어 통합 행렬과 커버리지를 저장합니다."""
    started = time.perf_counter()
    parent = helper.safe_mkdir(BASE / OUT_KIND)
    out = helper.checked_path(parent / OUT_NAME)
    if out.exists():
        raise FileExistsError(f'기존 결과를 덮어쓰지 않습니다: {out}')
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

    sources = []
    frames = []
    key_sets = []
    feature_columns = []
    category_status_cols = []
    for cat, path, feats in TABLES:
        if not Path(path).exists():
            raise FileNotFoundError(f'후보군 결과 parquet이 없습니다: {path}')
        sources.append(helper.fingerprint(Path(path)))
        df = pd.read_parquet(path)
        for k in ('seed', 'sample'):
            df[k] = df[k].astype(int)
        df['system_id'] = df['system_id'].astype(str)
        df['model'] = df['model'].astype(str)
        missing = [c for c in feats if c not in df.columns]
        if missing or 'status' not in df.columns:
            raise ValueError(f'{cat}: 기대한 열이 없습니다: {missing or "status"}')
        status_col = f'{cat}_status'
        sub = df[KEYS + feats].copy()
        sub[status_col] = df['status'].values
        # 후보군 접두사로 feature 이름 충돌을 방지합니다.
        rename = {c: f'{cat}:{c}' for c in feats}
        sub = sub.rename(columns=rename)
        feature_columns.extend(rename.values())
        category_status_cols.append(status_col)
        frames.append(sub.set_index(KEYS))
        key_sets.append(set(map(tuple, df[KEYS].to_numpy().tolist())))

    # 모든 표가 같은 pose 키 집합을 가져야 합니다.
    base_keys = key_sets[0]
    for (cat, _p, _f), ks in zip(TABLES, key_sets):
        if ks != base_keys:
            raise ValueError(f'{cat}: pose 키 집합이 다른 표와 일치하지 않습니다(차이 {len(ks ^ base_keys)}개).')

    unified = pd.concat(frames, axis=1).reset_index()
    if unified.duplicated(KEYS).any() or len(unified) != len(base_keys):
        raise RuntimeError('통합 과정에서 행 수·중복이 맞지 않습니다.')
    unified.insert(4, 'pose_id', unified['system_id'] + '|' + unified['model'] + '|'
                   + unified['seed'].astype(str) + '|' + unified['sample'].astype(str))

    buf = io.BytesIO()
    unified.to_parquet(buf, engine='pyarrow', index=False, compression='snappy')
    with helper.checked_path(out / 'unified_features.parquet').open('xb') as handle:
        handle.write(buf.getvalue())
    restored = pd.read_parquet(out / 'unified_features.parquet', engine='pyarrow')
    pd.testing.assert_frame_equal(restored, unified, check_dtype=False)

    # 후보군별 커버리지(계산 완료 pose 수와 상태 분포).
    cov_rows = []
    for (cat, _p, _f), status_col in zip(TABLES, category_status_cols):
        counts = Counter(unified[status_col])
        cov_rows.append({'category': cat, 'computed': int(counts.get('computed', 0)),
                         'total': int(len(unified)), 'status_counts_json': audit.encoded(dict(counts))})
    audit.save_csv(out / 'category_coverage.csv', cov_rows, ('category', 'computed', 'total'))

    # pose별 완성도: 몇 개 후보군이 computed인지.
    computed_per_pose = (unified[category_status_cols] == 'computed').sum(axis=1)
    pose_rows = [{'categories_computed': int(k), 'num_poses': int(v)}
                 for k, v in sorted(Counter(computed_per_pose).items())]
    audit.save_csv(out / 'pose_completeness.csv', pose_rows, ('categories_computed', 'num_poses'))

    changed = []
    for source in sources:
        try:
            info = Path(source['path']).stat()
            if (info.st_size, info.st_mtime_ns) != (source['size_bytes'], source['mtime_ns']):
                changed.append(source['path'])
        except OSError:
            changed.append(source['path'])
    status = 'INTEGRATION_COMPLETE' if not changed else 'CHECK_REQUIRED'
    report = {
        'version': VERSION, 'status': status, 'output_dir': str(out), 'sources': sources,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'support_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (_support_path, roles._support_path, audit._support_path)},
        'python': sys.executable, 'pyarrow_version': pa.__version__,
        'poses': int(len(unified)), 'feature_count': len(feature_columns),
        'categories': [c for c, _p, _f in TABLES], 'feature_columns': feature_columns,
        'category_coverage': cov_rows, 'pose_completeness': pose_rows,
        'changed_sources': changed, 'elapsed_seconds': time.perf_counter() - started,
        'gt_used': False, 'coordinates_modified': False,
        'notes': ['표본(선정 160 pose) 통합이며 전체 460k가 아닙니다.',
                  'feature를 0으로 채우지 않고 후보군별 상태·널을 보존합니다.',
                  'cat1은 표본(07) 결과이며 확대(08) 결과와 구분합니다.',
                  '이 표는 탐색·선별의 기반이며 학습·선별을 아직 시작하지 않습니다.']}
    helper.write_json(out / 'integration_report.json', report)
    helper.write_json(out / 'RUN_COMPLETE.json', {'outputs_written': True, 'integration_finished': True,
        'status': status, 'poses': int(len(unified)), 'feature_count': len(feature_columns),
        'sources_unchanged': not changed})
    print('STATUS:', status, flush=True)
    print('POSES:', len(unified), '| FEATURES:', len(feature_columns), flush=True)
    print('CATEGORY_COVERAGE:', roles.j({c: int(Counter(unified[s]).get('computed', 0))
                                         for (c, _p, _f), s in zip(TABLES, category_status_cols)}), flush=True)
    print('POSE_COMPLETENESS:', roles.j({r['categories_computed']: r['num_poses'] for r in pose_rows}), flush=True)
    return 0 if status == 'INTEGRATION_COMPLETE' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    return run(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
