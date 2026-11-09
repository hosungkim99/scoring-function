#!/usr/bin/env python3
"""누수 방지 데이터 분할(B②)을 라벨을 보기 전에 고정합니다.

PLINDER 포켓 유사성 cluster(pocket_lddt__50__strong__component)를 1차 그룹 키로 삼아, 같은
cluster가 dev/test 양쪽에 걸치지 않도록 group-split 합니다. 같은 시스템의 100 pose는 시스템이
cluster에 속하므로 자동으로 한 분할에 들어갑니다. 이미 반복해서 본 진단용 40 시스템의 cluster는
dev로 고정합니다. dev 내부는 cluster 단위 GroupKFold(사이즈 균형)로 교차검증 폴드를 나눕니다.
리간드(CCD)·ECOD 교차 겹침은 강제하지 않고 감사 리포트로 남깁니다. feature·GT 라벨을 읽지 않으며,
시스템 메타데이터만으로 분할을 정의합니다. 05→04→03 함수에서 저장·경로 보호를 재사용합니다.

출력: out/data_split/18_run_make_split/ 아래 고정 폴더.
"""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import importlib.util

_support_path = Path(__file__).with_name('05_audit_input_roles.py')
_spec = importlib.util.spec_from_file_location('roles_support', _support_path)
roles = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(roles)
audit = roles.audit
helper = roles.helper

VERSION = 'data_split_0.1.0'
OUT_KIND = 'data_split'
OUT_NAME = '18_run_make_split'
DEFAULT_SUMMARY = Path('<SHARED_ROOT>/distillation/data/common_4603_summary.csv')
DEFAULT_SELECTION = audit.DEFAULT_SELECTION  # 03번 선정 결과(진단용 40 시스템)
CLUSTER_COL = 'pocket_lddt__50__strong__component'
ECOD_COL = 'system_pocket_ECOD_t_name'
LIGAND_COL = 'ligand_ccd_code'


def hkey(text, seed):
    """결정적 해시 키(순서 섞기용)."""
    return int(hashlib.sha1(f'{text}|{seed}'.encode()).hexdigest(), 16)


def run(args):
    """포켓 cluster 기준으로 dev/test와 dev 내부 CV 폴드를 고정합니다."""
    started = time.perf_counter()
    parent = helper.safe_mkdir(helper.ALLOWED_ROOT / 'Scoring_Function' / 'out' / OUT_KIND)
    out = helper.checked_path(parent / OUT_NAME)
    if out.exists():
        raise FileExistsError(f'기존 결과를 덮어쓰지 않습니다: {out}')
    out.mkdir()
    for name in ('tmp', 'cache'):
        helper.safe_mkdir(out / name)
    for key in ('TMPDIR', 'TMP', 'TEMP'):
        os.environ[key] = str(out / 'tmp')
    os.environ['XDG_CACHE_HOME'] = str(out / 'cache')
    print('OUTPUT_DIR:', out, flush=True)

    import pandas as pd
    sources = [helper.fingerprint(args.summary_csv),
               helper.fingerprint(args.selection_run / 'selected_systems.csv')]

    df = pd.read_csv(args.summary_csv)
    needed = ['system_id', CLUSTER_COL]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f'요약 CSV에 필요한 열이 없습니다: {c}')
    if df['system_id'].duplicated().any():
        raise ValueError('요약 CSV에 system_id 중복이 있습니다.')
    if df[CLUSTER_COL].isna().any():
        raise ValueError('포켓 cluster 열에 결측이 있습니다. 분할 그룹을 만들 수 없습니다.')
    df['system_id'] = df['system_id'].astype(str)
    df['pocket_cluster'] = df[CLUSTER_COL].astype(str)
    df['ecod'] = df[ECOD_COL].astype(str) if ECOD_COL in df.columns else None
    df['ligand_ccd'] = df[LIGAND_COL].astype(str) if LIGAND_COL in df.columns else None

    # 진단용 40 시스템.
    with (args.selection_run / 'selected_systems.csv').open(newline='', encoding='utf-8-sig') as handle:
        diag_systems = {r['system_id'] for r in csv.DictReader(handle)}
    missing_diag = diag_systems - set(df['system_id'])
    if missing_diag:
        raise ValueError(f'진단용 시스템이 요약 CSV에 없습니다: {sorted(missing_diag)[:5]}')

    # cluster -> 시스템 목록.
    cluster_systems = defaultdict(list)
    for sid, cl in zip(df['system_id'], df['pocket_cluster']):
        cluster_systems[cl].append(sid)
    total_systems = len(df)

    # 진단 40이 속한 cluster는 dev로 고정.
    diag_clusters = {cl for cl, sids in cluster_systems.items() if diag_systems & set(sids)}

    # 나머지 cluster를 결정적으로 섞어 test 목표 비율까지 채움.
    rest = [cl for cl in cluster_systems if cl not in diag_clusters]
    rest.sort(key=lambda c: hkey(c, args.seed))
    target_test = int(round(args.test_frac * total_systems))
    test_clusters, acc = set(), 0
    for cl in rest:
        if acc >= target_test:
            break
        test_clusters.add(cl)
        acc += len(cluster_systems[cl])
    dev_clusters = [cl for cl in cluster_systems if cl not in test_clusters]

    # dev cluster를 사이즈 균형으로 k개 폴드에 배정(큰 것부터 가장 작은 폴드로).
    fold_size = [0] * args.folds
    fold_of_cluster = {}
    for cl in sorted(dev_clusters, key=lambda c: (-len(cluster_systems[c]), hkey(c, args.seed))):
        f = min(range(args.folds), key=lambda i: (fold_size[i], i))
        fold_of_cluster[cl] = f
        fold_size[f] += len(cluster_systems[cl])

    # 시스템별 분할 표. ecod/ligand는 한 번만 매핑합니다.
    ecod_map = dict(zip(df['system_id'], df['ecod'])) if df['ecod'] is not None else {}
    lig_map = dict(zip(df['system_id'], df['ligand_ccd'])) if df['ligand_ccd'] is not None else {}
    rows = []
    for sid, cl in zip(df['system_id'], df['pocket_cluster']):
        is_test = cl in test_clusters
        rows.append({'system_id': sid, 'pocket_cluster': cl,
                     'ecod': ecod_map.get(sid), 'ligand_ccd': lig_map.get(sid),
                     'split': 'test' if is_test else 'dev',
                     'cv_fold': -1 if is_test else fold_of_cluster[cl],
                     'is_diagnostic': sid in diag_systems})
    audit.save_csv(out / 'system_split.csv', rows,
                   ('system_id', 'pocket_cluster', 'split', 'cv_fold', 'is_diagnostic', 'ecod', 'ligand_ccd'))

    dev_sys = [r['system_id'] for r in rows if r['split'] == 'dev']
    test_sys = [r['system_id'] for r in rows if r['split'] == 'test']
    # 누수 감사: 포켓 cluster는 구성상 교차 0. 리간드/ECOD 교차 겹침은 보고만.
    dev_lig = {lig_map.get(s) for s in dev_sys if lig_map.get(s)}
    test_lig = {lig_map.get(s) for s in test_sys if lig_map.get(s)}
    dev_ecod = {ecod_map.get(s) for s in dev_sys if ecod_map.get(s) and ecod_map.get(s) != 'nan'}
    test_ecod = {ecod_map.get(s) for s in test_sys if ecod_map.get(s) and ecod_map.get(s) != 'nan'}
    cross_cluster = len({r['pocket_cluster'] for r in rows if r['split'] == 'dev'}
                        & {r['pocket_cluster'] for r in rows if r['split'] == 'test'})
    fold_rows = [{'cv_fold': i, 'dev_systems': fold_size[i]} for i in range(args.folds)]
    audit.save_csv(out / 'fold_sizes.csv', fold_rows, ('cv_fold', 'dev_systems'))

    changed = []
    for source in sources:
        try:
            info = Path(source['path']).stat()
            if (info.st_size, info.st_mtime_ns) != (source['size_bytes'], source['mtime_ns']):
                changed.append(source['path'])
        except OSError:
            changed.append(source['path'])
    status = 'SPLIT_COMPLETE' if cross_cluster == 0 and not changed else 'CHECK_REQUIRED'
    report = {
        'version': VERSION, 'status': status, 'output_dir': str(out),
        'summary_csv': str(args.summary_csv), 'selection_run': str(args.selection_run), 'sources': sources,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'support_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (_support_path, roles._support_path, audit._support_path)},
        'python': sys.executable, 'group_key': CLUSTER_COL, 'seed': args.seed,
        'test_frac_target': args.test_frac, 'folds': args.folds,
        'total_systems': total_systems, 'total_clusters': len(cluster_systems),
        'dev_systems': len(dev_sys), 'test_systems': len(test_sys),
        'test_frac_actual': round(len(test_sys) / total_systems, 4),
        'diagnostic_systems': len(diag_systems), 'diagnostic_clusters_forced_dev': len(diag_clusters),
        'fold_sizes': fold_size,
        'leakage_audit': {
            'pocket_cluster_cross_split': cross_cluster,
            'ligand_ccd_shared_dev_test': len(dev_lig & test_lig),
            'ecod_shared_dev_test': len(dev_ecod & test_ecod),
            'note': '포켓 cluster는 구성상 교차 0. 리간드 CCD·ECOD 공유는 보고값이며 강제 분리는 하지 않았습니다.'},
        'changed_sources': changed, 'elapsed_seconds': time.perf_counter() - started,
        'gt_used': False, 'features_used': False,
        'policy': [
            '분할은 라벨·feature를 보기 전에 고정했습니다.',
            '같은 시스템의 100 pose는 시스템이 cluster에 속하므로 한 분할에 들어갑니다.',
            '진단용 40 시스템의 cluster는 dev로 고정했습니다.',
            '최종 테스트는 feature·모델 선택에 사용하지 않습니다.'],
        'limitations': [
            '포켓 유사성 1차 그룹입니다. 리간드·서열 유사성 누수는 감사값으로만 보고하며 추후 더 엄격히 할 수 있습니다.',
            'CATH는 결측이 많아 사용하지 않았고 ECOD는 보조 점검에만 썼습니다.']}
    helper.write_json(out / 'split_report.json', report)
    helper.write_json(out / 'RUN_COMPLETE.json', {'outputs_written': True, 'split_finished': True,
        'status': status, 'dev_systems': len(dev_sys), 'test_systems': len(test_sys),
        'sources_unchanged': not changed})
    print('STATUS:', status, flush=True)
    print('TOTAL:', total_systems, '| DEV:', len(dev_sys), '| TEST:', len(test_sys),
          '| TEST_FRAC:', report['test_frac_actual'], flush=True)
    print('CLUSTERS:', len(cluster_systems), '| DIAG_CLUSTERS(dev고정):', len(diag_clusters), flush=True)
    print('FOLD_SIZES:', fold_size, flush=True)
    print('LEAKAGE_AUDIT:', roles.j(report['leakage_audit']), flush=True)
    return 0 if status == 'SPLIT_COMPLETE' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary-csv', type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument('--selection-run', type=Path, default=DEFAULT_SELECTION)
    parser.add_argument('--test-frac', type=float, default=0.2)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261022)
    args = parser.parse_args()
    if not (0 < args.test_frac < 1) or args.folds < 2:
        parser.error('test-frac은 0~1, folds는 2 이상이어야 합니다.')
    return run(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
