#!/usr/bin/env python3
"""전체 460k pose의 sc_rmsd 정답 라벨 테이블을 생성합니다(Process A 라벨 단계).

sc_rmsd는 평가 CSV(results_common4603_v3final_qs_filtered.csv)에 이미 계산되어 있으므로 새로
계산하지 않고 (system_id, model, seed, sample) 키로 정리만 합니다. near-native 기준은 09·10과
동일하게 sc_rmsd < 2.0(엄격)이며, sc_rmsd 결측은 라벨 불명으로 보존합니다(0으로 채우지 않음).
fallback(결측 시 ligand_rmsd 사용)은 09·10에서 '정책 미확정'으로 미적용했으므로 기본 라벨
near_native에는 반영하지 않고, Process B에서 영향만 비교해볼 수 있도록 near_native_fallback을
별도 열로만 함께 제공합니다. 02 전체 pose 인덱스와 교차검증하여 커버리지(라벨 있는/결측 pose 수,
인덱스-라벨 불일치)를 보고합니다. GT 구조·좌표는 읽지 않으며 sc_rmsd 수치만 사용합니다.
02에서 pose 세계(알려진 시스템·EXPECTED 조합)를, 05→04→03에서 경로 보호·저장 함수를 재사용합니다.

입력: 평가 CSV(읽기 전용), 요약 CSV(알려진 시스템), 02 전체 pose 인덱스(교차검증, 선택).
출력: out/labels/24_run_build_labels/ 아래 고정 폴더. labels.parquet이 주 산출물입니다.
"""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import Counter
import hashlib
import importlib.util
import io
import json
from math import comb
import os
from pathlib import Path
import time

# 05→04→03 모듈(경로 보호·저장 함수)과 02 모듈(pose 세계 상수)을 재사용합니다.
_support_path = Path(__file__).with_name('05_audit_input_roles.py')
_spec = importlib.util.spec_from_file_location('roles_support', _support_path)
roles = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(roles)
audit = roles.audit
helper = roles.helper

_index_path = Path(__file__).with_name('02_build_pose_index.py')
_ispec = importlib.util.spec_from_file_location('pose_index_support', _index_path)
poseidx = importlib.util.module_from_spec(_ispec)
_ispec.loader.exec_module(poseidx)

VERSION = 'labels_0.1.0'
OUT_KIND = 'labels'
OUT_NAME = '24_run_build_labels'
KEYS = ['system_id', 'model', 'seed', 'sample']
THRESHOLD = 2.0          # near-native 기준: sc_rmsd < 2.0 (엄격, 09·10과 동일)
TOPK = 5
EXPECTED_PER_SYSTEM = 100
DEFAULT_EVAL = poseidx.PROJECT / 'eval_results/results_common4603_v3final_qs_filtered.csv'
DEFAULT_SUMMARY = poseidx.PROJECT / 'data/common_4603_summary.csv'
DEFAULT_INDEX = helper.ALLOWED_ROOT / 'Scoring_Function' / 'out' / 'pose_index' / '02_run_pose_index_full' / 'pose_index.parquet'


def random_topk_success(n_label, n_near_native, k=TOPK):
    """라벨 있는 pose 풀에서 무작위 Top-k가 near-native를 하나 이상 포함할 확률."""
    if n_label <= 0:
        return None
    if n_near_native <= 0:
        return 0.0
    if n_label <= k:
        return 1.0
    return 1.0 - comb(n_label - n_near_native, k) / comb(n_label, k)


def run(args):
    """평가 CSV의 sc_rmsd를 전체 pose 라벨 테이블로 정리하고 커버리지를 검증합니다."""
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
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    print('OUTPUT_DIR:', out, flush=True)

    import numpy as np
    import pandas as pd
    import pyarrow as pa
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)

    sources = [helper.fingerprint(args.eval_csv), helper.fingerprint(args.summary_csv)]
    if args.index_parquet.exists():
        sources.append(helper.fingerprint(args.index_parquet))

    # 1) 알려진 시스템 집합(요약 CSV). 02의 로더를 재사용해 중복·형식을 함께 검증합니다.
    known_systems = poseidx.load_systems(args.summary_csv)
    known_set = set(known_systems)
    expected_rows = len(known_systems) * EXPECTED_PER_SYSTEM

    # 2) 평가 CSV에서 키와 라벨 열만 읽습니다(원본은 수정하지 않음).
    ev = pd.read_csv(args.eval_csv,
                     usecols=KEYS + ['sc_rmsd', 'ligand_rmsd', 'status', 'mapping_status'],
                     dtype={'system_id': str, 'model': str})
    ev = ev.rename(columns={'status': 'eval_status', 'mapping_status': 'eval_mapping_status'})
    total_eval_rows = len(ev)

    # 3) pose 세계로 필터: 알려진 시스템 + EXPECTED(모델·seed·sample 범위) 조합만 남깁니다.
    ev['seed_i'] = pd.to_numeric(ev['seed'], errors='coerce')
    ev['sample_i'] = pd.to_numeric(ev['sample'], errors='coerce')
    bad_key = ev['seed_i'].isna() | ev['sample_i'].isna()
    n_bad_key = int(bad_key.sum())
    ev = ev[~bad_key].copy()
    ev['seed'] = ev['seed_i'].astype(int)
    ev['sample'] = ev['sample_i'].astype(int)
    ev = ev.drop(columns=['seed_i', 'sample_i'])

    in_known = ev['system_id'].isin(known_set)
    n_unknown_system = int((~in_known).sum())
    combo = list(zip(ev['model'], ev['seed'], ev['sample']))
    in_expected = pd.Series([c in poseidx.EXPECTED for c in combo], index=ev.index)
    n_out_of_range = int((in_known & ~in_expected).sum())
    ev = ev[in_known & in_expected].copy()

    # 4) 키 중복 제거(평가 CSV 반복 행 방지). 첫 행을 유지하고 개수를 보고합니다.
    dup_mask = ev.duplicated(KEYS, keep='first')
    n_duplicate_rows = int(dup_mask.sum())
    ev = ev[~dup_mask].copy()

    # 5) 라벨 열 구성. sc_rmsd가 주 라벨이며 결측은 라벨 불명으로 보존합니다.
    ev['sc_rmsd'] = pd.to_numeric(ev['sc_rmsd'], errors='coerce')
    ev['ligand_rmsd'] = pd.to_numeric(ev['ligand_rmsd'], errors='coerce')
    ev['label_known'] = ev['sc_rmsd'].notna()
    ev['ligand_rmsd_known'] = ev['ligand_rmsd'].notna()
    ev['near_native'] = ev['label_known'] & (ev['sc_rmsd'] < THRESHOLD)
    # fallback: sc_rmsd 결측 시에만 ligand_rmsd로 대체(기본 라벨에는 미반영, 비교용 별도 열).
    ev['near_native_fallback'] = np.where(
        ev['label_known'], ev['near_native'],
        ev['ligand_rmsd_known'] & (ev['ligand_rmsd'] < THRESHOLD))
    ev['pose_id'] = (ev['system_id'] + '|' + ev['model'] + '|'
                     + ev['seed'].astype(str) + '|' + ev['sample'].astype(str))

    labels = ev[KEYS + ['pose_id', 'sc_rmsd', 'ligand_rmsd', 'label_known', 'ligand_rmsd_known',
                        'near_native', 'near_native_fallback', 'eval_status', 'eval_mapping_status']].copy()
    labels = labels.sort_values(KEYS, kind='mergesort').reset_index(drop=True)

    # 6) 주 산출물 저장(덮어쓰기 금지 + 재읽기 검증).
    buf = io.BytesIO()
    labels.to_parquet(buf, engine='pyarrow', index=False, compression='snappy')
    with helper.checked_path(out / 'labels.parquet').open('xb') as handle:
        handle.write(buf.getvalue())
    restored = pd.read_parquet(out / 'labels.parquet', engine='pyarrow')
    pd.testing.assert_frame_equal(restored, labels, check_dtype=False)

    # 7) 시스템별 커버리지·oracle 상한.
    system_rows = []
    for sid, g in labels.groupby('system_id'):
        n = len(g)
        n_label = int(g['label_known'].sum())
        n_nn = int(g['near_native'].sum())
        rtop = random_topk_success(n_label, n_nn)
        system_rows.append({'system_id': sid, 'n_pose': n, 'n_label': n_label,
                            'n_missing': n - n_label, 'n_near_native': n_nn,
                            'oracle_near_native': bool(n_nn > 0),
                            'best_sc_rmsd': round(float(g['sc_rmsd'].min()), 4) if n_label else None,
                            'random_top5_success': round(rtop, 4) if rtop is not None else None})
    system_rows.sort(key=lambda r: r['system_id'])
    audit.save_csv(out / 'system_labels.csv', system_rows,
                   ('system_id', 'n_pose', 'n_label', 'n_missing', 'n_near_native',
                    'oracle_near_native', 'best_sc_rmsd', 'random_top5_success'))

    n_sys = len(system_rows)
    n_oracle = sum(1 for r in system_rows if r['oracle_near_native'])
    rand_oracle = [r['random_top5_success'] for r in system_rows
                   if r['oracle_near_native'] and r['random_top5_success'] is not None]

    # 8) 결측 라벨 원인 분포(평가 상태별).
    miss = labels[~labels['label_known']]
    miss_status = Counter((str(a), str(b)) for a, b in
                          zip(miss['eval_status'].fillna('NA'), miss['eval_mapping_status'].fillna('NA')))
    miss_rows = [{'eval_status': a, 'eval_mapping_status': b, 'count': c}
                 for (a, b), c in sorted(miss_status.items(), key=lambda x: -x[1])]
    audit.save_csv(out / 'missing_label_status.csv',
                   miss_rows or [{'eval_status': None, 'eval_mapping_status': None, 'count': 0}],
                   ('eval_status', 'eval_mapping_status', 'count'))

    # 9) 02 전체 pose 인덱스와 교차검증(있을 때만). 인덱스=eval에서 파생되므로 불일치는 0이어야 합니다.
    index_check = {'performed': False}
    if args.index_parquet.exists():
        idx = pd.read_parquet(args.index_parquet, columns=KEYS + ['status'])
        idx['system_id'] = idx['system_id'].astype(str)
        idx['model'] = idx['model'].astype(str)
        idx['seed'] = idx['seed'].astype(int)
        idx['sample'] = idx['sample'].astype(int)
        idx_keys = set(map(tuple, idx[KEYS].to_numpy().tolist()))
        idx_ok_keys = set(map(tuple, idx[idx['status'] == 'ok'][KEYS].to_numpy().tolist()))
        label_keys = set(map(tuple, labels[KEYS].to_numpy().tolist()))
        index_check = {
            'performed': True, 'index_unique_poses': len(idx_keys),
            'index_ok_poses': len(idx_ok_keys),
            'index_poses_without_label': len(idx_keys - label_keys),
            'label_poses_without_index': len(label_keys - idx_keys),
            'index_ok_poses_without_label': len(idx_ok_keys - label_keys)}

    # 10) 입력 변경 확인.
    changed = []
    for source in sources:
        try:
            info = Path(source['path']).stat()
            if (info.st_size, info.st_mtime_ns) != (source['size_bytes'], source['mtime_ns']):
                changed.append(source['path'])
        except OSError:
            changed.append(source['path'])

    n_label_known = int(labels['label_known'].sum())
    n_label_missing = int((~labels['label_known']).sum())
    n_near_native = int(labels['near_native'].sum())
    n_nn_fallback = int(labels['near_native_fallback'].sum())
    coverage_ok = (len(labels) == expected_rows and (not index_check['performed']
                   or (index_check['index_poses_without_label'] == 0
                       and index_check['label_poses_without_index'] == 0)))
    status = 'LABELS_COMPLETE' if coverage_ok and not changed else 'CHECK_REQUIRED'
    report = {
        'version': VERSION, 'status': status, 'output_dir': str(out),
        'eval_csv': str(args.eval_csv), 'summary_csv': str(args.summary_csv),
        'index_parquet': str(args.index_parquet), 'sources': sources,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'support_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (_support_path, _index_path, roles._support_path, audit._support_path)},
        'python': sys.executable, 'pyarrow_version': pa.__version__,
        'threshold_sc_rmsd': THRESHOLD, 'top_k': TOPK,
        'known_systems': len(known_systems), 'expected_rows': expected_rows,
        'total_eval_rows': total_eval_rows, 'label_rows': int(len(labels)),
        'dropped_bad_key': n_bad_key, 'dropped_unknown_system': n_unknown_system,
        'dropped_out_of_range': n_out_of_range, 'dropped_duplicate_rows': n_duplicate_rows,
        'poses_label_known': n_label_known, 'poses_label_missing': n_label_missing,
        'poses_near_native': n_near_native, 'poses_near_native_fallback': n_nn_fallback,
        'systems': n_sys, 'oracle_systems': n_oracle,
        'oracle_rate': round(n_oracle / n_sys, 4) if n_sys else None,
        'random_top5_mean_over_oracle': round(float(sum(rand_oracle) / len(rand_oracle)), 4) if rand_oracle else None,
        'index_crosscheck': index_check,
        'coverage_ok': coverage_ok, 'changed_sources': changed,
        'elapsed_seconds': time.perf_counter() - started,
        'gt_role': 'sc_rmsd는 정답 라벨로만 사용하며 feature 생성·pose 선택에는 쓰지 않음',
        'fallback_policy': ['기본 라벨 near_native는 sc_rmsd < 2 엄격 기준이며 결측은 불명으로 보존합니다.',
                            'near_native_fallback(결측 시 ligand_rmsd < 2)은 비교용 별도 열이며 기본 라벨에 반영하지 않았습니다.'],
        'limitations': [
            'sc_rmsd 결측 pose의 실제 near-native 여부는 확정하지 않으며 결측 원인 분포만 보고합니다.',
            '이 테이블은 평가·학습 라벨이며 feature·pose 선택에 사용하지 않습니다.',
            '인덱스 교차검증은 02 전체 인덱스가 있을 때만 수행합니다.']}
    helper.write_json(out / 'labels_report.json', report)
    helper.write_json(out / 'RUN_COMPLETE.json', {'outputs_written': True, 'labels_finished': True,
        'status': status, 'label_rows': int(len(labels)), 'poses_label_known': n_label_known,
        'poses_near_native': n_near_native, 'oracle_systems': n_oracle, 'coverage_ok': coverage_ok,
        'sources_unchanged': not changed})
    print('STATUS:', status, flush=True)
    print('LABEL_ROWS:', len(labels), '/', expected_rows, flush=True)
    print('DROPPED bad_key/unknown_sys/out_of_range/dup:', n_bad_key, n_unknown_system,
          n_out_of_range, n_duplicate_rows, flush=True)
    print('POSES label_known/missing/near_native:', n_label_known, n_label_missing, n_near_native, flush=True)
    print('SYSTEMS:', n_sys, '| ORACLE_SYSTEMS:', n_oracle, '| ORACLE_RATE:', report['oracle_rate'], flush=True)
    print('RANDOM_TOP5_MEAN(oracle):', report['random_top5_mean_over_oracle'], flush=True)
    if index_check['performed']:
        print('INDEX_CROSSCHECK:', roles.j(index_check), flush=True)
    return 0 if status == 'LABELS_COMPLETE' else 2


def main():
    """입력 위치를 지정하며 출력 위치는 사용자 경로로 고정합니다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-csv', type=Path, default=DEFAULT_EVAL)
    parser.add_argument('--summary-csv', type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument('--index-parquet', type=Path, default=DEFAULT_INDEX,
                        help='02 전체 pose 인덱스(교차검증용, 없으면 생략).')
    args = parser.parse_args()
    for key in ('eval_csv', 'summary_csv', 'index_parquet'):
        setattr(args, key, getattr(args, key).absolute())
    return run(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
