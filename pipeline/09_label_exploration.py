#!/usr/bin/env python3
"""08번 접촉 feature와 GT 라벨(sc_rmsd)의 관계를 탐색 분석합니다.

sc_rmsd는 평가·분석 라벨로만 사용하며 feature 생성·pose 선택에는 쓰지 않습니다(08은 이미 GT
미사용). 엄격한 sc_rmsd < 2 기준으로 near-native를 정의하고, 결측은 라벨 불명으로 분리하여
두 가지 분모(결측=non-near-native 보수적 / 라벨 있는 pose만 조건부)로 모두 보고합니다. ligand_rmsd
fallback은 정책 미확정이므로 적용하지 않습니다. 이 단계는 탐색이며 학습·split·정규화를 하지
않습니다. feature 방향 선택의 낙관적 편향을 피하기 위해 각 feature의 오름·내림 두 방향 결과를
모두 남깁니다. 같은 폴더의 05→04→03 함수에서 경로 보호·저장 함수를 재사용합니다.

입력: 08 contact_features.parquet, 평가 CSV(읽기 전용 원본). 원본은 수정하지 않습니다.
출력: out/label_exploration/09_run_label_exploration/ 아래 고정 폴더.
"""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from math import comb
import os
from pathlib import Path
import time

_support_path = Path(__file__).with_name('05_audit_input_roles.py')
_spec = importlib.util.spec_from_file_location('roles_support', _support_path)
roles = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(roles)
audit = roles.audit        # 04 모듈
helper = roles.helper      # 03 모듈

VERSION = 'label_exploration_0.1.0'
OUT_KIND = 'label_exploration'
OUT_NAME = '09_run_label_exploration'
DEFAULT_FEATURES = helper.ALLOWED_ROOT / 'Scoring_Function' / 'out' / 'contact_features_expanded' / '08_run_contact_features_expanded'
DEFAULT_EVAL = Path('<SHARED_ROOT>/distillation/eval_results/results_common4603_v3final_qs_filtered.csv')
KEYS = ('system_id', 'model', 'seed', 'sample')
THRESHOLD = 2.0
TOPK = 5
FEATURES = ['pl_min_distance_A'] + [f'{p}_lt_{c}A' for c in (3, 4, 5)
            for p in ('pl_pairs', 'ligand_atoms', 'protein_atoms', 'protein_residues')]


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
    """08 feature와 sc_rmsd 라벨을 조인하여 분별력 탐색 결과를 저장합니다."""
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
    from scipy.stats import spearmanr

    features_dir = args.features_run
    sources = [helper.fingerprint(p) for p in (
        features_dir / 'RUN_COMPLETE.json', features_dir / 'contact_features.parquet', args.eval_csv)]
    complete = json.loads((features_dir / 'RUN_COMPLETE.json').read_text(encoding='utf-8'))
    if complete.get('status') != 'FEATURES_COMPLETE' or complete.get('outputs_written') is not True:
        raise ValueError('08번 접촉 feature 추출이 정상 완료된 결과가 아닙니다.')

    feats = pd.read_parquet(features_dir / 'contact_features.parquet')
    comp = feats[feats['status'] == 'computed'].copy()
    for k in ('seed', 'sample'):
        comp[k] = comp[k].astype(int)
    comp['system_id'] = comp['system_id'].astype(str)
    comp['model'] = comp['model'].astype(str)

    ev = pd.read_csv(args.eval_csv, usecols=list(KEYS) + ['ligand_rmsd', 'sc_rmsd', 'status', 'mapping_status'])
    ev = ev.rename(columns={'status': 'eval_status', 'mapping_status': 'eval_mapping_status'})
    for k in ('seed', 'sample'):
        ev[k] = ev[k].astype(int)
    ev['system_id'] = ev['system_id'].astype(str)
    ev['model'] = ev['model'].astype(str)
    ev['_in_eval'] = 1

    merged = comp.merge(ev, on=list(KEYS), how='left')
    merged['_in_eval'] = merged['_in_eval'].fillna(0).astype(int)
    merged['sc_rmsd_num'] = pd.to_numeric(merged['sc_rmsd'], errors='coerce')
    merged['label_known'] = merged['sc_rmsd_num'].notna()
    merged['near_native'] = merged['label_known'] & (merged['sc_rmsd_num'] < THRESHOLD)

    unmatched = int((merged['_in_eval'] == 0).sum())
    # 시스템별 커버리지와 oracle 상한.
    system_rows = []
    groups = {sid: grp for sid, grp in merged.groupby('system_id')}
    for sid in sorted(groups):
        grp = groups[sid]
        n = len(grp)
        n_label = int(grp['label_known'].sum())
        n_nn = int(grp['near_native'].sum())
        system_rows.append({'system_id': sid, 'n_pose': n, 'n_label': n_label,
                            'n_missing': n - n_label, 'n_near_native': n_nn,
                            'oracle_near_native': bool(n_nn > 0),
                            'best_sc_rmsd': round(float(grp['sc_rmsd_num'].min()), 4) if n_label else None,
                            'random_top5_success': (round(random_topk_success(n_label, n_nn), 4)
                                                    if random_topk_success(n_label, n_nn) is not None else None)})
    oracle_systems = [r['system_id'] for r in system_rows if r['oracle_near_native']]
    n_sys = len(system_rows)
    n_oracle = len(oracle_systems)

    # feature 방향별 Top-k 성공률과 시스템 내 Spearman.
    ranking_rows = []
    for f in FEATURES:
        # 시스템 내 Spearman(feature, sc_rmsd): 라벨 있는 pose가 충분한 시스템만.
        rho = []
        for sid in sorted(groups):
            sub = groups[sid]
            lab = sub[sub['label_known']]
            if len(lab) >= 5 and lab[f].nunique() > 1 and lab['sc_rmsd_num'].nunique() > 1:
                r, _ = spearmanr(lab[f].values, lab['sc_rmsd_num'].values)
                if np.isfinite(r):
                    rho.append(r)
        mean_rho = round(float(np.mean(rho)), 4) if rho else None
        median_rho = round(float(np.median(rho)), 4) if rho else None
        for direction in ('asc', 'desc'):
            succ_all = succ_oracle = 0
            for sid in sorted(groups):
                sub = groups[sid]
                top = sub.sort_values(f, ascending=(direction == 'asc'), kind='mergesort').head(TOPK)
                success = bool(top['near_native'].any())
                succ_all += success
                if sub['near_native'].any():
                    succ_oracle += success
            ranking_rows.append({
                'feature': f, 'direction': direction,
                'top5_success_systems': succ_all, 'top5_success_rate_all': round(succ_all / n_sys, 4),
                'top5_success_rate_oracle': round(succ_oracle / n_oracle, 4) if n_oracle else None,
                'mean_within_spearman': mean_rho, 'median_within_spearman': median_rho,
                'n_systems_spearman': len(rho)})

    audit.save_csv(out / 'system_oracle.csv', system_rows,
                   ('system_id', 'n_pose', 'n_label', 'n_missing', 'n_near_native', 'oracle_near_native'))
    audit.save_csv(out / 'feature_ranking.csv', ranking_rows,
                   ('feature', 'direction', 'top5_success_rate_all', 'top5_success_rate_oracle'))

    # 결측 원인 참고: 라벨 불명 pose의 평가 상태 분포.
    missing = merged[~merged['label_known']]
    missing_status = Counter((str(a), str(b)) for a, b in
                             zip(missing['eval_status'].fillna('NA'), missing['eval_mapping_status'].fillna('NA')))
    missing_rows = [{'eval_status': a, 'eval_mapping_status': b, 'count': c}
                    for (a, b), c in sorted(missing_status.items(), key=lambda x: -x[1])]
    audit.save_csv(out / 'missing_label_status.csv', missing_rows or [{'eval_status': None, 'eval_mapping_status': None, 'count': 0}],
                   ('eval_status', 'eval_mapping_status', 'count'))

    changed = []
    for source in sources:
        try:
            info = Path(source['path']).stat()
            if (info.st_size, info.st_mtime_ns) != (source['size_bytes'], source['mtime_ns']):
                changed.append(source['path'])
        except OSError:
            changed.append(source['path'])

    best = max((r for r in ranking_rows if r['top5_success_rate_oracle'] is not None),
               key=lambda r: r['top5_success_rate_oracle'], default=None)
    rand_oracle = [r['random_top5_success'] for r in system_rows
                   if r['oracle_near_native'] and r['random_top5_success'] is not None]
    status = 'EXPLORATION_COMPLETE' if not changed else 'CHECK_REQUIRED'
    report = {
        'version': VERSION, 'status': status, 'output_dir': str(out),
        'features_run': str(features_dir), 'eval_csv': str(args.eval_csv), 'sources': sources,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'support_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (_support_path, roles._support_path, audit._support_path)},
        'python': sys.executable, 'threshold_sc_rmsd': THRESHOLD, 'top_k': TOPK,
        'computed_poses': int(len(comp)), 'merged_poses': int(len(merged)),
        'unmatched_in_eval': unmatched, 'systems': n_sys, 'oracle_systems': n_oracle,
        'oracle_rate': round(n_oracle / n_sys, 4) if n_sys else None,
        'poses_label_known': int(merged['label_known'].sum()),
        'poses_label_missing': int((~merged['label_known']).sum()),
        'poses_near_native': int(merged['near_native'].sum()),
        'random_top5_mean_over_oracle': round(float(sum(rand_oracle) / len(rand_oracle)), 4) if rand_oracle else None,
        'best_single_feature': ({'feature': best['feature'], 'direction': best['direction'],
                                 'top5_success_rate_oracle': best['top5_success_rate_oracle'],
                                 'top5_success_rate_all': best['top5_success_rate_all']} if best else None),
        'changed_sources': changed, 'elapsed_seconds': time.perf_counter() - started,
        'gt_role': 'sc_rmsd는 평가·분석 라벨로만 사용하며 feature·선택에 쓰지 않음',
        'limitations': [
            'sc_rmsd < 2 엄격 기준이며 결측은 두 가지 분모로 분리해 보고합니다.',
            'ligand_rmsd fallback은 정책 미확정이므로 적용하지 않았습니다.',
            '단일 feature 두 방향 결과를 모두 기록하며 best는 동일 데이터 선택이라 낙관적입니다.',
            '선정 40시스템 중 08에서 계산된 38시스템만 대상이며 1p8j/8bxs는 보류되어 제외됩니다.',
            '탐색 분석이며 학습·split·정규화·일반화 성능 평가가 아닙니다.',
            'Top-k oracle 상한과 실제 성공률을 구분해 보고합니다.']}
    helper.write_json(out / 'exploration_report.json', report)
    helper.write_json(out / 'RUN_COMPLETE.json', {'outputs_written': True, 'exploration_finished': True,
        'status': status, 'systems': n_sys, 'oracle_systems': n_oracle, 'sources_unchanged': not changed})
    print('STATUS:', status, flush=True)
    print('SYSTEMS:', n_sys, '| ORACLE_SYSTEMS:', n_oracle, '| ORACLE_RATE:', report['oracle_rate'], flush=True)
    print('POSES label_known/missing/near_native:', report['poses_label_known'],
          report['poses_label_missing'], report['poses_near_native'], flush=True)
    print('RANDOM_TOP5_MEAN(oracle):', report['random_top5_mean_over_oracle'], flush=True)
    if best:
        print('BEST_SINGLE_FEATURE(optimistic):', best['feature'], best['direction'],
              'oracle_rate=', best['top5_success_rate_oracle'], flush=True)
    return 0 if status == 'EXPLORATION_COMPLETE' else 2


def main():
    """입력 위치를 지정하며 출력 위치는 사용자 경로로 고정합니다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features-run', type=Path, default=DEFAULT_FEATURES)
    parser.add_argument('--eval-csv', type=Path, default=DEFAULT_EVAL)
    args = parser.parse_args()
    return run(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
