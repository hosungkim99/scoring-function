#!/usr/bin/env python3
"""08번 접촉 feature(category 1)와 sc_rmsd 라벨을 난이도 층화로 한 차례 정리합니다.

구조 재처리·feature 재추출·새 표본 선정을 하지 않고 기존 결과(08 contact_features.parquet,
평가 CSV)만 읽습니다. sc_rmsd는 분석 라벨로만 쓰며 엄격한 sc_rmsd < 2 기준을 유지하고
ligand_rmsd fallback은 적용하지 않습니다. 난이도 구간과 분모는 결과를 보기 전에 코드 상수로
고정하며, 라벨이 전부 결측인 시스템은 '난이도 불명'으로, 일부 결측이면서 near-native가 확인되지
않은 시스템은 '미확인'으로 분리합니다. feature와 random은 동일한 후보 집합·결측 처리·시스템
분모에서 비교하고, 13개 feature의 오름/내림 두 방향을 모두 기록합니다. 동점은 pose_id 해시
오름차순으로 결정적으로 깨어 특정 생성 모델에 유리하지 않게 합니다. 같은 표본에서 고른 최선의
feature·방향은 탐색 결과로만 표시합니다. 05→04→03 함수에서 경로 보호·저장 함수를 재사용합니다.

출력: out/label_stratified/10_run_label_stratified/ 아래 고정 폴더.
"""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import Counter, defaultdict
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
audit = roles.audit
helper = roles.helper

VERSION = 'label_stratified_0.1.0'
OUT_KIND = 'label_stratified'
OUT_NAME = '10_run_label_stratified'
DEFAULT_FEATURES = helper.ALLOWED_ROOT / 'Scoring_Function' / 'out' / 'contact_features_expanded' / '08_run_contact_features_expanded'
DEFAULT_EVAL = Path('<SHARED_ROOT>/distillation/eval_results/results_common4603_v3final_qs_filtered.csv')
KEYS = ('system_id', 'model', 'seed', 'sample')
THRESHOLD = 2.0          # near-native 기준: sc_rmsd < 2 (엄격)
TOPK = 5
# 사전 고정 난이도 구간: fraction = near-native 수 / 유효 라벨 수(분모=유효 라벨).
EASY_MIN = 0.70          # fraction >= 0.70 → easy
MEDIUM_MIN = 0.20        # 0.20 <= fraction < 0.70 → medium, 0 < fraction < 0.20 → hard
FEATURES = ['pl_min_distance_A'] + [f'{p}_lt_{c}A' for c in (3, 4, 5)
            for p in ('pl_pairs', 'ligand_atoms', 'protein_atoms', 'protein_residues')]


def random_topk_success(n, m, k=TOPK):
    """크기 n 풀에서 near-native m개일 때 무작위 Top-k가 하나 이상 포함할 확률."""
    if n <= 0:
        return None
    if m <= 0:
        return 0.0
    if n <= k:
        return 1.0
    return 1.0 - comb(n - m, k) / comb(n, k)


def stratum_of(n_label, n_missing, n_near_native):
    """결과를 보기 전에 고정한 기준으로 시스템 난이도를 분류합니다."""
    if n_label == 0:
        return 'unlabeled'                       # 난이도 불명
    if n_near_native == 0:
        # near-native가 확인되지 않음. 일부 결측이면 '성공 불가'로 단정하지 않습니다.
        return 'none_full_label' if n_missing == 0 else 'none_partial_label'
    frac = n_near_native / n_label
    if frac >= EASY_MIN:
        return 'easy'
    if frac >= MEDIUM_MIN:
        return 'medium'
    return 'hard'


def tie_key(pose_id):
    """동점을 결정적으로 깨되 생성 모델과 무관한 해시 키를 만듭니다."""
    return int(hashlib.sha1(str(pose_id).encode()).hexdigest(), 16)


def run(args):
    """08 feature와 라벨을 조인하여 난이도 층화 비교를 저장합니다."""
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

    ev = pd.read_csv(args.eval_csv, usecols=list(KEYS) + ['sc_rmsd', 'ligand_rmsd', 'status', 'mapping_status'])
    ev = ev.rename(columns={'status': 'eval_status', 'mapping_status': 'eval_mapping_status'})
    for k in ('seed', 'sample'):
        ev[k] = ev[k].astype(int)
    ev['system_id'] = ev['system_id'].astype(str)
    ev['model'] = ev['model'].astype(str)

    merged = comp.merge(ev, on=list(KEYS), how='left')
    merged['sc_rmsd_num'] = pd.to_numeric(merged['sc_rmsd'], errors='coerce')
    merged['label_known'] = merged['sc_rmsd_num'].notna()
    merged['near_native'] = merged['label_known'] & (merged['sc_rmsd_num'] < THRESHOLD)
    merged['pose_id'] = (merged['system_id'] + '|' + merged['model'] + '|'
                         + merged['seed'].astype(str) + '|' + merged['sample'].astype(str))
    merged['_tie'] = merged['pose_id'].map(tie_key)

    # 1) 시스템별 난이도·라벨 커버리지(사전 고정 기준).
    sys_rows = []
    groups = {sid: g for sid, g in merged.groupby('system_id')}
    for sid in sorted(groups):
        g = groups[sid]
        n = len(g)
        n_label = int(g['label_known'].sum())
        n_nn = int(g['near_native'].sum())
        n_missing = n - n_label
        frac = (n_nn / n_label) if n_label else None
        sys_rows.append({'system_id': sid, 'n_pose': n, 'n_label': n_label, 'n_missing': n_missing,
                         'n_near_native': n_nn,
                         'near_native_fraction': round(frac, 4) if frac is not None else None,
                         'stratum': stratum_of(n_label, n_missing, n_nn),
                         'best_sc_rmsd': round(float(g['sc_rmsd_num'].min()), 4) if n_label else None})
    audit.save_csv(out / 'system_difficulty.csv', sys_rows,
                   ('system_id', 'n_pose', 'n_label', 'n_missing', 'n_near_native', 'near_native_fraction', 'stratum'))
    stratum_of_system = {r['system_id']: r['stratum'] for r in sys_rows}

    # 2) feature vs random: 두 풀(all/labeled)에서 동일 분모로 Top-k 비교.
    #    pool_all   : computed 100개 전체, 결측=non-near-native(보수적).
    #    pool_label : 유효 라벨 pose만.
    # near-native가 존재하는 시스템만 Top-k 성공 비교 대상(그 외 분모 왜곡 방지).
    def eligible_systems(pool):
        res = []
        for sid in sorted(groups):
            g = groups[sid]
            if pool == 'labeled':
                g = g[g['label_known']]
            if g['near_native'].any():
                res.append(sid)
        return res

    ranking_rows = []
    model_balance = Counter()
    for pool in ('all', 'labeled'):
        elig = eligible_systems(pool)
        by_stratum = defaultdict(list)
        for sid in elig:
            by_stratum[stratum_of_system[sid]].append(sid)
        # random 기대(시스템별) — 동일 풀 기준.
        rand_by_sys = {}
        for sid in elig:
            g = groups[sid]
            if pool == 'labeled':
                g = g[g['label_known']]
            rand_by_sys[sid] = random_topk_success(len(g), int(g['near_native'].sum()))
        for f in FEATURES:
            for direction in ('asc', 'desc'):
                succ = {}
                for sid in elig:
                    g = groups[sid]
                    if pool == 'labeled':
                        g = g[g['label_known']]
                    ordered = g.sort_values([f, '_tie'], ascending=[direction == 'asc', True], kind='mergesort')
                    top = ordered.head(TOPK)
                    succ[sid] = bool(top['near_native'].any())
                    for mdl in top['model']:
                        model_balance[(f, direction, pool, mdl)] += 1
                # 층별 집계 + 전체(eligible) 집계.
                for stratum in list(by_stratum) + ['all_eligible']:
                    sids = elig if stratum == 'all_eligible' else by_stratum[stratum]
                    if not sids:
                        continue
                    frate = sum(succ[s] for s in sids) / len(sids)
                    rrate = sum(rand_by_sys[s] for s in sids) / len(sids)
                    ranking_rows.append({
                        'pool': pool, 'stratum': stratum, 'feature': f, 'direction': direction,
                        'n_systems': len(sids), 'feature_top5_rate': round(frate, 4),
                        'random_top5_rate': round(rrate, 4), 'lift': round(frate - rrate, 4)})
    audit.save_csv(out / 'feature_stratum.csv', ranking_rows,
                   ('pool', 'stratum', 'feature', 'direction', 'n_systems', 'feature_top5_rate', 'random_top5_rate', 'lift'))

    balance_rows = [{'feature': f, 'direction': d, 'pool': p, 'model': m, 'topk_selected': c}
                    for (f, d, p, m), c in sorted(model_balance.items())]
    audit.save_csv(out / 'model_balance.csv', balance_rows, ('feature', 'direction', 'pool', 'model', 'topk_selected'))

    # 층 요약 표.
    stratum_counts = Counter(r['stratum'] for r in sys_rows)
    stratum_rows = []
    for stratum in ('easy', 'medium', 'hard', 'none_full_label', 'none_partial_label', 'unlabeled'):
        members = [r for r in sys_rows if r['stratum'] == stratum]
        if not members:
            stratum_rows.append({'stratum': stratum, 'n_systems': 0, 'mean_label_coverage': None,
                                 'mean_near_native_fraction': None})
            continue
        cov = [r['n_label'] / r['n_pose'] for r in members]
        frac = [r['near_native_fraction'] for r in members if r['near_native_fraction'] is not None]
        stratum_rows.append({'stratum': stratum, 'n_systems': len(members),
                             'mean_label_coverage': round(sum(cov) / len(cov), 4),
                             'mean_near_native_fraction': round(sum(frac) / len(frac), 4) if frac else None})
    audit.save_csv(out / 'stratum_summary.csv', stratum_rows,
                   ('stratum', 'n_systems', 'mean_label_coverage', 'mean_near_native_fraction'))

    changed = []
    for source in sources:
        try:
            info = Path(source['path']).stat()
            if (info.st_size, info.st_mtime_ns) != (source['size_bytes'], source['mtime_ns']):
                changed.append(source['path'])
        except OSError:
            changed.append(source['path'])

    # 탐색용 best(동일 표본 선택이라 낙관적) — pool_all, all_eligible 기준.
    cand = [r for r in ranking_rows if r['pool'] == 'all' and r['stratum'] == 'all_eligible']
    best = max(cand, key=lambda r: r['feature_top5_rate'], default=None)
    status = 'STRATIFIED_COMPLETE' if not changed else 'CHECK_REQUIRED'
    report = {
        'version': VERSION, 'status': status, 'output_dir': str(out),
        'features_run': str(features_dir), 'eval_csv': str(args.eval_csv), 'sources': sources,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'support_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (_support_path, roles._support_path, audit._support_path)},
        'python': sys.executable,
        'predefined_criteria': {
            'near_native': 'sc_rmsd < 2.0 (strict)', 'top_k': TOPK,
            'difficulty_fraction': 'near_native / n_label (분모=유효 라벨)',
            'strata': {'easy': f'fraction >= {EASY_MIN}', 'medium': f'{MEDIUM_MIN} <= fraction < {EASY_MIN}',
                       'hard': f'0 < fraction < {MEDIUM_MIN}',
                       'none_full_label': 'n_label>0, 결측 0, near-native 0',
                       'none_partial_label': 'n_label>0, 결측>0, near-native 0 (성공불가로 단정하지 않음)',
                       'unlabeled': 'n_label==0 (난이도 불명)'},
            'pools': {'all': 'computed 100개 전체, 결측=non-near-native',
                      'labeled': '유효 라벨 pose만'},
            'tie_break': '동점은 pose_id SHA1 해시 오름차순(모델 무관)',
            'ligand_rmsd_fallback': '미적용'},
        'stratum_counts': dict(stratum_counts),
        'systems': len(sys_rows),
        'best_single_feature_exploratory': ({'feature': best['feature'], 'direction': best['direction'],
                                             'pool': best['pool'], 'stratum': best['stratum'],
                                             'feature_top5_rate': best['feature_top5_rate'],
                                             'random_top5_rate': best['random_top5_rate'],
                                             'lift': best['lift']} if best else None),
        'changed_sources': changed, 'elapsed_seconds': time.perf_counter() - started,
        'confirmed': ['난이도 구간·분모·동점 기준을 사전 고정했고 두 풀에서 동일 조건으로 비교했습니다.',
                      '라벨 전부 결측(unlabeled)과 일부 결측·near-native 미확인(none_partial_label)을 분리했습니다.'],
        'not_determined': ['sc_rmsd 결측 pose의 실제 near-native 여부와 결측 원인(평가 매핑 실패 등)은 확정하지 않았습니다.',
                           '단일 feature·선형 ranking 결과이며 다중 feature·학습 모델의 성능은 아닙니다.',
                           '선정 40시스템은 난이도 대표성이 없어 전체 데이터셋 성능으로 일반화할 수 없습니다.'],
        'notes': ['category 1 feature를 삭제하거나 추가 재분석을 계속하지 않습니다. 이 정리는 1회성입니다.',
                  'best는 동일 표본에서 고른 탐색 결과이며 일반화 성능이 아닙니다.']}
    helper.write_json(out / 'stratified_report.json', report)
    helper.write_json(out / 'RUN_COMPLETE.json', {'outputs_written': True, 'analysis_finished': True,
        'status': status, 'systems': len(sys_rows), 'sources_unchanged': not changed})
    print('STATUS:', status, flush=True)
    print('STRATUM_COUNTS:', roles.j(dict(stratum_counts)), flush=True)
    if best:
        print('BEST(exploratory, all/all_eligible):', best['feature'], best['direction'],
              'feat=', best['feature_top5_rate'], 'random=', best['random_top5_rate'], 'lift=', best['lift'], flush=True)
    return 0 if status == 'STRATIFIED_COMPLETE' else 2


def main():
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
