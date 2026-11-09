#!/usr/bin/env python3
"""타깃별 Top-5 성공률 평가 하네스(B①). 어떤 스코어링이든 공통 규칙으로 채점합니다.

성공 기준은 타깃(시스템)별 Top-K에 sc_rmsd < threshold(기본 2.0, 엄격 <) pose가 하나 이상
포함되는지입니다. 두 후보 풀에서 평가합니다 — all: 100개 전체를 랭킹하고 라벨 결측은
비near-native로 간주(보수적), labeled: 라벨 있는 pose만. 시스템은 near-native 보유(eligible),
라벨은 있으나 near-native 없음(labeled_no_nn), 라벨 전부 결측(fully_unlabeled)으로 분류해
분모를 분리 보고합니다. oracle 상한(= eligible/대상)과 random Top-K 기대(해석식 + 선택적
경험식)를 같은 후보집합·분모에서 함께 내보내 어떤 스코어링도 동일 조건에서 비교합니다. 동점은
pose_id SHA1 해시 오름차순으로 결정적으로 깨어 특정 생성 모델에 치우치지 않게 합니다. 18의
dev/test split이 있으면 분해해 보고하되 test는 모델·feature 선택에 쓰지 않습니다(표시만).

스코어링이 없으면 baseline(oracle·random)만 산출하며, 이 경우 labeled 풀 eligible random 평균과
oracle_rate는 24 라벨 리포트 값을 재현해야 합니다(정합성 자체검증). 스코어링은 --scores parquet
(키 또는 pose_id + 점수 열)과 --score-col·--direction으로 주입합니다. GT 좌표는 읽지 않고
24가 만든 sc_rmsd 라벨만 사용합니다. 05→04→03 함수에서 경로 보호·저장을 재사용합니다.

출력: out/eval/25_run_evaluate[_TAG]/ 아래 고정 폴더(덮어쓰기 금지, --tag로 분리).
"""
import sys
sys.dont_write_bytecode = True
import argparse
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

VERSION = 'eval_0.1.0'
OUT_KIND = 'eval'
OUT_BASE = '25_run_evaluate'
KEYS = ['system_id', 'model', 'seed', 'sample']
DEFAULT_LABELS = helper.ALLOWED_ROOT / 'Scoring_Function' / 'out' / 'labels' / '24_run_build_labels' / 'labels.parquet'
DEFAULT_SPLIT = helper.ALLOWED_ROOT / 'Scoring_Function' / 'out' / 'data_split' / '18_run_make_split' / 'system_split.csv'
THRESHOLD = 2.0
TOPK = 5
POOLS = ('all', 'labeled')


def random_topk_success(n, m, k):
    """크기 n 풀에서 near-native m개일 때 무작위 Top-k가 하나 이상 포함할 확률(해석식)."""
    if n <= 0:
        return None
    if m <= 0:
        return 0.0
    if n <= k:
        return 1.0
    return 1.0 - comb(n - m, k) / comb(n, k)


def tie_key(pose_id):
    """동점을 결정적으로 깨되 생성 모델과 무관한 해시 키."""
    return int(hashlib.sha1(str(pose_id).encode()).hexdigest(), 16)


def empirical_random(nn_values, k, rng, trials):
    """near_native 불리언 배열에서 무작위 Top-k 경험적 성공률(trials회 평균)."""
    import numpy as np
    n = len(nn_values)
    if trials <= 0 or n == 0:
        return None
    m = int(nn_values.sum())
    if m == 0:
        return 0.0
    if n <= k or m == n:
        return 1.0
    idx = np.arange(n)
    hits = 0
    for _ in range(trials):
        pick = rng.choice(idx, size=k, replace=False)
        if nn_values[pick].any():
            hits += 1
    return hits / trials


def scoring_topk_success(g, ascending, k):
    """점수 순 Top-k에 near-native가 있는지. NaN 점수는 최하위로 보냅니다."""
    ordered = g.sort_values(['_score', '_tie'], ascending=[ascending, True],
                            na_position='last', kind='mergesort')
    return bool(ordered.head(k)['near_native'].to_numpy().any())


def aggregate(rows, pool, has_scores, selector):
    """선택된 시스템 집합에서 pool별 지표를 집계합니다."""
    sub = [r for r in rows if selector(r)]
    n = len(sub)
    if n == 0:
        return None
    elig = [r for r in sub if r['eligible']]
    out = {'pool': pool, 'n_systems': n, 'n_eligible': len(elig),
           'oracle_ceiling': round(len(elig) / n, 4),
           'random_analytic': round(sum(r[f'{pool}_rand_analytic'] for r in sub) / n, 4)}
    emp = [r[f'{pool}_rand_emp'] for r in sub if r[f'{pool}_rand_emp'] is not None]
    out['random_empirical'] = round(sum(emp) / len(emp), 4) if emp else None
    if has_scores:
        out['scoring_success'] = round(sum(r[f'{pool}_score_success'] for r in sub) / n, 4)
        out['lift_vs_random'] = round(out['scoring_success'] - out['random_analytic'], 4)
    return out


def run(args):
    """라벨과 (선택)스코어링을 받아 Top-K 성공률을 baseline과 함께 평가합니다."""
    started = time.perf_counter()
    out_name = OUT_BASE + (f'_{args.tag}' if args.tag else '')
    parent = helper.safe_mkdir(helper.ALLOWED_ROOT / 'Scoring_Function' / 'out' / OUT_KIND)
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

    import numpy as np
    import pandas as pd

    sources = [helper.fingerprint(args.labels)]
    labels = pd.read_parquet(args.labels)
    labels['system_id'] = labels['system_id'].astype(str)
    labels['sc_rmsd'] = pd.to_numeric(labels['sc_rmsd'], errors='coerce')
    labels['label_known'] = labels['sc_rmsd'].notna()
    # threshold를 인자로 재적용(기본 2.0은 24 라벨과 동일).
    labels['near_native'] = labels['label_known'] & (labels['sc_rmsd'] < args.threshold)
    if 'pose_id' not in labels.columns:
        labels['pose_id'] = (labels['system_id'] + '|' + labels['model'].astype(str) + '|'
                             + labels['seed'].astype(str) + '|' + labels['sample'].astype(str))
    labels['_tie'] = labels['pose_id'].map(tie_key)

    # 스코어링 주입(선택). 키 또는 pose_id로 병합하며 커버리지를 보고합니다.
    has_scores = args.scores is not None
    score_coverage = None
    if has_scores:
        sources.append(helper.fingerprint(args.scores))
        sc = pd.read_parquet(args.scores)
        if args.score_col not in sc.columns:
            raise ValueError(f'점수 열이 없습니다: {args.score_col}')
        if set(KEYS) <= set(sc.columns):
            on = KEYS
            sc = sc[KEYS + [args.score_col]].copy()
            sc['system_id'] = sc['system_id'].astype(str)
            for k in ('seed', 'sample'):
                sc[k] = sc[k].astype(int)
            sc['model'] = sc['model'].astype(str)
        elif 'pose_id' in sc.columns:
            on = ['pose_id']
            sc = sc[['pose_id', args.score_col]].copy()
        else:
            raise ValueError('스코어 파일에 KEYS 또는 pose_id가 필요합니다.')
        sc = sc.rename(columns={args.score_col: '_score'})
        if sc.duplicated(on).any():
            raise ValueError('스코어 파일에 pose 키 중복이 있습니다.')
        labels = labels.merge(sc, on=on, how='left')
        n_scored = int(labels['_score'].notna().sum())
        score_coverage = {'scored_poses': n_scored, 'total_poses': int(len(labels)),
                          'coverage': round(n_scored / len(labels), 4) if len(labels) else None}
    else:
        labels['_score'] = np.nan

    # split 주입(선택). 없으면 전체만 보고합니다.
    split_map = {}
    if args.split is not None and Path(args.split).exists():
        sources.append(helper.fingerprint(args.split))
        sp = pd.read_csv(args.split, dtype={'system_id': str})
        split_map = {r['system_id']: (str(r['split']), int(r['cv_fold']))
                     for _, r in sp.iterrows()}

    rng = np.random.default_rng(args.seed)
    rows = []
    for sid, g_all in labels.groupby('system_id', sort=True):
        g_lab = g_all[g_all['label_known']]
        n_label = int(len(g_lab))
        nn_full = int(g_all['near_native'].to_numpy().sum())
        if n_label == 0:
            category = 'fully_unlabeled'
        elif nn_full > 0:
            category = 'eligible'
        else:
            category = 'labeled_no_nn'
        split, fold = split_map.get(sid, (None, None))
        row = {'system_id': sid, 'split': split, 'cv_fold': fold, 'category': category,
               'eligible': nn_full > 0, 'n_pose': int(len(g_all)), 'n_label': n_label,
               'n_near_native': nn_full}
        for pool, gg in (('all', g_all), ('labeled', g_lab)):
            nn = gg['near_native'].to_numpy()
            row[f'{pool}_rand_analytic'] = random_topk_success(len(gg), int(nn.sum()), args.topk) or 0.0
            row[f'{pool}_rand_emp'] = empirical_random(nn, args.topk, rng, args.random_trials)
            if has_scores:
                row[f'{pool}_score_success'] = (scoring_topk_success(gg, args.direction == 'asc', args.topk)
                                                if len(gg) else False)
        rows.append(row)

    # 전체·분모별 집계.
    overall = []
    denoms = [('all_systems', lambda r: True),
              ('labeled_systems', lambda r: r['n_label'] > 0),
              ('eligible_systems', lambda r: r['eligible'])]
    for pool in POOLS:
        for dname, sel in denoms:
            agg = aggregate(rows, pool, has_scores, sel)
            if agg:
                agg['denominator'] = dname
                overall.append(agg)
    ov_header = ['pool', 'denominator', 'n_systems', 'n_eligible', 'oracle_ceiling',
                 'random_analytic', 'random_empirical'] + (['scoring_success', 'lift_vs_random'] if has_scores else [])
    audit.save_csv(out / 'overall_metrics.csv', overall, tuple(ov_header))

    # split별 집계(있을 때).
    split_rows = []
    if split_map:
        for sval in ('dev', 'test'):
            for pool in POOLS:
                for dname, sel in denoms:
                    agg = aggregate(rows, pool, has_scores,
                                    lambda r, sv=sval, s=sel: r['split'] == sv and s(r))
                    if agg:
                        agg['split'] = sval
                        agg['denominator'] = dname
                        split_rows.append(agg)
        sp_header = ['split', 'pool', 'denominator', 'n_systems', 'n_eligible', 'oracle_ceiling',
                     'random_analytic', 'random_empirical'] + (['scoring_success', 'lift_vs_random'] if has_scores else [])
        audit.save_csv(out / 'split_metrics.csv', split_rows, tuple(sp_header))

    # 시스템별 상세.
    sys_header = ['system_id', 'split', 'cv_fold', 'category', 'eligible', 'n_pose', 'n_label', 'n_near_native',
                  'all_rand_analytic', 'labeled_rand_analytic']
    if has_scores:
        sys_header += ['all_score_success', 'labeled_score_success']
    audit.save_csv(out / 'system_eval.csv', rows, tuple(sys_header))

    changed = []
    for source in sources:
        try:
            info = Path(source['path']).stat()
            if (info.st_size, info.st_mtime_ns) != (source['size_bytes'], source['mtime_ns']):
                changed.append(source['path'])
        except OSError:
            changed.append(source['path'])

    from collections import Counter
    cat_counts = Counter(r['category'] for r in rows)
    status = 'EVAL_COMPLETE' if not changed else 'CHECK_REQUIRED'
    report = {
        'version': VERSION, 'status': status, 'output_dir': str(out),
        'labels': str(args.labels), 'scores': str(args.scores) if has_scores else None,
        'score_col': args.score_col if has_scores else None,
        'direction': args.direction if has_scores else None,
        'split_csv': str(args.split) if split_map else None, 'sources': sources,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'support_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (_support_path, roles._support_path, audit._support_path)},
        'python': sys.executable, 'threshold_sc_rmsd': args.threshold, 'top_k': args.topk,
        'random_trials': args.random_trials, 'seed': args.seed, 'tie_break': 'pose_id SHA1 오름차순',
        'systems': len(rows), 'category_counts': dict(cat_counts),
        'score_coverage': score_coverage, 'overall_metrics': overall,
        'split_metrics': split_rows if split_map else None,
        'changed_sources': changed, 'elapsed_seconds': time.perf_counter() - started,
        'gt_role': 'sc_rmsd 라벨로만 채점하며 feature·pose 선택에 사용하지 않음',
        'policy': [
            '성공=타깃별 Top-K에 sc_rmsd<threshold pose 하나 이상(엄격 <).',
            '두 풀(all=결측 비near-native 보수적, labeled=라벨 pose만)을 모두 보고.',
            'fully_unlabeled(난이도 불명)와 labeled_no_nn을 eligible과 분리.',
            'oracle 상한과 random(해석식+경험식)을 같은 후보집합·분모에서 비교.',
            'test split은 모델·feature 선택에 사용하지 않으며 표시만 합니다.'],
        'notes': ['스코어링 미주입 시 baseline만 산출합니다(labeled/eligible random은 24 값 재현 자체검증).',
                  '경험적 random은 해석식 검증용이며 random_trials=0이면 생략합니다.']}
    helper.write_json(out / 'eval_report.json', report)
    helper.write_json(out / 'RUN_COMPLETE.json', {'outputs_written': True, 'eval_finished': True,
        'status': status, 'systems': len(rows), 'has_scores': has_scores,
        'sources_unchanged': not changed})
    print('STATUS:', status, flush=True)
    print('SYSTEMS:', len(rows), '| CATEGORY:', roles.j(dict(cat_counts)), flush=True)
    if has_scores:
        print('SCORE_COVERAGE:', roles.j(score_coverage), flush=True)
    for agg in overall:
        tail = (f" | scoring={agg['scoring_success']} lift={agg['lift_vs_random']}"
                if has_scores else '')
        print(f"[{agg['pool']}/{agg['denominator']}] n={agg['n_systems']} "
              f"oracle={agg['oracle_ceiling']} rand={agg['random_analytic']}{tail}", flush=True)
    return 0 if status == 'EVAL_COMPLETE' else 2


def main():
    """라벨·스코어링·split 위치를 지정하며 출력은 사용자 경로로 고정합니다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, default=DEFAULT_LABELS)
    parser.add_argument('--scores', type=Path, default=None, help='pose별 점수 parquet(KEYS 또는 pose_id + 점수 열).')
    parser.add_argument('--score-col', type=str, default='score')
    parser.add_argument('--direction', choices=('asc', 'desc'), default='asc',
                        help='asc=점수 낮을수록 좋음, desc=높을수록 좋음.')
    parser.add_argument('--split', type=Path, default=DEFAULT_SPLIT)
    parser.add_argument('--threshold', type=float, default=THRESHOLD)
    parser.add_argument('--topk', type=int, default=TOPK)
    parser.add_argument('--random-trials', type=int, default=20, help='경험적 random 반복 수(0이면 생략).')
    parser.add_argument('--seed', type=int, default=20261028)
    parser.add_argument('--tag', type=str, default=None, help='출력 폴더 접미사(반복 평가 분리용).')
    args = parser.parse_args()
    for key in ('labels', 'scores', 'split'):
        v = getattr(args, key)
        if v is not None:
            setattr(args, key, v.absolute())
    if args.topk < 1 or args.threshold <= 0:
        parser.error('topk는 1 이상, threshold는 양수여야 합니다.')
    return run(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
