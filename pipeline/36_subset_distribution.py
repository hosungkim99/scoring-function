#!/usr/bin/env python3
"""Process B 보강(11). "feature 집합 선택이 얼마나 중요한가"를 분포로 답한다.

*** 사전 등록 — 이 실행은 최고 집합을 고르지 않는다. ***

이유가 산수에 있다. 집합 하나의 test k=1 표준편차가 약 0.006~0.009 이므로, 실력이
전부 같아도 N 개를 재서 최고를 고르면 최고값은 약 sigma * sqrt(2 ln N) 만큼 부푼다.

    N= 4 -> +0.012 | N= 7 -> +0.014 | N=20 -> +0.017 | N=100 -> +0.021

35번에서 관측한 네 집합의 전체 폭이 0.0145 였다. 즉 7개만 재도 부풀림이 관측 폭과
같아진다. 33번이 dev 에서 1,294개를 쓸어 얻은 +0.0069 도 그 탐색의 부풀림(약 +0.015)
안에 있었고, 실제로 test 에서 -0.0056 으로 뒤집혔다. 같은 실수를 test 에서 반복하면
되돌릴 데이터가 없다.

그래서 이 실행의 답은 최고값이 아니라 **분산 분해**다.

    관측 분산 = 집합 간 진짜 분산 + 시드 잡음 / 반복수
    집합 간 진짜 분산 <= 0  ->  "집합 선택은 측정 가능한 영향이 없다"

재는 것
  ① 지정 3종 — 34번의 union_top1 / union_top2 / union_top5. 35번이 빠뜨린 집합이며
     35번과 같은 시드 5회로 재어 바로 비교된다. 7집합 기록을 완성한다.
  ② 무작위 균일 — 전체 pool 에서 크기별로 균일 추출. "아무 집합이나 되는가"
  ③ 무작위 고정축 — 단독 가능 7개를 반드시 포함하고 나머지를 무작위로 채움.
     "좋은 7개 위에 무엇을 더 얹든 상관없는가" — adopted_33 vs union_all12 가
     구분되지 않은 이유를 직접 답하는 층이다.

35번이 이미 잰 4종(adopted_33 / union_all12 / anchor_only / top10_32)은 다시 재지 않고
35번 출력 dev_vs_test.csv 에서 읽어 같은 표에 올린다. 시드와 코드가 같아 그대로 비교된다.

고정 조건은 35번과 동일하다(HELD_CONSTANT). 변인은 feature 목록뿐이다.
36번의 대조군 값이 35번 보고서와 일치하는지 실행 중에 확인한다.

출력: out/subset_dist/36_run_subsets[_TAG]/
"""
import sys
sys.dont_write_bytecode = True
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time

# 35번을 통해 34~28번까지 불러옵니다(측정·평가 코드를 그대로 공유합니다).
_t35_path = Path(__file__).with_name('35_test_evaluation.py')
_tspec = importlib.util.spec_from_file_location('test_eval_support', _t35_path)
test_eval = importlib.util.module_from_spec(_tspec)
_tspec.loader.exec_module(test_eval)
union = test_eval.union
combination = test_eval.combination
selection = test_eval.selection
combined = test_eval.combined
graded = test_eval.graded
ranking = test_eval.ranking
baseline = test_eval.baseline
roles = test_eval.roles
audit = test_eval.audit
helper = test_eval.helper

VERSION = 'subset_dist_0.1.0'
OUT_KIND = 'subset_dist'
OUT_BASE = '36_run_subsets'
KEYS = baseline.KEYS
BASE = baseline.BASE
DEFAULT_MATRIX = baseline.DEFAULT_MATRIX
RULES = combined.RULES
HEADLINE_RULE = selection.HEADLINE_RULE
HEADLINE_K = selection.HEADLINE_K
HELD_CONSTANT = test_eval.HELD_CONSTANT
DEFAULT_UNION = BASE / 'union' / '34_run_union_final_full'
DEFAULT_TEST_EVAL = BASE / 'test_eval' / '35_run_test_final'
# 35번이 빠뜨린 집합. 34번 union_results.csv 에서 읽습니다.
DESIGNATED = ('union_top1', 'union_top2', 'union_top5')
# 35번이 이미 잰 집합. 다시 재지 않고 결과만 읽어 같은 표에 올립니다.
REUSED = ('adopted_33', 'union_all12', 'anchor_only', 'top10_32')
# 이 실행이 답하는 칸. 어려운 층은 test 표본이 61개뿐이라 참고로만 냅니다.
DECISION_METRICS = ('lift|ruleA|k1', 'lift|ruleA|k3', 'lift|ruleA|k5')
PREREGISTRATION = (
    '이 실행은 최고 집합을 고르지 않는다. 최고값은 보고하되 선택 후보가 아니다.',
    '헤드라인은 분산 분해다: 집합 간 진짜 분산이 0 이하면 "집합 선택은 영향 없음".',
    '최고값에 붙는 부풀림(sigma * sqrt(2 ln N))을 같은 표에 함께 적는다.',
    '무작위 부분집합은 고정 시드로 뽑으며, 뽑은 뒤 성적을 보고 교체하지 않는다.',
    '35번이 이미 잰 4종은 재측정하지 않고 그 결과를 읽어 비교에 쓴다.',
    'dev 와 test 의 순위 상관을 함께 내어 dev 최대값이 test 로 전이되는지 본다.')


def read_union_sets(union_dir, labels):
    """34번 union_results.csv 에서 지정 집합을 읽습니다(손으로 적지 않습니다)."""
    import csv
    path = Path(union_dir) / 'union_results.csv'
    if not path.exists():
        raise FileNotFoundError(f'34번 결과가 없습니다: {path}')
    table = {}
    with path.open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            joined = (row.get('features_joined') or '').strip()
            if row.get('label') and joined:
                table[row['label']] = sorted(f for f in joined.split('|') if f)
    missing = [name for name in labels if name not in table]
    if missing:
        raise ValueError(f'union_results.csv 에 없는 집합: {missing} '
                         f'(있는 것: {sorted(table)})')
    return {name: table[name] for name in labels}, table


def read_prior_results(test_eval_dir, labels):
    """35번 dev_vs_test.csv 에서 이미 잰 집합의 결과를 읽습니다."""
    import csv
    path = Path(test_eval_dir) / 'dev_vs_test.csv'
    if not path.exists():
        raise FileNotFoundError(f'35번 결과가 없습니다: {path}')
    rows = {}
    with path.open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            if row.get('label'):
                rows[row['label']] = row
    missing = [name for name in labels if name not in rows]
    if missing:
        raise ValueError(f'35번 결과에 없는 집합: {missing}')
    report_path = Path(test_eval_dir) / 'test_report.json'
    report = (json.loads(report_path.read_text(encoding='utf-8'))
              if report_path.exists() else {})
    return {name: rows[name] for name in labels}, report


def build_subsets(pool, anchor, args):
    """무작위 부분집합을 고정 시드로 뽑습니다. 뽑은 뒤 바꾸지 않습니다."""
    import numpy as np
    rng = np.random.default_rng(args.subset_seed)
    pool = sorted(pool)
    anchor = sorted(anchor)
    rest = sorted(set(pool) - set(anchor))
    out = []
    for size in args.uniform_sizes:
        if size > len(pool):
            continue
        for i in range(args.random_per_size):
            pick = sorted(rng.choice(pool, size=size, replace=False).tolist())
            out.append({'label': f'uniform{size}_{i + 1}', 'kind': 'random_uniform',
                        'group': f'uniform|{size}', 'size': size, 'features': pick})
    for size in args.anchored_sizes:
        extra = size - len(anchor)
        if extra < 0 or extra > len(rest):
            continue
        for i in range(args.random_per_size):
            pick = sorted(anchor + (rng.choice(rest, size=extra, replace=False).tolist()
                                    if extra else []))
            out.append({'label': f'anchored{size}_{i + 1}', 'kind': 'random_anchored',
                        'group': f'anchored|{size}', 'size': size, 'features': pick})
    return out


def dedup(entries):
    """같은 feature 집합이 두 번 측정되지 않게 합니다. 분포가 왜곡되기 때문입니다."""
    seen, kept, dropped = {}, [], []
    for row in entries:
        key = frozenset(row['features'])
        if key in seen:
            dropped.append({'label': row['label'], 'duplicate_of': seen[key]})
            continue
        seen[key] = row['label']
        kept.append(row)
    return kept, dropped


def rankdata(values):
    """동점은 평균 순위로. scipy 없이 씁니다."""
    import numpy as np
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind='mergesort')
    ranks = np.empty(arr.size, dtype=np.float64)
    i = 0
    while i < arr.size:
        j = i
        while j + 1 < arr.size and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a, b):
    """순위 상관. 점이 3개 미만이거나 한쪽이 전부 동점이면 None."""
    import numpy as np
    if len(a) < 3:
        return None
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _betacf(a, b, x):
    """불완전 베타의 연분수(Lentz). betainc 전용 보조 함수입니다."""
    maxit, eps, fpmin = 300, 3.0e-16, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a, b, x):
    """정규화 불완전 베타 I_x(a,b). scipy 를 쓰지 않습니다."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def f_tail(f, df1, df2):
    """F(df1, df2) 에서 P(F > f). 분산분석 p 값에 씁니다."""
    if f is None or f <= 0.0 or df1 < 1 or df2 < 1:
        return None
    return betainc(df2 / 2.0, df1 / 2.0, df2 / (df2 + df1 * f))


def decompose(means, stds, repeats):
    """관측 분산을 집합 간 분산과 시드 잡음으로 나눕니다.

    시드 repeats 회의 평균이므로 평균의 잡음 분산은 within/repeats 입니다.
    집합 간 분산이 0 이하로 나오면 집합 선택의 영향이 잡음에 묻힌 것입니다.
    """
    import numpy as np
    means = [m for m in means if m is not None]
    stds = [s for s in stds if s is not None]
    if len(means) < 2:
        return None
    n = len(means)
    observed = float(np.var(means, ddof=1))
    median = float(np.median(means))
    peak = float(np.max(means))
    base = {
        'n': n,
        'min': round(float(np.min(means)), 4),
        'q1': round(float(np.percentile(means, 25)), 4),
        'median': round(median, 4),
        'q3': round(float(np.percentile(means, 75)), 4),
        'max': round(peak, 4),
        'range': round(peak - float(np.min(means)), 4),
        'max_minus_median': round(peak - median, 4),
        'observed_std': round(math.sqrt(observed), 4)}

    # 시드 반복이 없거나 표준편차가 없으면 잡음을 뺄 수 없어 판정하지 않습니다.
    if not stds or repeats < 2:
        base.update({'seed_std': None, 'mean_se': None, 'between_std': None,
                     'f_ratio': None, 'df1': n - 1, 'df2': 0, 'p_value': None,
                     'selection_inflation': None,
                     'verdict': '판정 불가(시드 반복 부족)'})
        return base

    # 일원 분산분석. 집합을 처리(treatment), 시드를 반복(replicate)으로 봅니다.
    #   MS_between = repeats * var(집합별 평균)        자유도 n-1
    #   MS_within  = mean(집합별 분산)                 자유도 n*(repeats-1)
    # 두 값의 비가 F 이고, 귀무가설은 "모든 집합이 같다"입니다. 부호만 보던
    # 이전 규칙은 잡음 추정값끼리의 차여서 0 근처에서 거짓 양성을 냈습니다.
    ms_within = float(np.mean([s ** 2 for s in stds]))
    ms_between = repeats * observed
    df1, df2 = n - 1, len(stds) * (repeats - 1)
    f_ratio = (ms_between / ms_within) if ms_within > 0 else None
    p_value = f_tail(f_ratio, df1, df2) if f_ratio is not None else None
    component = (ms_between - ms_within) / repeats
    seed_std = math.sqrt(ms_within)
    mean_se = seed_std / math.sqrt(repeats)
    base.update({
        'seed_std': round(seed_std, 4),
        'mean_se': round(mean_se, 4),
        'between_std': round(math.sqrt(component), 4) if component > 0 else 0.0,
        'f_ratio': round(f_ratio, 3) if f_ratio is not None else None,
        'df1': df1, 'df2': df2,
        'p_value': (round(p_value, 5) if p_value is not None else None),
        # 집합 n 개 중 최고를 고를 때 중앙값 위로 자동으로 붙는 허수 크기.
        # 비교 대상은 range(최대-최소)가 아니라 max_minus_median 입니다.
        'selection_inflation': round(mean_se * math.sqrt(2.0 * math.log(n)), 4)})
    if p_value is None:
        base['verdict'] = '판정 불가(잡음 0)'
    elif p_value >= 0.05:
        base['verdict'] = f'집합 선택의 영향이 잡음에 묻힘 (p={p_value:.3f})'
    else:
        base['verdict'] = f'집합 간 차이가 잡음보다 큼 (p={p_value:.4f})'
    return base


def measure(features, ctx, args, seeds):
    """35번과 같은 조건으로 dev out-of-fold 와 test 를 재고 시드 평균을 냅니다."""
    import numpy as np
    dev, test = ctx['dev'], ctx['test']
    idx = np.array([ctx['index_of'][f] for f in features], dtype=np.int64)
    dev_seeds, test_seeds = [], []
    for sd in seeds:
        variant = copy.copy(args)
        variant.seed = sd
        dev_score, _, _ = selection.train_oof(dev['X'], dev['rel'], dev['codes'],
                                              dev['fold_of'], dev['known'],
                                              ctx['folds'], idx, variant)
        rates, hard_rates, _ = selection.evaluate_scores(dev['groups'], dev_score,
                                                         args.topk)
        dev_seeds.append(test_eval.cells(rates, hard_rates, dev['controls'], args.topk))
        test_score, _, _ = test_eval.train_full(dev['X'][:, idx], dev['rel'],
                                                dev['codes'], dev['known'], variant,
                                                test['X'][:, idx])
        rates, hard_rates, _ = selection.evaluate_scores(test['groups'], test_score,
                                                         args.topk)
        test_seeds.append(test_eval.cells(rates, hard_rates, test['controls'],
                                          args.topk))
    out = {}
    for split, per_seed in (('dev', dev_seeds), ('test', test_seeds)):
        for key, value in test_eval.summarize(per_seed).items():
            stat, metric = key.split('|', 1)
            out[f'{split}|{stat}|{metric}'] = value
    return out


def prepare_splits(frame, feats, args, bins):
    """35번과 같은 방식으로 dev/test 를 준비합니다(값 일치를 뒤에서 확인합니다)."""
    import numpy as np
    import pandas as pd
    parts = {}
    for name in ('dev', 'test'):
        sub = frame[frame['split'] == name].reset_index(drop=True)
        if sub.empty:
            raise ValueError(f'{name} 분할이 비어 있습니다.')
        codes = pd.factorize(sub['system_id'], sort=False)[0]
        assert np.all(np.diff(codes) >= 0), f'{name} 시스템 그룹이 연속이 아닙니다.'
        sc = sub['sc_rmsd'].to_numpy(dtype=np.float64, na_value=np.nan)
        rel = combined.relevance_from_sc(sc, bins, 'graded')
        known = sub['label_known'].to_numpy(dtype=bool)
        nn = sub['near_native'].to_numpy(dtype=bool)
        level = combined.near_native_level(bins, 'graded')
        if int(np.sum((rel[known] >= level) != nn[known])):
            raise ValueError(f'{name}: 등급과 near_native 가 불일치합니다.')
        groups = selection.group_randoms(selection.prepare_groups(sub, codes), args.topk)
        controls = selection.evaluate_controls(groups, args.topk, args.shuffles, args.seed)
        parts[name] = {'frame': sub, 'codes': codes, 'rel': rel, 'known': known,
                       'X': sub[feats].to_numpy(dtype=np.float32, na_value=np.nan),
                       'fold_of': (sub['cv_fold'].astype(int).to_numpy()
                                   if name == 'dev' else None),
                       'groups': groups, 'controls': controls}
    return parts


def run(args):
    """지정 3종을 재고, 무작위 부분집합의 분포를 내고, 분산을 분해합니다."""
    if not args.reuse_test:
        print('이 실행은 test 923 타깃을 다시 엽니다. --reuse-test 를 주세요.',
              file=sys.stderr, flush=True)
        print('열기 전에 확인할 것:', file=sys.stderr, flush=True)
        print('  1) 이 실행은 최고 집합을 고르지 않는다는 것에 동의하는가',
              file=sys.stderr, flush=True)
        print('  2) 표본 시드(--subset-seed)를 바꿔 다시 돌려 좋은 집합을 찾는 것은 '
              '낚시이며, 하지 않을 각오가 됐는가', file=sys.stderr, flush=True)
        print('  3) 답이 "집합 선택은 영향 없음"으로 나와도 그것을 결과로 받아들일 '
              '준비가 됐는가', file=sys.stderr, flush=True)
        return 2
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
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(args.threads)
    print('OUTPUT_DIR:', out, flush=True)
    print('', flush=True)
    print('*** 사전 등록 — 이 실행은 최고 집합을 고르지 않습니다. ***', flush=True)
    for line in PREREGISTRATION:
        print(f'  - {line}', flush=True)
    print('', flush=True)

    import numpy as np
    import pandas as pd
    import xgboost

    matrix_path = args.matrix / 'unified_all.parquet'
    catalog_path = args.matrix / 'feature_catalog.csv'
    for p in (matrix_path, catalog_path):
        if not p.exists():
            raise FileNotFoundError(f'입력이 없습니다: {p}')
    sources = [helper.fingerprint(matrix_path), helper.fingerprint(catalog_path)]
    print('xgboost', xgboost.__version__, flush=True)

    designated, union_table = read_union_sets(args.union_dir, DESIGNATED)
    prior, prior_report = read_prior_results(args.test_eval_dir, REUSED)
    anchor = union_table.get('anchor_only')
    if not anchor:
        raise ValueError('anchor_only 를 읽지 못했습니다.')
    print('고정 조건 — feature 집합만 변인', flush=True)
    for line in HELD_CONSTANT:
        print(f'  고정  {line}', flush=True)
    print('', flush=True)
    for label, feats_of in designated.items():
        print(f'  지정   {label:14s} feature {len(feats_of):2d}개 (이번에 측정)', flush=True)
    for label in REUSED:
        print(f'  재사용 {label:14s} feature {prior[label]["n_features"]:>2s}개 '
              f'(35번 결과 읽음)', flush=True)

    catalog = pd.read_csv(catalog_path).set_index('feature')
    meta_cols = ['system_id', 'model', 'seed', 'sample', 'pose_id', 'sc_rmsd',
                 'near_native', 'label_known', 'split', 'cv_fold']
    frame = pd.read_parquet(matrix_path, columns=meta_cols + catalog.index.tolist())
    frame['near_native'] = frame['near_native'].fillna(False).astype(bool)
    frame['label_known'] = frame['label_known'].fillna(False).astype(bool)
    if 'split' not in frame.columns or frame['split'].isna().all():
        raise ValueError('split 열이 없습니다.')
    frame = frame.sort_values(KEYS, kind='mergesort').reset_index(drop=True)
    frame['_tie'] = frame['pose_id'].map(baseline.tie_key)

    feats, _ = baseline.select_features(frame, catalog, set())
    index_of = {name: i for i, name in enumerate(feats)}
    bins = combined.BIN_SETS[args.relevance_bins]

    randoms = build_subsets(feats, anchor, args)
    named = [{'label': label, 'kind': 'designated', 'group': 'designated',
              'size': len(fs), 'features': fs} for label, fs in designated.items()]
    entries, duplicates = dedup(named + randoms)
    needed = {f for row in entries for f in row['features']}
    missing = sorted(needed - set(feats))
    if missing:
        raise ValueError(f'현재 행렬에 없는 feature: {missing}')
    print('', flush=True)
    print(f'feature pool {len(feats)}개 | 단독 가능 축 {len(anchor)}개', flush=True)
    print(f'측정할 집합 {len(entries)}개 '
          f'(지정 {len(named)} + 무작위 {len(entries) - len(named)})', flush=True)
    for row in duplicates:
        print(f"  [제외] {row['label']} 는 {row['duplicate_of']} 와 같은 집합", flush=True)

    parts = prepare_splits(frame, feats, args, bins)
    dev, test = parts['dev'], parts['test']
    span = int(pd.Series(dev['fold_of']).groupby(dev['codes']).nunique().max())
    if span != 1:
        raise ValueError('dev 의 한 시스템이 여러 폴드에 걸쳐 있습니다.')
    folds = sorted(int(f) for f in np.unique(dev['fold_of']))
    print('', flush=True)
    for name in ('dev', 'test'):
        ctrl = parts[name]['controls']
        print(f'{name.upper()}: systems {int(len(set(parts[name]["frame"]["system_id"])))} '
              f'| eligible {len(parts[name]["groups"])} | 어려운 층 {ctrl["hard_systems"]}',
              flush=True)

    # 35번과 같은 파이프라인인지 대조군 값으로 확인합니다.
    consistency = {'checked': False, 'mismatches': []}
    prior_splits = (prior_report.get('splits') or {})
    if prior_splits:
        consistency['checked'] = True
        for name in ('dev', 'test'):
            want = (prior_splits.get(name) or {})
            got = parts[name]['controls']
            if want.get('eligible') not in (None, len(parts[name]['groups'])):
                consistency['mismatches'].append(f'{name}.eligible')
            if want.get('hard_systems') not in (None, got['hard_systems']):
                consistency['mismatches'].append(f'{name}.hard_systems')
            for k, value in (want.get('random_analytic') or {}).items():
                mine = got['analytic'].get(int(k))
                if mine is not None and abs(float(value) - mine) > 1e-9:
                    consistency['mismatches'].append(f'{name}.random_analytic.k{k}')
        if consistency['mismatches']:
            raise ValueError('35번과 대조군이 다릅니다(파이프라인 불일치): '
                             + ', '.join(consistency['mismatches']))
        print('35번 대조군과 일치 확인 — 같은 파이프라인입니다.', flush=True)

    ctx = {'dev': dev, 'test': test, 'index_of': index_of, 'folds': folds}
    named_seeds = [args.seed + i for i in range(args.repeats)]
    random_seeds = [args.seed + i for i in range(args.random_repeats)]
    print('', flush=True)
    print(f'측정 (트리 {args.n_estimators}) — 지정 시드 {len(named_seeds)}회 / '
          f'무작위 시드 {len(random_seeds)}회', flush=True)

    k1 = f'mean|lift|{HEADLINE_RULE}|k1'
    k5 = f'mean|lift|{HEADLINE_RULE}|k{HEADLINE_K}'
    rows = []
    for pos, row in enumerate(entries, start=1):
        seeds = named_seeds if row['kind'] == 'designated' else random_seeds
        got = measure(sorted(row['features']), ctx, args, seeds)
        record = {'label': row['label'], 'kind': row['kind'], 'group': row['group'],
                  'n_features': len(row['features']), 'repeats': len(seeds),
                  'measured_here': True,
                  'features_joined': '|'.join(sorted(row['features']))}
        record.update(got)
        rows.append(record)
        print(f"  [{pos:3d}/{len(entries)}] {row['label']:16s} n={record['n_features']:2d} "
              f"| dev k1 {record[f'dev|{k1}']:+.4f} | test k1 {record[f'test|{k1}']:+.4f} "
              f"| test k5 {record[f'test|{k5}']:+.4f}", flush=True)

    # 35번이 이미 잰 4종을 같은 표에 올립니다(재측정하지 않습니다).
    metric_names = ([f'{q}|{r}|k{k}' for q in ('success', 'lift')
                     for r in RULES for k in args.topk]
                    + [f'hard_lift|{HEADLINE_RULE}|k{k}' for k in args.topk])
    for label in REUSED:
        src = prior[label]
        record = {'label': label, 'kind': 'reused_from_35', 'group': 'reused',
                  'n_features': int(src['n_features']),
                  'repeats': args.repeats, 'measured_here': False,
                  'features_joined': src.get('features_joined', '')}
        for split in ('dev', 'test'):
            for metric in metric_names:
                for stat in ('mean', 'std'):
                    key = f'{split}|{stat}|{metric}'
                    raw = src.get(key, '')
                    record[key] = float(raw) if raw not in ('', None) else None
        rows.append(record)

    header = ['label', 'kind', 'group', 'n_features', 'repeats', 'measured_here']
    header += [f'{sp}|{st}|{m}' for sp in ('dev', 'test')
               for m in metric_names for st in ('mean', 'std')]
    header += ['features_joined']
    audit.save_csv(out / 'subset_results.csv', rows, tuple(header))

    # 분산 분해 — 이 실행의 헤드라인.
    dist_rows = []
    groups_seen = []
    for row in rows:
        if row['group'] not in groups_seen:
            groups_seen.append(row['group'])
    for group in groups_seen + ['random_all', 'all_measured']:
        if group == 'random_all':
            members = [r for r in rows if r['kind'].startswith('random_')]
        elif group == 'all_measured':
            members = [r for r in rows if r['measured_here']]
        else:
            members = [r for r in rows if r['group'] == group]
        if len(members) < 2:
            continue
        rep = members[0]['repeats']
        for split in ('dev', 'test'):
            for metric in DECISION_METRICS:
                got = decompose([m.get(f'{split}|mean|{metric}') for m in members],
                                [m.get(f'{split}|std|{metric}') for m in members], rep)
                if got is None:
                    continue
                dist_rows.append({'group': group, 'split': split, 'metric': metric,
                                  'repeats': rep, **got})
    audit.save_csv(out / 'distribution.csv', dist_rows,
                   ('group', 'split', 'metric', 'repeats', 'n', 'min', 'q1', 'median',
                    'q3', 'max', 'range', 'max_minus_median', 'observed_std',
                    'seed_std', 'mean_se', 'between_std', 'f_ratio', 'df1', 'df2',
                    'p_value', 'selection_inflation', 'verdict'))

    # dev 에서 좋았던 집합이 test 에서도 좋은가 — 전이 확인.
    transfer_rows = []
    measured = [r for r in rows if r['measured_here']]
    for scope, members in (('random_only',
                            [r for r in rows if r['kind'].startswith('random_')]),
                           ('all', rows), ('measured_here', measured)):
        for metric in DECISION_METRICS:
            pairs = [(r[f'dev|mean|{metric}'], r[f'test|mean|{metric}'], r['label'])
                     for r in members
                     if r.get(f'dev|mean|{metric}') is not None
                     and r.get(f'test|mean|{metric}') is not None]
            if len(pairs) < 3:
                continue
            dev_v = [p[0] for p in pairs]
            test_v = [p[1] for p in pairs]
            rho = spearman(dev_v, test_v)
            best_dev = max(pairs, key=lambda p: p[0])
            best_test = max(pairs, key=lambda p: p[1])
            transfer_rows.append({
                'scope': scope, 'metric': metric, 'n': len(pairs),
                'spearman': round(rho, 3) if rho is not None else None,
                'dev_best_label': best_dev[2], 'dev_best_dev': round(best_dev[0], 4),
                'dev_best_test': round(best_dev[1], 4),
                'test_best_label': best_test[2], 'test_best_dev': round(best_test[0], 4),
                'test_best_test': round(best_test[1], 4),
                'dev_winner_also_test_winner': best_dev[2] == best_test[2]})
    audit.save_csv(out / 'transfer.csv', transfer_rows,
                   ('scope', 'metric', 'n', 'spearman', 'dev_best_label',
                    'dev_best_dev', 'dev_best_test', 'test_best_label',
                    'test_best_dev', 'test_best_test', 'dev_winner_also_test_winner'))
    transfer = transfer_rows

    changed = []
    for source in sources:
        try:
            info = Path(source['path']).stat()
            if (info.st_size, info.st_mtime_ns) != (source['size_bytes'],
                                                    source['mtime_ns']):
                changed.append(source['path'])
        except OSError:
            changed.append(source['path'])
    status = 'SUBSET_DIST_COMPLETE' if not changed else 'CHECK_REQUIRED'

    headline = [r for r in dist_rows
                if r['group'] == 'random_all' and r['split'] == 'test']
    report = {
        'version': VERSION, 'status': status, 'output_dir': str(out),
        'preregistration': list(PREREGISTRATION),
        'picks_nothing': True,
        'question': 'feature 집합 선택이 성능에 측정 가능한 영향을 주는가',
        'held_constant': list(HELD_CONSTANT), 'varied': 'feature 목록',
        'decision_metrics': list(DECISION_METRICS),
        'hard_layer_is_reference_only': True,
        'test_reopened': True,
        'anova_calibration': {
            'design': '균형 일원 분산분석. 처리=집합, 반복=시드.',
            'checked_by': '정규 모의실험 3,000회 (집합 40개, 시드 2회, sigma 0.009)',
            'false_positive_at_0.05': 0.0513, 'false_positive_at_0.01': 0.0077,
            'power_at_between_sd_0.02': 1.0,
            'note': ('귀무가설에서 거짓양성률이 명목 5% 와 일치하고, 집합 간 '
                     '표준편차가 0.02 면 100% 검출합니다. 시드 2회로 충분합니다.'),
            'beta_function_checked_against': ['I_0.5(1,1)=0.5', 'I_x(2,1)=x^2',
                                              'F(2,2) 꼬리=1/(1+f)',
                                              'F(5,10) 임계 3.326 -> 0.05']},
        'matrix': str(args.matrix), 'union_dir': str(args.union_dir),
        'test_eval_dir': str(args.test_eval_dir), 'sources': sources,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'support_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (_t35_path, test_eval._union_path,
                                     union._combination_path,
                                     combination._selection_path,
                                     selection._combined_path, combined._graded_path,
                                     graded._ranking_path, ranking._baseline_path,
                                     baseline._support_path, roles._support_path,
                                     audit._support_path)},
        'python': sys.executable, 'xgboost': xgboost.__version__,
        'feature_pool': len(feats), 'anchor_axis': anchor,
        'sampling': {'subset_seed': args.subset_seed,
                     'uniform_sizes': list(args.uniform_sizes),
                     'anchored_sizes': list(args.anchored_sizes),
                     'per_size': args.random_per_size,
                     'duplicates_dropped': duplicates},
        'designated_added': {k: v for k, v in designated.items()},
        'reused_from_35': list(REUSED),
        'pipeline_consistency_with_35': consistency,
        'locked_config': {'features': 'raw', 'relevance': 'graded',
                          'relevance_bins': args.relevance_bins,
                          'objective': 'rank:ndcg', 'selection_rule': HEADLINE_RULE},
        'model_params': {'n_estimators': args.n_estimators,
                         'max_depth': args.max_depth,
                         'learning_rate': args.learning_rate,
                         'named_seeds': named_seeds, 'random_seeds': random_seeds},
        'topk': list(args.topk), 'shuffles': args.shuffles,
        'splits': {name: {'systems': int(len(set(parts[name]['frame']['system_id']))),
                          'eligible': len(parts[name]['groups']),
                          'hard_systems': parts[name]['controls']['hard_systems'],
                          'random_analytic': parts[name]['controls']['analytic'],
                          'random_analytic_hard': parts[name]['controls']['hard_analytic']}
                   for name in ('dev', 'test')},
        'results': rows, 'distribution': dist_rows, 'transfer': transfer,
        'headline_test_random_all': headline,
        'changed_sources': changed,
        'elapsed_seconds': time.perf_counter() - started,
        'notes': ['distribution.csv 의 p_value 가 이 실행의 판정입니다. 일원 '
                  '분산분석으로 "모든 집합이 같다"를 검정하며, p >= 0.05 면 집합 '
                  '선택의 영향이 시드 잡음에 묻혔다는 뜻입니다. between_std 는 크기 '
                  '추정값이고 0 근처에서는 흔들리므로 부호만으로 판정하지 않습니다.',
                  'selection_inflation 은 그 집합 수에서 최고값에 자동으로 붙는 허수 '
                  '크기(평균의 표준오차 x sqrt(2 ln n))입니다. max_minus_median 이 이 '
                  '값보다 작으면 1위는 고르기로 생긴 것입니다.',
                  'anchored 층은 단독 가능 7개를 고정하고 나머지만 바꾼 것이라, '
                  'adopted_33 과 union_all12 가 구분되지 않은 이유를 직접 답합니다.',
                  'transfer.csv 의 spearman 이 낮으면 dev 에서 고른 집합이 test 에서 '
                  '좋을 이유가 없다는 뜻이고, 33번에서 일어난 일이 바로 그것입니다.',
                  'uniform 층이 anchored 층보다 크게 낮으면 "feature 선별 자체는 '
                  '효과가 있었다"는 뜻입니다. 선별은 되고 미세조정은 안 된다는 구도입니다.']}
    helper.write_json(out / 'distribution_report.json', report)
    helper.write_json(out / 'RUN_COMPLETE.json', {
        'outputs_written': True, 'subset_dist_finished': True, 'status': status,
        'picks_nothing': True, 'subsets_measured': len(entries),
        'rows_total': len(rows),
        'headline': [{'metric': r['metric'], 'n': r['n'], 'range': r['range'],
                      'between_std': r['between_std'],
                      'selection_inflation': r['selection_inflation'],
                      'verdict': r['verdict']} for r in headline],
        'sources_unchanged': not changed})

    print('', flush=True)
    print('STATUS:', status, flush=True)
    print('', flush=True)
    print('=== 분산 분해 (test) — 이 실행의 답 ===', flush=True)
    print('  최대-중앙 <= 허수  이면 1위는 고르기로 생긴 것이고 실력이 아닙니다.',
          flush=True)
    print(f"{'층':18s}{'칸':18s}{'n':>3s}{'최소':>9s}{'중앙':>9s}{'최대':>9s}"
          f"{'최대-중앙':>10s}{'허수':>8s}{'집합간σ':>9s}{'F':>8s}{'p':>8s}  판정",
          flush=True)
    for row in dist_rows:
        if row['split'] != 'test':
            continue
        infl = row['selection_inflation']
        btw = row['between_std']
        fval, pval = row['f_ratio'], row['p_value']
        print(f"{row['group']:18s}{row['metric']:18s}{row['n']:3d}"
              f"{row['min']:+9.4f}{row['median']:+9.4f}{row['max']:+9.4f}"
              f"{row['max_minus_median']:10.4f}"
              f"{(infl if infl is not None else 0):8.4f}"
              f"{(btw if btw is not None else 0):9.4f}"
              f"{(fval if fval is not None else 0):8.2f}"
              f"{(pval if pval is not None else 1):8.4f}  {row['verdict']}", flush=True)

    print('', flush=True)
    print('=== dev 최고가 test 최고인가 (전이) ===', flush=True)
    print(f"{'범위':14s}{'칸':18s}{'n':>3s}{'상관':>7s}  "
          f"{'dev 최고':16s}{'그 집합 test':>13s}  {'test 최고':16s}", flush=True)
    for row in transfer_rows:
        rho = row['spearman']
        print(f"{row['scope']:14s}{row['metric']:18s}{row['n']:3d}"
              f"{(rho if rho is not None else 0):7.2f}  "
              f"{row['dev_best_label']:16s}{row['dev_best_test']:+13.4f}  "
              f"{row['test_best_label']:16s}", flush=True)

    print('', flush=True)
    print('=== 지정 3종 + 35번 4종 (7집합 완성, 규칙 A) ===', flush=True)
    print(f"{'집합':14s}{'n':>3s}{'출처':>10s}", end='', flush=True)
    for k in args.topk:
        print(f"{'dev k' + str(k):>9s}{'test k' + str(k):>9s}", end='', flush=True)
    print('', flush=True)
    for row in rows:
        if row['kind'].startswith('random_'):
            continue
        src = '이번' if row['measured_here'] else '35번'
        line = f"{row['label']:14s}{row['n_features']:3d}{src:>10s}"
        for k in args.topk:
            metric = f'mean|lift|{HEADLINE_RULE}|k{k}'
            dv, tv = row.get(f'dev|{metric}'), row.get(f'test|{metric}')
            line += f"{(dv if dv is not None else 0):+9.4f}"
            line += f"{(tv if tv is not None else 0):+9.4f}"
        print(line, flush=True)

    print('', flush=True)
    print('*** 이 표의 최고값은 선택 후보가 아닙니다. 사전 등록대로 아무것도 고르지 '
          '않습니다. ***', flush=True)
    return 0 if status == 'SUBSET_DIST_COMPLETE' else 2


def parse_sizes(text):
    """7,11,15,21 형태를 정수 튜플로."""
    out = []
    for piece in str(text).split(','):
        piece = piece.strip()
        if not piece:
            continue
        value = int(piece)
        if value < 1:
            raise argparse.ArgumentTypeError('크기는 1 이상이어야 합니다.')
        out.append(value)
    if not out:
        raise argparse.ArgumentTypeError('크기를 하나 이상 주세요.')
    return tuple(dict.fromkeys(out))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-test', action='store_true',
                        help='test 923 타깃을 다시 엽니다. 이 플래그 없이는 '
                             '실행되지 않습니다.')
    parser.add_argument('--matrix', type=Path, default=DEFAULT_MATRIX)
    parser.add_argument('--union-dir', type=Path, default=DEFAULT_UNION,
                        help='34번 결과 폴더(지정 3종을 읽습니다).')
    parser.add_argument('--test-eval-dir', type=Path, default=DEFAULT_TEST_EVAL,
                        help='35번 결과 폴더(이미 잰 4종을 읽습니다).')
    parser.add_argument('--relevance-bins', choices=tuple(combined.BIN_SETS),
                        default='asymmetric')
    parser.add_argument('--topk', type=combined.parse_topk, default=(1, 3, 5))
    parser.add_argument('--shuffles', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=5,
                        help='지정 3종의 시드 반복(35번과 같게 두세요).')
    parser.add_argument('--random-repeats', type=int, default=2,
                        help='무작위 부분집합의 시드 반복. 분포를 보는 데는 2회로 '
                             '충분하고, 시드 잡음은 분산 분해에서 빼냅니다.')
    parser.add_argument('--random-per-size', type=int, default=4,
                        help='크기마다 뽑을 무작위 집합 수.')
    parser.add_argument('--uniform-sizes', type=parse_sizes, default=(7, 11, 15, 21),
                        help='전체 pool 에서 균일 추출할 크기들.')
    parser.add_argument('--anchored-sizes', type=parse_sizes, default=(11, 15, 21),
                        help='단독 가능 7개를 고정하고 채울 크기들.')
    parser.add_argument('--subset-seed', type=int, default=20261106,
                        help='무작위 부분집합 추출 시드. 바꾸면 다른 표본이 됩니다.')
    parser.add_argument('--n-estimators', type=int, default=400)
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261102)
    parser.add_argument('--tag', type=str, default=None)
    args = parser.parse_args()
    for name in ('shuffles', 'repeats', 'random_repeats', 'random_per_size',
                 'n_estimators', 'max_depth', 'threads'):
        if getattr(args, name) < 1:
            parser.error(f'--{name.replace("_", "-")} 는 1 이상이어야 합니다.')
    if args.random_repeats < 2:
        parser.error('--random-repeats 는 2 이상이어야 합니다(분산 분해에 필요).')
    args.matrix = args.matrix.absolute()
    args.union_dir = args.union_dir.absolute()
    args.test_eval_dir = args.test_eval_dir.absolute()
    return run(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
