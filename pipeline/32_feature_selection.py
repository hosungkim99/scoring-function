#!/usr/bin/env python3
"""Process B 마감 2단계(⑦). feature 선별 — 몇 개가 실제로 필요한가.

확정된 설정으로 고정합니다(31번 2×2 결과, 개발일지 §10.12·§11.10).

    feature = 원본만(z-score 결합 안 함)   relevance = 등급 + 비대칭 경계
    목적함수 = rank:ndcg (그룹 = 시스템)    선택 규칙 = 생성기 분산(ruleA)

이 단계가 필요한 이유는 성능이 아니라 세 가지입니다.

  (가) 과정 B의 목적 자체가 "최종 scoring function에 쓸 feature를 찾는 것"입니다.
       쓰이는 부분집합을 특정하는 것이 산출물입니다.
  (나) 추출 비용이 큰 후보군이 있습니다. P2Rank(cat5)는 약 850 CPU-시간, gnina는
       약 5,100 코어-시간이 들었습니다. 기여가 없으면 다음 작업은 건너뛸 수 있습니다.
  (다) feature가 적으면 dev 과적합 위험이 낮아지고 설명이 단순해집니다.

두 가지를 한 작업에서 냅니다.

  1) 중요도 사다리 — 전체 학습에서 얻은 중요도 순으로 상위 N개만 남겨 재학습.
     N = 5, 10, 15, 20, 30, 42, 전체. 어디서 평탄해지는지 봅니다.
  2) 후보군 제외 — 8개 카테고리를 하나씩 빼고 재학습. 각 군의 실제 기여를 봅니다.
  3) 단일 feature 비교선 — feature 하나만 쓴 모델을 같은 평가로 재서, 앙상블 중요도
     순위와 비교합니다. 둘의 차이가 "단독으로는 약한데 조합에서 살아나는 feature"를
     가립니다. 트리 중요도는 "많이 쓰였다"만 알려주고 그 구분을 못 해줍니다.
     이것이 필요한 더 중요한 이유는 **공정한 비교선이 아직 없다**는 점입니다. 과거
     "단일 feature 최고 +0.170"은 (a) 모델 공헌의 부호를 뒤집는 망가진 선택 규칙,
     (b) 포화된 k=5, (c) 108개 변형 중 in-sample 최대값으로 측정한 값이라 현재
     숫자와 비교할 수 없습니다. 과정 B의 핵심 질문("조합이 단일 feature를 이기는가")에
     같은 조건의 답을 여기서 처음 얻습니다.
     단독 순위는 **k=1 기준**으로 매깁니다. k=5는 포화돼 분해능이 없습니다(§10.6).

**선택 규칙을 미리 등록합니다.** 결과를 보고 고르면 사후 선택이 되므로, 코드에
박아두고 실행 뒤에 바꾸지 않습니다.

    후보  = 전체 + 사다리 (drop_* 는 후보군 기여를 재는 진단용이라 후보가 아닙니다)
    조건1 = model / ruleA / k=5 lift 가 최고에서 tol_k5 이내      (공식 수용 기준)
    조건2 = 어려운 층 k=5 lift 가 최고에서 tol_hard 이내          (우선순위 반영)
    선택  = 두 조건을 만족하는 것 중 **feature 수가 가장 적은** 설정

조건2가 있는 이유: §11.10에서 설정을 고를 때 "어려운 타깃 우선"을 기준으로 삼기로
이미 결정했는데, 전체 k=5 만 보는 규칙은 그 결정을 반영하지 못합니다.

**허용폭은 고정값이 아니라 측정값입니다.** 같은 설정을 시드만 바꿔 --repeats 회
돌리고, 기준 설정(전체)의 시드 간 표준편차를 그대로 허용폭으로 씁니다.

    tol_k5   = max(기준 설정의 lift 표준편차,      TOL_FLOOR_K5)
    tol_hard = max(기준 설정의 어려운층 표준편차, TOL_FLOOR_HARD)

이전에는 제가 눈대중으로 정한 0.005를 썼습니다. 실행이 2분대라는 것이 확인되어
반복 측정이 공짜가 됐으므로, 잡음 크기를 추정하지 않고 재서 씁니다. 규칙은 미리
고정되어 있고 숫자만 데이터가 정합니다.

대조군(무작위)은 모델 점수를 쓰지 않으므로 설정과 무관합니다. 한 번만 계산해
모든 설정에 공유합니다(평가 비용 약 1/7).

주의: 이 실행도 dev 예산을 씁니다. 지금까지 dev에서 설정 9종을 돌렸고 여기서
15종이 더해집니다. test 923은 마지막에 1회만 쓰며, dev 숫자는 선택 편향을 포함한
낙관값으로 읽어야 합니다(개발일지 §10.10 주의 블록).

출력: out/feature_selection/32_run_selection[_TAG]/
"""
import sys
sys.dont_write_bytecode = True
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

# 31번을 통해 30·29·28번까지 불러옵니다(선택 규칙·평가·QA 코드를 복사하지 않습니다).
_combined_path = Path(__file__).with_name('31_combined_model.py')
_cspec = importlib.util.spec_from_file_location('combined_support', _combined_path)
combined = importlib.util.module_from_spec(_cspec)
_cspec.loader.exec_module(combined)
graded = combined.graded
ranking = combined.ranking
baseline = combined.baseline
roles = combined.roles
audit = combined.audit
helper = combined.helper

VERSION = 'selection_0.1.0'
OUT_KIND = 'feature_selection'
OUT_BASE = '32_run_selection'
KEYS = baseline.KEYS
BASE = baseline.BASE
DEFAULT_MATRIX = baseline.DEFAULT_MATRIX
RULES = combined.RULES
# 사다리 눈금. 42는 "타깃 안에서 구별 가능한 feature 수"(57 - 시스템 내 상수 15).
LADDER = (5, 10, 15, 20, 30, 42)
# 기준 칸. 허용폭은 실행 중에 시드 간 표준편차로 측정하며, 아래 값은 반복이 1회일
# 때나 표준편차가 비정상적으로 작게 나올 때만 쓰는 하한입니다.
HEADLINE_RULE = 'ruleA'
HEADLINE_K = 5
TOL_FLOOR_K5 = 0.002
TOL_FLOOR_HARD = 0.010
# 어려운 층 = 무작위 Top-5 성공확률 0.5 미만(§10 전반에서 쓴 정의).
HARD_CUT = 0.5


def prepare_groups(dev, codes):
    """시스템별 배열을 미리 잘라 둡니다. 설정마다 groupby를 반복하지 않기 위함입니다."""
    import numpy as np
    _, starts, sizes = ranking.group_layout(codes)
    nn_all = dev['near_native'].to_numpy()
    gen_all = dev['model'].to_numpy()
    tie_all = dev['_tie'].to_numpy()
    sid_all = dev['system_id'].to_numpy()
    groups = []
    for start, size in zip(starts, sizes):
        sl = slice(int(start), int(start) + int(size))
        nn = nn_all[sl]
        m = int(nn.sum())
        if m == 0:
            continue
        groups.append({'system_id': sid_all[start], 'start': int(start), 'size': int(size),
                       'nn': nn, 'gen': gen_all[sl], 'tie': tie_all[sl], 'm': m})
    return groups


def group_randoms(groups, topk):
    """시스템별 해석식 무작위 성공확률. 난이도 축과 기준선으로 씁니다."""
    for g in groups:
        g['rand'] = {k: baseline.random_topk(g['size'], g['m'], k) for k in topk}
    return groups


def hits_from_order(group, order, topk, kmax):
    """한 정렬에서 규칙별·k별 성공 여부."""
    out = {}
    for rule in RULES:
        idx = combined.rule_order(group['gen'][order], rule, kmax)
        sel = order[idx]
        for k in topk:
            out[(rule, k)] = bool(group['nn'][sel[:k]].any())
    return out


def evaluate_scores(groups, score, topk):
    """모델 점수로 규칙 x k 성공률과 어려운 층 성공률을 계산합니다."""
    import numpy as np
    kmax = max(topk)
    total = {(r, k): 0.0 for r in RULES for k in topk}
    hard = {(r, k): 0.0 for r in RULES for k in topk}
    n_hard = 0
    for g in groups:
        chunk = score[g['start']:g['start'] + g['size']]
        chunk = np.where(np.isnan(chunk), -np.inf, chunk)
        order = np.lexsort((g['tie'], -chunk))
        got = hits_from_order(g, order, topk, kmax)
        is_hard = g['rand'][HEADLINE_K] < HARD_CUT
        if is_hard:
            n_hard += 1
        for key, val in got.items():
            total[key] += float(val)
            if is_hard:
                hard[key] += float(val)
    n = len(groups)
    return ({key: val / n for key, val in total.items()},
            {key: (val / n_hard if n_hard else None) for key, val in hard.items()},
            n_hard)


def evaluate_controls(groups, topk, shuffles, seed):
    """모델 점수를 쓰지 않는 대조군. 설정과 무관하므로 한 번만 계산합니다."""
    import numpy as np
    kmax = max(topk)
    rng = np.random.default_rng(seed)
    pseudo = {(r, k): 0.0 for r in RULES for k in topk}
    shuf = {(r, k): 0.0 for r in RULES for k in topk}
    analytic = {k: 0.0 for k in topk}
    hard_analytic = {k: 0.0 for k in topk}
    hard_shuf = {(r, k): 0.0 for r in RULES for k in topk}
    n_hard = 0
    for g in groups:
        is_hard = g['rand'][HEADLINE_K] < HARD_CUT
        if is_hard:
            n_hard += 1
        for k in topk:
            analytic[k] += g['rand'][k]
            if is_hard:
                hard_analytic[k] += g['rand'][k]
        got = hits_from_order(g, np.argsort(g['tie'], kind='mergesort'), topk, kmax)
        for key, val in got.items():
            pseudo[key] += float(val)
        acc = {key: 0.0 for key in shuf}
        for _ in range(shuffles):
            got = hits_from_order(g, rng.permutation(g['size']), topk, kmax)
            for key, val in got.items():
                acc[key] += float(val)
        for key, val in acc.items():
            shuf[key] += val / shuffles
            if is_hard:
                hard_shuf[key] += val / shuffles
    n = len(groups)
    return {
        'analytic': {k: v / n for k, v in analytic.items()},
        'hard_analytic': {k: (v / n_hard if n_hard else None)
                          for k, v in hard_analytic.items()},
        'pseudorandom': {key: v / n for key, v in pseudo.items()},
        'shuffle': {key: v / n for key, v in shuf.items()},
        'hard_shuffle': {key: (v / n_hard if n_hard else None)
                         for key, v in hard_shuf.items()},
        'systems': n, 'hard_systems': n_hard}


def train_oof(X, rel, codes, fold_of, known, folds, feat_idx, args):
    """주어진 feature 부분집합으로 out-of-fold 점수와 평균 중요도를 만듭니다."""
    import numpy as np
    sub = X[:, feat_idx] if feat_idx is not None else X
    score = np.full(X.shape[0], np.nan, dtype=np.float64)
    imp = np.zeros(sub.shape[1], dtype=np.float64)
    groups_used = []
    for k in folds:
        tr = (fold_of != k) & known
        va = fold_of == k
        _, dead = graded.pairable_systems(codes, rel, tr)
        if dead.size:
            tr = tr & ~np.isin(codes, dead)
        model = graded.build_ranker(args)
        model.fit(sub[tr], rel[tr], qid=codes[tr])
        score[va] = model.predict(sub[va])
        groups_used.append(int(np.unique(codes[tr]).size))
        got = getattr(model, 'feature_importances_', None)
        if got is not None and len(got) == sub.shape[1]:
            imp += np.asarray(got, dtype=np.float64)
    return score, imp / len(folds), groups_used


def pick_config(rows, tol_k5, tol_hard):
    """사전 등록된 규칙으로 설정을 고릅니다. 결과를 본 뒤 바꾸지 않습니다.

    drop_* 는 후보군의 기여를 재기 위한 진단용 설정이므로 최종 집합 후보에서 뺍니다.
    후보에 넣으면 기준선을 밀어 올려 허용폭 계산까지 왜곡합니다.
    """
    key = f'mean|lift|{HEADLINE_RULE}|k{HEADLINE_K}'
    hkey = f'mean|hard_lift|{HEADLINE_RULE}|k{HEADLINE_K}'
    pool = [r for r in rows if r['kind'] != 'drop'
            and r.get(key) is not None and r.get(hkey) is not None]
    if not pool:
        return None, None, None
    best = max(r[key] for r in pool)
    best_hard = max(r[hkey] for r in pool)
    ok = [r for r in pool
          if r[key] >= best - tol_k5 and r[hkey] >= best_hard - tol_hard]
    if not ok:
        return None, best, best_hard
    return min(ok, key=lambda r: (r['n_features'], -r[key])), best, best_hard


def run(args):
    """중요도 사다리와 후보군 제외를 한 작업에서 평가합니다."""
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
        os.environ[key] = str(args.threads)
    print('OUTPUT_DIR:', out, flush=True)

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

    catalog = pd.read_csv(catalog_path).set_index('feature')
    meta_cols = ['system_id', 'model', 'seed', 'sample', 'pose_id', 'sc_rmsd', 'near_native',
                 'label_known', 'split', 'cv_fold']
    frame = pd.read_parquet(matrix_path, columns=meta_cols + catalog.index.tolist())
    frame['near_native'] = frame['near_native'].fillna(False).astype(bool)
    frame['label_known'] = frame['label_known'].fillna(False).astype(bool)
    if 'split' not in frame.columns or frame['split'].isna().all():
        raise ValueError('split 열이 없습니다. 26번 통합에 18번 분할이 붙었는지 확인하세요.')
    test_systems = int(frame.loc[frame['split'] == 'test', 'system_id'].nunique())
    dev = frame[frame['split'] == 'dev'].copy()
    del frame
    dev = dev.sort_values(KEYS, kind='mergesort').reset_index(drop=True)
    dev['cv_fold'] = dev['cv_fold'].astype(int)
    dev['_tie'] = dev['pose_id'].map(baseline.tie_key)
    folds = sorted(int(f) for f in dev['cv_fold'].unique())
    print(f'DEV poses {len(dev)} | systems {dev["system_id"].nunique()} | folds {folds} '
          f'| TEST systems {test_systems} (봉인)', flush=True)

    bins = combined.BIN_SETS[args.relevance_bins]
    sc = dev['sc_rmsd'].to_numpy(dtype=np.float64, na_value=np.nan)
    rel = combined.relevance_from_sc(sc, bins, 'graded')
    known = dev['label_known'].to_numpy(dtype=bool)
    nn = dev['near_native'].to_numpy(dtype=bool)
    nn_level = combined.near_native_level(bins, 'graded')
    mismatch = int(np.sum((rel[known] >= nn_level) != nn[known]))
    if mismatch:
        raise ValueError(f'등급 {nn_level} 이상과 near_native가 {mismatch}개에서 불일치합니다.')
    print(f'RELEVANCE graded/{args.relevance_bins} {bins} | 등급>={nn_level} == near_native '
          f'확인 ({int(known.sum())} pose)', flush=True)

    feats, dropped = baseline.select_features(dev, catalog, set())
    audit.save_csv(out / 'dropped_features.csv',
                   dropped or [{'feature': None, 'category': None, 'reason': None}],
                   ('feature', 'category', 'reason'))
    if not feats:
        raise ValueError('사용할 feature가 없습니다.')
    codes = pd.factorize(dev['system_id'], sort=False)[0]
    assert np.all(np.diff(codes) >= 0), '시스템 그룹이 연속이 아닙니다.'
    fold_of = dev['cv_fold'].to_numpy()
    if int(pd.Series(fold_of).groupby(codes).nunique().max()) != 1:
        raise ValueError('한 시스템이 여러 폴드에 걸쳐 있습니다.')
    X = dev[feats].to_numpy(dtype=np.float32, na_value=np.nan)
    print(f'FEATURES {len(feats)} | dropped {len(dropped)} | MATRIX {X.shape} '
          f'({X.nbytes / 1e6:.0f} MB)', flush=True)

    groups = group_randoms(prepare_groups(dev, codes), args.topk)
    print(f'ELIGIBLE dev systems {len(groups)}', flush=True)
    print('대조군 계산(설정과 무관, 1회)...', flush=True)
    controls = evaluate_controls(groups, args.topk, args.shuffles, args.seed)
    print(f"  무작위(해석) k{args.topk}: "
          f"{[round(controls['analytic'][k], 4) for k in args.topk]} "
          f"| 어려운 층 {controls['hard_systems']}개", flush=True)

    # 1) 전체 feature 기준선 + 중요도. 사다리의 순서는 여기서 나옵니다.
    print('\n[1/3] 전체 feature 학습(중요도 산출)...', flush=True)
    score, imp, groups_used = train_oof(X, rel, codes, fold_of, known, folds, None, args)
    order = np.argsort(-imp)
    ranked = [feats[i] for i in order]
    imp_rows = [{'rank': r + 1, 'feature': feats[i],
                 'category': str(catalog.loc[feats[i], 'category']),
                 'importance': round(float(imp[i]), 6)} for r, i in enumerate(order)]
    audit.save_csv(out / 'feature_importance.csv', imp_rows,
                   ('rank', 'feature', 'category', 'importance'))
    print('  학습 시스템:', groups_used, flush=True)
    print('  중요도 상위 12:', ', '.join(ranked[:12]), flush=True)
    zero = [r['feature'] for r in imp_rows if r['importance'] <= 0.0]
    if zero:
        print(f'  중요도 0인 feature {len(zero)}개: {zero}', flush=True)

    index_of = {name: i for i, name in enumerate(feats)}
    configs = [('all', None, list(feats))]
    for n in LADDER:
        if n < len(feats):
            configs.append((f'top{n}', None, ranked[:n]))
    cats = sorted({str(catalog.loc[f, 'category']) for f in feats})
    for cat in cats:
        keep = [f for f in feats if str(catalog.loc[f, 'category']) != cat]
        if keep:
            configs.append((f'drop_{cat}', cat, keep))
    print(f'\n[2/3] 설정 {len(configs)}종 평가 '
          f'(사다리 {sum(1 for c in configs if c[0].startswith("top"))} + '
          f'제외 {len(cats)} + 전체 1)', flush=True)

    seeds = [args.seed + i for i in range(args.repeats)]
    print(f'  시드 {seeds} (반복 {len(seeds)}회, 허용폭을 이 반복의 표준편차에서 가져옵니다)',
          flush=True)
    rows = []
    for name, cat, keep in configs:
        idx = (None if name == 'all'
               else np.array([index_of[f] for f in keep], dtype=np.int64))
        per_seed = []
        for sd in seeds:
            if name == 'all' and sd == args.seed:
                sub_score = score          # 중요도 산출 때 학습한 것을 재사용
            else:
                variant = copy.copy(args)
                variant.seed = sd
                sub_score, _, _ = train_oof(X, rel, codes, fold_of, known, folds,
                                            idx, variant)
            rates, hard_rates, _ = evaluate_scores(groups, sub_score, args.topk)
            cell = {}
            for rule in RULES:
                for k in args.topk:
                    cell[f'success|{rule}|k{k}'] = rates[(rule, k)]
                    cell[f'lift|{rule}|k{k}'] = rates[(rule, k)] - controls['analytic'][k]
            for k in args.topk:
                hv = hard_rates[(HEADLINE_RULE, k)]
                hr = controls['hard_analytic'][k]
                cell[f'hard_lift|{HEADLINE_RULE}|k{k}'] = (
                    hv - hr if hv is not None and hr is not None else None)
            per_seed.append(cell)
        row = {'config': name, 'kind': ('ladder' if name.startswith('top')
                                        else 'drop' if cat else 'reference'),
               'dropped_category': cat or '', 'n_features': len(keep),
               'repeats': len(seeds),
               'features_sha1': hashlib.sha1(','.join(sorted(keep)).encode()).hexdigest()[:12]}
        for metric in per_seed[0]:
            vals = [c[metric] for c in per_seed if c[metric] is not None]
            row[f'mean|{metric}'] = round(float(np.mean(vals)), 4) if vals else None
            row[f'std|{metric}'] = (round(float(np.std(vals, ddof=1)), 4)
                                    if len(vals) > 1 else None)
        rows.append(row)
        mk = f'mean|lift|{HEADLINE_RULE}|k{HEADLINE_K}'
        sk = f'std|lift|{HEADLINE_RULE}|k{HEADLINE_K}'
        spread = f"+-{row[sk]:.4f}" if row[sk] is not None else "+-   -  "
        print(f"  {name:26s} n={len(keep):3d} "
              f"| k5 lift {row[mk]:+.4f} {spread} "
              f"| k1 lift {row[f'mean|lift|{HEADLINE_RULE}|k1']:+.4f} "
              f"| 어려운층 k5 "
              f"{row[f'mean|hard_lift|{HEADLINE_RULE}|k{HEADLINE_K}']:+.4f}", flush=True)

    metrics = [f'{q}|{r}|k{k}' for q in ('success', 'lift')
               for r in RULES for k in args.topk]
    metrics += [f'hard_lift|{HEADLINE_RULE}|k{k}' for k in args.topk]
    header = ['config', 'kind', 'dropped_category', 'n_features', 'repeats', 'features_sha1']
    header += [f'{stat}|{m}' for m in metrics for stat in ('mean', 'std')]
    audit.save_csv(out / 'selection_results.csv', rows, tuple(header))

    # 3) 단일 feature 비교선. 학습 비용은 1열이라 거의 없고, 평가가 비용의 대부분입니다.
    print('', flush=True)
    print(f'[3/3] 단일 feature {len(feats)}종 비교선...', flush=True)
    imp_rank = {feats[i]: r + 1 for r, i in enumerate(order)}
    # 단독 비교는 k=1(전체)과 어려운층 k=5 로 봅니다. 전체 k=5 는 무작위가 0.91대라
    # feature 57개를 줄세울 분해능이 없습니다(§10.6).
    hkey = f'hard_lift|{HEADLINE_RULE}|k{HEADLINE_K}'
    k1key = f'lift|{HEADLINE_RULE}|k1'
    single_rows = []
    for i, name in enumerate(feats):
        s_score, _, _ = train_oof(X, rel, codes, fold_of, known, folds,
                                  np.array([i], dtype=np.int64), args)
        rates, hard_rates, _ = evaluate_scores(groups, s_score, args.topk)
        row = {'feature': name, 'category': str(catalog.loc[name, 'category']),
               'importance_rank': imp_rank[name], 'importance': round(float(imp[i]), 6)}
        for rule in RULES:
            for k in args.topk:
                row[f'lift|{rule}|k{k}'] = round(rates[(rule, k)] - controls['analytic'][k], 4)
        for k in args.topk:
            hv = hard_rates[(HEADLINE_RULE, k)]
            hr = controls['hard_analytic'][k]
            row[f'hard_lift|{HEADLINE_RULE}|k{k}'] = (round(hv - hr, 4)
                                                      if hv is not None and hr is not None
                                                      else None)
        single_rows.append(row)

    # 단독 순위는 k=1 로 매깁니다. k=5 는 무작위가 0.91대라 분해능이 없습니다.
    for rank, row in enumerate(sorted(single_rows, key=lambda r: -r[k1key]), 1):
        row['single_rank_k1'] = rank
    for row in single_rows:
        # 양수 = 단독 순위가 중요도 순위보다 뒤 = 조합에 의존하는 feature
        row['synergy_rank_delta'] = row['single_rank_k1'] - row['importance_rank']
    single_header = (['feature', 'category', 'importance_rank', 'importance',
                      'single_rank_k1', 'synergy_rank_delta']
                     + [f'lift|{r}|k{k}' for r in RULES for k in args.topk]
                     + [f'hard_lift|{HEADLINE_RULE}|k{k}' for k in args.topk])
    audit.save_csv(out / 'single_feature.csv',
                   sorted(single_rows, key=lambda r: -r[k1key]), tuple(single_header))
    audit.save_csv(out / 'synergy.csv',
                   sorted(single_rows, key=lambda r: -r['synergy_rank_delta']),
                   ('feature', 'category', 'importance_rank', 'single_rank_k1',
                    'synergy_rank_delta', 'importance', k1key, hkey))

    # 앙상블 쪽은 시드 반복의 평균을 씁니다. 단일 feature 는 비교선이라 1회만 돌립니다.
    ref_row = next(r for r in rows if r['config'] == 'all')
    ens_k1 = ref_row[f'mean|lift|{HEADLINE_RULE}|k1']
    ens_k5 = ref_row[f'mean|hard_lift|{HEADLINE_RULE}|k{HEADLINE_K}']
    best_single = max(single_rows, key=lambda r: r[k1key])
    best_single_k5 = max(single_rows, key=lambda r: r[hkey])
    headline = {
        'best_single_k1': {'feature': best_single['feature'], 'lift': best_single[k1key]},
        'best_single_k5_hard': {'feature': best_single_k5['feature'],
                                'lift': best_single_k5[hkey]},
        'ensemble_k1': ens_k1, 'ensemble_k5_hard': ens_k5,
        'combination_gain_k1': round(ens_k1 - best_single[k1key], 4),
        'combination_gain_k5_hard': round(ens_k5 - best_single_k5[hkey], 4)}
    print(f"  단일 최고(k=1)      : {best_single['feature']} {best_single[k1key]:+.4f}",
          flush=True)
    print(f"  조합 모델(k=1)      : {ens_k1:+.4f} "
          f"-> 조합 이득 {headline['combination_gain_k1']:+.4f}", flush=True)
    print(f"  단일 최고(어려운층 k5): {best_single_k5['feature']} "
          f"{best_single_k5[hkey]:+.4f}", flush=True)
    print(f"  조합 모델(어려운층 k5): {ens_k5:+.4f} "
          f"-> 조합 이득 {headline['combination_gain_k5_hard']:+.4f}", flush=True)

    # 허용폭을 기준 설정(전체)의 시드 간 표준편차에서 가져옵니다. 규칙은 미리 고정되어
    # 있고 숫자만 데이터가 정합니다.
    key = f'mean|lift|{HEADLINE_RULE}|k{HEADLINE_K}'
    hard_key = f'mean|hard_lift|{HEADLINE_RULE}|k{HEADLINE_K}'
    std_k5 = ref_row[f'std|lift|{HEADLINE_RULE}|k{HEADLINE_K}']
    std_hard = ref_row[f'std|hard_lift|{HEADLINE_RULE}|k{HEADLINE_K}']
    tol_k5 = max(std_k5 or 0.0, TOL_FLOOR_K5)
    tol_hard = max(std_hard or 0.0, TOL_FLOOR_HARD)
    chosen, best, best_hard = pick_config(rows, tol_k5, tol_hard)
    print('', flush=True)
    print(f'측정된 잡음(전체 설정 시드 {args.repeats}회): '
          f'k5 표준편차 {std_k5} / 어려운층 {std_hard}', flush=True)
    print(f'허용폭: k5 {tol_k5:.4f} (하한 {TOL_FLOOR_K5}) / '
          f'어려운층 {tol_hard:.4f} (하한 {TOL_FLOOR_HARD})', flush=True)
    chosen_features = None
    if chosen is not None:
        chosen_features = dict(((c[0], c[2]) for c in configs))[chosen['config']]
        helper.write_json(out / 'chosen_features.json', {
            'rule': {'candidates': '전체 + 사다리 (drop_* 제외)',
                     'condition_1': f'model/{HEADLINE_RULE}/k{HEADLINE_K} lift '
                                    f'>= 최고 - tol_k5',
                     'condition_2': f'어려운층 k{HEADLINE_K} lift >= 최고 - tol_hard',
                     'tie_break': 'feature 수 최소, 같으면 lift 높은 쪽',
                     'tol_k5': round(tol_k5, 4), 'tol_hard': round(tol_hard, 4),
                     'tol_source': '기준 설정(전체)의 시드 간 표준편차, 하한 적용',
                     'floors': {'k5': TOL_FLOOR_K5, 'hard': TOL_FLOOR_HARD},
                     'repeats': args.repeats, 'pre_registered': True},
            'config': chosen['config'], 'n_features': chosen['n_features'],
            'best_lift': best, 'best_hard_lift': best_hard,
            'chosen_lift': chosen[key], 'chosen_hard_lift': chosen[hard_key],
            'features': chosen_features,
            'by_category': {cat: [f for f in chosen_features
                                  if str(catalog.loc[f, 'category']) == cat]
                            for cat in cats}})

    changed = []
    for source in sources:
        try:
            info = Path(source['path']).stat()
            if (info.st_size, info.st_mtime_ns) != (source['size_bytes'], source['mtime_ns']):
                changed.append(source['path'])
        except OSError:
            changed.append(source['path'])

    status = 'SELECTION_COMPLETE' if not changed else 'CHECK_REQUIRED'
    report = {
        'version': VERSION, 'status': status, 'output_dir': str(out),
        'matrix': str(args.matrix), 'sources': sources,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'support_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (_combined_path, combined._graded_path,
                                     graded._ranking_path, ranking._baseline_path,
                                     baseline._support_path, roles._support_path,
                                     audit._support_path)},
        'python': sys.executable, 'xgboost': xgboost.__version__,
        'locked_config': {'features': 'raw', 'relevance': 'graded',
                          'relevance_bins': args.relevance_bins, 'objective': 'rank:ndcg',
                          'selection_rule': HEADLINE_RULE},
        'relevance_bins_values': [list(b) for b in bins], 'near_native_level': nn_level,
        'model_params': {'n_estimators': args.n_estimators, 'max_depth': args.max_depth,
                         'learning_rate': args.learning_rate, 'seed': args.seed},
        'topk': list(args.topk), 'shuffles': args.shuffles, 'ladder': list(LADDER),
        'repeats': args.repeats, 'seeds': seeds,
        'measured_noise': {'reference_config': 'all',
                           'std_lift_k5': std_k5, 'std_hard_lift_k5': std_hard},
        'tolerances': {'k5': round(tol_k5, 4), 'hard': round(tol_hard, 4),
                       'floors': {'k5': TOL_FLOOR_K5, 'hard': TOL_FLOOR_HARD},
                       'source': '기준 설정의 시드 간 표준편차(고정값 아님)'},
        'features_available': len(feats), 'features_dropped': len(dropped),
        'zero_importance_features': zero,
        'dev_poses': int(len(dev)), 'dev_systems': int(np.unique(codes).size),
        'eligible_systems': len(groups), 'hard_systems': controls['hard_systems'],
        'test_systems_untouched': test_systems,
        'controls': {'analytic': controls['analytic'],
                     'pseudorandom': {f'{r}|k{k}': v for (r, k), v
                                      in controls['pseudorandom'].items()},
                     'shuffle': {f'{r}|k{k}': v for (r, k), v in controls['shuffle'].items()}},
        'results': rows, 'best_headline_lift': best, 'best_hard_lift': best_hard,
        'single_feature': sorted(single_rows, key=lambda r: -r[k1key]),
        'headline_comparison': headline,
        'chosen': chosen, 'chosen_features': chosen_features,
        'changed_sources': changed, 'elapsed_seconds': time.perf_counter() - started,
        'policy': ['test 분할은 학습·평가에 사용하지 않았습니다.',
                   '31번 2×2에서 확정한 설정(원본 feature + 비대칭 등급 경계)으로 '
                   '고정했습니다.',
                   '선택 규칙을 코드에 사전 등록하고 결과를 본 뒤 바꾸지 않았습니다.',
                   '허용폭은 제가 정한 고정값이 아니라 기준 설정의 시드 간 표준편차를 '
                   '측정해 쓴 값입니다. 후보 풀에서 drop_* 진단 설정을 제외했습니다.',
                   '어려운 층 조건을 넣은 이유는 설정 선택 기준을 "어려운 타깃 우선"으로 '
                   '이미 정해두었기 때문입니다(§11.10).',
                   '무작위 대조군은 모델 점수를 쓰지 않아 설정과 무관하므로 1회만 '
                   '계산해 공유했습니다.',
                   '중요도 순서는 전체 feature 학습에서 폴드별로 얻어 평균한 것이며, '
                   '각 폴드 모델은 자기 학습 분할만 봤습니다.',
                   '동점은 pose_id SHA1 오름차순으로 결정적으로 깼습니다.'],
        'notes': ['사다리에서 lift가 평탄해지는 지점이 "실제로 필요한 feature 수"입니다.',
                  'drop_* 행의 lift 하락폭이 그 후보군의 기여입니다. cat5(P2Rank)와 '
                  'cat2_gnina는 추출 비용이 커서(각 850·5,100 코어-시간) 이 숫자가 '
                  '다음 작업의 비용 판단 근거가 됩니다.',
                  'headline_comparison 이 과정 B의 핵심 질문에 대한 답입니다. '
                  'combination_gain 이 양수여야 조합에 가치가 있습니다. 과거 '
                  '"단일 feature 최고 +0.170"과는 비교하지 마십시오 — 그 값은 망가진 '
                  '선택 규칙·포화된 k=5·in-sample 최대값으로 측정된 것입니다.',
                  'synergy_rank_delta 가 양수이고 큰 feature가 "단독으로는 약한데 '
                  '조합에서 살아나는" 것입니다. 음수면 단독으로도 강한 feature입니다.',
                  '이 실행도 dev 예산을 씁니다(설정 9종 -> 24종). dev 숫자는 선택 '
                  '편향을 포함한 낙관값이며 test 923에서 낮아지는 것이 정상입니다.']}
    helper.write_json(out / 'selection_report.json', report)
    helper.write_json(out / 'RUN_COMPLETE.json', {
        'outputs_written': True, 'selection_finished': True, 'status': status,
        'chosen_config': chosen['config'] if chosen else None,
        'chosen_n_features': chosen['n_features'] if chosen else None,
        'best_headline_lift': best, 'headline_comparison': headline,
        'sources_unchanged': not changed})

    print('\nSTATUS:', status, flush=True)
    print(f'기준 칸: model/{HEADLINE_RULE}/k{HEADLINE_K} | 후보 최고 {best:+.4f} '
          f'| 어려운층 최고 {best_hard:+.4f}', flush=True)
    if chosen is not None:
        print(f"사전 등록 규칙에 따른 선택: **{chosen['config']}** "
              f"(feature {chosen['n_features']}개, k5 lift {chosen[key]:+.4f}, "
              f"어려운층 {chosen[hard_key]:+.4f})", flush=True)
        for cat in cats:
            picked = [f for f in chosen_features if str(catalog.loc[f, 'category']) == cat]
            if picked:
                print(f'  {cat:20s} {len(picked):2d}개: {picked}', flush=True)
        # 타깃 안에서 상수인 feature는 pose를 구별하지 못하고 조건 변수로만 쓰입니다.
        within_const = [f for f in chosen_features
                        if dev.groupby('system_id')[f].nunique().max() <= 1]
        print(f'  이 중 타깃 내 상수 {len(within_const)}개'
              f'(pose 구별 불가, 조건 변수로만 쓰임): {within_const}', flush=True)
    else:
        print('두 조건을 동시에 만족하는 설정이 없습니다. 허용폭 하한을 확인하세요.',
              flush=True)

    print('\n=== 후보군 제외 영향 (기준 칸) ===', flush=True)
    print(f"{'제외한 군':22s}{'n':>4s}{'k5 lift':>10s}{'전체대비':>10s}"
          f"{'어려운층':>10s}{'어려운층대비':>13s}{'k1대비':>10s}", flush=True)
    k1m = f'mean|lift|{HEADLINE_RULE}|k1'
    for row in rows:
        if row['kind'] != 'drop':
            continue
        print(f"{row['dropped_category']:22s}{row['n_features']:4d}"
              f"{row[key]:+10.4f}{row[key] - ref_row[key]:+10.4f}"
              f"{row[hard_key]:+10.4f}{row[hard_key] - ref_row[hard_key]:+13.4f}"
              f"{row[k1m] - ref_row[k1m]:+10.4f}", flush=True)

    print('', flush=True)
    print('=== 조합 의존도 상위 (단독 약함 + 앙상블에서 많이 쓰임) ===', flush=True)
    print(f"{'feature':30s}{'중요도순위':>10s}{'단독순위':>9s}{'차이':>7s}"
          f"{'단독 k1':>10s}", flush=True)
    for row in sorted(single_rows, key=lambda r: -r['synergy_rank_delta'])[:10]:
        print(f"{row['feature'][:29]:30s}{row['importance_rank']:10d}"
              f"{row['single_rank_k1']:9d}{row['synergy_rank_delta']:+7d}"
              f"{row[k1key]:+10.4f}", flush=True)

    print('', flush=True)
    print('=== 단독으로도 강한 feature 상위 (k=1 기준) ===', flush=True)
    print(f"{'feature':30s}{'단독 k1':>10s}{'단독 k5':>10s}{'중요도순위':>10s}", flush=True)
    for row in sorted(single_rows, key=lambda r: -r[k1key])[:10]:
        print(f"{row['feature'][:29]:30s}{row[k1key]:+10.4f}{row[hkey]:+10.4f}"
              f"{row['importance_rank']:10d}", flush=True)
    return 0 if status == 'SELECTION_COMPLETE' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, default=DEFAULT_MATRIX,
                        help='26번 통합 결과 폴더.')
    parser.add_argument('--relevance-bins', choices=tuple(combined.BIN_SETS),
                        default='asymmetric', help='확정 설정은 asymmetric입니다.')
    parser.add_argument('--topk', type=combined.parse_topk, default=(1, 3, 5))
    parser.add_argument('--shuffles', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=3,
                        help='사다리·제외 설정을 시드만 바꿔 반복할 횟수. '
                             '허용폭을 이 반복의 표준편차에서 가져옵니다. '
                             '단일 feature 비교선은 반복하지 않습니다.')
    parser.add_argument('--n-estimators', type=int, default=400)
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261102)
    parser.add_argument('--tag', type=str, default=None, help='출력 폴더 접미사.')
    args = parser.parse_args()
    if args.n_estimators < 1 or args.max_depth < 1 or args.threads < 1 or args.shuffles < 1:
        parser.error('모델·대조군 인자는 양수여야 합니다.')
    if args.repeats < 1:
        parser.error('--repeats 는 1 이상이어야 합니다.')
    args.matrix = args.matrix.absolute()
    return run(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
