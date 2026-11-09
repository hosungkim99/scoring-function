#!/usr/bin/env python3
"""Process B 조합 탐색(⑧). 단독으로는 무의미하지만 묶으면 유의미해지는 조합 찾기.

역할 (개발일지 §10.15 사전 등록)

1. **단독으로는 무작위보다 못한 feature만 후보로 두고, 그중 몇 개를 묶었을 때 유의미한
   성능이 나오는 조합을 찾는다** — 쌍에 한정하지 않고 크기 2는 전수, 크기 3 이상은
   beam search로 탐색한다.
2. **찾은 조합마다 원소를 하나씩 빼보아 전원이 필요한 기약(irreducible) 조합만 남기고**,
   영향력을 두 가지로 측정해 등수를 매긴다 — ① 그 조합만 썼을 때의 성능 ② 단독 유의미
   feature 위에 얹었을 때의 증분.
3. **결과는 등수·구성·영향력이 적힌 조합 목록으로 내고**, 최종 feature 집합을 교체할지는
   사전 등록된 기준으로 판정한다.

왜 32번으로는 안 되는가
  32번이 비교한 것은 7개 후보뿐이고 전부 '중요도 순으로 앞에서 자른' 중첩 집합이었다.
  사다리가 평탄했다는 결과의 뜻은 '중요도 순으로 자르는 방식은 몇 개를 자르든 상관없다'
  이고, '어떤 부분집합도 더 낫지 않다'가 아니다. 중요도 8위를 버리고 25위를 넣은 4개
  집합이 top10을 이길 수 있으며 사다리는 그런 집합에 도달할 수 없다.

영향력을 두 가지로 재는 이유
  ① 조합 자체의 성능  "이 조합만으로 얼마나 가는가"        <- 과학적 질문
  ② 기준 위의 증분    "이미 gnina가 있는데 더 보태는가"    <- 공학적 질문
  ①만 재면 함정이 있다. 단독 +0.02를 내는 조합이라도 그 정보가 gnina가 이미 잡아내는
  것과 겹치면 최종 모델에 넣을 이유가 없다. 반대로 ①이 작아도 ②가 크면 넣어야 한다.
  **등수는 ②로 매긴다.** 최종 모델에 무엇을 넣을지가 이 작업의 목적이기 때문이다.

알려진 한계 (사전 기록)
  탐색은 ①로 하고 등수는 ②로 매기므로, '단독 성능은 낮지만 기준 위에 얹으면 크게
  보태는 조합'은 beam에 들어오지 못해 놓칠 수 있다. ①로 탐색하는 것이 '단독으로는
  무의미하지만 조합하면 유의미해지는'이라는 질문에 충실하므로 이번에는 ①만 쓴다.
  결선에서 ②가 전부 작게 나오면 ②로 다시 탐색할 근거가 생긴다.

출력: out/combination/33_run_combination[_TAG]/
"""
import sys
sys.dont_write_bytecode = True
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
from itertools import combinations
import os
from pathlib import Path
import time

# 32번을 통해 31·30·29·28번까지 불러옵니다(평가·대조군·학습 코드를 복사하지 않습니다).
_selection_path = Path(__file__).with_name('32_feature_selection.py')
_sspec = importlib.util.spec_from_file_location('selection_support', _selection_path)
selection = importlib.util.module_from_spec(_sspec)
_sspec.loader.exec_module(selection)
combined = selection.combined
graded = selection.graded
ranking = selection.ranking
baseline = selection.baseline
roles = selection.roles
audit = selection.audit
helper = selection.helper

VERSION = 'combination_0.1.0'
OUT_KIND = 'combination'
OUT_BASE = '33_run_combination'
KEYS = baseline.KEYS
BASE = baseline.BASE
DEFAULT_MATRIX = baseline.DEFAULT_MATRIX
RULES = combined.RULES
HEADLINE_RULE = selection.HEADLINE_RULE
HEADLINE_K = selection.HEADLINE_K
HARD_CUT = selection.HARD_CUT
DEFAULT_SELECTION = BASE / 'feature_selection' / '32_run_selection_locked_rep5'
# 탐색·등수 기준. k=1 은 §8에서 주 품질 지표로 정한 것이고 분해능이 가장 큽니다.
SEARCH_K = 1
# 사전 등록된 교체 기준(개발일지 §10.15). 결과를 본 뒤 바꾸지 않습니다.
REPLACE_SIGMA_K1 = 2.0
REPLACE_SIGMA_HARD = 1.0
TOL_FLOOR_K1 = 0.002
TOL_FLOOR_HARD = 0.010


def load_single_feature(folder):
    """32번의 단일 feature 결과에서 후보 풀과 기준 집합을 읽습니다."""
    path = Path(folder) / 'single_feature.csv'
    if not path.exists():
        raise FileNotFoundError(f'32번 단일 feature 결과가 없습니다: {path}')
    k1 = f'lift|{HEADLINE_RULE}|k1'
    hard = f'hard_lift|{HEADLINE_RULE}|k{HEADLINE_K}'
    rows = []
    with path.open(encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            rows.append({'feature': row['feature'], 'category': row['category'],
                         'k1': float(row[k1]), 'hard': float(row[hard])})
    # 단독 불가 = 두 지표 모두 0 이하. 기준 = 두 지표 모두 양수.
    pool = [r['feature'] for r in rows if r['k1'] <= 0 and r['hard'] <= 0]
    anchor = [r['feature'] for r in rows if r['k1'] > 0 and r['hard'] > 0]
    return rows, pool, anchor


def key_of(members):
    """집합을 순서와 무관한 식별자로."""
    return '|'.join(sorted(members))


def measure(subset, cache, ctx):
    """조합 하나를 학습·평가합니다. 같은 집합은 다시 학습하지 않습니다."""
    import numpy as np
    token = key_of(subset)
    if token in cache:
        return cache[token]
    idx = np.array([ctx['index_of'][f] for f in subset], dtype=np.int64)
    score, _, _ = selection.train_oof(ctx['X'], ctx['rel'], ctx['codes'], ctx['fold_of'],
                                      ctx['known'], ctx['folds'], idx, ctx['args'])
    rates, hard_rates, _ = selection.evaluate_scores(ctx['groups'], score, ctx['topk'])
    out = {'members': sorted(subset), 'size': len(subset)}
    for rule in RULES:
        for k in ctx['topk']:
            out[f'lift|{rule}|k{k}'] = round(rates[(rule, k)] - ctx['analytic'][k], 4)
    for k in ctx['topk']:
        hv, hr = hard_rates[(HEADLINE_RULE, k)], ctx['hard_analytic'][k]
        out[f'hard_lift|{HEADLINE_RULE}|k{k}'] = (round(hv - hr, 4) if hv is not None
                                                  and hr is not None else None)
    out['objective'] = out[f'lift|{HEADLINE_RULE}|k{SEARCH_K}']
    cache[token] = out
    return out


def search(pool, ctx, args):
    """크기 2는 전수, 크기 3 이상은 beam search. 목표는 조합 단독 k=1 lift(①)."""
    cache, evaluated, frontier = {}, [], []
    pairs = list(combinations(sorted(pool), 2))
    print(f'  크기 2: 전수 {len(pairs)}쌍', flush=True)
    for i, pair in enumerate(pairs, 1):
        evaluated.append(measure(pair, cache, ctx))
        if i % 100 == 0:
            print(f'    {i}/{len(pairs)}', flush=True)
    frontier = sorted([r for r in evaluated if r['size'] == 2],
                      key=lambda r: -r['objective'])[:args.beam]
    print(f'    상위 {len(frontier)}개 유지, 최고 {frontier[0]["objective"]:+.4f} '
          f'({frontier[0]["members"]})', flush=True)

    for size in range(3, args.max_size + 1):
        seen, batch = set(), []
        for parent in frontier:
            for feat in pool:
                if feat in parent['members']:
                    continue
                child = tuple(sorted(parent['members'] + [feat]))
                if child in seen:
                    continue
                seen.add(child)
                batch.append(child)
        print(f'  크기 {size}: 확장 {len(batch)}개', flush=True)
        level = [measure(child, cache, ctx) for child in batch]
        evaluated.extend(level)
        frontier = sorted(level, key=lambda r: -r['objective'])[:args.beam]
        print(f'    상위 {len(frontier)}개 유지, 최고 {frontier[0]["objective"]:+.4f} '
              f'({frontier[0]["members"]})', flush=True)
    return evaluated, cache


def leave_one_out(subset, cache, ctx, margin):
    """원소를 하나씩 빼본다. 전부 하락하면 기약이다."""
    base = measure(subset, cache, ctx)['objective']
    rows, worst_keep = [], None
    for feat in subset:
        rest = [f for f in subset if f != feat]
        if len(rest) < 1:
            continue
        got = measure(rest, cache, ctx)['objective']
        drop = base - got
        rows.append({'members': key_of(subset), 'removed': feat,
                     'objective_full': round(base, 4), 'objective_without': round(got, 4),
                     'drop': round(drop, 4), 'needed': bool(drop > margin)})
        if drop <= margin:
            worst_keep = feat if worst_keep is None else worst_keep
    return rows, all(r['needed'] for r in rows), worst_keep


def run(args):
    """조합을 탐색하고 기약성을 검증한 뒤 영향력 순으로 등수를 매깁니다."""
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

    single_rows, pool, anchor = load_single_feature(args.selection_dir)
    chosen_path = Path(args.selection_dir) / 'chosen_features.json'
    incumbent = (json.loads(chosen_path.read_text(encoding='utf-8'))['features']
                 if chosen_path.exists() else None)
    print(f'후보 풀(단독 불가) {len(pool)}개 | 기준 집합(단독 가능) {len(anchor)}개', flush=True)
    print(f'  기준 집합: {anchor}', flush=True)
    if incumbent:
        print(f'  현 잠정 답(32번 top10) {len(incumbent)}개', flush=True)
    if len(pool) < 2:
        raise ValueError('후보 풀이 2개 미만입니다.')

    catalog = pd.read_csv(catalog_path).set_index('feature')
    meta_cols = ['system_id', 'model', 'seed', 'sample', 'pose_id', 'sc_rmsd', 'near_native',
                 'label_known', 'split', 'cv_fold']
    frame = pd.read_parquet(matrix_path, columns=meta_cols + catalog.index.tolist())
    frame['near_native'] = frame['near_native'].fillna(False).astype(bool)
    frame['label_known'] = frame['label_known'].fillna(False).astype(bool)
    test_systems = int(frame.loc[frame['split'] == 'test', 'system_id'].nunique())
    dev = frame[frame['split'] == 'dev'].copy()
    del frame
    dev = dev.sort_values(KEYS, kind='mergesort').reset_index(drop=True)
    dev['cv_fold'] = dev['cv_fold'].astype(int)
    dev['_tie'] = dev['pose_id'].map(baseline.tie_key)
    folds = sorted(int(f) for f in dev['cv_fold'].unique())
    print(f'DEV poses {len(dev)} | systems {dev["system_id"].nunique()} '
          f'| TEST systems {test_systems} (봉인)', flush=True)

    bins = combined.BIN_SETS[args.relevance_bins]
    sc = dev['sc_rmsd'].to_numpy(dtype=np.float64, na_value=np.nan)
    rel = combined.relevance_from_sc(sc, bins, 'graded')
    known = dev['label_known'].to_numpy(dtype=bool)
    nn = dev['near_native'].to_numpy(dtype=bool)
    nn_level = combined.near_native_level(bins, 'graded')
    if int(np.sum((rel[known] >= nn_level) != nn[known])):
        raise ValueError('등급과 near_native가 불일치합니다.')

    feats, _ = baseline.select_features(dev, catalog, set())
    missing = [f for f in pool + anchor if f not in feats]
    if missing:
        raise ValueError(f'32번 결과에 있으나 현재 행렬에 없는 feature: {missing}')
    codes = pd.factorize(dev['system_id'], sort=False)[0]
    assert np.all(np.diff(codes) >= 0), '시스템 그룹이 연속이 아닙니다.'
    fold_of = dev['cv_fold'].to_numpy()
    if int(pd.Series(fold_of).groupby(codes).nunique().max()) != 1:
        raise ValueError('한 시스템이 여러 폴드에 걸쳐 있습니다.')
    X = dev[feats].to_numpy(dtype=np.float32, na_value=np.nan)
    index_of = {name: i for i, name in enumerate(feats)}

    groups = selection.group_randoms(selection.prepare_groups(dev, codes), args.topk)
    print(f'ELIGIBLE dev systems {len(groups)}', flush=True)
    print('대조군 계산(설정과 무관, 1회)...', flush=True)
    controls = selection.evaluate_controls(groups, args.topk, args.shuffles, args.seed)
    print(f"  무작위(해석) k{args.topk}: "
          f"{[round(controls['analytic'][k], 4) for k in args.topk]} "
          f"| 어려운 층 {controls['hard_systems']}개", flush=True)

    # 탐색은 가볍게(기본 트리 150), 결선은 본설정으로.
    search_args = copy.copy(args)
    search_args.n_estimators = args.search_estimators
    ctx = {'X': X, 'rel': rel, 'codes': codes, 'fold_of': fold_of, 'known': known,
           'folds': folds, 'groups': groups, 'topk': args.topk,
           'analytic': controls['analytic'], 'hard_analytic': controls['hard_analytic'],
           'index_of': index_of, 'args': search_args}

    print(f'\n[1/4] 조합 탐색 (트리 {args.search_estimators}, beam {args.beam}, '
          f'최대 크기 {args.max_size})', flush=True)
    evaluated, cache = search(pool, ctx, args)
    header = (['members', 'size', 'objective']
              + [f'lift|{r}|k{k}' for r in RULES for k in args.topk]
              + [f'hard_lift|{HEADLINE_RULE}|k{k}' for k in args.topk])
    flat = [dict(r, members=key_of(r['members'])) for r in evaluated]
    audit.save_csv(out / 'combination_search.csv',
                   sorted(flat, key=lambda r: -r['objective']), tuple(header))
    best_by_size = []
    for size in sorted({r['size'] for r in evaluated}):
        best = max((r for r in evaluated if r['size'] == size), key=lambda r: r['objective'])
        best_by_size.append(dict(best, members=key_of(best['members'])))
    audit.save_csv(out / 'best_by_size.csv', best_by_size, tuple(header))
    print(f'  평가한 조합 {len(evaluated)}개 (중복 제외 {len(cache)}개)', flush=True)
    print('  크기별 최고:', flush=True)
    for row in best_by_size:
        print(f"    크기 {row['size']}: {row['objective']:+.4f}  {row['members']}", flush=True)

    # 기약성 검증. 여유폭은 탐색 단계 잡음의 하한을 씁니다.
    pick = sorted(evaluated, key=lambda r: -r['objective'])[:args.finalists]
    print(f'\n[2/4] 기약성 검증 (상위 {len(pick)}개, 여유폭 {TOL_FLOOR_K1})', flush=True)
    loo_rows, irreducible = [], []
    for row in pick:
        rows, is_irr, spare = leave_one_out(row['members'], cache, ctx, TOL_FLOOR_K1)
        loo_rows.extend(rows)
        mark = '기약' if is_irr else f'잉여 {spare}'
        print(f"  {key_of(row['members']):60s} {row['objective']:+.4f}  {mark}", flush=True)
        if is_irr:
            irreducible.append(row)
    audit.save_csv(out / 'leave_one_out.csv',
                   loo_rows or [{'members': None, 'removed': None, 'objective_full': None,
                                 'objective_without': None, 'drop': None, 'needed': None}],
                   ('members', 'removed', 'objective_full', 'objective_without',
                    'drop', 'needed'))
    print(f'  기약 조합 {len(irreducible)} / {len(pick)}', flush=True)

    # 결선: 트리 본설정 + 시드 반복으로 ①②를 재측정.
    print(f'\n[3/4] 결선 (트리 {args.n_estimators}, 시드 {args.repeats}회, '
          f'조합 {len(irreducible)}개)', flush=True)
    seeds = [args.seed + i for i in range(args.repeats)]
    final_ctx = dict(ctx)
    anchor_only = None
    finals = []

    def repeat_measure(subset):
        """시드 반복 측정. 조합 단독(①)과 기준+조합(②)을 함께 냅니다."""
        vals = {'k1': [], 'hard': [], 'with_k1': [], 'with_hard': []}
        with_set = sorted(set(subset) | set(anchor))
        for sd in seeds:
            variant = copy.copy(args)
            variant.seed = sd
            final_ctx['args'] = variant
            alone = measure(subset, {}, final_ctx)
            both = measure(with_set, {}, final_ctx)
            vals['k1'].append(alone[f'lift|{HEADLINE_RULE}|k1'])
            vals['hard'].append(alone[f'hard_lift|{HEADLINE_RULE}|k{HEADLINE_K}'])
            vals['with_k1'].append(both[f'lift|{HEADLINE_RULE}|k1'])
            vals['with_hard'].append(both[f'hard_lift|{HEADLINE_RULE}|k{HEADLINE_K}'])
        return vals

    def stat(values):
        arr = [v for v in values if v is not None]
        if not arr:
            return None, None
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else None
        return mean, std

    # 기준 집합 단독과 현 잠정 답을 같은 조건으로 먼저 잰다.
    for label, subset in (('anchor', anchor), ('incumbent_top10', incumbent)):
        if not subset:
            continue
        vals = {'k1': [], 'hard': []}
        for sd in seeds:
            variant = copy.copy(args)
            variant.seed = sd
            final_ctx['args'] = variant
            got = measure(sorted(subset), {}, final_ctx)
            vals['k1'].append(got[f'lift|{HEADLINE_RULE}|k1'])
            vals['hard'].append(got[f'hard_lift|{HEADLINE_RULE}|k{HEADLINE_K}'])
        m1, s1 = stat(vals['k1'])
        mh, sh = stat(vals['hard'])
        row = {'label': label, 'size': len(subset), 'members': key_of(subset),
               'k1_mean': round(m1, 4), 'k1_std': round(s1, 4) if s1 is not None else None,
               'hard_mean': round(mh, 4), 'hard_std': round(sh, 4) if sh is not None else None}
        finals.append(row)
        print(f"  [{label}] n={len(subset)} k1 {m1:+.4f}"
              f"{'' if s1 is None else f' +-{s1:.4f}'} | 어려운층 {mh:+.4f}", flush=True)
        if label == 'anchor':
            anchor_only = row

    ranked = []
    for row in irreducible:
        vals = repeat_measure(row['members'])
        a1, a1s = stat(vals['k1'])
        ah, _ = stat(vals['hard'])
        w1, w1s = stat(vals['with_k1'])
        wh, whs = stat(vals['with_hard'])
        inc1 = (w1 - anchor_only['k1_mean']) if anchor_only else None
        inch = (wh - anchor_only['hard_mean']) if anchor_only else None
        ranked.append({
            'members': key_of(row['members']), 'size': row['size'],
            'alone_k1': round(a1, 4), 'alone_k1_std': round(a1s, 4) if a1s else None,
            'alone_hard': round(ah, 4),
            'with_anchor_k1': round(w1, 4),
            'with_anchor_k1_std': round(w1s, 4) if w1s else None,
            'with_anchor_hard': round(wh, 4),
            'with_anchor_hard_std': round(whs, 4) if whs else None,
            'increment_k1': round(inc1, 4) if inc1 is not None else None,
            'increment_hard': round(inch, 4) if inch is not None else None})
        print(f"  {key_of(row['members'])[:58]:60s} 단독 {a1:+.4f} | "
              f"기준위 {w1:+.4f} | 증분 "
              f"{'-' if inc1 is None else f'{inc1:+.4f}'}", flush=True)

    # 등수는 ②(기준 위 증분) 기준.
    ranked.sort(key=lambda r: -(r['increment_k1'] if r['increment_k1'] is not None else -9))
    for rank, row in enumerate(ranked, 1):
        row['rank'] = rank
    audit.save_csv(out / 'ranked_combinations.csv', ranked,
                   ('rank', 'members', 'size', 'increment_k1', 'increment_hard',
                    'with_anchor_k1', 'with_anchor_k1_std', 'with_anchor_hard',
                    'with_anchor_hard_std', 'alone_k1', 'alone_k1_std', 'alone_hard'))
    audit.save_csv(out / 'finals.csv', finals,
                   ('label', 'size', 'members', 'k1_mean', 'k1_std',
                    'hard_mean', 'hard_std'))

    # 사전 등록된 교체 판정.
    print('\n[4/4] 교체 판정 (사전 등록 기준)', flush=True)
    inc_row = next((r for r in finals if r['label'] == 'incumbent_top10'), None)
    verdict = {'rule': {'sigma_k1': REPLACE_SIGMA_K1, 'sigma_hard': REPLACE_SIGMA_HARD,
                        'floors': {'k1': TOL_FLOOR_K1, 'hard': TOL_FLOOR_HARD},
                        'pre_registered': True, 'source': '개발일지 §10.15'},
               'incumbent': inc_row, 'challenger': None, 'replace': False, 'reason': None}
    if inc_row is None:
        verdict['reason'] = '현 잠정 답을 같은 조건으로 재지 못했습니다(32번 결과 없음).'
    elif not ranked:
        verdict['reason'] = '기약 조합이 없어 도전자가 없습니다.'
    else:
        tol1 = max(inc_row['k1_std'] or 0.0, TOL_FLOOR_K1)
        tolh = max(inc_row['hard_std'] or 0.0, TOL_FLOOR_HARD)
        need1 = inc_row['k1_mean'] + REPLACE_SIGMA_K1 * tol1
        floorh = inc_row['hard_mean'] - REPLACE_SIGMA_HARD * tolh
        print(f"  기준: k1 >= {need1:+.4f} (현 {inc_row['k1_mean']:+.4f} + "
              f"{REPLACE_SIGMA_K1}x{tol1:.4f})", flush=True)
        print(f"        어려운층 >= {floorh:+.4f}", flush=True)
        passed = [r for r in ranked
                  if r['with_anchor_k1'] >= need1 and r['with_anchor_hard'] >= floorh]
        if passed:
            win = min(passed, key=lambda r: (r['size'], -r['with_anchor_k1']))
            verdict.update({'challenger': win, 'replace': True,
                            'reason': '사전 등록 기준 전부 통과',
                            'new_features': sorted(set(win['members'].split('|')) | set(anchor))})
            print(f"  **교체**: 기준 집합 + {win['members']} "
                  f"(k1 {win['with_anchor_k1']:+.4f})", flush=True)
        else:
            best = max(ranked, key=lambda r: r['with_anchor_k1'])
            verdict.update({'challenger': best, 'replace': False,
                            'reason': f"최선 도전자 k1 {best['with_anchor_k1']:+.4f} < "
                                      f"기준 {need1:+.4f}"})
            print(f"  **유지**: top10. 최선 도전자 {best['members']} "
                  f"k1 {best['with_anchor_k1']:+.4f} < 기준 {need1:+.4f}", flush=True)
    helper.write_json(out / 'replacement_verdict.json', verdict)

    changed = []
    for source in sources:
        try:
            info = Path(source['path']).stat()
            if (info.st_size, info.st_mtime_ns) != (source['size_bytes'], source['mtime_ns']):
                changed.append(source['path'])
        except OSError:
            changed.append(source['path'])

    status = 'COMBINATION_COMPLETE' if not changed else 'CHECK_REQUIRED'
    report = {
        'version': VERSION, 'status': status, 'output_dir': str(out),
        'matrix': str(args.matrix), 'selection_dir': str(args.selection_dir),
        'sources': sources,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'support_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (_selection_path, selection._combined_path,
                                     combined._graded_path, graded._ranking_path,
                                     ranking._baseline_path, baseline._support_path,
                                     roles._support_path, audit._support_path)},
        'python': sys.executable, 'xgboost': xgboost.__version__,
        'locked_config': {'features': 'raw', 'relevance': 'graded',
                          'relevance_bins': args.relevance_bins, 'objective': 'rank:ndcg',
                          'selection_rule': HEADLINE_RULE},
        'search': {'pool': pool, 'pool_size': len(pool), 'anchor': anchor,
                   'incumbent': incumbent, 'beam': args.beam, 'max_size': args.max_size,
                   'search_estimators': args.search_estimators,
                   'objective': f'lift|{HEADLINE_RULE}|k{SEARCH_K} (조합 단독)',
                   'ranking_metric': 'increment_k1 (기준 집합 위 증분)',
                   'evaluated': len(evaluated), 'unique': len(cache)},
        'model_params': {'n_estimators': args.n_estimators, 'max_depth': args.max_depth,
                         'learning_rate': args.learning_rate, 'seed': args.seed,
                         'repeats': args.repeats, 'seeds': seeds},
        'topk': list(args.topk), 'eligible_systems': len(groups),
        'hard_systems': controls['hard_systems'],
        'test_systems_untouched': test_systems,
        'best_by_size': best_by_size, 'irreducible': len(irreducible),
        'ranked_combinations': ranked, 'finals': finals, 'verdict': verdict,
        'changed_sources': changed, 'elapsed_seconds': time.perf_counter() - started,
        'policy': ['test 분할은 학습·평가에 사용하지 않았습니다.',
                   '후보 풀과 기준 집합을 32번 결과 파일에서 읽었습니다(손으로 적지 않음).',
                   '탐색 목표는 조합 단독 k=1 lift, 등수는 기준 집합 위 증분입니다.',
                   '교체 기준을 §10.15에 실행 전 등록하고 결과를 본 뒤 바꾸지 않았습니다.',
                   '기약성 검증으로 잉여 원소가 끼인 조합을 걸렀습니다.',
                   '탐색은 가벼운 트리로, 결선은 본설정 + 시드 반복으로 했습니다.',
                   '무작위 대조군은 설정과 무관하므로 1회만 계산해 공유했습니다.'],
        'notes': ['ranked_combinations.csv 의 increment_k1 이 등수 기준입니다.',
                  '탐색을 ①로 하고 등수를 ②로 매기므로 "단독은 약하나 기준 위에서 크게 '
                  '보태는 조합"은 놓칠 수 있습니다(§10.15 알려진 한계). 증분이 전부 작게 '
                  '나오면 ②로 다시 탐색할 근거가 됩니다.',
                  'replacement_verdict.json 이 최종 feature 집합 교체 여부입니다.']}
    helper.write_json(out / 'combination_report.json', report)
    helper.write_json(out / 'RUN_COMPLETE.json', {
        'outputs_written': True, 'combination_finished': True, 'status': status,
        'evaluated': len(evaluated), 'irreducible': len(irreducible),
        'replace': verdict['replace'], 'new_features': verdict.get('new_features'),
        'sources_unchanged': not changed})

    print('', flush=True)
    print('STATUS:', status, flush=True)
    print(f'평가 조합 {len(evaluated)} | 기약 {len(irreducible)} | '
          f'교체 {"예" if verdict["replace"] else "아니오"}', flush=True)
    if ranked:
        print('', flush=True)
        print('=== 영향력 순위 (기준 집합 위 증분) ===', flush=True)
        print(f"{'등수':>4s}{'크기':>4s}{'증분 k1':>10s}{'증분 어려운층':>14s}"
              f"{'단독 k1':>10s}  조합", flush=True)
        for row in ranked:
            print(f"{row['rank']:4d}{row['size']:4d}"
                  f"{(row['increment_k1'] if row['increment_k1'] is not None else 0):+10.4f}"
                  f"{(row['increment_hard'] if row['increment_hard'] is not None else 0):+14.4f}"
                  f"{row['alone_k1']:+10.4f}  {row['members']}", flush=True)
    return 0 if status == 'COMBINATION_COMPLETE' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, default=DEFAULT_MATRIX,
                        help='26번 통합 결과 폴더.')
    parser.add_argument('--selection-dir', type=Path, default=DEFAULT_SELECTION,
                        help='32번 결과 폴더(후보 풀·기준 집합·현 잠정 답을 여기서 읽습니다).')
    parser.add_argument('--relevance-bins', choices=tuple(combined.BIN_SETS),
                        default='asymmetric', help='확정 설정은 asymmetric입니다.')
    parser.add_argument('--beam', type=int, default=8, help='크기별로 유지할 상위 조합 수.')
    parser.add_argument('--max-size', type=int, default=6, help='탐색할 최대 조합 크기.')
    parser.add_argument('--finalists', type=int, default=20,
                        help='기약성 검증에 넘길 상위 조합 수.')
    parser.add_argument('--search-estimators', type=int, default=150,
                        help='탐색 단계 트리 수(결선은 --n-estimators).')
    parser.add_argument('--topk', type=combined.parse_topk, default=(1, 3, 5))
    parser.add_argument('--shuffles', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=5, help='결선 시드 반복 수.')
    parser.add_argument('--n-estimators', type=int, default=400)
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20261102)
    parser.add_argument('--tag', type=str, default=None, help='출력 폴더 접미사.')
    args = parser.parse_args()
    for name in ('beam', 'max_size', 'finalists', 'search_estimators', 'n_estimators',
                 'max_depth', 'threads', 'repeats', 'shuffles'):
        if getattr(args, name) < 1:
            parser.error(f'--{name.replace("_", "-")} 는 1 이상이어야 합니다.')
    if args.max_size < 2:
        parser.error('--max-size 는 2 이상이어야 합니다.')
    args.matrix = args.matrix.absolute()
    args.selection_dir = args.selection_dir.absolute()
    return run(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
