#!/usr/bin/env python3
"""Process B 최종 집합 확정(⑨). 조합을 여러 개 쓰면 더 좋은가.

33번이 기약 조합 12개를 찾았고 교체 규칙이 그중 **하나**를 골랐다. 규칙이 단일 승자
방식이었기 때문이며, 여러 세트를 함께 쓰는 경우는 고려하지 않았다. 이 스크립트가
그 누락을 메운다.

짚어둘 것이 두 가지 있다.

  1. XGBoost는 '조합'을 단위로 보관하지 않는다. feature 집합을 주면 트리가 알아서
     상호작용을 찾는다. 따라서 **"조합 N세트를 쓴다" = "N세트의 feature를 합집합으로
     넣는다"**와 같다.
  2. 33번이 찾은 12개는 서로 다른 그룹이 아니다. `protein_residues_lt_4A`가 12개
     전부에 들어 있고, 상위권은 거기에 `lig_bond_len_min_A`와 `receptor_clash_pairs`가
     붙은 변형들이다. 즉 **한 덩어리의 변형이 12개**이며 합집합은 약 14개다.

그래서 예상은 '평탄하게 나온다'이다. 32번에서 feature 수를 10 -> 20 -> 57로 늘려도
잡음 범위 안에서 같았다. 다만 그것은 '중요도 순으로 자른 20개'에 대한 결과이고
'조합 탐색으로 고른 21개'는 다른 집합이므로 32번 결과로 단정할 수 없다. 재서 정한다.

사전 등록된 판정 기준 (결과를 본 뒤 바꾸지 않는다)

    합집합 후보가 현 채택 집합(33번이 고른 것)을 **k=1 에서 측정 표준편차의 2배 이상**
    이기고, **어려운 층에서 1배 이상 나쁘지 않으면** 교체한다. 아니면 현 집합을 유지한다.
    통과자가 여럿이면 feature 수가 가장 적은 쪽.

2배를 쓰는 이유는 이것도 여러 후보 중 최대값을 고르는 일이기 때문이다(33번 §10.15와
같은 근거).

출력: out/union/34_run_union[_TAG]/
"""
import sys
sys.dont_write_bytecode = True
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

# 33번을 통해 32~28번까지 불러옵니다(평가·대조군·학습·측정 코드를 공유합니다).
_combination_path = Path(__file__).with_name('33_combination_search.py')
_kspec = importlib.util.spec_from_file_location('combination_support', _combination_path)
combination = importlib.util.module_from_spec(_kspec)
_kspec.loader.exec_module(combination)
selection = combination.selection
combined = combination.combined
graded = combination.graded
ranking = combination.ranking
baseline = combination.baseline
roles = combination.roles
audit = combination.audit
helper = combination.helper

VERSION = 'union_0.1.0'
OUT_KIND = 'union'
OUT_BASE = '34_run_union'
KEYS = baseline.KEYS
BASE = baseline.BASE
DEFAULT_MATRIX = baseline.DEFAULT_MATRIX
HEADLINE_RULE = selection.HEADLINE_RULE
HEADLINE_K = selection.HEADLINE_K
DEFAULT_SELECTION = BASE / 'feature_selection' / '32_run_selection_locked_rep5'
DEFAULT_COMBINATION = BASE / 'combination' / '33_run_combination_search'
# 합집합 사다리의 눈금. 상위 N개 조합을 합친다.
UNION_STEPS = (1, 2, 3, 5)
# 사전 등록된 교체 기준.
REPLACE_SIGMA_K1 = 2.0
REPLACE_SIGMA_HARD = 1.0
TOL_FLOOR_K1 = 0.002
TOL_FLOOR_HARD = 0.010


def load_combination_results(folder):
    """33번의 등수표와 교체 판정을 읽습니다."""
    folder = Path(folder)
    ranked_path = folder / 'ranked_combinations.csv'
    verdict_path = folder / 'replacement_verdict.json'
    if not ranked_path.exists():
        raise FileNotFoundError(f'33번 등수표가 없습니다: {ranked_path}')
    ranked = []
    with ranked_path.open(encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            ranked.append({'rank': int(row['rank']),
                           'members': [m for m in row['members'].split('|') if m],
                           'size': int(row['size']),
                           'increment_k1': float(row['increment_k1'])
                           if row['increment_k1'] not in ('', 'None') else None})
    ranked.sort(key=lambda r: r['rank'])
    verdict = (json.loads(verdict_path.read_text(encoding='utf-8'))
               if verdict_path.exists() else {})
    return ranked, verdict


def build_candidates(anchor, ranked, adopted, top10, steps):
    """비교할 feature 집합들을 만듭니다. 조합 N세트 = 그 feature들의 합집합."""
    out = [{'label': 'anchor_only', 'kind': 'baseline', 'n_sets': 0,
            'features': sorted(anchor)}]
    if adopted:
        out.append({'label': 'adopted_33', 'kind': 'incumbent', 'n_sets': 1,
                    'features': sorted(adopted)})
    if top10:
        out.append({'label': 'top10_32', 'kind': 'reference', 'n_sets': 0,
                    'features': sorted(top10)})
    for n in steps:
        if n > len(ranked):
            continue
        members = set(anchor)
        for row in ranked[:n]:
            members |= set(row['members'])
        out.append({'label': f'union_top{n}', 'kind': 'union', 'n_sets': n,
                    'features': sorted(members)})
    if len(ranked) > max(steps, default=0):
        members = set(anchor)
        for row in ranked:
            members |= set(row['members'])
        out.append({'label': f'union_all{len(ranked)}', 'kind': 'union',
                    'n_sets': len(ranked), 'features': sorted(members)})
    # 같은 feature 집합이 두 번 들어오면 뒤쪽을 버립니다(라벨만 다른 중복 측정 방지).
    seen, unique = set(), []
    for row in out:
        token = '|'.join(row['features'])
        if token in seen:
            row['duplicate_of'] = next(r['label'] for r in unique
                                       if '|'.join(r['features']) == token)
            continue
        seen.add(token)
        unique.append(row)
    return unique, out


def repeat_measure(features, ctx, args, seeds):
    """시드를 바꿔 여러 번 재고 **모든 (규칙 x k) 칸**의 평균·표준편차를 냅니다.

    이전 판에서는 k=1 과 어려운층만 보관하고 나머지를 버렸습니다. 판정에 그 두 칸만
    쓰기 때문이었는데, 그러면 최종 집합의 k=3·k=5 성적을 나중에 확인할 수 없습니다.
    계산은 어차피 measure() 가 전부 하므로 보관만 합니다(추가 비용 없음).
    """
    import numpy as np
    per_seed = []
    for sd in seeds:
        variant = copy.copy(args)
        variant.seed = sd
        ctx['args'] = variant
        per_seed.append(combination.measure(sorted(features), {}, ctx))
    out = {}
    for metric in per_seed[0]:
        if metric in ('members', 'size', 'objective'):
            continue
        values = [c[metric] for c in per_seed if c[metric] is not None]
        out[f'mean|{metric}'] = round(float(np.mean(values)), 4) if values else None
        out[f'std|{metric}'] = (round(float(np.std(values, ddof=1)), 4)
                                if len(values) > 1 else None)
    # 판정(judge)과 출력이 쓰는 이름은 그대로 유지합니다.
    k1 = f'lift|{HEADLINE_RULE}|k1'
    hard = f'hard_lift|{HEADLINE_RULE}|k{HEADLINE_K}'
    out['k1_mean'] = out[f'mean|{k1}']
    out['k1_std'] = out[f'std|{k1}']
    out['hard_mean'] = out[f'mean|{hard}']
    out['hard_std'] = out[f'std|{hard}']
    return out


def judge(rows, incumbent_label):
    """사전 등록된 기준으로 교체 여부를 정합니다."""
    inc = next((r for r in rows if r['label'] == incumbent_label), None)
    if inc is None or inc['k1_mean'] is None:
        return {'replace': False, 'reason': '현 채택 집합을 재지 못했습니다.',
                'incumbent': inc, 'challenger': None}
    tol1 = max(inc['k1_std'] or 0.0, TOL_FLOOR_K1)
    tolh = max(inc['hard_std'] or 0.0, TOL_FLOOR_HARD)
    need1 = inc['k1_mean'] + REPLACE_SIGMA_K1 * tol1
    floorh = inc['hard_mean'] - REPLACE_SIGMA_HARD * tolh
    passed = [r for r in rows
              if r['label'] != incumbent_label and r['k1_mean'] is not None
              and r['k1_mean'] >= need1
              and (r['hard_mean'] is None or r['hard_mean'] >= floorh)]
    result = {'incumbent': inc, 'threshold_k1': round(need1, 4),
              'threshold_hard': round(floorh, 4),
              'tol_k1': round(tol1, 4), 'tol_hard': round(tolh, 4)}
    if not passed:
        best = max((r for r in rows if r['label'] != incumbent_label
                    and r['k1_mean'] is not None),
                   key=lambda r: r['k1_mean'], default=None)
        if best is None:
            reason = '비교할 후보가 없습니다.'
        else:
            # 어느 조건에서 걸렸는지 구분해 적습니다. k1 은 넘겼는데 어려운 층에서
            # 걸린 경우를 'k1 부족'으로 쓰면 원인을 잘못 기록하게 됩니다.
            fails = []
            if best['k1_mean'] < need1:
                fails.append(f"k1 {best['k1_mean']:+.4f} < 기준 {need1:+.4f}")
            if best['hard_mean'] is not None and best['hard_mean'] < floorh:
                fails.append(f"어려운층 {best['hard_mean']:+.4f} < 하한 {floorh:+.4f}")
            if not fails:
                # 최선 후보는 통과했으나 다른 조건으로 걸린 경우(있을 수 없지만 방어).
                fails.append('조건 평가 불일치')
            reason = f"최선 후보 {best['label']}: " + ' / '.join(fails)
        result.update({'replace': False, 'challenger': best, 'reason': reason})
    else:
        win = min(passed, key=lambda r: (len(r['features']), -r['k1_mean']))
        result.update({'replace': True, 'challenger': win,
                       'reason': '사전 등록 기준 통과'})
    return result


def run(args):
    """합집합 사다리를 같은 조건으로 재고 최종 feature 집합을 확정합니다."""
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

    _, pool, anchor = combination.load_single_feature(args.selection_dir)
    ranked, verdict33 = load_combination_results(args.combination_dir)
    adopted = verdict33.get('new_features')
    chosen_path = Path(args.selection_dir) / 'chosen_features.json'
    top10 = (json.loads(chosen_path.read_text(encoding='utf-8'))['features']
             if chosen_path.exists() else None)
    print(f'기준 집합(단독 가능) {len(anchor)}개 | 33번 기약 조합 {len(ranked)}개', flush=True)
    if adopted:
        print(f'33번 채택 집합 {len(adopted)}개', flush=True)
    else:
        print('경고: 33번이 교체하지 않아 채택 집합이 없습니다. '
              '기준 집합을 현 채택으로 둡니다.', flush=True)

    # 조합들이 서로 다른 그룹인지, 한 덩어리의 변형인지 먼저 보여줍니다.
    freq = {}
    for row in ranked:
        for feat in row['members']:
            freq[feat] = freq.get(feat, 0) + 1
    union_all = sorted({f for row in ranked for f in row['members']})
    print(f'조합 {len(ranked)}개의 합집합 {len(union_all)}개 | 등장 빈도:', flush=True)
    for feat, count in sorted(freq.items(), key=lambda kv: -kv[1]):
        print(f'  {feat:34s} {count}/{len(ranked)}', flush=True)
    audit.save_csv(out / 'member_frequency.csv',
                   [{'feature': f, 'appears_in': c, 'of_total': len(ranked),
                     'share': round(c / len(ranked), 3)}
                    for f, c in sorted(freq.items(), key=lambda kv: -kv[1])],
                   ('feature', 'appears_in', 'of_total', 'share'))

    candidates, requested = build_candidates(anchor, ranked, adopted, top10,
                                             tuple(args.union_steps))
    dropped = [r for r in requested if 'duplicate_of' in r]
    print(f'\n비교할 집합 {len(candidates)}개'
          f'{f" (중복 {len(dropped)}개 제외)" if dropped else ""}', flush=True)
    for row in candidates:
        print(f"  {row['label']:14s} 조합 {row['n_sets']}세트 "
              f"feature {len(row['features']):2d}개", flush=True)
    for row in dropped:
        print(f"  [제외] {row['label']} 는 {row['duplicate_of']} 와 같은 집합", flush=True)

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

    bins = combined.BIN_SETS[args.relevance_bins]
    sc = dev['sc_rmsd'].to_numpy(dtype=np.float64, na_value=np.nan)
    rel = combined.relevance_from_sc(sc, bins, 'graded')
    known = dev['label_known'].to_numpy(dtype=bool)
    nn = dev['near_native'].to_numpy(dtype=bool)
    if int(np.sum((rel[known] >= combined.near_native_level(bins, 'graded')) != nn[known])):
        raise ValueError('등급과 near_native가 불일치합니다.')

    feats, _ = baseline.select_features(dev, catalog, set())
    needed = {f for row in candidates for f in row['features']}
    missing = sorted(needed - set(feats))
    if missing:
        raise ValueError(f'현재 행렬에 없는 feature: {missing}')
    codes = pd.factorize(dev['system_id'], sort=False)[0]
    assert np.all(np.diff(codes) >= 0), '시스템 그룹이 연속이 아닙니다.'
    fold_of = dev['cv_fold'].to_numpy()
    if int(pd.Series(fold_of).groupby(codes).nunique().max()) != 1:
        raise ValueError('한 시스템이 여러 폴드에 걸쳐 있습니다.')
    X = dev[feats].to_numpy(dtype=np.float32, na_value=np.nan)
    print(f'DEV poses {len(dev)} | systems {dev["system_id"].nunique()} '
          f'| TEST systems {test_systems} (봉인)', flush=True)

    groups = selection.group_randoms(selection.prepare_groups(dev, codes), args.topk)
    print(f'ELIGIBLE dev systems {len(groups)}', flush=True)
    controls = selection.evaluate_controls(groups, args.topk, args.shuffles, args.seed)
    ctx = {'X': X, 'rel': rel, 'codes': codes, 'fold_of': fold_of, 'known': known,
           'folds': folds, 'groups': groups, 'topk': args.topk,
           'analytic': controls['analytic'], 'hard_analytic': controls['hard_analytic'],
           'index_of': {name: i for i, name in enumerate(feats)}, 'args': args}

    seeds = [args.seed + i for i in range(args.repeats)]
    print(f'\n측정 (트리 {args.n_estimators}, 시드 {len(seeds)}회)', flush=True)
    rows = []
    print(f"  {'집합':14s}{'n':>3s}{'k1':>9s}{'±':>8s}{'k3':>9s}{'±':>8s}"
          f"{'k5':>9s}{'±':>8s}{'어려운층':>10s}{'±':>8s}  (규칙 {HEADLINE_RULE})",
          flush=True)
    for row in candidates:
        got = repeat_measure(row['features'], ctx, args, seeds)
        merged = dict(row, **got, n_features=len(row['features']),
                      features_joined='|'.join(row['features']))
        rows.append(merged)
        line = f"  {row['label']:14s}{len(row['features']):3d}"
        for k in args.topk:
            mean = got.get(f'mean|lift|{HEADLINE_RULE}|k{k}')
            std = got.get(f'std|lift|{HEADLINE_RULE}|k{k}')
            line += f"{(mean if mean is not None else 0):+9.4f}"
            line += f"{(std if std is not None else 0):8.4f}"
        line += f"{got['hard_mean']:+10.4f}{(got['hard_std'] or 0):8.4f}"
        print(line, flush=True)

    incumbent_label = 'adopted_33' if adopted else 'anchor_only'
    result = judge(rows, incumbent_label)
    cells = [f'lift|{r}|k{k}' for r in combined.RULES for k in args.topk]
    cells += [f'hard_lift|{HEADLINE_RULE}|k{k}' for k in args.topk]
    header = ['label', 'kind', 'n_sets', 'n_features', 'k1_mean', 'k1_std',
              'hard_mean', 'hard_std']
    header += [f'{stat}|{m}' for m in cells for stat in ('mean', 'std')]
    header += ['features_joined']
    audit.save_csv(out / 'union_results.csv', rows, tuple(header))

    final = result['challenger'] if result['replace'] else result['incumbent']
    if final is None:
        raise ValueError('최종 집합을 정할 수 없습니다. 32·33번 결과를 확인하세요.')
    final_features = sorted(final['features'])
    combo_only = sorted(set(final_features) - set(anchor))
    helper.write_json(out / 'final_feature_set.json', {
        'decision': 'replaced' if result['replace'] else 'kept',
        'label': final['label'], 'n_features': len(final_features),
        'features': final_features,
        'standalone': sorted(f for f in final_features if f in anchor),
        'combination_only': combo_only,
        'rule': {'sigma_k1': REPLACE_SIGMA_K1, 'sigma_hard': REPLACE_SIGMA_HARD,
                 'floors': {'k1': TOL_FLOOR_K1, 'hard': TOL_FLOOR_HARD},
                 'tie_break': 'feature 수 최소', 'pre_registered': True,
                 'threshold_k1': result.get('threshold_k1'),
                 'threshold_hard': result.get('threshold_hard')},
        'reason': result['reason']})

    changed = []
    for source in sources:
        try:
            info = Path(source['path']).stat()
            if (info.st_size, info.st_mtime_ns) != (source['size_bytes'], source['mtime_ns']):
                changed.append(source['path'])
        except OSError:
            changed.append(source['path'])

    status = 'UNION_COMPLETE' if not changed else 'CHECK_REQUIRED'
    report = {
        'version': VERSION, 'status': status, 'output_dir': str(out),
        'matrix': str(args.matrix), 'selection_dir': str(args.selection_dir),
        'combination_dir': str(args.combination_dir), 'sources': sources,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'support_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (_combination_path, combination._selection_path,
                                     selection._combined_path, combined._graded_path,
                                     graded._ranking_path, ranking._baseline_path,
                                     baseline._support_path, roles._support_path,
                                     audit._support_path)},
        'python': sys.executable, 'xgboost': xgboost.__version__,
        'anchor': sorted(anchor), 'pool_size': len(pool),
        'combinations_from_33': len(ranked), 'combination_union': union_all,
        'member_frequency': {f: c for f, c in sorted(freq.items(), key=lambda kv: -kv[1])},
        'candidates': [{k: v for k, v in r.items() if k != 'features_joined'}
                       for r in rows],
        'duplicates_dropped': [{'label': r['label'], 'duplicate_of': r['duplicate_of']}
                               for r in dropped],
        'incumbent_label': incumbent_label, 'verdict': result,
        'final_features': final_features,
        'final_standalone': sorted(f for f in final_features if f in anchor),
        'final_combination_only': combo_only,
        'model_params': {'n_estimators': args.n_estimators, 'max_depth': args.max_depth,
                         'learning_rate': args.learning_rate, 'seeds': seeds},
        'topk': list(args.topk), 'eligible_systems': len(groups),
        'hard_systems': controls['hard_systems'],
        'test_systems_untouched': test_systems,
        'changed_sources': changed, 'elapsed_seconds': time.perf_counter() - started,
        'policy': ['test 분할은 학습·평가에 사용하지 않았습니다.',
                   '후보 집합을 32·33번 결과 파일에서 읽었습니다(손으로 적지 않음).',
                   '"조합 N세트 사용" = "그 feature들의 합집합"으로 구현했습니다. '
                   'XGBoost는 조합을 단위로 보관하지 않습니다.',
                   '같은 feature 집합이 라벨만 달라 중복되면 한 번만 측정했습니다.',
                   '교체 기준을 실행 전 등록하고 결과를 본 뒤 바꾸지 않았습니다.'],
        'notes': ['member_frequency.csv 가 "조합들이 서로 다른 그룹인가"를 보여줍니다. '
                  '한 feature가 대부분의 조합에 들어 있으면 서로 다른 그룹이 아니라 '
                  '한 덩어리의 변형입니다.',
                  'final_feature_set.json 이 test 평가에 쓸 최종 집합입니다.',
                  'dev 에서 본 설정이 누적되어 있으므로 test 숫자는 dev 보다 낮게 나오는 '
                  '것이 정상입니다.']}
    helper.write_json(out / 'union_report.json', report)
    helper.write_json(out / 'RUN_COMPLETE.json', {
        'outputs_written': True, 'union_finished': True, 'status': status,
        'decision': 'replaced' if result['replace'] else 'kept',
        'final_label': final['label'], 'final_n_features': len(final_features),
        'sources_unchanged': not changed})

    print('', flush=True)
    print('STATUS:', status, flush=True)
    inc = result.get('incumbent')
    if inc and inc.get('k1_mean') is not None:
        print(f"기준: k1 >= {result.get('threshold_k1')} "
              f"(현 채택 {inc['k1_mean']:+.4f} + "
              f"{REPLACE_SIGMA_K1}x{result.get('tol_k1')}) "
              f"| 어려운층 >= {result.get('threshold_hard')}", flush=True)
    print(f"판정: {'교체' if result['replace'] else '유지'} — {result['reason']}", flush=True)
    print('', flush=True)
    print(f'=== 최종 feature 집합 ({len(final_features)}개, {final["label"]}) ===', flush=True)
    print(f"단독 {len(report['final_standalone'])}개:", flush=True)
    for feat in report['final_standalone']:
        print(f'  {feat}', flush=True)
    print(f'조합 전용 {len(combo_only)}개:', flush=True)
    for feat in combo_only:
        print(f'  {feat}', flush=True)
    return 0 if status == 'UNION_COMPLETE' else 2


def parse_steps(text):
    """'1,2,3,5' -> (1, 2, 3, 5)."""
    values = []
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit() or int(part) < 1:
            raise argparse.ArgumentTypeError(f'합집합 눈금은 1 이상 정수여야 합니다: {part}')
        values.append(int(part))
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError('눈금이 비었거나 중복입니다.')
    return tuple(sorted(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, default=DEFAULT_MATRIX)
    parser.add_argument('--selection-dir', type=Path, default=DEFAULT_SELECTION,
                        help='32번 결과 폴더(기준 집합·top10).')
    parser.add_argument('--combination-dir', type=Path, default=DEFAULT_COMBINATION,
                        help='33번 결과 폴더(등수표·채택 집합).')
    parser.add_argument('--union-steps', type=parse_steps, default=UNION_STEPS,
                        help='합집합 사다리 눈금. 기본 1,2,3,5 (+전체).')
    parser.add_argument('--relevance-bins', choices=tuple(combined.BIN_SETS),
                        default='asymmetric')
    parser.add_argument('--topk', type=combined.parse_topk, default=(1, 3, 5))
    parser.add_argument('--shuffles', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--n-estimators', type=int, default=400)
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20261102)
    parser.add_argument('--tag', type=str, default=None)
    args = parser.parse_args()
    for name in ('shuffles', 'repeats', 'n_estimators', 'max_depth', 'threads'):
        if getattr(args, name) < 1:
            parser.error(f'--{name.replace("_", "-")} 는 1 이상이어야 합니다.')
    args.matrix = args.matrix.absolute()
    args.selection_dir = args.selection_dir.absolute()
    args.combination_dir = args.combination_dir.absolute()
    return run(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
