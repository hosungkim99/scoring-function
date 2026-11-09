#!/usr/bin/env python3
"""Process B 마지막 단계(⑩). test 923 타깃 1회 평가.

**test 는 한 번만 쓸 수 있다.** dev 에서 설정 9종(28·29x3·30·31x4) + 선별 24종(32) +
조합 1,294종(33) + 합집합 7종(34)을 비교했고, 그중 최고를 고르는 행위가 dev 숫자를
부풀렸다. test 는 그 편향을 교정하는 유일한 장치다. 여기서 나온 숫자를 보고 모델을
다시 고르면 그 장치가 사라진다.

비교 설계 — **feature 집합만 바꾸고 나머지는 전부 고정한다.**

    고정  원본 feature / 비대칭 등급 경계 / rank:ndcg(그룹=시스템) /
          생성기 분산 선택(규칙 A) / 같은 분할·폴드·시드·트리 인자 / 같은 평가 코드
    변인  feature 목록 하나

집합 4종을 같은 조건으로 잰다. 네 집합 모두 34번이 이미 dev 에서 같은 조건으로 잰
것이며, feature 목록을 손으로 적지 않고 34번의 `union_results.csv` 에서 읽는다.

    adopted_33    11개. 34번이 사전 등록 규칙으로 유지한 집합       <- 비교 대상 A
    union_all12   21개. 기약 조합 12개 전부의 합집합                <- 비교 대상 B
    anchor_only    7개. 단독 가능 feature 만. 조합의 기여를 보는 비교선
    top10_32      10개. 32번이 중요도 사다리로 고른 집합. 참고용

A 와 B 가 이 실행의 비교 대상이다. dev 에서는 네 기준 칸(k=1/3/5/어려운층) 어디서도
2σ 로 갈리지 않았다. test 는 923 타깃이라 dev 보다 잡음이 크므로 **갈리지 않을
가능성이 높고, 그 결과도 유효한 답이다**("둘은 구분되지 않는다").

anchor_only 는 조합의 기여를 보는 비교선이다. dev 에서 조합 4개가 단독 7개 위에
+0.0069 를 보탰는데 그 값은 1,294개 중 최대값이라 **선택 편향 범위 안**이다. test 에서
사라지면 "조합 탐색으로 얻은 개선은 dev 과적합이었다"가 결론이고 그것도 유효한 결과다.

어려운 층은 무작위 Top-5 성공확률 < 0.5 인 시스템이다(dev 207개). test 에서는 약 1/4
규모로 줄어 잡음이 더 크므로 **A·B 를 가르는 근거로 쓰지 않는다.** 참고 수치다.

측정 방법
  dev  : 18번 폴드로 out-of-fold (34번과 같은 방식). 비교 기준선
  test : dev 전체로 모델 1개를 학습해 test 를 예측. 폴드 불필요(test 가 홀드아웃)
  둘 다 시드를 --repeats 회 바꿔 평균·표준편차를 낸다. 시드 중 최고를 고르지 않는다.

사전 예고 (test 를 보기 전에 적어둔 값, 개발일지 §10.10)
  test 는 dev 보다 k=5 약 0.006 / k=1 약 0.011 낮게 나오는 것이 정상이다.
  낮게 나왔다고 실패로 오판하거나 추가 튜닝으로 test 를 소모하지 않는다.

출력: out/test_eval/35_run_test[_TAG]/
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

# 34번을 통해 33~28번까지 불러옵니다(평가·대조군·학습 코드를 공유합니다).
_union_path = Path(__file__).with_name('34_union_test.py')
_uspec = importlib.util.spec_from_file_location('union_support', _union_path)
union = importlib.util.module_from_spec(_uspec)
_uspec.loader.exec_module(union)
combination = union.combination
selection = union.selection
combined = union.combined
graded = union.graded
ranking = union.ranking
baseline = union.baseline
roles = union.roles
audit = union.audit
helper = union.helper

VERSION = 'test_eval_0.1.0'
OUT_KIND = 'test_eval'
OUT_BASE = '35_run_test'
KEYS = baseline.KEYS
BASE = baseline.BASE
DEFAULT_MATRIX = baseline.DEFAULT_MATRIX
RULES = combined.RULES
HEADLINE_RULE = selection.HEADLINE_RULE
HEADLINE_K = selection.HEADLINE_K
DEFAULT_UNION = BASE / 'union' / '34_run_union_final'
DEFAULT_SELECTION = BASE / 'feature_selection' / '32_run_selection_locked_rep5'
# 사전 예고치(개발일지 §10.10). test 를 보기 전에 정한 값이며 판정 기준이 아니라
# "이 정도 낮아지는 것은 정상"이라는 참고선이다.
EXPECTED_DROP_K1 = 0.011
EXPECTED_DROP_K5 = 0.006
# 비교 대상 2종. 순서가 보고서의 A/B 순서입니다.
CANDIDATES = ('adopted_33', 'union_all12')
# 하나의 기준이 필요한 자리(공헌도 분해 출력 등)에서 쓸 집합.
REFERENCE_LABEL = 'adopted_33'
# 34번 union_results.csv 에서 읽을 집합과 그 역할.
CONFIG_SPEC = (
    ('adopted_33', 'candidate_a', '34번이 사전 등록 규칙으로 유지한 집합'),
    ('union_all12', 'candidate_b', '기약 조합 12개 전부의 합집합'),
    ('anchor_only', 'baseline_for_combination', '단독 가능 feature 만. 조합 기여 비교선'),
    ('top10_32', 'context_only', '32번이 중요도 사다리로 고른 집합. 참고용'))
# 집합을 뺀 나머지 조건. 보고서에 그대로 적어 통제 변인을 명시합니다.
HELD_CONSTANT = (
    'feature 원본값(z-score 변환 없음)',
    '등급 경계 비대칭 (<1A 11 / <2A 10 / <3A 2 / <5A 1)',
    'rank:ndcg, 그룹 = 시스템',
    '선택 규칙 ruleA (plain·ruleB 도 함께 계산해 보고)',
    'dev/test 분할과 18번 폴드',
    '시드 집합과 반복 횟수',
    '트리 수·깊이·학습률',
    '동점 처리(pose_id SHA1 오름차순)',
    '평가·대조군 코드(32번 함수 공유)')


def load_configs(union_dir, selection_dir):
    """34번 union_results.csv 에서 집합 4종을 읽습니다(손으로 적지 않습니다)."""
    import csv
    union_path = Path(union_dir) / 'union_results.csv'
    if not union_path.exists():
        raise FileNotFoundError(f'34번 결과가 없습니다: {union_path}')
    table = {}
    with union_path.open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            joined = (row.get('features_joined') or '').strip()
            if row.get('label') and joined:
                table[row['label']] = sorted(f for f in joined.split('|') if f)
    missing = [label for label, _, _ in CONFIG_SPEC if label not in table]
    if missing:
        raise ValueError(f'union_results.csv 에 없는 집합: {missing} '
                         f'(있는 것: {sorted(table)})')
    configs, seen = [], {}
    for label, role, note in CONFIG_SPEC:
        feats = table[label]
        if not feats:
            raise ValueError(f'{label} 의 feature 목록이 비어 있습니다.')
        key = tuple(feats)
        if key in seen:
            raise ValueError(f'{label} 과 {seen[key]} 의 feature 집합이 같습니다. '
                             f'통제 비교가 성립하지 않습니다.')
        seen[key] = label
        configs.append({'label': label, 'role': role, 'note': note,
                        'features': feats})
    # 34번이 확정한 집합·32번이 고른 집합과 대조해 파일이 섞이지 않았는지 확인합니다.
    final_path = Path(union_dir) / 'final_feature_set.json'
    final_data = (json.loads(final_path.read_text(encoding='utf-8'))
                  if final_path.exists() else {})
    standalone = sorted(final_data.get('standalone') or table['anchor_only'])
    combo = sorted(final_data.get('combination_only') or [])
    if final_data.get('features'):
        confirmed = sorted(final_data['features'])
        got = table.get(final_data.get('label'))
        if got is not None and got != confirmed:
            raise ValueError(f"34번 확정 집합({final_data.get('label')})이 "
                             f'union_results.csv 와 다릅니다.')
    if standalone and combo:
        if sorted(set(standalone) | set(combo)) != table['adopted_33']:
            raise ValueError('단독/조합 분해가 adopted_33 과 맞지 않습니다.')
    chosen_path = Path(selection_dir) / 'chosen_features.json'
    if chosen_path.exists():
        chosen = sorted(json.loads(chosen_path.read_text(encoding='utf-8'))['features'])
        if chosen != table['top10_32']:
            raise ValueError('32번 chosen_features.json 이 top10_32 와 다릅니다.')
    return configs, final_data, standalone, combo


def train_full(X, rel, codes, known, args, X_pred):
    """dev 전체로 모델 하나를 학습해 test 를 예측합니다. 폴드가 필요 없습니다."""
    import numpy as np
    keep = known.copy()
    _, dead = graded.pairable_systems(codes, rel, keep)
    if dead.size:
        keep = keep & ~np.isin(codes, dead)
    model = graded.build_ranker(args)
    model.fit(X[keep], rel[keep], qid=codes[keep])
    return (model.predict(X_pred), int(np.unique(codes[keep]).size), int(keep.sum()))


def cells(rates, hard_rates, controls, topk):
    """규칙 x k 성공률과 lift 를 한 묶음으로."""
    out = {}
    for rule in RULES:
        for k in topk:
            out[f'success|{rule}|k{k}'] = rates[(rule, k)]
            out[f'lift|{rule}|k{k}'] = rates[(rule, k)] - controls['analytic'][k]
    for k in topk:
        value, ref = hard_rates[(HEADLINE_RULE, k)], controls['hard_analytic'][k]
        out[f'hard_lift|{HEADLINE_RULE}|k{k}'] = (value - ref if value is not None
                                                  and ref is not None else None)
    return out


def summarize(per_seed):
    """시드별 측정값의 평균·표준편차. 시드 중 최고를 고르지 않습니다."""
    import numpy as np
    out = {}
    for metric in per_seed[0]:
        values = [c[metric] for c in per_seed if c[metric] is not None]
        out[f'mean|{metric}'] = round(float(np.mean(values)), 4) if values else None
        out[f'std|{metric}'] = (round(float(np.std(values, ddof=1)), 4)
                                if len(values) > 1 else None)
    return out


def run(args):
    """dev out-of-fold 와 test 를 같은 조건으로 재고 나란히 보고합니다."""
    if not args.open_test:
        print('test 는 한 번만 쓸 수 있습니다. 열려면 --open-test 를 주세요.',
              file=sys.stderr, flush=True)
        print('열기 전에 확인할 것:', file=sys.stderr, flush=True)
        print('  1) 최종 feature 집합이 확정됐는가 (34번 final_feature_set.json)',
              file=sys.stderr, flush=True)
        print('  2) dev 에서 더 바꿀 것이 없는가', file=sys.stderr, flush=True)
        print('  3) test 결과를 보고 모델을 다시 고르지 않을 각오가 됐는가',
              file=sys.stderr, flush=True)
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
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(args.threads)
    print('OUTPUT_DIR:', out, flush=True)
    print('', flush=True)
    print('*** test 923 타깃을 엽니다. 한 번만 쓸 수 있습니다. ***', flush=True)
    print('*** feature 집합만 변인입니다. 나머지 조건은 네 집합이 전부 같습니다. ***',
          flush=True)
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

    configs, final_data, standalone, combo = load_configs(args.union_dir,
                                                          args.selection_dir)
    print('비교 설계 — feature 집합만 변인, 나머지는 전부 고정', flush=True)
    for line in HELD_CONSTANT:
        print(f'  고정  {line}', flush=True)
    print('', flush=True)
    for row in configs:
        mark = {'candidate_a': '비교 대상 A', 'candidate_b': '비교 대상 B'}.get(
            row['role'], '비교선     ')
        print(f'  {mark}  {row["label"]:12s} feature {len(row["features"]):2d}개 '
              f'— {row["note"]}', flush=True)
    print(f'  (adopted_33 분해: 단독 {len(standalone)}개 + 조합 전용 {len(combo)}개)',
          flush=True)

    catalog = pd.read_csv(catalog_path).set_index('feature')
    meta_cols = ['system_id', 'model', 'seed', 'sample', 'pose_id', 'sc_rmsd', 'near_native',
                 'label_known', 'split', 'cv_fold']
    frame = pd.read_parquet(matrix_path, columns=meta_cols + catalog.index.tolist())
    frame['near_native'] = frame['near_native'].fillna(False).astype(bool)
    frame['label_known'] = frame['label_known'].fillna(False).astype(bool)
    if 'split' not in frame.columns or frame['split'].isna().all():
        raise ValueError('split 열이 없습니다.')
    frame = frame.sort_values(KEYS, kind='mergesort').reset_index(drop=True)
    frame['_tie'] = frame['pose_id'].map(baseline.tie_key)

    feats, _ = baseline.select_features(frame, catalog, set())
    needed = {f for row in configs for f in row['features']}
    missing = sorted(needed - set(feats))
    if missing:
        raise ValueError(f'현재 행렬에 없는 feature: {missing}')
    index_of = {name: i for i, name in enumerate(feats)}
    bins = combined.BIN_SETS[args.relevance_bins]

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
        hard_rand = [(round(controls['hard_analytic'][k], 4)
                      if controls['hard_analytic'][k] is not None else None)
                     for k in args.topk]
        print(f'{name.upper()}: poses {len(sub)} | systems {int(np.unique(codes).size)} '
              f'| eligible {len(groups)} | 어려운 층 {controls["hard_systems"]}', flush=True)
        print(f'  무작위(해석)    k{args.topk}: '
              f'{[round(controls["analytic"][k], 4) for k in args.topk]}', flush=True)
        print(f'  어려운층 무작위 k{args.topk}: {hard_rand}', flush=True)

    dev, test = parts['dev'], parts['test']
    if dev['fold_of'] is not None:
        span = int(pd.Series(dev['fold_of']).groupby(dev['codes']).nunique().max())
        if span != 1:
            raise ValueError('dev 의 한 시스템이 여러 폴드에 걸쳐 있습니다.')
    folds = sorted(int(f) for f in np.unique(dev['fold_of']))
    seeds = [args.seed + i for i in range(args.repeats)]
    print('', flush=True)
    print(f'측정 (트리 {args.n_estimators}, 시드 {len(seeds)}회, 폴드 {folds})', flush=True)

    rows, score_store = [], {}
    for cfg in configs:
        idx = np.array([index_of[f] for f in cfg['features']], dtype=np.int64)
        dev_seeds, test_seeds, train_info = [], [], []
        for sd in seeds:
            variant = copy.copy(args)
            variant.seed = sd
            # dev: out-of-fold (34번과 같은 방식)
            dev_score, _, _ = selection.train_oof(dev['X'], dev['rel'], dev['codes'],
                                                  dev['fold_of'], dev['known'], folds,
                                                  idx, variant)
            rates, hard_rates, _ = selection.evaluate_scores(dev['groups'], dev_score,
                                                             args.topk)
            dev_seeds.append(cells(rates, hard_rates, dev['controls'], args.topk))
            # test: dev 전체로 학습한 모델 1개로 예측
            test_score, groups_used, poses_used = train_full(
                dev['X'][:, idx], dev['rel'], dev['codes'], dev['known'], variant,
                test['X'][:, idx])
            rates, hard_rates, _ = selection.evaluate_scores(test['groups'], test_score,
                                                             args.topk)
            test_seeds.append(cells(rates, hard_rates, test['controls'], args.topk))
            train_info.append({'seed': sd, 'train_groups': groups_used,
                               'train_poses': poses_used})
            if sd == args.seed:
                score_store[cfg['label']] = test_score
        row = {'label': cfg['label'], 'role': cfg['role'],
               'n_features': len(cfg['features']),
               'features_joined': '|'.join(cfg['features']),
               'train_groups': train_info[0]['train_groups'],
               'train_poses': train_info[0]['train_poses']}
        for split, per_seed in (('dev', dev_seeds), ('test', test_seeds)):
            for key, value in summarize(per_seed).items():
                stat, metric = key.split('|', 1)
                row[f'{split}|{stat}|{metric}'] = value
        rows.append(row)
        k1 = f'mean|lift|{HEADLINE_RULE}|k1'
        k5 = f'mean|lift|{HEADLINE_RULE}|k{HEADLINE_K}'
        print(f"  {cfg['label']:10s} n={len(cfg['features']):2d} "
              f"| dev k1 {row[f'dev|{k1}']:+.4f} k5 {row[f'dev|{k5}']:+.4f} "
              f"| test k1 {row[f'test|{k1}']:+.4f} k5 {row[f'test|{k5}']:+.4f}", flush=True)

    # 비교 대상 2종의 test 점수를 남깁니다(첫 시드).
    measured = {r['label'] for r in rows}
    if REFERENCE_LABEL not in CANDIDATES:
        raise ValueError(f'REFERENCE_LABEL({REFERENCE_LABEL})은 CANDIDATES 안에 '
                         f'있어야 합니다: {CANDIDATES}')
    absent = [label for label in CANDIDATES if label not in measured]
    if absent:
        raise ValueError(f'비교 대상이 측정되지 않았습니다: {absent}')
    buf_frame = test['frame'][KEYS + ['pose_id']].copy()
    for label in CANDIDATES:
        buf_frame[f'score|{label}'] = score_store[label]
    import io
    buf = io.BytesIO()
    buf_frame.to_parquet(buf, engine='pyarrow', index=False, compression='snappy')
    with helper.checked_path(out / 'scores_test.parquet').open('xb') as handle:
        handle.write(buf.getvalue())

    header = ['label', 'role', 'n_features', 'train_groups', 'train_poses']
    metrics = [f'{q}|{r}|k{k}' for q in ('success', 'lift')
               for r in RULES for k in args.topk]
    metrics += [f'hard_lift|{HEADLINE_RULE}|k{k}' for k in args.topk]
    header += [f'{sp}|{st}|{m}' for sp in ('dev', 'test')
               for m in metrics for st in ('mean', 'std')]
    header += ['features_joined']
    audit.save_csv(out / 'dev_vs_test.csv', rows, tuple(header))

    strat_rows = []
    for split in ('dev', 'test'):
        part = parts[split]
        axis = f'rand_analytic_k{HEADLINE_K}'
        _ = axis
        for name, lo, hi in baseline.STRATA:
            sel = [g for g in part['groups']
                   if lo <= g['rand'][HEADLINE_K] < hi]
            if not sel:
                continue
            strat_rows.append({'split': split, 'stratum': name, 'systems': len(sel),
                               'rand_top5': round(float(sum(g['rand'][HEADLINE_K]
                                                            for g in sel) / len(sel)), 4),
                               'room_targets': round(sum(1.0 - g['rand'][HEADLINE_K]
                                                         for g in sel), 1)})
    audit.save_csv(out / 'strata_profile.csv', strat_rows,
                   ('split', 'stratum', 'systems', 'rand_top5', 'room_targets'))

    # 공헌도 분해 — 선택 규칙의 이득과 모델의 이득을 분리합니다(평가 명세 §8 의무).
    contrib = []
    for label in CANDIDATES:
        cand = next(r for r in rows if r['label'] == label)
        for split in ('dev', 'test'):
            ctrl = parts[split]['controls']
            n = ctrl['systems']
            for rule in RULES:
                for k in args.topk:
                    base_rand = ctrl['shuffle'][('plain', k)]
                    rule_rand = ctrl['shuffle'][(rule, k)]
                    model = cand[f'{split}|mean|success|{rule}|k{k}']
                    contrib.append({
                        'label': label,
                        'split': split, 'rule': rule, 'topk': k, 'systems': n,
                        'rand_analytic': round(ctrl['analytic'][k], 4),
                        'rand_plain_empirical': round(base_rand, 4),
                        'rand_with_rule': round(rule_rand, 4),
                        'model_with_rule': round(model, 4),
                        'diversity_gain': round(rule_rand - base_rand, 4),
                        'model_gain': round(model - rule_rand, 4),
                        'total_gain': round(model - base_rand, 4),
                        'model_targets': round((model - rule_rand) * n, 1)})
    audit.save_csv(out / 'contribution.csv', contrib,
                   ('label', 'split', 'rule', 'topk', 'systems', 'rand_analytic',
                    'rand_plain_empirical', 'rand_with_rule', 'model_with_rule',
                    'diversity_gain', 'model_gain', 'total_gain', 'model_targets'))

    # 조합의 기여가 test 에서 살아남았는지 — 이 실행의 핵심 질문.
    anchor = next((r for r in rows if r['role'] == 'baseline_for_combination'), None)
    combo_check = None
    if anchor is not None:
        k1 = f'mean|lift|{HEADLINE_RULE}|k1'
        hk = f'mean|hard_lift|{HEADLINE_RULE}|k{HEADLINE_K}'
        s1 = f'std|lift|{HEADLINE_RULE}|k1'
        combo_check = {}
        for label in CANDIDATES:
            cand = next(r for r in rows if r['label'] == label)
            per_split = {}
            for split in ('dev', 'test'):
                gain = cand[f'{split}|{k1}'] - anchor[f'{split}|{k1}']
                spread = ((cand[f'{split}|{s1}'] or 0.0) ** 2
                          + (anchor[f'{split}|{s1}'] or 0.0) ** 2) ** 0.5
                per_split[split] = {
                    'anchor_k1': anchor[f'{split}|{k1}'],
                    'candidate_k1': cand[f'{split}|{k1}'],
                    'combination_gain_k1': round(gain, 4),
                    'combined_std': round(spread, 4),
                    'sigma': round(gain / spread, 2) if spread > 1e-9 else None,
                    'anchor_hard': anchor[f'{split}|{hk}'],
                    'candidate_hard': cand[f'{split}|{hk}'],
                    'combination_gain_hard': round(cand[f'{split}|{hk}']
                                                   - anchor[f'{split}|{hk}'], 4)}
            combo_check[label] = per_split

    k1m = f'mean|lift|{HEADLINE_RULE}|k1'
    k5m = f'mean|lift|{HEADLINE_RULE}|k{HEADLINE_K}'
    drops = {}
    for row_of in rows:
        d1 = round(row_of[f'dev|{k1m}'] - row_of[f'test|{k1m}'], 4)
        d5 = round(row_of[f'dev|{k5m}'] - row_of[f'test|{k5m}'], 4)
        drops[row_of['label']] = {
            'drop_k1': d1, 'drop_k5': d5,
            'k1_within_expectation': bool(d1 <= EXPECTED_DROP_K1 * 2),
            'k5_within_expectation': bool(d5 <= EXPECTED_DROP_K5 * 2)}
    drop = {'k1': drops[REFERENCE_LABEL]['drop_k1'],
            'k5': drops[REFERENCE_LABEL]['drop_k5']}
    expectation = {
        'expected_drop_k1': EXPECTED_DROP_K1, 'expected_drop_k5': EXPECTED_DROP_K5,
        'reference': REFERENCE_LABEL, 'per_config': drops,
        'actual_drop_k1': drop['k1'], 'actual_drop_k5': drop['k5'],
        'k1_within_expectation': drops[REFERENCE_LABEL]['k1_within_expectation'],
        'k5_within_expectation': drops[REFERENCE_LABEL]['k5_within_expectation'],
        'note': ('사전 예고치는 판정 기준이 아니라 "이 정도 낮아지는 것은 정상"이라는 '
                 '참고선입니다. 예고보다 크게 낮아지면 dev 과적합이 예상보다 심했다는 '
                 '뜻이고, 그것도 유효한 결과입니다.')}

    # 정면 비교 — 이 실행의 주된 질문. feature 집합만 다른 두 집합을 칸마다 맞댑니다.
    a_label, b_label = CANDIDATES
    cand_a = next(r for r in rows if r['label'] == a_label)
    cand_b = next(r for r in rows if r['label'] == b_label)
    a_mean, a_std = f'{a_label}|mean', f'{a_label}|std'
    b_mean, b_std = f'{b_label}|mean', f'{b_label}|std'
    compare_cells = []
    for split in ('dev', 'test'):
        for metric in ([f'lift|{HEADLINE_RULE}|k{k}' for k in args.topk]
                       + [f'hard_lift|{HEADLINE_RULE}|k{k}' for k in args.topk]):
            av, bv = cand_a[f'{split}|mean|{metric}'], cand_b[f'{split}|mean|{metric}']
            if av is None or bv is None:
                continue
            asd = cand_a[f'{split}|std|{metric}'] or 0.0
            bsd = cand_b[f'{split}|std|{metric}'] or 0.0
            diff = bv - av
            spread = (asd ** 2 + bsd ** 2) ** 0.5
            sigma = (diff / spread) if spread > 1e-9 else None
            compare_cells.append({
                'split': split, 'metric': metric,
                'hard_layer': metric.startswith('hard_'),
                a_mean: av, a_std: round(asd, 4),
                b_mean: bv, b_std: round(bsd, 4),
                'diff_b_minus_a': round(diff, 4),
                'combined_std': round(spread, 4),
                'sigma': round(sigma, 2) if sigma is not None else None,
                'verdict': ('구분됨' if sigma is not None and abs(sigma) >= 2.0
                            else '구분 안 됨'),
                'higher': ((a_label if diff < 0 else b_label)
                           if abs(diff) > 1e-9 else '동일')})
    decided = [c for c in compare_cells
               if c['split'] == 'test' and not c['hard_layer']
               and c['verdict'] == '구분됨']
    head_to_head = {
        'a': a_label, 'b': b_label,
        'varied': f"feature 목록 ({cand_a['n_features']}개 vs {cand_b['n_features']}개)",
        'held_constant': list(HELD_CONSTANT),
        'decision_cells': [f'lift|{HEADLINE_RULE}|k{k}' for k in args.topk],
        'hard_layer_is_reference_only': True,
        'sigma_threshold': 2.0,
        'cells': compare_cells,
        'test_cells_separating_them': [c['metric'] for c in decided],
        'conclusion': ('test 에서도 두 집합은 구분되지 않는다'
                       if not decided else
                       '구분되는 칸이 있다: ' + ', '.join(c['metric'] for c in decided)),
        'note': ('어려운 층은 test 에서 표본이 작아 잡음이 크므로 판정 칸에 넣지 '
                 '않았습니다. 참고 수치입니다.')}
    audit.save_csv(out / 'head_to_head.csv', compare_cells,
                   ('split', 'metric', 'hard_layer', a_mean, a_std, b_mean, b_std,
                    'diff_b_minus_a', 'combined_std', 'sigma', 'verdict', 'higher'))

    changed = []
    for source in sources:
        try:
            info = Path(source['path']).stat()
            if (info.st_size, info.st_mtime_ns) != (source['size_bytes'], source['mtime_ns']):
                changed.append(source['path'])
        except OSError:
            changed.append(source['path'])

    status = 'TEST_EVAL_COMPLETE' if not changed else 'CHECK_REQUIRED'
    report = {
        'version': VERSION, 'status': status, 'output_dir': str(out),
        'test_opened': True, 'test_single_use_warning_acknowledged': True,
        'matrix': str(args.matrix), 'union_dir': str(args.union_dir),
        'selection_dir': str(args.selection_dir), 'sources': sources,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'support_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (_union_path, union._combination_path,
                                     combination._selection_path,
                                     selection._combined_path, combined._graded_path,
                                     graded._ranking_path, ranking._baseline_path,
                                     baseline._support_path, roles._support_path,
                                     audit._support_path)},
        'python': sys.executable, 'xgboost': xgboost.__version__,
        'comparison': {'candidates': list(CANDIDATES),
                       'reference_for_single_cells': REFERENCE_LABEL,
                       'varied': 'feature 목록',
                       'held_constant': list(HELD_CONSTANT),
                       'source': str(Path(args.union_dir) / 'union_results.csv')},
        'configs': [{'label': c['label'], 'role': c['role'], 'note': c['note'],
                     'n_features': len(c['features']), 'features': c['features']}
                    for c in configs],
        'adopted_33_decomposition': {'standalone': standalone,
                                     'combination_only': combo,
                                     'decision_in_34': final_data.get('decision'),
                                     'source': str(Path(args.union_dir)
                                                   / 'final_feature_set.json')},
        'locked_config': {'features': 'raw', 'relevance': 'graded',
                          'relevance_bins': args.relevance_bins,
                          'objective': 'rank:ndcg', 'selection_rule': HEADLINE_RULE},
        'model_params': {'n_estimators': args.n_estimators, 'max_depth': args.max_depth,
                         'learning_rate': args.learning_rate, 'seeds': seeds},
        'topk': list(args.topk), 'shuffles': args.shuffles,
        'splits': {name: {'poses': int(len(parts[name]['frame'])),
                          'systems': int(len(set(parts[name]['frame']['system_id']))),
                          'eligible': len(parts[name]['groups']),
                          'hard_systems': parts[name]['controls']['hard_systems'],
                          'hard_cut': selection.HARD_CUT,
                          'hard_definition': (f'무작위 Top-{HEADLINE_K} 성공확률 < '
                                              f'{selection.HARD_CUT} 인 시스템'),
                          'random_analytic': parts[name]['controls']['analytic'],
                          'random_analytic_hard': parts[name]['controls']['hard_analytic']}
                   for name in ('dev', 'test')},
        'results': rows, 'strata_profile': strat_rows, 'contribution': contrib,
        'head_to_head': head_to_head,
        'combination_survived': combo_check, 'dev_to_test_drop': expectation,
        'changed_sources': changed, 'elapsed_seconds': time.perf_counter() - started,
        'policy': ['test 는 이 실행에서 처음 열렸고, 한 번만 쓸 수 있습니다.',
                   'feature 집합만 변인이고 나머지 조건은 held_constant 에 적힌 대로 '
                   '네 집합이 전부 동일합니다.',
                   'adopted_33 과 union_all12 가 비교 대상이고 anchor_only 와 '
                   'top10_32 는 비교선입니다.',
                   '네 집합의 feature 목록을 34번 union_results.csv 에서 읽었고 '
                   'final_feature_set.json 과 chosen_features.json 에 대조했습니다.',
                   '판정 칸은 전체 k=1/3/5 입니다. 어려운 층은 test 표본이 작아 '
                   '참고 수치로만 적었습니다.',
                   'dev 는 18번 폴드로 out-of-fold, test 는 dev 전체로 학습한 모델 1개로 '
                   '예측했습니다.',
                   '시드를 바꿔 평균·표준편차를 냈고 시드 중 최고를 고르지 않았습니다.',
                   '선택 규칙의 이득과 모델의 이득을 분리해 보고했습니다(§8 의무).',
                   '동점은 pose_id SHA1 오름차순으로 결정적으로 깼습니다.',
                   '사전 예고치(k=1 0.011 / k=5 0.006)는 test 를 보기 전에 정한 참고선입니다.'],
        'notes': ['head_to_head 가 이 실행의 주된 답입니다. adopted_33 과 union_all12 를 '
                  'feature 집합만 바꿔 맞댄 결과이며, 구분되지 않는다는 결과도 유효한 '
                  '답입니다(그 경우 feature 수가 적은 adopted_33 이 간결합니다).',
                  'combination_survived 는 둘째 질문입니다. dev 에서 +0.0069 였던 '
                  '조합의 기여가 test 에서 유지되는지 봅니다. 사라지면 "조합 탐색으로 얻은 '
                  '개선은 dev 과적합이었다"가 결론이며 그것도 유효한 결과입니다.',
                  'dev_to_test_drop 을 사전 예고치와 비교하십시오. 예고보다 크면 선택 '
                  '편향이 예상보다 컸다는 뜻입니다.',
                  'contribution.csv 의 model_gain 이 모델 자체의 공헌이고 diversity_gain 은 '
                  '생성기 4종이라는 데이터 구조가 주는 이득입니다. 둘을 합쳐 보고하면 '
                  '데이터 구조의 공로를 모델에 돌리게 됩니다.']}
    helper.write_json(out / 'test_report.json', report)
    helper.write_json(out / 'RUN_COMPLETE.json', {
        'outputs_written': True, 'test_eval_finished': True, 'status': status,
        'test_opened': True, 'candidates': list(CANDIDATES),
        'test_k1_lift': {r['label']: r[f'test|{k1m}'] for r in rows},
        'test_k5_lift': {r['label']: r[f'test|{k5m}'] for r in rows},
        'head_to_head_conclusion': head_to_head['conclusion'],
        'separating_cells': head_to_head['test_cells_separating_them'],
        'combination_survived': {label: per.get('test')
                                 for label, per in (combo_check or {}).items()},
        'sources_unchanged': not changed})

    print('', flush=True)
    print('STATUS:', status, flush=True)
    print('', flush=True)
    print('=== 집합 4종 — dev 대비 test (규칙 A, 전체 타깃) ===', flush=True)
    head = f"{'집합':13s}{'n':>3s}"
    for k in args.topk:
        head += f"{'dev k' + str(k):>9s}{'test k' + str(k):>9s}"
    print(head, flush=True)
    for row_of in rows:
        line = f"{row_of['label']:13s}{row_of['n_features']:3d}"
        for k in args.topk:
            metric = f'mean|lift|{HEADLINE_RULE}|k{k}'
            line += f"{row_of[f'dev|{metric}']:+9.4f}{row_of[f'test|{metric}']:+9.4f}"
        print(line, flush=True)

    print('', flush=True)
    print(f'=== 정면 비교: {a_label} vs {b_label} (feature 집합만 다름) ===', flush=True)
    print(f"{'분할':6s}{'칸':24s}{a_label:>13s}{b_label:>13s}"
          f"{'차이':>10s}{'σ':>7s}  판정", flush=True)
    for cell in compare_cells:
        mark = ' (참고)' if cell['hard_layer'] else ''
        sig = cell['sigma']
        print(f"{cell['split']:6s}{cell['metric']:24s}"
              f"{cell[a_mean]:+13.4f}{cell[b_mean]:+13.4f}"
              f"{cell['diff_b_minus_a']:+10.4f}"
              f"{(sig if sig is not None else 0):7.2f}  {cell['verdict']}{mark}",
              flush=True)
    print('', flush=True)
    print(f"판정 칸: 전체 k={list(args.topk)} (어려운 층은 참고)", flush=True)
    print(f"결론: {head_to_head['conclusion']}", flush=True)

    print('', flush=True)
    print(f'=== dev 에서 test 로의 하락 (사전 예고 k=1 {EXPECTED_DROP_K1} / '
          f'k=5 {EXPECTED_DROP_K5}) ===', flush=True)
    print(f"{'집합':13s}{'k1 하락':>10s}{'k5 하락':>10s}  예고 범위(2배 이내)", flush=True)
    for row_of in rows:
        d = drops[row_of['label']]
        ok = ('k1 OK' if d['k1_within_expectation'] else 'k1 초과')
        ok += ' / ' + ('k5 OK' if d['k5_within_expectation'] else 'k5 초과')
        print(f"{row_of['label']:13s}{d['drop_k1']:+10.4f}{d['drop_k5']:+10.4f}  {ok}",
              flush=True)

    if combo_check:
        print('', flush=True)
        print('=== 조합의 기여가 test 에서 살아남았는가 (기준선 anchor_only) ===',
              flush=True)
        print(f"{'집합':13s}{'분할':6s}{'단독7개':>10s}{'후보':>10s}"
              f"{'조합 기여':>11s}{'σ':>7s}", flush=True)
        for label, per in combo_check.items():
            for split in ('dev', 'test'):
                c = per[split]
                print(f"{label:13s}{split:6s}{c['anchor_k1']:+10.4f}"
                      f"{c['candidate_k1']:+10.4f}{c['combination_gain_k1']:+11.4f}"
                      f"{(c['sigma'] if c['sigma'] is not None else 0):7.2f}", flush=True)

    print('', flush=True)
    print(f'=== 공헌도 분해 (test, 규칙 A, {REFERENCE_LABEL}) ===', flush=True)
    print(f"{'k':>3s}{'다양성':>10s}{'모델':>10s}{'합계':>10s}{'모델타깃':>10s}", flush=True)
    for row in contrib:
        if (row['label'] != REFERENCE_LABEL or row['split'] != 'test'
                or row['rule'] != HEADLINE_RULE):
            continue
        print(f"{row['topk']:3d}{row['diversity_gain']:+10.4f}{row['model_gain']:+10.4f}"
              f"{row['total_gain']:+10.4f}{row['model_targets']:+10.1f}", flush=True)
    return 0 if status == 'TEST_EVAL_COMPLETE' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--open-test', action='store_true',
                        help='test 923 타깃을 엽니다. 한 번만 쓸 수 있으므로 '
                             '이 플래그 없이는 실행되지 않습니다.')
    parser.add_argument('--matrix', type=Path, default=DEFAULT_MATRIX)
    parser.add_argument('--union-dir', type=Path, default=DEFAULT_UNION,
                        help='34번 결과 폴더(union_results.csv 에서 집합 4종을 읽습니다).')
    parser.add_argument('--selection-dir', type=Path, default=DEFAULT_SELECTION,
                        help='32번 결과 폴더(top10_32 대조 확인용).')
    parser.add_argument('--relevance-bins', choices=tuple(combined.BIN_SETS),
                        default='asymmetric')
    parser.add_argument('--topk', type=combined.parse_topk, default=(1, 3, 5))
    parser.add_argument('--shuffles', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--n-estimators', type=int, default=400)
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261102)
    parser.add_argument('--tag', type=str, default=None)
    args = parser.parse_args()
    for name in ('shuffles', 'repeats', 'n_estimators', 'max_depth', 'threads'):
        if getattr(args, name) < 1:
            parser.error(f'--{name.replace("_", "-")} 는 1 이상이어야 합니다.')
    args.matrix = args.matrix.absolute()
    args.union_dir = args.union_dir.absolute()
    args.selection_dir = args.selection_dir.absolute()
    return run(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
