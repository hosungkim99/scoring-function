#!/usr/bin/env python3
"""Process B 산출물(12). 확정된 scoring function 을 쓸 수 있는 파일로 굳힙니다.

이 스크립트는 **설계 결정을 하나도 새로 하지 않습니다.** 28~36번이 정한 레시피를 그대로
받아 학습된 모델과 추론 코드로 만듭니다. 고를 것이 없으므로 test 에서 아무것도 선택하지
않습니다. test 는 "저장한 모델이 35번 수치를 재현하는가"를 확인하는 데만 쓰입니다.

받아오는 레시피 (전부 앞 단계에서 확정)
  feature 7개      34번 union_results.csv 의 anchor_only (36번에서 추가가 무의미함을 확인)
  목적함수         rank:ndcg, 그룹 = system_id                      (29번)
  등급 경계        비대칭 <1A 11 / <2A 10 / <3A 2 / <5A 1            (30·31번)
  트리 인자        n_estimators 400 / max_depth 6 / lr 0.05          (28번 이후 고정)
  선택 규칙        규칙 A — 생성기별 1위 먼저, 남은 자리는 점수순     (31번)
  동점 처리        pose_id SHA1 오름차순                              (28번)

출하 모델을 미리 정합니다 (사전 등록)
  시드 5개로 각각 학습해 전부 저장하되, **기준 시드(--seed) 모델 하나를 출하 모델로
  지정**합니다. 5개 평균(앙상블)의 성적도 함께 재어 보고하지만, 성적을 보고 출하 모델을
  바꾸지 않습니다. 둘 다 측정해 적는 것이지 고르는 것이 아닙니다.

검증 (이 스크립트의 절반은 검증입니다)
  ① 저장한 model.json 을 디스크에서 다시 읽어 점수가 학습 직후와 같은지
  ② 그 모델로 test 를 다시 재어 35번의 anchor_only 수치가 재현되는지
  ③ feature 순서를 일부러 섞으면 점수가 달라지는지
     (달라져야 feature_order.json 이 실제로 필요한 가드임이 증명됩니다)
  ④ predict.py 를 실제로 돌려 이 스크립트의 Top-5 와 같은 답이 나오는지

출력: out/final_model/37_run_freeze[_TAG]/
  model.json              출하 모델 (기준 시드)
  model_seed<N>.json      시드별 모델 5개
  feature_order.json      7개 이름과 순서. 순서가 틀리면 오류 없이 틀린 점수가 납니다
  spec.json               레시피 전문 (등급 경계·목적함수·규칙 A·동점 키)
  provenance.json         학습에 쓴 시스템 수·입력 해시·성적·출처 단계
  predict.py              독립 실행 추론 스크립트 (28~36번 불필요)
  selfcheck.json          위 검증 4종의 결과
  freeze_report.json / RUN_COMPLETE.json
"""
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

# 35번을 통해 34~28번까지 불러옵니다(학습·평가 코드를 그대로 씁니다).
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

VERSION = 'freeze_0.1.0'
OUT_KIND = 'final_model'
OUT_BASE = '37_run_freeze'
KEYS = baseline.KEYS
BASE = baseline.BASE
DEFAULT_MATRIX = baseline.DEFAULT_MATRIX
RULES = combined.RULES
HEADLINE_RULE = selection.HEADLINE_RULE
HEADLINE_K = selection.HEADLINE_K
DEFAULT_UNION = BASE / 'union' / '34_run_union_final_full'
DEFAULT_TEST_EVAL = BASE / 'test_eval' / '35_run_test_final'
# 굳힐 집합. 36번에서 "이 위에 무엇을 얹어도 같다"가 확인된 집합입니다.
FINAL_LABEL = 'anchor_only'
# 재현 점검 허용 오차. 같은 데이터·같은 시드·같은 인자이므로 원래는 정확히 같아야 하지만,
# xgboost hist 는 스레드 수에 따라 마지막 자리가 흔들릴 수 있어 여유를 둡니다.
REPRODUCE_TOL = 0.002
# 저장·재적재 점수 일치 허용 오차. 같은 트리이므로 부동소수 오차만 허용합니다.
ROUNDTRIP_TOL = 1e-5
PREREGISTRATION = (
    '이 실행은 레시피를 받아 적을 뿐이며 설계 결정을 새로 하지 않는다.',
    f'굳히는 집합은 {FINAL_LABEL}(7개)이고 36번 결과에 따라 사전에 정해졌다.',
    '출하 모델은 기준 시드 모델 하나로 미리 지정한다. 성적을 보고 바꾸지 않는다.',
    '5개 평균(앙상블) 성적도 함께 적지만 선택 후보가 아니라 참고 수치다.',
    'test 는 저장한 모델이 35번 수치를 재현하는지 확인하는 데만 쓴다.')


def load_final_features(union_dir):
    """34번 union_results.csv 에서 최종 집합을 읽습니다(손으로 적지 않습니다)."""
    import csv
    path = Path(union_dir) / 'union_results.csv'
    if not path.exists():
        raise FileNotFoundError(f'34번 결과가 없습니다: {path}')
    found = None
    with path.open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            if row.get('label') == FINAL_LABEL:
                joined = (row.get('features_joined') or '').strip()
                found = sorted(f for f in joined.split('|') if f)
                break
    if not found:
        raise ValueError(f'union_results.csv 에서 {FINAL_LABEL} 을 읽지 못했습니다: {path}')
    return found


def load_reference(test_eval_dir):
    """35번이 보고한 anchor_only 수치를 읽습니다(재현 점검의 기준값)."""
    import csv
    path = Path(test_eval_dir) / 'dev_vs_test.csv'
    if not path.exists():
        raise FileNotFoundError(f'35번 결과가 없습니다: {path}')
    row = None
    with path.open(encoding='utf-8', newline='') as handle:
        for item in csv.DictReader(handle):
            if item.get('label') == FINAL_LABEL:
                row = item
                break
    if row is None:
        raise ValueError(f'35번 결과에 {FINAL_LABEL} 이 없습니다.')
    out = {}
    for split in ('dev', 'test'):
        for stat in ('mean', 'std'):
            for k in (1, 3, 5):
                key = f'{split}|{stat}|lift|{HEADLINE_RULE}|k{k}'
                raw = row.get(key, '')
                out[key] = float(raw) if raw not in ('', None) else None
    out['n_features'] = int(row['n_features'])
    out['features_joined'] = row.get('features_joined', '')
    report_path = Path(test_eval_dir) / 'test_report.json'
    report = (json.loads(report_path.read_text(encoding='utf-8'))
              if report_path.exists() else {})
    return out, report


def train_and_keep(X, rel, codes, known, args):
    """dev 전체로 모델 하나를 학습하고 모델 객체를 그대로 돌려줍니다.

    35번 train_full 과 같은 전처리(쌍이 안 생기는 시스템 제외)를 씁니다. 다른 점은
    모델을 버리지 않고 반환한다는 것뿐입니다.
    """
    import numpy as np
    keep = known.copy()
    _, dead = graded.pairable_systems(codes, rel, keep)
    if dead.size:
        keep = keep & ~np.isin(codes, dead)
    model = graded.build_ranker(args)
    model.fit(X[keep], rel[keep], qid=codes[keep])
    return model, {'train_groups': int(np.unique(codes[keep]).size),
                   'train_poses': int(keep.sum()),
                   'dropped_groups': int(dead.size)}


def lift_of(scores, groups, controls, topk):
    """점수 배열에서 규칙별 x k lift 를 냅니다(35번과 같은 함수를 씁니다)."""
    rates, hard_rates, _ = selection.evaluate_scores(groups, scores, topk)
    return test_eval.cells(rates, hard_rates, controls, topk)


def topk_table(frame, scores, groups, k):
    """규칙 A 로 시스템마다 상위 k개를 골라 표로 만듭니다(predict.py 와 대조용).

    groups 에 든 시스템만 돕니다. 근접 pose 가 하나도 없는 시스템은 평가 대상이
    아니므로 groups 에서 빠져 있고(32번 prepare_groups), 대조 범위도 그에 맞춥니다.
    """
    import numpy as np
    rows = []
    pose_id = frame['pose_id'].to_numpy()
    for g in groups:
        start, size = g['start'], g['size']
        chunk = np.where(np.isnan(scores[start:start + size]), -np.inf,
                         scores[start:start + size])
        order = np.lexsort((g['tie'], -chunk))
        gens = [str(g['gen'][i]) for i in order]
        picks = combined.rule_order(gens, HEADLINE_RULE, k)
        for rank, pos in enumerate(picks, start=1):
            local = int(order[pos])
            idx = start + local
            rows.append({'system_id': str(g['system_id']), 'rank': rank,
                         'pose_id': str(pose_id[idx]),
                         'generator': str(g['gen'][local]),
                         'score': round(float(scores[idx]), 6)})
    return rows


def eligible_rows(groups):
    """groups 에 든 시스템의 행 번호를 모읍니다(대조용 입력을 같은 범위로 만들기)."""
    import numpy as np
    pieces = [np.arange(g['start'], g['start'] + g['size'], dtype=np.int64)
              for g in groups]
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)


def run(args):
    """레시피를 받아 모델을 학습·저장하고, 저장한 파일로 재현을 검증합니다."""
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
    print('*** 사전 등록 — 이 실행은 아무것도 고르지 않습니다. ***', flush=True)
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

    final_features = load_final_features(args.union_dir)
    reference, ref_report = load_reference(args.test_eval_dir)
    if reference['n_features'] != len(final_features):
        raise ValueError(f'34번({len(final_features)}개)과 35번'
                         f'({reference["n_features"]}개)의 집합 크기가 다릅니다.')
    ref_feats = sorted(f for f in (reference['features_joined'] or '').split('|') if f)
    if ref_feats and ref_feats != final_features:
        raise ValueError('34번과 35번의 feature 목록이 다릅니다.')
    print(f'굳힐 집합: {FINAL_LABEL} — feature {len(final_features)}개', flush=True)
    for i, name in enumerate(final_features, start=1):
        print(f'  {i}. {name}', flush=True)

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
    missing = sorted(set(final_features) - set(feats))
    if missing:
        raise ValueError(f'현재 행렬에 없는 feature: {missing}')
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
        # 학습·추론에 쓰는 행렬은 최종 집합의 열만, 사전에 정한 순서로 만듭니다.
        parts[name] = {'frame': sub, 'codes': codes, 'rel': rel, 'known': known,
                       'X': sub[final_features].to_numpy(dtype=np.float32,
                                                         na_value=np.nan),
                       'groups': groups, 'controls': controls}
        print(f'{name.upper()}: poses {len(sub)} | systems '
              f'{int(np.unique(codes).size)} | eligible {len(groups)} '
              f'| 어려운 층 {controls["hard_systems"]}', flush=True)

    dev, test = parts['dev'], parts['test']
    # 35번과 같은 파이프라인인지 대조군으로 확인합니다.
    consistency = {'checked': False, 'mismatches': []}
    for name in ('dev', 'test'):
        want = ((ref_report.get('splits') or {}).get(name) or {})
        if not want:
            continue
        consistency['checked'] = True
        if want.get('eligible') not in (None, len(parts[name]['groups'])):
            consistency['mismatches'].append(f'{name}.eligible')
        for k, value in (want.get('random_analytic') or {}).items():
            mine = parts[name]['controls']['analytic'].get(int(k))
            if mine is not None and abs(float(value) - mine) > 1e-9:
                consistency['mismatches'].append(f'{name}.random_analytic.k{k}')
    if consistency['mismatches']:
        raise ValueError('35번과 대조군이 다릅니다(파이프라인 불일치): '
                         + ', '.join(consistency['mismatches']))
    if consistency['checked']:
        print('35번 대조군과 일치 확인 — 같은 파이프라인입니다.', flush=True)

    seeds = [args.seed + i for i in range(args.repeats)]
    designated_seed = seeds[0]
    print('', flush=True)
    print(f'학습 (트리 {args.n_estimators}, 시드 {seeds})', flush=True)
    print(f'출하 모델로 미리 지정한 시드: {designated_seed}', flush=True)

    import copy
    per_seed, dev_cells, test_cells = [], [], []
    dev_scores, test_scores = [], []
    designated_model, train_info = None, None
    for sd in seeds:
        variant = copy.copy(args)
        variant.seed = sd
        model, info = train_and_keep(dev['X'], dev['rel'], dev['codes'],
                                     dev['known'], variant)
        ds = model.predict(dev['X'])
        ts = model.predict(test['X'])
        dev_scores.append(ds)
        test_scores.append(ts)
        dcell = lift_of(ds, dev['groups'], dev['controls'], args.topk)
        tcell = lift_of(ts, test['groups'], test['controls'], args.topk)
        dev_cells.append(dcell)
        test_cells.append(tcell)
        path = helper.checked_path(out / f'model_seed{sd}.json')
        model.get_booster().save_model(str(path))
        per_seed.append({'seed': sd, 'file': path.name,
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'bytes': path.stat().st_size, **info,
                         'dev_in_sample_lift_k1':
                             dcell[f'lift|{HEADLINE_RULE}|k1'],
                         'test_lift_k1': tcell[f'lift|{HEADLINE_RULE}|k1'],
                         'test_lift_k3': tcell[f'lift|{HEADLINE_RULE}|k3'],
                         'test_lift_k5': tcell[f'lift|{HEADLINE_RULE}|k5']})
        print(f"  시드 {sd} | 학습 시스템 {info['train_groups']} "
              f"| dev(암기) k1 {dcell[f'lift|{HEADLINE_RULE}|k1']:+.4f} "
              f"| test k1 {tcell[f'lift|{HEADLINE_RULE}|k1']:+.4f}", flush=True)
        if sd == designated_seed:
            designated_model, train_info = model, info

    # 출하 모델을 model.json 으로 한 번 더 저장합니다(시드별 파일과 내용 동일).
    model_path = helper.checked_path(out / 'model.json')
    designated_model.get_booster().save_model(str(model_path))
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    seed_sha = next(r['sha256'] for r in per_seed if r['seed'] == designated_seed)

    # 시드 평균(앙상블)도 재어 적습니다. 선택 후보가 아니라 참고 수치입니다.
    ens_dev = np.mean(np.vstack(dev_scores), axis=0)
    ens_test = np.mean(np.vstack(test_scores), axis=0)
    ensemble = {'dev': lift_of(ens_dev, dev['groups'], dev['controls'], args.topk),
                'test': lift_of(ens_test, test['groups'], test['controls'], args.topk)}

    def agg(cells, key):
        vals = [c[key] for c in cells if c[key] is not None]
        return ((round(float(np.mean(vals)), 4),
                 round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else None)
                if vals else (None, None))

    # ── 검증 ① 저장한 파일을 다시 읽어 점수가 같은가
    booster = xgboost.Booster()
    booster.load_model(str(model_path))
    dmat = xgboost.DMatrix(test['X'], feature_names=None)
    reloaded = booster.predict(dmat)
    direct = designated_model.predict(test['X'])
    gap = float(np.max(np.abs(reloaded - direct)))
    roundtrip_ok = bool(gap <= ROUNDTRIP_TOL)

    # ── 검증 ② 다시 읽은 모델로 test 를 재어 35번 수치가 재현되는가
    reloaded_cells = lift_of(reloaded, test['groups'], test['controls'], args.topk)
    reproduce = []
    for k in args.topk:
        key = f'lift|{HEADLINE_RULE}|k{k}'
        mine_mean, mine_std = agg(test_cells, key)
        want = reference[f'test|mean|lift|{HEADLINE_RULE}|k{k}']
        diff = (round(mine_mean - want, 4)
                if (mine_mean is not None and want is not None) else None)
        reproduce.append({
            'metric': key, 'reference_35': want, 'here_mean': mine_mean,
            'here_std': mine_std, 'diff': diff,
            'designated_seed_value': reloaded_cells[key],
            'within_tolerance': bool(diff is not None and abs(diff) <= REPRODUCE_TOL)})
    reproduce_ok = all(r['within_tolerance'] for r in reproduce)

    # ── 검증 ③ feature 순서를 섞으면 점수가 달라지는가
    order_guard = {'checked': False}
    if len(final_features) > 1:
        shuffled = test['X'][:, [1, 0] + list(range(2, len(final_features)))]
        bad = booster.predict(xgboost.DMatrix(shuffled))
        changed = float(np.max(np.abs(bad - reloaded)))
        order_guard = {
            'checked': True, 'swapped_columns': [final_features[0], final_features[1]],
            'max_score_change': round(changed, 6),
            'scores_changed': bool(changed > ROUNDTRIP_TOL),
            'note': ('열 순서를 바꾸면 점수가 달라지고 오류는 나지 않습니다. '
                     'feature_order.json 을 반드시 지켜야 하는 이유입니다.')}

    # 추론에 필요한 파일들을 씁니다.
    helper.write_json(out / 'feature_order.json', {
        'features': final_features, 'n_features': len(final_features),
        'order_matters': True,
        'note': ('xgboost 는 열 이름을 보지 않고 위치만 봅니다. 이 순서대로 행렬을 '
                 '만들지 않으면 오류 없이 틀린 점수가 나옵니다.'),
        'source': str(Path(args.union_dir) / 'union_results.csv'),
        'source_label': FINAL_LABEL})
    spec = {
        'version': VERSION,
        'name': 'co-folding pose 재순위화기 (GT-free Top-k 선별)',
        'scoring_function': {
            'stage1_model': {
                'kind': 'xgboost booster (rank:ndcg)', 'file': 'model.json',
                'features': final_features,
                'objective': 'rank:ndcg', 'group': 'system_id',
                'n_estimators': args.n_estimators, 'max_depth': args.max_depth,
                'learning_rate': args.learning_rate,
                'subsample': 0.8, 'colsample_bytree': 0.8,
                'tree_method': 'hist', 'ndcg_exp_gain': False,
                'lambdarank_pair_method': 'topk',
                'missing': 'NaN 그대로 넣습니다. 0 으로 채우면 안 됩니다.'},
            'stage2_rule': {
                'name': HEADLINE_RULE,
                'definition': ('점수 내림차순으로 순회하며 생성기가 처음 나오면 먼저 '
                               '담고, 이미 쓴 생성기는 뒤로 보냅니다. 그 순서에서 앞 k개.'),
                'generator_column': 'model',
                'why': ('모델은 1등은 잘 고르지만 2~5등을 고르면 비슷한 포즈끼리 '
                        '뭉칩니다. 이 규칙이 test k=3/5 이득의 대부분을 법니다.'),
                'tie_break': 'pose_id SHA1 오름차순 (생성기와 무관해야 공평합니다)'}},
        'training_only': {
            'relevance_bins': {'scheme': args.relevance_bins,
                               'bins': [list(b) for b in bins],
                               'why': ('평가는 2A 이진이므로 성공 경계만 크게 벌리고 '
                                       '1A 과 1.9A 는 거의 구별하지 않습니다.')},
            'label_source': 'sc_rmsd', 'mask': 'label_known',
            'note': 'sc_rmsd 는 학습에만 쓰입니다. 추론에는 정답이 필요 없습니다.'},
        'evaluation': {'metric': '타깃별 Top-k 안에 sc_rmsd < 2A pose 가 1개 이상',
                       'strict_less_than': True,
                       'selection_rule_for_reported_numbers': HEADLINE_RULE},
        'designated_seed': designated_seed,
        'ensemble_is_reference_only': True,
        'provenance_steps': {
            '28': '기준선. NaN 유지, 0 채우기 금지 결정',
            '29': '랭킹 목적함수와 그룹=시스템',
            '30': '등급 relevance',
            '31': '2x2 결합과 선택 규칙 A/B',
            '32': 'feature 수 사다리 (5개와 57개가 같음)',
            '33': '조합 탐색 (1,294개)',
            '34': '합집합 비교, 7집합',
            '35': 'test 923 평가 (조합 기여 재현 실패 확인)',
            '36': '부분집합 분포 (7개 위 추가는 무의미)',
            '37': '이 실행 — 굳히기'}}
    helper.write_json(out / 'spec.json', spec)

    provenance = {
        'version': VERSION, 'frozen_at_elapsed_s': None,
        'final_label': FINAL_LABEL, 'features': final_features,
        'matrix': str(args.matrix), 'sources': sources,
        'union_dir': str(args.union_dir), 'test_eval_dir': str(args.test_eval_dir),
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
        'trained_on': {'split': 'dev',
                       'systems_in_split': int(len(set(dev['frame']['system_id']))),
                       'poses_in_split': int(len(dev['frame'])),
                       'groups_used': train_info['train_groups'],
                       'poses_used': train_info['train_poses'],
                       'groups_dropped_no_pair': train_info['dropped_groups'],
                       'note': ('등급이 전부 같은 시스템은 기울기가 0이라 학습에서 '
                                '빠집니다(30번 pairable_systems).')},
        'held_out': {'split': 'test',
                     'systems': int(len(set(test['frame']['system_id']))),
                     'eligible': len(test['groups']),
                     'hard_systems': test['controls']['hard_systems'],
                     'never_used_for_selection': True},
        'models': per_seed, 'designated': {'seed': designated_seed,
                                           'file': 'model.json',
                                           'sha256': model_sha,
                                           'same_as': f'model_seed{designated_seed}.json',
                                           'sha256_matches_seed_file':
                                               bool(model_sha == seed_sha)},
        'performance': {}}
    # dev 는 학습에 쓴 분할이므로 여기서 잰 dev 값은 in-sample(암기) 수치입니다.
    # 성능 추정이 아니므로 이름에 못 박고, 정직한 dev 값(35번 out-of-fold)을 옆에 둡니다.
    label_of = {'dev': 'dev_in_sample', 'test': 'test'}
    for split, cells in (('dev', dev_cells), ('test', test_cells)):
        tag = label_of[split]
        for k in args.topk:
            key = f'lift|{HEADLINE_RULE}|k{k}'
            mean, std = agg(cells, key)
            entry = {
                'seed_mean': mean, 'seed_std': std,
                'designated_seed': next(c[key] for c, s in zip(cells, seeds)
                                        if s == designated_seed),
                'ensemble_reference_only': ensemble[split][key]}
            if split == 'dev':
                oof = reference[f'dev|mean|lift|{HEADLINE_RULE}|k{k}']
                entry.update({
                    'is_in_sample': True,
                    'not_a_performance_estimate': True,
                    'dev_oof_from_35': oof,
                    'memorization_gap': (round(mean - oof, 4)
                                         if (mean is not None and oof is not None)
                                         else None),
                    'note': ('dev 전체로 학습한 뒤 그 dev 를 다시 예측한 값입니다. '
                             '모델이 이미 본 데이터라 성능이 아니라 암기 정도를 '
                             '나타냅니다. dev 성능은 dev_oof_from_35 를 보십시오.')})
            provenance['performance'][f'{tag}|{key}'] = entry
            hk = f'hard_lift|{HEADLINE_RULE}|k{k}'
            hmean, hstd = agg(cells, hk)
            hentry = {'seed_mean': hmean, 'seed_std': hstd, 'reference_only': True,
                      'note': 'test 어려운 층은 61개뿐이라 판정 근거로 쓰지 않습니다.'}
            if split == 'dev':
                hentry['is_in_sample'] = True
                hentry['not_a_performance_estimate'] = True
            provenance['performance'][f'{tag}|{hk}'] = hentry
    # 정직한 dev 성능은 35번 out-of-fold 값입니다(이 실행이 다시 재지 않습니다).
    provenance['dev_performance_out_of_fold'] = {
        f'lift|{HEADLINE_RULE}|k{k}': {
            'mean': reference[f'dev|mean|lift|{HEADLINE_RULE}|k{k}'],
            'std': reference[f'dev|std|lift|{HEADLINE_RULE}|k{k}']}
        for k in args.topk}
    provenance['dev_performance_out_of_fold']['source'] = str(
        Path(args.test_eval_dir) / 'dev_vs_test.csv')
    provenance['dev_performance_out_of_fold']['note'] = (
        '35번이 18번 폴드로 out-of-fold 측정한 값입니다. 이것이 dev 성능이고, '
        'performance 의 dev_in_sample 항목은 암기 수치입니다.')
    provenance['random_baseline'] = {
        split: parts[split]['controls']['analytic'] for split in ('dev', 'test')}
    provenance['frozen_at_elapsed_s'] = round(time.perf_counter() - started, 1)
    helper.write_json(out / 'provenance.json', provenance)

    # predict.py 를 씁니다(28~36번을 불러오지 않는 독립 스크립트).
    predict_path = helper.checked_path(out / 'predict.py')
    predict_path.write_text(PREDICT_SOURCE, encoding='utf-8')

    # ── 검증 ④ predict.py 를 실제로 돌려 같은 Top-5 가 나오는가
    mine_top = topk_table(test['frame'], reloaded, test['groups'], HEADLINE_K)
    sample_path = helper.checked_path(out / 'tmp' / 'verify_poses.parquet')
    cols = ['system_id', 'model', 'seed', 'sample', 'pose_id'] + final_features
    # predict.py 는 파일에 있는 모든 시스템을 묶으므로, 평가 대상(groups)과 같은
    # 범위만 써야 행 수가 맞습니다. 923개 중 근접 pose 가 있는 718개입니다.
    keep_rows = eligible_rows(test['groups'])
    verify_frame = test['frame'].iloc[keep_rows][cols].reset_index(drop=True)
    verify_frame.to_parquet(sample_path, index=False, compression='snappy')
    print('', flush=True)
    print(f'predict.py 대조 입력: {len(verify_frame)}행 / '
          f'{verify_frame["system_id"].nunique()}타깃 (평가 대상과 동일)', flush=True)
    verify_out = helper.checked_path(out / 'tmp' / 'verify_top5.csv')
    proc = subprocess.run(
        [sys.executable, '-B', str(predict_path), '--poses', str(sample_path),
         '--model-dir', str(out), '--k', str(HEADLINE_K), '--out', str(verify_out)],
        capture_output=True, text=True, timeout=1800)
    predict_check = {'returncode': proc.returncode,
                     'stderr_tail': (proc.stderr or '')[-600:]}
    if proc.returncode == 0 and verify_out.exists():
        import csv as _csv
        with verify_out.open(encoding='utf-8', newline='') as handle:
            theirs = [{'system_id': r['system_id'], 'rank': int(r['rank']),
                       'pose_id': r['pose_id']} for r in _csv.DictReader(handle)]
        mine_keys = [(r['system_id'], r['rank'], r['pose_id']) for r in mine_top]
        their_keys = [(r['system_id'], r['rank'], r['pose_id']) for r in theirs]
        same = mine_keys == their_keys
        predict_check.update({'rows_here': len(mine_keys), 'rows_predict': len(their_keys),
                              'identical': bool(same),
                              'first_mismatch': None if same else next(
                                  (f'{a} != {b}' for a, b in zip(mine_keys, their_keys)
                                   if a != b), 'length differs')})
    else:
        predict_check['identical'] = False
        predict_check['first_mismatch'] = 'predict.py 실행 실패'
    predict_ok = bool(predict_check.get('identical'))

    selfcheck = {
        'roundtrip_save_load': {'passed': roundtrip_ok,
                                'max_score_gap': gap, 'tolerance': ROUNDTRIP_TOL,
                                'what': '저장한 model.json 을 다시 읽어 점수가 같은가'},
        'reproduce_35': {'passed': reproduce_ok, 'tolerance': REPRODUCE_TOL,
                         'cells': reproduce,
                         'what': '다시 읽은 모델의 test 성적이 35번 수치와 같은가'},
        'feature_order_guard': order_guard,
        'predict_py_matches': predict_check,
        'all_passed': bool(roundtrip_ok and reproduce_ok and predict_ok
                           and order_guard.get('scores_changed', False))}
    helper.write_json(out / 'selfcheck.json', selfcheck)

    changed = []
    for source in sources:
        try:
            info = Path(source['path']).stat()
            if (info.st_size, info.st_mtime_ns) != (source['size_bytes'],
                                                    source['mtime_ns']):
                changed.append(source['path'])
        except OSError:
            changed.append(source['path'])
    status = ('FREEZE_COMPLETE' if (selfcheck['all_passed'] and not changed)
              else 'CHECK_REQUIRED')

    helper.write_json(out / 'freeze_report.json', {
        'version': VERSION, 'status': status, 'output_dir': str(out),
        'preregistration': list(PREREGISTRATION), 'picks_nothing': True,
        'spec': spec, 'provenance': provenance, 'selfcheck': selfcheck,
        'changed_sources': changed,
        'elapsed_seconds': time.perf_counter() - started,
        'how_to_use': [
            '이 폴더를 그대로 복사해 쓰십시오. 28~36번 스크립트는 필요하지 않습니다.',
            'python predict.py --poses <poses.parquet> --k 5 --out top5.csv',
            'poses 파일에 system_id, model, seed, sample, pose_id 와 '
            'feature_order.json 의 7개 열이 있어야 합니다.',
            'model 열은 생성기 이름이며 규칙 A 가 이 열로 다양성을 셉니다.',
            '결측은 빈 값(NaN)으로 두십시오. 0 으로 채우면 틀린 점수가 납니다.'],
        'limits': [
            'provenance 의 dev_in_sample 수치는 학습 데이터 재예측이며 성능이 '
            '아닙니다. dev 성능은 dev_performance_out_of_fold(35번 측정)를 '
            '인용하십시오. 두 값의 차가 memorization_gap 입니다.',
            '인용할 대표 수치는 test 의 seed_mean 입니다. designated_seed 값은 '
            '출하 모델 1회 실측이라 ±seed_std 만큼의 표본 잡음을 함께 보십시오.',
            '학습은 dev 3,680 타깃 1회입니다. 새 타깃으로 재학습하지 않습니다.',
            'test 923 성적이 기대값이며, 다른 분포의 타깃에서는 달라질 수 있습니다.',
            'feature 7개 중 4개는 gnina·P2Rank 출력입니다. 두 도구가 필요합니다.',
            'k=3/5 이득의 대부분은 규칙 A 몫이고 모델 몫은 k=1 에 몰려 있습니다.',
            '앙상블(시드 5개 평균) 수치는 참고용이며 출하 모델이 아닙니다.']})
    helper.write_json(out / 'RUN_COMPLETE.json', {
        'outputs_written': True, 'freeze_finished': True, 'status': status,
        'picks_nothing': True, 'final_label': FINAL_LABEL,
        'n_features': len(final_features), 'designated_seed': designated_seed,
        'model_file': 'model.json', 'model_sha256': model_sha,
        'selfcheck_all_passed': selfcheck['all_passed'],
        'test_lift': {f'k{k}': provenance['performance'][
            f'test|lift|{HEADLINE_RULE}|k{k}']['seed_mean'] for k in args.topk},
        'sources_unchanged': not changed})

    print('', flush=True)
    print('STATUS:', status, flush=True)
    print('', flush=True)
    print('=== 검증 ===', flush=True)
    print(f"  ① 저장-재적재 점수 일치     {'통과' if roundtrip_ok else '실패'} "
          f"(최대 차 {gap:.2e}, 허용 {ROUNDTRIP_TOL})", flush=True)
    print(f"  ② 35번 수치 재현            {'통과' if reproduce_ok else '실패'} "
          f"(허용 {REPRODUCE_TOL})", flush=True)
    for row in reproduce:
        mark = 'OK' if row['within_tolerance'] else '초과'
        print(f"       {row['metric']:20s} 35번 {row['reference_35']:+.4f} "
              f"| 여기 {row['here_mean']:+.4f} | 차 {row['diff']:+.4f}  {mark}",
              flush=True)
    guard = order_guard.get('scores_changed', False)
    print(f"  ③ feature 순서 가드         {'통과' if guard else '실패'} "
          f"(열 2개 교환 시 점수 최대 변화 "
          f"{order_guard.get('max_score_change', 0)})", flush=True)
    print(f"  ④ predict.py 동일 Top-{HEADLINE_K}       "
          f"{'통과' if predict_ok else '실패'} "
          f"({predict_check.get('rows_predict', 0)}행 대조)", flush=True)

    print('', flush=True)
    print('=== 출하 모델 성적 (규칙 A) ===', flush=True)
    print('  dev(암기)는 학습 데이터를 다시 예측한 값으로 성능이 아닙니다.', flush=True)
    print('  dev 성능은 35번 out-of-fold 열을 보십시오.', flush=True)
    print(f"{'칸':7s}{'dev(OOF/35번)':>14s}{'dev(암기)':>11s}{'암기격차':>10s}"
          f"{'test 평균':>11s}{'test ±':>9s}{'출하시드':>10s}{'앙상블(참고)':>13s}",
          flush=True)
    for k in args.topk:
        key = f'lift|{HEADLINE_RULE}|k{k}'
        d = provenance['performance'][f'dev_in_sample|{key}']
        t = provenance['performance'][f'test|{key}']
        oof = d['dev_oof_from_35']
        gap = d['memorization_gap']
        print(f"k={k:<5d}{(oof if oof is not None else 0):+14.4f}"
              f"{d['seed_mean']:+11.4f}{(gap if gap is not None else 0):+10.4f}"
              f"{t['seed_mean']:+11.4f}{(t['seed_std'] or 0):9.4f}"
              f"{t['designated_seed']:+10.4f}"
              f"{t['ensemble_reference_only']:+13.4f}", flush=True)
    print('', flush=True)
    print(f"무작위 기준 test: "
          f"{[round(test['controls']['analytic'][k], 4) for k in args.topk]}", flush=True)
    print(f"평가 가능 타깃 test: {len(test['groups'])}개 "
          f"(lift x 이 수 = 더 맞춘 타깃 수)", flush=True)

    print('', flush=True)
    print('=== 쓰는 법 ===', flush=True)
    print(f'  cd {out}', flush=True)
    print(f'  python predict.py --poses <poses.parquet> --k {HEADLINE_K} '
          f'--out top5.csv', flush=True)
    print('', flush=True)
    print('*** 이 실행은 아무것도 고르지 않았습니다. 레시피를 파일로 옮겼을 뿐입니다. ***',
          flush=True)
    return 0 if status == 'FREEZE_COMPLETE' else 2


PREDICT_SOURCE = '''#!/usr/bin/env python3
"""저장된 scoring function 으로 pose 묶음에서 Top-k 를 고릅니다. (독립 실행)

필요한 것: xgboost, numpy, pandas. 그리고 이 폴더의 model.json / feature_order.json /
spec.json. 28~36번 파이프라인 스크립트는 **필요하지 않습니다.**

2단 구조입니다.
  1단  feature -> 점수      model.json (rank:ndcg 로 학습된 트리)
  2단  점수 -> Top-k        규칙 A (생성기별 1위 먼저, 남은 자리는 점수순)

쓰는 법
  python predict.py --poses poses.parquet --k 5 --out top5.csv
  python predict.py --poses poses.csv --k 3
  python predict.py --poses poses.parquet --ensemble      # 시드 5개 평균(참고용)

poses 파일에 있어야 하는 열
  system_id   타깃 식별자. 이 열로 묶어 타깃마다 Top-k 를 고릅니다
  model       생성기 이름(af3/bt2/of3/pt2 등). 규칙 A 가 이 열로 다양성을 셉니다
  pose_id     동점을 깨는 키. SHA1 오름차순으로 깹니다
  seed sample 있으면 결과에 함께 싣습니다(없어도 됩니다)
  + feature_order.json 의 feature 열 전부

주의
  결측은 빈 값으로 두십시오. 0 으로 채우면 틀린 점수가 납니다
  (학습 때 NaN 을 NaN 으로 다뤘기 때문입니다).
  열 순서는 이 스크립트가 feature_order.json 을 보고 맞춥니다. 직접 맞추지 마십시오.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xgboost


def tie_key(pose_id):
    """동점을 결정적으로 깨되 생성기와 무관한 키. 학습 단계와 같은 함수입니다."""
    return int(hashlib.sha1(str(pose_id).encode()).hexdigest(), 16)


def rule_a_order(generators, kmax):
    """규칙 A — 점수 내림차순 배열에서 고를 위치를 순서대로 돌려줍니다.

    생성기가 처음 나오면 먼저 담고, 이미 쓴 생성기는 뒤로 보냅니다. 그 순서의 앞 kmax개.
    31번 rule_order 의 ruleA 와 같은 동작입니다.
    """
    seen, first, rest = set(), [], []
    for i, gen in enumerate(generators):
        if gen in seen:
            rest.append(i)
        else:
            seen.add(gen)
            first.append(i)
    return (first + rest)[:kmax]


def load_bundle(model_dir, ensemble):
    """모델과 feature 순서, 규칙 이름을 읽습니다."""
    model_dir = Path(model_dir)
    order_path = model_dir / "feature_order.json"
    spec_path = model_dir / "spec.json"
    for path in (order_path, spec_path):
        if not path.exists():
            raise FileNotFoundError(f"필요한 파일이 없습니다: {path}")
    features = json.loads(order_path.read_text(encoding="utf-8"))["features"]
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    rule = spec["scoring_function"]["stage2_rule"]["name"]
    if rule != "ruleA":
        raise ValueError(f"이 스크립트는 규칙 A 만 구현합니다(spec: {rule}).")
    gen_col = spec["scoring_function"]["stage2_rule"].get("generator_column", "model")
    if ensemble:
        paths = sorted(model_dir.glob("model_seed*.json"))
        if not paths:
            raise FileNotFoundError("model_seed*.json 이 없습니다.")
    else:
        paths = [model_dir / "model.json"]
        if not paths[0].exists():
            raise FileNotFoundError(f"model.json 이 없습니다: {paths[0]}")
    boosters = []
    for path in paths:
        booster = xgboost.Booster()
        booster.load_model(str(path))
        boosters.append(booster)
    return features, boosters, [p.name for p in paths], gen_col


def read_poses(path):
    """parquet 또는 csv 를 읽습니다."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"pose 파일이 없습니다: {path}")
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def score_poses(frame, features, boosters):
    """1단 — feature 행렬을 사전에 정한 순서로 만들어 점수를 냅니다."""
    missing = [f for f in features if f not in frame.columns]
    if missing:
        raise ValueError(f"pose 파일에 없는 feature 열: {missing}")
    matrix = frame[features].to_numpy(dtype=np.float32, na_value=np.nan)
    dmat = xgboost.DMatrix(matrix)
    parts = [booster.predict(dmat) for booster in boosters]
    return np.mean(np.vstack(parts), axis=0) if len(parts) > 1 else parts[0]


def pick_topk(frame, scores, k, gen_col):
    """2단 — 타깃마다 규칙 A 로 상위 k개를 고릅니다."""
    needed = ["system_id", gen_col, "pose_id"]
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise ValueError(f"pose 파일에 없는 열: {missing}")
    extra = [c for c in ("seed", "sample") if c in frame.columns]
    ties = frame["pose_id"].map(tie_key).to_numpy()
    rows = []
    for system, idx in frame.groupby("system_id", sort=False).indices.items():
        idx = np.asarray(idx)
        chunk = np.where(np.isnan(scores[idx]), -np.inf, scores[idx])
        order = np.lexsort((ties[idx], -chunk))
        gens = frame[gen_col].to_numpy()[idx][order]
        for rank, pos in enumerate(rule_a_order(list(gens), k), start=1):
            row_at = idx[order[pos]]
            item = {"system_id": str(system), "rank": rank,
                    "pose_id": str(frame["pose_id"].to_numpy()[row_at]),
                    "generator": str(frame[gen_col].to_numpy()[row_at]),
                    "score": round(float(scores[row_at]), 6)}
            for col in extra:
                item[col] = frame[col].to_numpy()[row_at]
            rows.append(item)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses", required=True, help="pose 표 (parquet 또는 csv).")
    parser.add_argument("--model-dir", default=str(Path(__file__).resolve().parent),
                        help="model.json 등이 있는 폴더. 기본은 이 스크립트 폴더.")
    parser.add_argument("--k", type=int, default=5, help="타깃마다 고를 pose 수.")
    parser.add_argument("--out", default=None, help="결과 csv 경로. 없으면 화면에 씁니다.")
    parser.add_argument("--ensemble", action="store_true",
                        help="시드 5개 평균을 씁니다(참고용. 출하 모델은 단일입니다).")
    parser.add_argument("--scores-out", default=None,
                        help="pose 별 점수 전체를 쓸 csv 경로(선택).")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k 는 1 이상이어야 합니다.")

    features, boosters, used, gen_col = load_bundle(args.model_dir, args.ensemble)
    frame = read_poses(args.poses)
    print(f"모델: {used}", file=sys.stderr)
    print(f"feature {len(features)}개 | pose {len(frame)}행 "
          f"| 타깃 {frame['system_id'].nunique()}개", file=sys.stderr)

    scores = score_poses(frame, features, boosters)
    nan_count = int(np.isnan(scores).sum())
    if nan_count:
        print(f"경고: 점수가 NaN 인 pose {nan_count}개 (가장 뒤로 보냅니다)",
              file=sys.stderr)
    rows = pick_topk(frame, scores, args.k, gen_col)

    if args.scores_out:
        dump = frame[[c for c in ("system_id", gen_col, "seed", "sample", "pose_id")
                      if c in frame.columns]].copy()
        dump["score"] = scores
        dump.to_csv(args.scores_out, index=False, encoding="utf-8")
        print(f"점수 전체: {args.scores_out}", file=sys.stderr)

    table = pd.DataFrame(rows)
    if args.out:
        table.to_csv(args.out, index=False, encoding="utf-8")
        print(f"Top-{args.k}: {args.out} ({len(table)}행)", file=sys.stderr)
    else:
        table.to_csv(sys.stdout, index=False)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, default=DEFAULT_MATRIX)
    parser.add_argument('--union-dir', type=Path, default=DEFAULT_UNION,
                        help='34번 결과 폴더(최종 집합을 읽습니다).')
    parser.add_argument('--test-eval-dir', type=Path, default=DEFAULT_TEST_EVAL,
                        help='35번 결과 폴더(재현 점검의 기준값).')
    parser.add_argument('--relevance-bins', choices=tuple(combined.BIN_SETS),
                        default='asymmetric')
    parser.add_argument('--topk', type=combined.parse_topk, default=(1, 3, 5))
    parser.add_argument('--shuffles', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=5,
                        help='학습할 시드 수. 첫 시드가 출하 모델입니다.')
    parser.add_argument('--n-estimators', type=int, default=400)
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261102,
                        help='기준 시드. 이 시드의 모델이 출하 모델이 됩니다.')
    parser.add_argument('--tag', type=str, default=None)
    args = parser.parse_args()
    for name in ('shuffles', 'repeats', 'n_estimators', 'max_depth', 'threads'):
        if getattr(args, name) < 1:
            parser.error(f'--{name.replace("_", "-")} 는 1 이상이어야 합니다.')
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
