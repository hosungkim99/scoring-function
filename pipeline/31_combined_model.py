#!/usr/bin/env python3
"""Process B 2차(⑥-4). 선택 규칙 내장 + k 동시 보고 + feature 표현 결합.

과정 B 1차 사이클(28~30번)이 남긴 교훈 세 개를 구조적으로 막는 스크립트입니다.
근거는 모두 실측이며 상세는 개발일지 §10에 있습니다.

1) 선택 규칙은 모델의 일부다 (§10.7)
   'Top-K = 점수 상위 K개'를 자명한 것으로 두고 하드코딩했는데, 그것이 모델 공헌의
   부호를 바꿨습니다. 같은 점수 파일에서 기본 규칙 -0.0049, 생성기 분산 규칙 +0.0089.
   그래서 이 스크립트는 세 규칙을 모두 계산하고, 각 규칙마다 '모델 없는' 대조군을 함께
   냅니다. 다양성 이득과 모델 이득을 분리하지 않으면 데이터 구조(생성기 4종)가 주는
   이득을 모델 공로로 오기합니다.

2) 포화된 지표로는 아무것도 측정할 수 없다 (§10.6)
   Top-5는 무작위가 0.9133으로 포화돼 다섯 실행 모두 음수였는데, 같은 점수로 Top-1을
   재면 전부 양수였습니다(최대 +190 타깃). k를 하나만 보면 또 틀립니다. 그래서
   --topk 1,3,5 를 한 실행에서 전부 냅니다.

3) 원본과 정규화는 서로 다른 층에서 이긴다 (§10.6)
   k=1 층별로 보면 원본이 어려운 층(very_hard +0.090 / medium +0.189), z-score가
   very_easy(+0.063)에서 이깁니다. 어려운 시스템은 결합 거리·clash 같은 절대적 물리
   타당성이 판별하고, 쉬운 시스템은 절대 임계값을 다 통과하므로 상대 비교만 남습니다.
   하나를 고를 이유가 없어 둘을 함께 줍니다(--features both).

목적함수(rank:ndcg)와 등급 relevance는 1차에서 확정된 설정이라 유지합니다. 등급 경계는
2A 우선순위를 복원하는 비대칭 가중을 플래그로 넣었습니다(§10.5의 미검증 처방).

평가 기준은 바뀌지 않습니다. 성공 = 타깃별 Top-K에 sc_rmsd < 2A pose 하나 이상.
난이도 층은 k를 바꿔도 같은 시스템 집단을 비교하도록 항상 k=5 무작위 확률로 고정합니다.

출력: out/combined/31_run_combined[_TAG]/
"""
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import time

# 30번을 통해 29·28번까지 불러옵니다(평가·QA·정규화 코드를 복사하지 않습니다).
_graded_path = Path(__file__).with_name('30_graded_model.py')
_gspec = importlib.util.spec_from_file_location('graded_support', _graded_path)
graded = importlib.util.module_from_spec(_gspec)
_gspec.loader.exec_module(graded)
ranking = graded.ranking
baseline = graded.baseline
roles = graded.roles
audit = graded.audit
helper = graded.helper

VERSION = 'combined_0.1.0'
OUT_KIND = 'combined'
OUT_BASE = '31_run_combined'
KEYS = baseline.KEYS
STRATA = baseline.STRATA
BASE = baseline.BASE
DEFAULT_MATRIX = baseline.DEFAULT_MATRIX
THRESHOLD = graded.THRESHOLD
# 등급 경계. default는 30번과 동일, asymmetric은 2A 경계 간격을 벌려 우선순위를 복원합니다
# (11 vs 10 = 10% 차이, 10 vs 2 = 5배). 둘 다 2A 컷의 등급 이상 == near_native 입니다.
BIN_SETS = {
    'default': ((1.0, 4), (2.0, 3), (3.0, 2), (5.0, 1)),
    'asymmetric': ((1.0, 11), (2.0, 10), (3.0, 2), (5.0, 1)),
}
RULES = ('plain', 'ruleA', 'ruleB')
SCORERS = ('model', 'pseudorandom', 'shuffle')
RULE_B_MAX = 2
Z_SUFFIX = '__z'


def relevance_from_sc(sc, bins, scheme):
    """sc_rmsd -> 학습용 등급. 평가 기준(2A 이진)과는 무관합니다."""
    import numpy as np
    rel = np.zeros(sc.shape, dtype=np.int16)
    if scheme == 'binary':
        rel[sc < THRESHOLD] = 1
        return rel
    for cut, level in sorted(bins, reverse=True):
        rel[sc < cut] = level
    return rel


def near_native_level(bins, scheme):
    """near_native(sc<2A)에 해당하는 최소 등급. 동치성 검증에 씁니다."""
    if scheme == 'binary':
        return 1
    return dict((c, l) for c, l in bins)[THRESHOLD]


def rule_order(generators, mode, kmax):
    """점수 내림차순 정렬 상태의 생성기 배열에서 고를 위치를 순서대로 돌려줍니다.

    모든 규칙이 접두사 성질을 가지므로 kmax까지 한 번 만들고 k별로 앞에서 잘라 씁니다.
      plain : 점수 상위 그대로
      ruleA : 생성기별 1위를 먼저, 남은 자리는 점수순
      ruleB : 점수순이되 생성기당 최대 RULE_B_MAX개
    """
    n = len(generators)
    if mode == 'plain':
        return list(range(min(kmax, n)))
    if mode == 'ruleA':
        seen, first, rest = set(), [], []
        for i, gen in enumerate(generators):
            if gen in seen:
                rest.append(i)
            else:
                seen.add(gen)
                first.append(i)
        return (first + rest)[:kmax]
    count, picked = {}, []
    for i, gen in enumerate(generators):
        if count.get(gen, 0) < RULE_B_MAX:
            picked.append(i)
            count[gen] = count.get(gen, 0) + 1
            if len(picked) == kmax:
                break
    if len(picked) < kmax:
        # 생성기 수 x 상한이 kmax보다 작으면 점수순으로 채웁니다.
        taken = set(picked)
        picked += [i for i in range(n) if i not in taken][:kmax - len(picked)]
    return picked


def build_feature_matrix(X_raw, feats, codes, starts, sizes, mode):
    """원본 / 시스템 내 z-score / 둘 다. 결합 시 z 열 이름에 __z 를 붙입니다."""
    import numpy as np
    if mode == 'raw':
        return X_raw, list(feats), []
    Z = ranking.zscore_within_group(X_raw, starts, sizes)
    # 시스템 안에서 상수였던 feature는 z 변환 후 전역 상수가 되어 어떤 분기도 만들지
    # 못합니다. 남겨두면 feature 중요도만 어지럽히므로 z쪽에서만 제외합니다.
    dead = []
    keep = []
    for i, name in enumerate(feats):
        col = Z[:, i]
        vals = col[~np.isnan(col)]
        if vals.size < 2 or float(vals.std()) < 1e-12:
            dead.append(name)
        else:
            keep.append(i)
    Zk = Z[:, keep]
    znames = [feats[i] + Z_SUFFIX for i in keep]
    if mode == 'zscore':
        return Zk, znames, dead
    return np.hstack([X_raw, Zk]), list(feats) + znames, dead


def evaluate(dev, codes, args):
    """시스템별로 (채점자 x 선택규칙 x k) 성공 여부를 전부 계산합니다."""
    import numpy as np
    rng = np.random.default_rng(args.seed)
    kmax = max(args.topk)
    rows = []
    for sid, grp in dev.groupby('system_id', sort=False):
        nn = grp['near_native'].to_numpy()
        m = int(nn.sum())
        if m == 0:
            continue
        n = len(grp)
        gen = grp['model'].to_numpy()
        tie = grp['_tie'].to_numpy()
        score = grp['score'].to_numpy(dtype=np.float64)
        score = np.where(np.isnan(score), -np.inf, score)
        rec = {'system_id': sid, 'n_pose': n, 'n_near_native': m}
        for k in args.topk:
            rec[f'rand_analytic_k{k}'] = baseline.random_topk(n, m, k)
        # lexsort는 마지막 키가 1순위입니다. 점수 내림차순 + 동점은 SHA1 오름차순.
        orders = {
            'model': [np.lexsort((tie, -score))],
            'pseudorandom': [np.argsort(tie, kind='mergesort')],
            'shuffle': [rng.permutation(n) for _ in range(args.shuffles)],
        }
        for scorer, olist in orders.items():
            for rule in RULES:
                acc = {k: 0.0 for k in args.topk}
                for order in olist:
                    idx = rule_order(gen[order], rule, kmax)
                    sel = order[idx]
                    for k in args.topk:
                        acc[k] += float(nn[sel[:k]].any())
                for k in args.topk:
                    rec[f'{scorer}|{rule}|k{k}'] = acc[k] / len(olist)
        rows.append(rec)
    return rows


def aggregate(rows, args):
    """전체·난이도층별 성공률과 공헌도 분해. 대조군은 shuffle(여러 번 평균)을 씁니다."""
    import numpy as np
    import pandas as pd
    frame = pd.DataFrame(rows)
    # 난이도 축은 k를 바꿔도 같은 집단을 비교하도록 k=5 무작위 확률로 고정합니다.
    axis = f'rand_analytic_k{5 if 5 in args.topk else max(args.topk)}'
    buckets = [(name, frame[(frame[axis] >= lo) & (frame[axis] < hi)])
               for name, lo, hi in STRATA]

    matrix, strat, contrib = [], [], []
    for scorer in SCORERS:
        for rule in RULES:
            for k in args.topk:
                col = f'{scorer}|{rule}|k{k}'
                rate = float(frame[col].mean())
                rnd = float(frame[f'rand_analytic_k{k}'].mean())
                matrix.append({'scorer': scorer, 'rule': rule, 'topk': k,
                               'systems': len(frame), 'success': round(rate, 4),
                               'rand_analytic': round(rnd, 4),
                               'lift_vs_analytic': round(rate - rnd, 4),
                               'net_targets': round((rate - rnd) * len(frame), 1)})
                for name, sel in buckets:
                    if sel.empty:
                        continue
                    mv = float(sel[col].mean())
                    rv = float(sel[f'rand_analytic_k{k}'].mean())
                    nl = baseline.normalized_lift(mv, rv)
                    strat.append({'stratum': name, 'scorer': scorer, 'rule': rule, 'topk': k,
                                  'systems': len(sel), 'success': round(mv, 4),
                                  'rand_analytic': round(rv, 4), 'lift': round(mv - rv, 4),
                                  'normalized_lift': round(nl, 4) if nl is not None else None,
                                  'room_targets': round(len(sel) * (1.0 - rv), 1),
                                  'gained_targets': round(len(sel) * (mv - rv), 1)})

    n = len(frame)
    for rule in RULES:
        for k in args.topk:
            base_rand = float(frame[f'shuffle|plain|k{k}'].mean())
            rule_rand = float(frame[f'shuffle|{rule}|k{k}'].mean())
            rule_model = float(frame[f'model|{rule}|k{k}'].mean())
            analytic = float(frame[f'rand_analytic_k{k}'].mean())
            contrib.append({
                'rule': rule, 'topk': k, 'systems': n,
                'rand_analytic': round(analytic, 4),
                'rand_plain_empirical': round(base_rand, 4),
                'rand_with_rule': round(rule_rand, 4),
                'model_with_rule': round(rule_model, 4),
                'diversity_gain': round(rule_rand - base_rand, 4),
                'model_gain': round(rule_model - rule_rand, 4),
                'total_gain': round(rule_model - base_rand, 4),
                'diversity_targets': round((rule_rand - base_rand) * n, 1),
                'model_targets': round((rule_model - rule_rand) * n, 1),
                'model_share': (round((rule_model - rule_rand) / (rule_model - base_rand), 3)
                                if abs(rule_model - base_rand) > 1e-9 else None)})
    return frame, matrix, strat, contrib


def run(args):
    """결합 feature + 등급 relevance로 학습하고 규칙·k 전 조합을 평가합니다."""
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
    from sklearn.metrics import roc_auc_score

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
    print('MATRIX poses:', len(frame), '| systems:', frame['system_id'].nunique(), flush=True)

    test_systems = int(frame.loc[frame['split'] == 'test', 'system_id'].nunique())
    dev = frame[frame['split'] == 'dev'].copy()
    del frame
    dev = dev.sort_values(KEYS, kind='mergesort').reset_index(drop=True)
    dev['cv_fold'] = dev['cv_fold'].astype(int)
    folds = sorted(int(f) for f in dev['cv_fold'].unique())
    print(f'DEV poses {len(dev)} | systems {dev["system_id"].nunique()} | folds {folds} '
          f'| TEST systems {test_systems} (봉인)', flush=True)
    print('GENERATORS:', sorted(dev['model'].unique()), flush=True)

    bins = BIN_SETS[args.relevance_bins]
    sc = dev['sc_rmsd'].to_numpy(dtype=np.float64, na_value=np.nan)
    rel = relevance_from_sc(sc, bins, args.relevance)
    rel_binary = relevance_from_sc(sc, bins, 'binary')
    known = dev['label_known'].to_numpy(dtype=bool)
    nn = dev['near_native'].to_numpy(dtype=bool)
    nn_level = near_native_level(bins, args.relevance)
    mismatch = int(np.sum((rel[known] >= nn_level) != nn[known]))
    if mismatch:
        raise ValueError(f'등급 {nn_level} 이상과 near_native가 {mismatch}개 pose에서 불일치합니다.')
    print(f'RELEVANCE {args.relevance}/{args.relevance_bins} 경계 {bins} '
          f'| 등급>={nn_level} == near_native 확인 ({int(known.sum())} pose)', flush=True)

    exclude = {c.strip() for c in (args.exclude_categories or '').split(',') if c.strip()}
    feats0, dropped = baseline.select_features(dev, catalog, exclude)
    audit.save_csv(out / 'dropped_features.csv',
                   dropped or [{'feature': None, 'category': None, 'reason': None}],
                   ('feature', 'category', 'reason'))
    if not feats0:
        raise ValueError('사용할 feature가 없습니다.')
    qa_rows = baseline.value_qa(dev, feats0, catalog)
    audit.save_csv(out / 'value_qa.csv',
                   qa_rows or [{'feature': None, 'category': None, 'rule': None, 'n_inf': 0,
                                'n_extreme': 0, 'n_rule_violation': 0, 'min': None, 'max': None}],
                   ('feature', 'category', 'rule', 'n_inf', 'n_extreme', 'n_rule_violation',
                    'min', 'max'))
    print(f'FEATURES base {len(feats0)} | dropped {len(dropped)} '
          f'| VALUE_QA 이상 {len(qa_rows)}건', flush=True)

    codes = pd.factorize(dev['system_id'], sort=False)[0]
    assert np.all(np.diff(codes) >= 0), '시스템 그룹이 연속이 아닙니다.'
    _, starts, sizes = ranking.group_layout(codes)
    fold_of = dev['cv_fold'].to_numpy()
    fold_span = pd.Series(fold_of).groupby(codes).nunique().max()
    if int(fold_span) != 1:
        raise ValueError(f'한 시스템이 여러 폴드에 걸쳐 있습니다(max {fold_span}).')

    X_raw = dev[feats0].to_numpy(dtype=np.float32, na_value=np.nan)
    X, feats, dead_z = build_feature_matrix(X_raw, feats0, codes, starts, sizes, args.features)
    print(f'MATRIX {args.features}: {X.shape} float32 ({X.nbytes / 1e6:.0f} MB) '
          f'| 결측 {int(np.isnan(X).sum())} 칸 | 시스템 {int(np.unique(codes).size)}', flush=True)
    if dead_z:
        print(f'  z 변환 후 전역 상수라 제외한 {len(dead_z)}개(시스템 내 상수): '
              f'{dead_z[:14]}{" ..." if len(dead_z) > 14 else ""}', flush=True)

    n_pair_bin, _ = graded.pairable_systems(codes, rel_binary, known)
    n_pair_rel, _ = graded.pairable_systems(codes, rel, known)
    print(f'PAIRABLE dev systems: 이진 {n_pair_bin} -> {args.relevance} {n_pair_rel} '
          f'(+{n_pair_rel - n_pair_bin}) / 전체 {int(np.unique(codes).size)}', flush=True)

    dev['score'] = np.nan
    fold_rows = []
    importances = np.zeros(len(feats), dtype=np.float64)
    effective = None
    for k in folds:
        tr = (fold_of != k) & known
        va = fold_of == k
        _, dead = graded.pairable_systems(codes, rel, tr)
        if dead.size:
            tr = tr & ~np.isin(codes, dead)
        model = graded.build_ranker(args)
        model.fit(X[tr], rel[tr], qid=codes[tr])
        if effective is None:
            effective = graded.booster_effective_params(
                model, {'ndcg_exp_gain', 'lambdarank_pair_method',
                        'lambdarank_num_pair_per_sample', 'objective'})
            print('BOOSTER 적용 인자:', effective, flush=True)
        dev.loc[va, 'score'] = model.predict(X[va])
        lab = va & known
        auc = None
        if len(np.unique(nn[lab])) > 1:
            auc = float(roc_auc_score(nn[lab], dev.loc[lab, 'score']))
        fold_rows.append({'cv_fold': int(k), 'train_poses': int(tr.sum()),
                          'train_groups': int(np.unique(codes[tr]).size),
                          'valid_poses': int(va.sum()), 'valid_labeled': int(lab.sum()),
                          'global_auc': round(auc, 4) if auc is not None else None})
        print(f'  fold {k}: train {int(tr.sum())} ({int(np.unique(codes[tr]).size)} systems) '
              f'valid {int(va.sum())} | 전역 AUC '
              f'{"-" if auc is None else round(auc, 4)} (참고용)', flush=True)
        imp = getattr(model, 'feature_importances_', None)
        if imp is not None and len(imp) == len(feats):
            importances += np.asarray(imp, dtype=np.float64)
    audit.save_csv(out / 'fold_metrics.csv', fold_rows,
                   ('cv_fold', 'train_poses', 'train_groups', 'valid_poses', 'valid_labeled',
                    'global_auc'))

    if importances.sum() > 0:
        order = np.argsort(-importances)
        imp_rows = []
        for i in order:
            name = feats[i]
            src = name[:-len(Z_SUFFIX)] if name.endswith(Z_SUFFIX) else name
            imp_rows.append({'feature': name,
                             'category': str(catalog.loc[src, 'category']),
                             'representation': 'zscore' if name.endswith(Z_SUFFIX) else 'raw',
                             'importance': round(float(importances[i] / len(folds)), 6)})
        audit.save_csv(out / 'feature_importance.csv', imp_rows,
                       ('feature', 'category', 'representation', 'importance'))
        print('TOP FEATURES:', ', '.join(r['feature'] for r in imp_rows[:8]), flush=True)
        share = {}
        for r in imp_rows:
            share[r['representation']] = share.get(r['representation'], 0.0) + r['importance']
        total = sum(share.values()) or 1.0
        print('표현별 중요도 비중:',
              {k2: round(v / total, 3) for k2, v in sorted(share.items())}, flush=True)

    dev['_tie'] = dev['pose_id'].map(baseline.tie_key)
    buf = io.BytesIO()
    dev[KEYS + ['pose_id', 'score']].to_parquet(buf, engine='pyarrow', index=False,
                                                compression='snappy')
    with helper.checked_path(out / 'scores.parquet').open('xb') as handle:
        handle.write(buf.getvalue())

    print(f'\n평가: 채점자 {len(SCORERS)} x 규칙 {len(RULES)} x k {args.topk} '
          f'(shuffle {args.shuffles}회)', flush=True)
    sys_rows = evaluate(dev, codes, args)
    if not sys_rows:
        raise ValueError('eligible 시스템이 없습니다.')
    sys_frame, matrix, strat, contrib = aggregate(sys_rows, args)
    sys_frame.to_csv(helper.checked_path(out / 'system_results.csv'), index=False,
                     encoding='utf-8')
    audit.save_csv(out / 'selection_matrix.csv', matrix,
                   ('scorer', 'rule', 'topk', 'systems', 'success', 'rand_analytic',
                    'lift_vs_analytic', 'net_targets'))
    audit.save_csv(out / 'selection_by_stratum.csv', strat,
                   ('stratum', 'scorer', 'rule', 'topk', 'systems', 'success', 'rand_analytic',
                    'lift', 'normalized_lift', 'room_targets', 'gained_targets'))
    audit.save_csv(out / 'contribution.csv', contrib,
                   ('rule', 'topk', 'systems', 'rand_analytic', 'rand_plain_empirical',
                    'rand_with_rule', 'model_with_rule', 'diversity_gain', 'model_gain',
                    'total_gain', 'diversity_targets', 'model_targets', 'model_share'))

    changed = []
    for source in sources:
        try:
            info = Path(source['path']).stat()
            if (info.st_size, info.st_mtime_ns) != (source['size_bytes'], source['mtime_ns']):
                changed.append(source['path'])
        except OSError:
            changed.append(source['path'])

    best = max((r for r in matrix if r['scorer'] == 'model'),
               key=lambda r: r['lift_vs_analytic'])
    status = 'COMBINED_COMPLETE' if not changed else 'CHECK_REQUIRED'
    report = {
        'version': VERSION, 'status': status, 'output_dir': str(out),
        'matrix': str(args.matrix), 'sources': sources,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'support_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (_graded_path, graded._ranking_path,
                                     ranking._baseline_path, baseline._support_path,
                                     roles._support_path, audit._support_path)},
        'python': sys.executable, 'xgboost': xgboost.__version__,
        'objective': 'rank', 'features': args.features, 'relevance': args.relevance,
        'relevance_bins': args.relevance_bins,
        'relevance_bins_values': [list(b) for b in bins],
        'near_native_level': nn_level,
        'booster_effective_params': effective,
        'model_params': {'n_estimators': args.n_estimators, 'max_depth': args.max_depth,
                         'learning_rate': args.learning_rate, 'seed': args.seed},
        'topk': list(args.topk), 'shuffles': args.shuffles, 'rules': list(RULES),
        'excluded_categories': sorted(exclude),
        'features_base': len(feats0), 'features_used': len(feats),
        'features_dropped': len(dropped), 'zscore_dropped': dead_z,
        'feature_list': feats, 'value_qa_flagged': len(qa_rows),
        'pairable_binary': n_pair_bin, 'pairable_relevance': n_pair_rel,
        'dev_poses': int(len(dev)), 'dev_systems': int(np.unique(codes).size),
        'test_systems_untouched': test_systems, 'eligible_systems': len(sys_rows),
        'best_model_cell': best,
        'fold_metrics': fold_rows, 'selection_matrix': matrix, 'contribution': contrib,
        'selection_by_stratum': strat,
        'changed_sources': changed, 'elapsed_seconds': time.perf_counter() - started,
        'policy': ['test 분할은 학습·평가에 사용하지 않았습니다.',
                   '18번 포켓 cluster 폴드를 그대로 썼고 한 시스템이 여러 폴드에 걸치지 '
                   '않는지 확인했습니다.',
                   '랭킹 그룹은 시스템이며 비교는 시스템 안에서만 일어납니다.',
                   '학습 등급은 sc_rmsd에서 만들고 평가 기준은 sc_rmsd < 2A 이진 Top-K로 '
                   '유지했습니다(동치성을 pose 단위로 검증).',
                   '라벨 결측 pose는 학습에서 제외하고 예측·순위에는 포함했습니다.',
                   'z-score에서 결측은 NaN으로 유지했습니다(평균 대입 없음).',
                   '선택 규칙마다 모델 없는 대조군(의사난수·셔플)을 함께 계산해 다양성 '
                   '이득과 모델 이득을 분리했습니다.',
                   '난이도 층은 k를 바꿔도 같은 집단을 비교하도록 k=5 무작위 확률로 '
                   '고정했습니다.',
                   '동점은 pose_id SHA1 오름차순으로 결정적으로 깼습니다.'],
        'notes': ['contribution.csv 의 model_gain 이 이 스크립트의 핵심 숫자입니다. '
                  'total_gain 은 데이터 구조(생성기 4종)가 주는 다양성 이득을 포함합니다.',
                  'Top-5는 무작위가 0.91대로 포화돼 단독 개발 지표가 될 수 없습니다. '
                  '모델 품질은 Top-1에서 판단하고 Top-5는 수용 기준으로 씁니다.',
                  'representation 비중으로 원본과 z-score 중 무엇이 쓰이는지 확인할 수 '
                  '있습니다. 한쪽이 거의 0이면 결합의 의미가 없다는 신호입니다.']}
    helper.write_json(out / 'combined_report.json', report)
    helper.write_json(out / 'RUN_COMPLETE.json', {
        'outputs_written': True, 'combined_finished': True, 'status': status,
        'features': args.features, 'relevance_bins': args.relevance_bins,
        'best_model_cell': best, 'eligible_systems': len(sys_rows),
        'features_used': len(feats), 'sources_unchanged': not changed})

    print('\nSTATUS:', status, flush=True)
    print(f'설정: features={args.features} relevance={args.relevance}/{args.relevance_bins} '
          f'feature수={len(feats)}', flush=True)
    print(f'ELIGIBLE dev systems: {len(sys_rows)}', flush=True)

    print('\n=== 선택 규칙 x k (모델 채점) ===', flush=True)
    print(f"{'rule':9s}{'k':>3s}{'success':>9s}{'무작위':>9s}{'lift':>9s}{'순득실':>9s}", flush=True)
    for row in matrix:
        if row['scorer'] != 'model':
            continue
        print(f"{row['rule']:9s}{row['topk']:3d}{row['success']:9.4f}"
              f"{row['rand_analytic']:9.4f}{row['lift_vs_analytic']:+9.4f}"
              f"{row['net_targets']:+9.1f}", flush=True)

    print('\n=== 공헌도 분해 (대조군 = 같은 규칙 + 점수 없는 셔플) ===', flush=True)
    print(f"{'rule':9s}{'k':>3s}{'다양성':>10s}{'모델':>10s}{'합계':>10s}"
          f"{'모델타깃':>10s}{'모델비중':>10s}", flush=True)
    for row in contrib:
        share = row['model_share']
        print(f"{row['rule']:9s}{row['topk']:3d}{row['diversity_gain']:+10.4f}"
              f"{row['model_gain']:+10.4f}{row['total_gain']:+10.4f}"
              f"{row['model_targets']:+10.1f}"
              f"{(f'{share:.3f}' if share is not None else '-'):>10s}", flush=True)

    print('\n=== 난이도 층별 (모델 채점, 규칙 ruleA) ===', flush=True)
    print(f"{'stratum':11s}{'k':>3s}{'systems':>8s}{'success':>9s}{'무작위':>9s}"
          f"{'lift':>9s}{'정규화':>9s}{'여백':>8s}{'획득':>8s}", flush=True)
    for row in strat:
        if row['scorer'] != 'model' or row['rule'] != 'ruleA':
            continue
        print(f"{row['stratum']:11s}{row['topk']:3d}{row['systems']:8d}{row['success']:9.3f}"
              f"{row['rand_analytic']:9.3f}{row['lift']:+9.3f}"
              f"{(row['normalized_lift'] if row['normalized_lift'] is not None else 0):9.3f}"
              f"{row['room_targets']:8.1f}{row['gained_targets']:+8.1f}", flush=True)
    return 0 if status == 'COMBINED_COMPLETE' else 2


def parse_topk(text):
    """'1,3,5' -> (1, 3, 5). 중복·비정수는 거부합니다."""
    values = []
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit() or int(part) < 1:
            raise argparse.ArgumentTypeError(f'topk는 1 이상의 정수여야 합니다: {part}')
        values.append(int(part))
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError('topk가 비었거나 중복입니다.')
    return tuple(sorted(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, default=DEFAULT_MATRIX,
                        help='26번 통합 결과 폴더.')
    parser.add_argument('--features', choices=('both', 'raw', 'zscore'), default='both',
                        help='both=원본⊕시스템내 z-score(기본), raw/zscore=절제용.')
    parser.add_argument('--relevance', choices=('graded', 'binary'), default='graded',
                        help='graded=sc_rmsd 등급, binary=절제용.')
    parser.add_argument('--relevance-bins', choices=tuple(BIN_SETS), default='asymmetric',
                        help='asymmetric=2Å 경계 우선(기본), default=30번과 동일.')
    parser.add_argument('--topk', type=parse_topk, default=(1, 3, 5),
                        help='쉼표 구분. 기본 1,3,5.')
    parser.add_argument('--shuffles', type=int, default=5,
                        help='선택 규칙별 무작위 대조군 반복 수.')
    parser.add_argument('--exclude-categories', type=str, default=None,
                        help='쉼표 구분 후보군 제외. 예: cat2_gnina')
    parser.add_argument('--n-estimators', type=int, default=400)
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261102)
    parser.add_argument('--tag', type=str, default=None, help='출력 폴더 접미사.')
    args = parser.parse_args()
    if args.n_estimators < 1 or args.max_depth < 1 or args.threads < 1:
        parser.error('모델 인자는 양수여야 합니다.')
    if args.shuffles < 1:
        parser.error('--shuffles 는 1 이상이어야 합니다.')
    args.matrix = args.matrix.absolute()
    return run(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
