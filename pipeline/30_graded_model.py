#!/usr/bin/env python3
"""Process B 등급 relevance 모델(⑥-3). sc_rmsd를 이진화하지 않고 학습 신호로 씁니다.

29번의 절제가 두 가지를 확정했습니다.

  z-score 정규화는 해롭다
    rank  : 원본 +0.140/+0.141/+0.163 -> z-score +0.020/+0.040/+0.081
    binary: 원본 +0.095/+0.116/+0.180 -> z-score +0.020/-0.023/-0.034
    결합 거리, clash 거리, 결합 길이는 절대값으로 의미를 갖는 양입니다. '이 시스템 안에서
    상대적으로'로 바꾸면 모든 pose가 나쁜 시스템에서 덜 나쁜 것이 +2시그마로 올라갑니다.
    그래서 30번은 정규화를 하지 않습니다.

  랭킹 목적함수는 분류보다 낫다
    rank_raw는 학습 시스템 1,375개로 28번(2,429~2,641개)의 어려운 층 성능을 넘었습니다.
    쌍 기반 학습이 데이터를 더 효율적으로 씁니다. 그래서 30번도 rank:ndcg를 씁니다.

남은 적자는 한 곳입니다. rank_raw는 어려운 층 207개에서 +30.6 타깃을 벌고 쉬운 층 2,670개에서
-43.1 타깃을 잃어 순 -12.6입니다. 그리고 쉬운 층이 바로 랭킹 학습에서 탈락한 시스템들입니다.
라벨된 pose가 전부 near-native인 시스템 1,405개는 이진 라벨에서 쌍이 0개라 기울기가 0입니다.
모델이 한 번도 본 적 없는 종류에 적용되고 있습니다.

30번의 변경은 하나입니다. sc_rmsd를 2A에서 잘라 버리는 대신 등급으로 씁니다.

    < 1.0 A -> 4,  < 2.0 A -> 3,  < 3.0 A -> 2,  < 5.0 A -> 1,  그 외 -> 0

전부 near-native인 시스템은 0.7A짜리(4)와 1.8A짜리(3)가 갈려 쌍이 생기고, near-native가
0개인 시스템도 2.5A(2)와 15A(0)가 갈려 쌍이 생깁니다. 지금까지 폐기하던 연속값 정보입니다.

평가는 전혀 바꾸지 않습니다. 성공 기준은 sc_rmsd < 2A 이진, Top-5 그대로입니다. 학습 신호만
세분화합니다. 이 구분이 흐려지면 29번과 비교가 불가능해지므로 코드에서도 분리해 둡니다.

주지표도 바꿥니다. 시스템별 AUC는 rank_raw 0.669 / bin_z 0.674로 거의 같은데 어려운 층
lift는 +0.14 / -0.03로 정반대였습니다. AUC는 순위 전체를 평균하지만 우리는 상위 5개만 봅니다.
그래서 시스템별 NDCG@5와 hit@5를 주지표로 둡니다.

출력: out/graded/30_run_graded[_TAG]/
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

# 29번을 통해 28번까지 불러옵니다(평가·QA·층별 집계 코드를 복사하지 않습니다).
_ranking_path = Path(__file__).with_name('29_ranking_model.py')
_rspec = importlib.util.spec_from_file_location('ranking_support', _ranking_path)
ranking = importlib.util.module_from_spec(_rspec)
_rspec.loader.exec_module(ranking)
baseline = ranking.baseline
roles = ranking.roles
audit = ranking.audit
helper = ranking.helper

VERSION = 'graded_0.1.0'
OUT_KIND = 'graded'
OUT_BASE = '30_run_graded'
KEYS = baseline.KEYS
TOPK = baseline.TOPK
STRATA = baseline.STRATA
BASE = baseline.BASE
DEFAULT_MATRIX = baseline.DEFAULT_MATRIX
# 성공 기준. 평가에서만 쓰며 학습 등급과 혼동하지 않도록 따로 둡니다.
THRESHOLD = 2.0
# 학습용 등급 경계. (상한 A, 등급) 이며 그 외는 0입니다. 2.0A가 성공 기준과 일치하도록
# 경계에 포함시켜, 등급 3 이상이 곧 near-native가 되게 맞췄습니다.
RELEVANCE_BINS = ((1.0, 4), (2.0, 3), (3.0, 2), (5.0, 1))
DEFAULT_COMPARE = ','.join([
    str(BASE / 'baseline' / '28_run_baseline_f63'),
    str(BASE / 'ranking' / '29_run_ranking_rank_raw'),
])
REPORT_NAMES = ('graded_report.json', 'ranking_report.json', 'baseline_report.json')


def relevance_from_sc(sc, scheme):
    """sc_rmsd -> 학습용 등급. 평가 기준(2A 이진)은 이 함수와 무관합니다.

    경계가 겹치지 않도록 느슨한 경계부터 덮어씁니다. sc_rmsd가 결측인 pose는 0이 되지만,
    그런 pose는 label_known이 False라 학습에 들어가지 않습니다.
    """
    import numpy as np
    rel = np.zeros(sc.shape, dtype=np.int8)
    if scheme == 'binary':
        rel[sc < THRESHOLD] = 1
        return rel
    for cut, level in sorted(RELEVANCE_BINS, reverse=True):
        rel[sc < cut] = level
    return rel


def pairable_systems(codes, rel, mask):
    """쌍이 하나라도 생기는 시스템. 등급이 전부 같으면 기울기가 0이라 학습에 기여하지 않습니다."""
    import numpy as np
    import pandas as pd
    agg = pd.Series(rel[mask]).groupby(codes[mask], sort=False).agg(['min', 'max'])
    dead = agg.index[agg['min'] == agg['max']].to_numpy()
    return int((agg['min'] != agg['max']).sum()), dead


def ndcg_at_k(ordered_gains, n_relevant, k=TOPK):
    """상위 k개만 보는 지표. 이득은 선형(0/1)이며 하위 순서는 보지 않습니다."""
    from math import log2
    top = ordered_gains[:k]
    dcg = sum(g / log2(i + 2) for i, g in enumerate(top))
    ideal = sum(1.0 / log2(i + 2) for i in range(min(n_relevant, k)))
    return (dcg / ideal) if ideal > 0 else None


def booster_effective_params(model, keys):
    """부스터가 실제로 적용한 인자를 확인합니다.

    xgboost는 모르는 인자를 조용히 무시할 수 있습니다. ndcg_exp_gain이 먹지 않았는데 먹은 줄
    알면 결과 해석이 틀어지므로, 설정을 꺼내 직접 확인합니다.
    """
    found = {}

    def walk(node):
        if isinstance(node, dict):
            for key, val in node.items():
                if key in keys and not isinstance(val, (dict, list)):
                    found[key] = val
                walk(val)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    try:
        walk(json.loads(model.get_booster().save_config()))
    except Exception as exc:
        found['_error'] = f'{type(exc).__name__}: {exc}'
    return found


def build_ranker(args):
    """등급 relevance를 선형 이득으로 다루고 리스트 상단에 집중하는 랭커."""
    from xgboost import XGBRanker
    return XGBRanker(
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        learning_rate=args.learning_rate, subsample=0.8, colsample_bytree=0.8,
        tree_method='hist', n_jobs=args.threads, random_state=args.seed,
        objective='rank:ndcg', eval_metric=f'ndcg@{TOPK}',
        lambdarank_pair_method='topk',
        lambdarank_num_pair_per_sample=max(TOPK, 8),
        # 기본 지수 이득은 <1A을 1~2A의 2배로 봅니다. 둘 다 성공인 우리 지표에는 선형이 맞습니다.
        ndcg_exp_gain=False)


def load_compare(paths):
    """앞선 실행들의 보고서를 읽어 층별 비교용으로 모읍니다."""
    rows = []
    for raw in paths:
        folder = Path(raw)
        for name in REPORT_NAMES:
            target = folder / name
            if target.exists():
                data = json.loads(target.read_text(encoding='utf-8'))
                rows.append({'label': folder.name, 'path': str(folder),
                             'objective': data.get('objective', 'binary'),
                             'normalize': data.get('normalize', 'none'),
                             'relevance': data.get('relevance', 'binary'),
                             'model_top5': data.get('model_top5'),
                             'random_top5': data.get('random_top5'),
                             'lift': data.get('lift'),
                             'net_targets': data.get('net_targets'),
                             'top5_by_stratum': data.get('top5_by_stratum')})
                break
    return rows


def run(args):
    """등급 relevance로 랭커를 학습하고 Top-5를 전체·난이도별로 평가합니다."""
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

    # 평가 라벨과 학습 등급이 어긋나지 않는지 확인합니다. 등급 3 이상이 곧 near-native여야
    # 하며, 어긋나면 두 기준이 다른 것을 재고 있다는 뜻이므로 즉시 멈춥니다.
    sc = dev['sc_rmsd'].to_numpy(dtype=np.float64, na_value=np.nan)
    rel_graded = relevance_from_sc(sc, 'graded')
    rel_binary = relevance_from_sc(sc, 'binary')
    known = dev['label_known'].to_numpy(dtype=bool)
    nn = dev['near_native'].to_numpy(dtype=bool)
    mismatch = int(np.sum((rel_graded[known] >= 3) != nn[known]))
    if mismatch:
        raise ValueError(f'등급 3 이상과 near_native가 {mismatch}개 pose에서 불일치합니다.')
    print(f'RELEVANCE 경계 {RELEVANCE_BINS} | 등급>=3 == near_native 확인 '
          f'({int(known.sum())} pose)', flush=True)

    exclude = {c.strip() for c in (args.exclude_categories or '').split(',') if c.strip()}
    feats, dropped = baseline.select_features(dev, catalog, exclude)
    audit.save_csv(out / 'dropped_features.csv',
                   dropped or [{'feature': None, 'category': None, 'reason': None}],
                   ('feature', 'category', 'reason'))
    print(f'FEATURES used {len(feats)} | dropped {len(dropped)} (정규화 없음)', flush=True)
    if not feats:
        raise ValueError('사용할 feature가 없습니다.')

    qa_rows = baseline.value_qa(dev, feats, catalog)
    audit.save_csv(out / 'value_qa.csv',
                   qa_rows or [{'feature': None, 'category': None, 'rule': None, 'n_inf': 0,
                                'n_extreme': 0, 'n_rule_violation': 0, 'min': None, 'max': None}],
                   ('feature', 'category', 'rule', 'n_inf', 'n_extreme', 'n_rule_violation',
                    'min', 'max'))
    print(f'VALUE_QA 이상 {len(qa_rows)}건 (삭제하지 않고 보고만)', flush=True)

    codes = pd.factorize(dev['system_id'], sort=False)[0]
    assert np.all(np.diff(codes) >= 0), '시스템 그룹이 연속이 아닙니다.'
    X = dev[feats].to_numpy(dtype=np.float32, na_value=np.nan)
    fold_of = dev['cv_fold'].to_numpy()
    rel = rel_graded if args.relevance == 'graded' else rel_binary
    print(f'MATRIX raw: {X.shape} float32 ({X.nbytes / 1e6:.0f} MB) '
          f'| 결측 {int(np.isnan(X).sum())} 칸 | 시스템 {int(np.unique(codes).size)}', flush=True)

    fold_span = pd.Series(fold_of).groupby(codes).nunique().max()
    if int(fold_span) != 1:
        raise ValueError(f'한 시스템이 여러 폴드에 걸쳐 있습니다(max {fold_span}).')

    # 등급 전환이 실제로 학습 가능 시스템을 늘렸는지가 이 실행의 전제입니다.
    # 늘지 않았다면 처방이 틀린 것이므로 숫자를 먼저 찍습니다.
    n_pair_bin, _ = pairable_systems(codes, rel_binary, known)
    n_pair_grad, _ = pairable_systems(codes, rel_graded, known)
    n_sys_total = int(np.unique(codes).size)
    recovered = n_pair_grad - n_pair_bin
    print(f'PAIRABLE dev systems: 이진 {n_pair_bin} -> 등급 {n_pair_grad} '
          f'(+{recovered}) / 전체 {n_sys_total}', flush=True)
    if args.relevance == 'graded' and recovered <= 0:
        print('  경고: 등급 전환으로 학습 가능 시스템이 늘지 않았습니다. '
              '이 처방의 전제가 성립하지 않습니다.', flush=True)
    grade_counts = {int(g): int((rel_graded[known] == g).sum())
                    for g in sorted(np.unique(rel_graded[known]))}
    print('GRADE 분포(라벨 있는 pose):', grade_counts, flush=True)

    dev['score'] = np.nan
    fold_rows = []
    importances = np.zeros(len(feats), dtype=np.float64)
    effective = None
    for k in folds:
        tr = (fold_of != k) & known
        va = fold_of == k
        _, dead = pairable_systems(codes, rel, tr)
        if dead.size:
            tr = tr & ~np.isin(codes, dead)
        model = build_ranker(args)
        model.fit(X[tr], rel[tr], qid=codes[tr])
        if effective is None:
            effective = booster_effective_params(
                model, {'ndcg_exp_gain', 'lambdarank_pair_method',
                        'lambdarank_num_pair_per_sample', 'objective'})
            print('BOOSTER 적용 인자:', effective, flush=True)
        dev.loc[va, 'score'] = model.predict(X[va])
        lab = va & known
        auc = None
        if len(np.unique(nn[lab])) > 1:
            auc = float(roc_auc_score(nn[lab], dev.loc[lab, 'score']))
        sys_auc, n_sys_auc = ranking.per_system_auc(codes, nn.astype(np.int8),
                                                   dev['score'].to_numpy(), lab)
        fold_rows.append({'cv_fold': int(k), 'train_poses': int(tr.sum()),
                          'train_groups': int(np.unique(codes[tr]).size),
                          'valid_poses': int(va.sum()), 'valid_labeled': int(lab.sum()),
                          'global_auc': round(auc, 4) if auc is not None else None,
                          'per_system_auc': round(sys_auc, 4) if sys_auc is not None else None})
        print(f'  fold {k}: train {int(tr.sum())} ({int(np.unique(codes[tr]).size)} systems) '
              f'valid {int(va.sum())} | 전역 AUC {"-" if auc is None else round(auc, 4)} '
              f'| 시스템별 AUC {"-" if sys_auc is None else round(sys_auc, 4)} (참고)', flush=True)
        imp = getattr(model, 'feature_importances_', None)
        if imp is not None and len(imp) == len(feats):
            importances += np.asarray(imp, dtype=np.float64)
    audit.save_csv(out / 'fold_metrics.csv', fold_rows,
                   ('cv_fold', 'train_poses', 'train_groups', 'valid_poses', 'valid_labeled',
                    'global_auc', 'per_system_auc'))

    if importances.sum() > 0:
        order = np.argsort(-importances)
        imp_rows = [{'feature': feats[i], 'category': str(catalog.loc[feats[i], 'category']),
                     'importance': round(float(importances[i] / len(folds)), 6)} for i in order]
        audit.save_csv(out / 'feature_importance.csv', imp_rows,
                       ('feature', 'category', 'importance'))
        print('TOP FEATURES:', ', '.join(r['feature'] for r in imp_rows[:8]), flush=True)

    dev['_tie'] = dev['pose_id'].map(baseline.tie_key)
    buf = io.BytesIO()
    dev[KEYS + ['pose_id', 'score']].to_parquet(buf, engine='pyarrow', index=False,
                                                compression='snappy')
    with helper.checked_path(out / 'scores.parquet').open('xb') as handle:
        handle.write(buf.getvalue())

    # 시스템별 Top-5. hit@5(성공 여부)와 NDCG@5(상위 안에서의 위치)를 함께 냅니다.
    sys_rows = []
    for sid, grp in dev.groupby('system_id'):
        m = int(grp['near_native'].sum())
        if m == 0:
            continue
        ordered = grp.sort_values(['score', '_tie'], ascending=[False, True],
                                  na_position='last', kind='mergesort')
        gains = ordered['near_native'].to_numpy(dtype=np.float64)
        ndcg = ndcg_at_k(gains, m)
        sys_rows.append({'system_id': sid, 'n_pose': int(len(grp)), 'n_near_native': m,
                         'random_top5': round(baseline.random_topk(len(grp), m), 4),
                         'model_top5': bool(gains[:TOPK].any()),
                         'ndcg_at_5': round(ndcg, 4) if ndcg is not None else None})
    audit.save_csv(out / 'system_top5.csv', sys_rows,
                   ('system_id', 'n_pose', 'n_near_native', 'random_top5', 'model_top5',
                    'ndcg_at_5'))
    if not sys_rows:
        raise ValueError('eligible 시스템이 없습니다.')

    n_sys = len(sys_rows)
    model_rate = float(np.mean([r['model_top5'] for r in sys_rows]))
    rnd_rate = float(np.mean([r['random_top5'] for r in sys_rows]))
    norm = baseline.normalized_lift(model_rate, rnd_rate)
    ndcg_vals = [r['ndcg_at_5'] for r in sys_rows if r['ndcg_at_5'] is not None]
    mean_ndcg = float(np.mean(ndcg_vals)) if ndcg_vals else None

    strat_rows = []
    for name, lo, hi in STRATA:
        sel = [r for r in sys_rows if lo <= r['random_top5'] < hi]
        if not sel:
            strat_rows.append({'stratum': name, 'range': f'{lo}-{hi}', 'systems': 0,
                               'model_top5': None, 'random_top5': None, 'lift': None,
                               'normalized_lift': None, 'ndcg_at_5': None,
                               'room_targets': None, 'gained_targets': None})
            continue
        mv = float(np.mean([r['model_top5'] for r in sel]))
        rv = float(np.mean([r['random_top5'] for r in sel]))
        nl = baseline.normalized_lift(mv, rv)
        nd = [r['ndcg_at_5'] for r in sel if r['ndcg_at_5'] is not None]
        strat_rows.append({'stratum': name, 'range': f'{lo}-{hi}', 'systems': len(sel),
                           'model_top5': round(mv, 4), 'random_top5': round(rv, 4),
                           'lift': round(mv - rv, 4),
                           'normalized_lift': round(nl, 4) if nl is not None else None,
                           'ndcg_at_5': round(float(np.mean(nd)), 4) if nd else None,
                           'room_targets': round(len(sel) * (1.0 - rv), 1),
                           'gained_targets': round(len(sel) * (mv - rv), 1)})
    audit.save_csv(out / 'top5_by_stratum.csv', strat_rows,
                   ('stratum', 'range', 'systems', 'model_top5', 'random_top5', 'lift',
                    'normalized_lift', 'ndcg_at_5', 'room_targets', 'gained_targets'))

    compare = load_compare(args.compare) if args.compare else []

    changed = []
    for source in sources:
        try:
            info = Path(source['path']).stat()
            if (info.st_size, info.st_mtime_ns) != (source['size_bytes'], source['mtime_ns']):
                changed.append(source['path'])
        except OSError:
            changed.append(source['path'])

    status = 'GRADED_COMPLETE' if not changed else 'CHECK_REQUIRED'
    report = {
        'version': VERSION, 'status': status, 'output_dir': str(out),
        'matrix': str(args.matrix), 'sources': sources,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'support_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (_ranking_path, ranking._baseline_path,
                                     baseline._support_path, roles._support_path,
                                     audit._support_path)},
        'python': sys.executable, 'xgboost': xgboost.__version__,
        'objective': 'rank', 'normalize': 'none', 'relevance': args.relevance,
        'relevance_bins': [list(b) for b in RELEVANCE_BINS],
        'grade_counts': grade_counts,
        'booster_effective_params': effective,
        'model_params': {'n_estimators': args.n_estimators, 'max_depth': args.max_depth,
                         'learning_rate': args.learning_rate, 'seed': args.seed},
        'excluded_categories': sorted(exclude),
        'features_used': len(feats), 'features_dropped': len(dropped), 'feature_list': feats,
        'value_qa_flagged': len(qa_rows),
        'pairable_binary': n_pair_bin, 'pairable_graded': n_pair_grad,
        'pairable_recovered': recovered, 'dev_systems_total': n_sys_total,
        'dev_poses': int(len(dev)), 'test_systems_untouched': test_systems,
        'eligible_systems': n_sys,
        'model_top5': round(model_rate, 4), 'random_top5': round(rnd_rate, 4),
        'lift': round(model_rate - rnd_rate, 4),
        'normalized_lift': round(norm, 4) if norm is not None else None,
        'mean_ndcg_at_5': round(mean_ndcg, 4) if mean_ndcg is not None else None,
        'net_targets': round(n_sys * (model_rate - rnd_rate), 1),
        'room_targets': round(sum(r['room_targets'] or 0 for r in strat_rows), 1),
        'fold_metrics': fold_rows, 'top5_by_stratum': strat_rows,
        'compare': [{k: v for k, v in c.items() if k != 'top5_by_stratum'} for c in compare],
        'changed_sources': changed, 'elapsed_seconds': time.perf_counter() - started,
        'policy': ['test 분할은 학습·평가에 사용하지 않았습니다.',
                   '시스템 내 정규화를 하지 않았습니다(29번 절제에서 해로움이 확인됨).',
                   '학습 등급은 sc_rmsd에서 만들고, 평가 기준은 sc_rmsd < 2A 이진 Top-5로 '
                   '29번과 동일하게 유지했습니다.',
                   '등급 3 이상이 near_native와 일치하는지 pose 단위로 검증했습니다.',
                   '18번 포켓 cluster 폴드를 그대로 썼고 한 시스템이 여러 폴드에 걸치지 '
                   '않는지 확인했습니다.',
                   '라벨 결측 pose는 학습에서 제외하고 예측·순위에는 포함했습니다.',
                   '동점은 pose_id SHA1 오름차순으로 결정적으로 깼습니다.',
                   'ndcg_exp_gain 등 인자가 실제로 적용됐는지 부스터 설정에서 확인했습니다.'],
        'notes': ['주지표는 난이도 층별 gained_targets 와 ndcg_at_5 입니다. 시스템별 AUC는 '
                  'rank_raw 0.669 / bin_z 0.674로 거의 같은데 어려운 층 lift가 '
                  '+0.14 / -0.03로 갈렸으므로 참고용으로만 둡니다.',
                  'pairable_recovered 가 0 이하면 등급 전환의 전제가 깨진 것이고, 쉬운 층 '
                  '적자의 원인이 학습 데이터가 아니라 feature 쪽이라는 신호입니다.']}
    helper.write_json(out / 'graded_report.json', report)
    helper.write_json(out / 'RUN_COMPLETE.json', {
        'outputs_written': True, 'graded_finished': True, 'status': status,
        'relevance': args.relevance, 'pairable_recovered': recovered,
        'model_top5': report['model_top5'], 'random_top5': report['random_top5'],
        'lift': report['lift'], 'net_targets': report['net_targets'],
        'mean_ndcg_at_5': report['mean_ndcg_at_5'],
        'features_used': len(feats), 'sources_unchanged': not changed})

    print('\nSTATUS:', status, flush=True)
    print(f'설정: objective=rank normalize=none relevance={args.relevance} '
          f'features={len(feats)}', flush=True)
    print(f'ELIGIBLE dev systems: {n_sys}', flush=True)
    print(f"MODEL {report['model_top5']} | RANDOM {report['random_top5']} "
          f"| LIFT {report['lift']:+} | 정규화 {report['normalized_lift']}", flush=True)
    print(f"순득실 {report['net_targets']:+} 타깃 / 남은 여백 {report['room_targets']} 타깃",
          flush=True)
    print(f"NDCG@5 평균 {report['mean_ndcg_at_5']} (주지표)", flush=True)

    print('\n=== 난이도 층별 ===', flush=True)
    print(f"{'stratum':11s}{'systems':>8s}{'model':>8s}{'random':>8s}{'lift':>8s}"
          f"{'정규화':>9s}{'NDCG@5':>9s}{'여백':>8s}{'획득':>8s}", flush=True)
    for r in strat_rows:
        if not r['systems']:
            continue
        print(f"{r['stratum']:11s}{r['systems']:8d}{r['model_top5']:8.3f}{r['random_top5']:8.3f}"
              f"{r['lift']:+8.3f}{r['normalized_lift']:9.3f}{r['ndcg_at_5']:9.3f}"
              f"{r['room_targets']:8.1f}{r['gained_targets']:+8.1f}", flush=True)

    if compare:
        print('\n=== 이전 실행 대비 (어려운 층 lift) ===', flush=True)
        print(f"{'run':28s}{'목적':8s}{'정규화':8s}{'등급':8s}"
              f"{'v_hard':>8s}{'hard':>8s}{'medium':>8s}{'전체':>9s}{'순득실':>9s}", flush=True)
        shown = compare + [{'label': out.name, 'objective': 'rank', 'normalize': 'none',
                            'relevance': args.relevance, 'lift': report['lift'],
                            'net_targets': report['net_targets'],
                            'top5_by_stratum': strat_rows}]
        for c in shown:
            by = {r['stratum']: r for r in (c['top5_by_stratum'] or [])}
            cells = []
            for name in ('very_hard', 'hard', 'medium'):
                val = (by.get(name) or {}).get('lift')
                cells.append(f'{val:+8.3f}' if val is not None else f'{"-":>8s}')
            net, lift = c.get('net_targets'), c.get('lift')
            print(f"{str(c['label'])[:27]:28s}{str(c['objective'] or '-'):8s}"
                  f"{str(c['normalize'] or '-'):8s}{str(c['relevance'] or '-'):8s}"
                  f"{''.join(cells)}"
                  f"{(f'{lift:+.4f}' if lift is not None else '-'):>9s}"
                  f"{(f'{net:+.1f}' if net is not None else '-'):>9s}", flush=True)
    return 0 if status == 'GRADED_COMPLETE' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, default=DEFAULT_MATRIX,
                        help='26번 통합 결과 폴더.')
    parser.add_argument('--relevance', choices=('graded', 'binary'), default='graded',
                        help='graded=sc_rmsd 등급, binary=29번 rank_raw 재현(절제용).')
    parser.add_argument('--exclude-categories', type=str, default=None,
                        help='쉼표 구분 후보군 제외. 예: cat2_gnina')
    parser.add_argument('--compare', type=str, default=DEFAULT_COMPARE,
                        help='쉼표 구분 이전 실행 폴더들(없으면 생략).')
    parser.add_argument('--no-compare', action='store_true', help='비교 출력을 끕니다.')
    parser.add_argument('--n-estimators', type=int, default=400)
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261102)
    parser.add_argument('--tag', type=str, default=None, help='출력 폴더 접미사.')
    args = parser.parse_args()
    if args.n_estimators < 1 or args.max_depth < 1 or args.threads < 1:
        parser.error('모델 인자는 양수여야 합니다.')
    args.matrix = args.matrix.absolute()
    args.compare = ([] if args.no_compare
                    else [p.strip() for p in args.compare.split(',') if p.strip()])
    return run(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
