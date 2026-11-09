#!/usr/bin/env python3
"""Process B 랭킹 모델(⑥-2). 28번이 드러낸 목적함수 오지정을 교정합니다.

28번 결과가 진단을 줬습니다. 전역 pose AUC는 0.80인데 Top-5 lift는 -0.0134였습니다.
near-native 비율이 시스템마다 크게 다르므로, '어느 시스템이 쉬운가'만 맞혀도 전역 AUC는
높게 나옵니다. 실제로 상위 feature에 시스템 안에서 상수인 cat8(lig_logp, lig_mol_weight)이
올라왔습니다. 모델은 우리가 원한 것(시스템 안에서 어느 pose가 좋은가)을 배우지 않았습니다.

세 가지를 함께 바꿉니다. 하나씩 바꾸면 효과가 잡음에 묻힙니다.

1) 시스템 내 정규화 (--normalize)
   feature를 시스템 단위로 z-score 또는 백분위 순위로 바꿉니다. 시스템 간 분산이 사라지므로
   모델이 절대값이 아니라 같은 타깃 안에서의 상대값만 보게 됩니다. 시스템 안에서 상수인
   feature는 모든 pose가 같은 값이 되어 순위에 영향을 주지 못합니다.

2) 랭킹 목적함수 (--objective rank)
   XGBRanker의 rank:ndcg를 쓰고 그룹을 시스템으로 묶습니다. 비교가 시스템 안에서만 일어나므로
   점수의 절대 크기가 무의미해지고 순서만 남습니다. 평가지표와 목적함수가 비로소 같은 것을
   봅니다. lambdarank_pair_method='topk'로 리스트 상단에 집중시킵니다. 우리 지표가 정확히
   '상위 5개'이기 때문입니다.
   쉬운 시스템에서 28번이 손실을 낸 것도 이 변경이 직접 겨냥합니다. near-native 95개 / 나쁜
   pose 5개인 시스템에서 역전 하나는 logloss에서 36.8만 항 중 5개였지만, 랭킹 손실에서는
   95x5=475개 쌍입니다.

3) 시스템별 AUC (항상 보고)
   전역 AUC는 28번에서 사람을 속였습니다. 시스템 내 AUC의 평균을 주지표로 둡니다. 이 값이
   0.5면 시스템 안에서 아무 판별력이 없다는 뜻이고, 전역 AUC가 아무리 높아도 소용없습니다.

평가·QA·층별 집계는 28번 함수를 그대로 import해 씁니다. 두 결과를 나란히 놓을 때 지표 정의가
달라지는 일을 막기 위함이며, 28번은 기준선 기록이므로 수정하지 않습니다.

출력: out/ranking/29_run_ranking[_TAG]/
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

# 28번을 모듈로 불러 평가·QA 코드를 공유합니다(복사하지 않습니다).
_baseline_path = Path(__file__).with_name('28_baseline_model.py')
_bspec = importlib.util.spec_from_file_location('baseline_support', _baseline_path)
baseline = importlib.util.module_from_spec(_bspec)
_bspec.loader.exec_module(baseline)
roles = baseline.roles
audit = baseline.audit
helper = baseline.helper

VERSION = 'ranking_0.1.0'
OUT_KIND = 'ranking'
OUT_BASE = '29_run_ranking'
KEYS = baseline.KEYS
TOPK = baseline.TOPK
STRATA = baseline.STRATA
BASE = baseline.BASE
DEFAULT_MATRIX = baseline.DEFAULT_MATRIX
DEFAULT_COMPARE = BASE / 'baseline' / '28_run_baseline_f63'


def group_layout(sorted_codes):
    """시스템별 시작 위치와 크기. 입력은 시스템 코드 오름차순으로 정렬되어 있어야 합니다."""
    import numpy as np
    uniq, starts, sizes = np.unique(sorted_codes, return_index=True, return_counts=True)
    return uniq, starts, sizes


def zscore_within_group(X, starts, sizes):
    """시스템 단위 z-score. 결측은 결측으로 남기고, 시스템 내 상수는 0으로 둡니다.

    reduceat로 그룹 합을 한 번에 구합니다. 시스템 안에서 상수인 feature는 표준편차가 0이라
    z-score가 정의되지 않는데, 이는 '이 타깃 안에서 구별 정보가 없다'는 뜻이므로 0을 줍니다.
    결측에 0을 주면 '평균값'이라는 거짓 정보가 되므로 결측은 NaN으로 유지합니다.
    """
    import numpy as np
    out = np.empty_like(X, dtype=np.float32)
    # 열을 쪼개 처리합니다. 36.8만 x 57을 float64로 한 번에 다루면 메모리가 불필요하게 큽니다.
    step = 16
    for lo in range(0, X.shape[1], step):
        block = X[:, lo:lo + step].astype(np.float64, copy=True)
        valid = ~np.isnan(block)
        filled = np.where(valid, block, 0.0)
        cnt = np.add.reduceat(valid.astype(np.float64), starts, axis=0)
        s1 = np.add.reduceat(filled, starts, axis=0)
        s2 = np.add.reduceat(filled * filled, starts, axis=0)
        safe = np.maximum(cnt, 1.0)
        mean = s1 / safe
        std = np.sqrt(np.maximum(s2 / safe - mean * mean, 0.0))
        usable = (cnt >= 2) & (std > 1e-12)
        mean_r = np.repeat(mean, sizes, axis=0)
        std_r = np.repeat(std, sizes, axis=0)
        usable_r = np.repeat(usable, sizes, axis=0)
        z = np.where(usable_r, (block - mean_r) / np.where(std_r > 0, std_r, 1.0), 0.0)
        out[:, lo:lo + step] = np.where(valid, z, np.nan).astype(np.float32)
    return out


def rank_within_group(X, feats, codes):
    """시스템 단위 백분위 순위(0~1). 이상치에 둔감한 정규화입니다."""
    import numpy as np
    import pandas as pd
    frame = pd.DataFrame(X, columns=feats)
    ranked = frame.groupby(codes, sort=False).rank(pct=True, na_option='keep')
    return ranked.to_numpy(dtype=np.float32)


def per_system_auc(codes, y, score, usable):
    """시스템 내 AUC의 평균. 28번의 전역 AUC가 사람을 속였으므로 이쪽을 주지표로 둡니다."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    frame = pd.DataFrame({'sys': codes[usable], 'y': y[usable], 's': score[usable]})
    vals = []
    for _, grp in frame.groupby('sys', sort=False):
        if grp['y'].nunique() < 2 or grp['s'].isna().all():
            continue
        vals.append(roc_auc_score(grp['y'], grp['s'].fillna(grp['s'].min() - 1.0)))
    if not vals:
        return None, 0
    return float(np.mean(vals)), len(vals)


def build_ranker(args):
    """시스템 내 순위를 직접 최적화하는 모델. 상단(Top-5)에 집중시킵니다."""
    from xgboost import XGBRanker
    common = dict(n_estimators=args.n_estimators, max_depth=args.max_depth,
                  learning_rate=args.learning_rate, subsample=0.8, colsample_bytree=0.8,
                  tree_method='hist', n_jobs=args.threads, random_state=args.seed,
                  objective='rank:ndcg', eval_metric=f'ndcg@{TOPK}')
    topk_args = dict(lambdarank_pair_method='topk',
                     lambdarank_num_pair_per_sample=max(TOPK, 8))
    try:
        return XGBRanker(**common, **topk_args), 'xgboost_ranker_topk'
    except TypeError:
        # 이 xgboost 버전이 lambdarank 세부 인자를 모르면 기본 쌍 생성으로 돌아갑니다.
        return XGBRanker(**common), 'xgboost_ranker_default'


def run(args):
    """시스템 내 정규화 + 랭킹 목적함수로 out-of-fold 점수를 만들고 Top-5를 평가합니다."""
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
    from sklearn.metrics import roc_auc_score, average_precision_score

    matrix_path = args.matrix / 'unified_all.parquet'
    catalog_path = args.matrix / 'feature_catalog.csv'
    for p in (matrix_path, catalog_path):
        if not p.exists():
            raise FileNotFoundError(f'입력이 없습니다: {p}')
    sources = [helper.fingerprint(matrix_path), helper.fingerprint(catalog_path)]

    catalog = pd.read_csv(catalog_path).set_index('feature')
    meta_cols = ['system_id', 'model', 'seed', 'sample', 'pose_id', 'near_native',
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
    # 랭킹 목적함수는 같은 그룹의 행이 붙어 있어야 하므로 시스템 기준으로 정렬합니다.
    dev = dev.sort_values(KEYS, kind='mergesort').reset_index(drop=True)
    dev['cv_fold'] = dev['cv_fold'].astype(int)
    folds = sorted(int(f) for f in dev['cv_fold'].unique())
    print(f'DEV poses {len(dev)} | systems {dev["system_id"].nunique()} | folds {folds} '
          f'| TEST systems {test_systems} (봉인)', flush=True)

    exclude = {c.strip() for c in (args.exclude_categories or '').split(',') if c.strip()}
    feats, dropped = baseline.select_features(dev, catalog, exclude)
    audit.save_csv(out / 'dropped_features.csv',
                   dropped or [{'feature': None, 'category': None, 'reason': None}],
                   ('feature', 'category', 'reason'))
    print(f'FEATURES used {len(feats)} | dropped {len(dropped)}', flush=True)
    for reason in sorted({d['reason'] for d in dropped if d['reason']}):
        names = [d['feature'] for d in dropped if d['reason'] == reason]
        print(f'  [{reason}] {len(names)}: {names[:6]}{" ..." if len(names) > 6 else ""}', flush=True)
    if not feats:
        raise ValueError('사용할 feature가 없습니다.')

    qa_rows = baseline.value_qa(dev, feats, catalog)
    audit.save_csv(out / 'value_qa.csv',
                   qa_rows or [{'feature': None, 'category': None, 'rule': None, 'n_inf': 0,
                                'n_extreme': 0, 'n_rule_violation': 0, 'min': None, 'max': None}],
                   ('feature', 'category', 'rule', 'n_inf', 'n_extreme', 'n_rule_violation',
                    'min', 'max'))
    print(f'VALUE_QA 이상 {len(qa_rows)}건 (삭제하지 않고 보고만)', flush=True)

    # 시스템 코드. 정렬된 상태이므로 코드는 단조증가하고 그룹은 연속입니다.
    codes = pd.factorize(dev['system_id'], sort=False)[0]
    assert np.all(np.diff(codes) >= 0), '시스템 그룹이 연속이 아닙니다.'
    _, starts, sizes = group_layout(codes)

    X_raw = dev[feats].to_numpy(dtype=np.float32, na_value=np.nan)
    y = dev['near_native'].to_numpy(dtype=np.int8)
    fold_of = dev['cv_fold'].to_numpy()
    known = dev['label_known'].to_numpy(dtype=bool)
    print(f'MATRIX raw: {X_raw.shape} float32 ({X_raw.nbytes / 1e6:.0f} MB) '
          f'| 결측 {int(np.isnan(X_raw).sum())} 칸 | 시스템 {len(sizes)}', flush=True)

    # 폴드가 시스템 단위로 배정되었는지 확인합니다. 한 시스템이 두 폴드에 걸치면
    # 랭킹 그룹이 학습/검증으로 쪼개져 누수가 생깁니다.
    fold_span = pd.Series(fold_of).groupby(codes).nunique().max()
    if int(fold_span) != 1:
        raise ValueError(f'한 시스템이 여러 폴드에 걸쳐 있습니다(max {fold_span}).')

    dead_after = []
    if args.normalize == 'zscore':
        X = zscore_within_group(X_raw, starts, sizes)
    elif args.normalize == 'rank':
        X = rank_within_group(X_raw, feats, codes)
    else:
        X = X_raw
    if args.normalize != 'none':
        flat = X[~np.isnan(X)]
        print(f'NORMALIZE {args.normalize}: 전체 평균 {float(flat.mean()):.4f} '
              f'표준편차 {float(flat.std()):.4f} | 결측 {int(np.isnan(X).sum())} 칸', flush=True)
        # 시스템 안에서 상수였던 feature(cat8 등)는 정규화 후 전역 상수가 되어 어떤 분기도
        # 만들지 못합니다. 죽은 것을 명시해 두면 feature 중요도를 잘못 읽는 일을 막습니다.
        dead_after = []
        for i, name in enumerate(feats):
            col = X[:, i]
            vals = col[~np.isnan(col)]
            if vals.size < 2 or float(vals.std()) < 1e-12:
                dead_after.append(name)
        print(f'  정규화 후 전역 상수 {len(dead_after)}개(순위에 기여하지 않음): '
              f'{dead_after[:14]}{" ..." if len(dead_after) > 14 else ""}', flush=True)

    dev['score'] = np.nan
    fold_rows = []
    importances = np.zeros(len(feats), dtype=np.float64)
    backend = None
    for k in folds:
        tr = (fold_of != k) & known
        va = fold_of == k
        if args.objective == 'rank':
            # 학습 부분집합에서 양성 또는 음성이 없는 시스템은 비교할 쌍이 없어
            # 기울기에 기여하지 않습니다. 그룹에서 빼 둡니다.
            lab = pd.Series(y[tr]).groupby(codes[tr], sort=False).agg(['min', 'max'])
            dead = lab.index[lab['min'] == lab['max']].to_numpy()
            if dead.size:
                tr = tr & ~np.isin(codes, dead)
            model, backend = build_ranker(args)
            model.fit(X[tr], y[tr], qid=codes[tr])
            groups_used = int(np.unique(codes[tr]).size)
        else:
            model, backend = baseline.build_model(args)
            model.fit(X[tr], y[tr])
            groups_used = int(np.unique(codes[tr]).size)
        pred = (model.predict_proba(X[va])[:, 1] if hasattr(model, 'predict_proba')
                else model.predict(X[va]))
        dev.loc[va, 'score'] = pred
        lab_mask = va & known
        auc = ap = None
        if len(np.unique(y[lab_mask])) > 1:
            auc = float(roc_auc_score(y[lab_mask], dev.loc[lab_mask, 'score']))
            ap = float(average_precision_score(y[lab_mask], dev.loc[lab_mask, 'score']))
        sys_auc, n_sys_auc = per_system_auc(codes, y, dev['score'].to_numpy(), lab_mask)
        fold_rows.append({'cv_fold': int(k), 'train_poses': int(tr.sum()),
                          'train_groups': groups_used, 'valid_poses': int(va.sum()),
                          'valid_labeled': int(lab_mask.sum()),
                          'global_auc': round(auc, 4) if auc is not None else None,
                          'per_system_auc': round(sys_auc, 4) if sys_auc is not None else None,
                          'avg_precision': round(ap, 4) if ap is not None else None})
        print(f'  fold {k}: train {int(tr.sum())} ({groups_used} systems) '
              f'valid {int(va.sum())} | 전역 AUC {"-" if auc is None else round(auc, 4)} '
              f'| 시스템별 AUC {"-" if sys_auc is None else round(sys_auc, 4)}', flush=True)
        imp = getattr(model, 'feature_importances_', None)
        if imp is not None and len(imp) == len(feats):
            importances += np.asarray(imp, dtype=np.float64)
    print('MODEL backend:', backend, flush=True)

    if importances.sum() > 0:
        order = np.argsort(-importances)
        imp_rows = [{'feature': feats[i], 'category': str(catalog.loc[feats[i], 'category']),
                     'importance': round(float(importances[i] / len(folds)), 6)} for i in order]
        audit.save_csv(out / 'feature_importance.csv', imp_rows,
                       ('feature', 'category', 'importance'))
        print('TOP FEATURES:', ', '.join(r['feature'] for r in imp_rows[:8]), flush=True)

    audit.save_csv(out / 'fold_metrics.csv', fold_rows,
                   ('cv_fold', 'train_poses', 'train_groups', 'valid_poses', 'valid_labeled',
                    'global_auc', 'per_system_auc', 'avg_precision'))

    scores = dev['score'].to_numpy()
    oof_sys_auc, n_oof_sys = per_system_auc(codes, y, scores, known)
    oof_global_auc = (float(roc_auc_score(y[known], scores[known]))
                      if len(np.unique(y[known])) > 1 else None)

    dev['_tie'] = dev['pose_id'].map(baseline.tie_key)
    buf = io.BytesIO()
    dev[KEYS + ['pose_id', 'score']].to_parquet(buf, engine='pyarrow', index=False,
                                                compression='snappy')
    with helper.checked_path(out / 'scores.parquet').open('xb') as handle:
        handle.write(buf.getvalue())

    sys_rows = []
    for sid, grp in dev.groupby('system_id'):
        m = int(grp['near_native'].sum())
        if m == 0:
            continue
        sys_rows.append({'system_id': sid, 'n_pose': int(len(grp)), 'n_near_native': m,
                         'random_top5': round(baseline.random_topk(len(grp), m), 4),
                         'model_top5': bool(baseline.topk_success(grp))})
    audit.save_csv(out / 'system_top5.csv', sys_rows,
                   ('system_id', 'n_pose', 'n_near_native', 'random_top5', 'model_top5'))
    if not sys_rows:
        raise ValueError('eligible 시스템이 없습니다.')

    n_sys = len(sys_rows)
    model_rate = float(np.mean([r['model_top5'] for r in sys_rows]))
    rnd_rate = float(np.mean([r['random_top5'] for r in sys_rows]))
    norm = baseline.normalized_lift(model_rate, rnd_rate)

    strat_rows = []
    for name, lo, hi in STRATA:
        sel = [r for r in sys_rows if lo <= r['random_top5'] < hi]
        if not sel:
            strat_rows.append({'stratum': name, 'range': f'{lo}-{hi}', 'systems': 0,
                               'model_top5': None, 'random_top5': None, 'lift': None,
                               'normalized_lift': None, 'room_targets': None,
                               'gained_targets': None})
            continue
        mv = float(np.mean([r['model_top5'] for r in sel]))
        rv = float(np.mean([r['random_top5'] for r in sel]))
        nl = baseline.normalized_lift(mv, rv)
        strat_rows.append({'stratum': name, 'range': f'{lo}-{hi}', 'systems': len(sel),
                           'model_top5': round(mv, 4), 'random_top5': round(rv, 4),
                           'lift': round(mv - rv, 4),
                           'normalized_lift': round(nl, 4) if nl is not None else None,
                           # 타깃 개수로 환산해야 비율 표가 숨기는 실제 득실이 보입니다.
                           'room_targets': round(len(sel) * (1.0 - rv), 1),
                           'gained_targets': round(len(sel) * (mv - rv), 1)})
    audit.save_csv(out / 'top5_by_stratum.csv', strat_rows,
                   ('stratum', 'range', 'systems', 'model_top5', 'random_top5', 'lift',
                    'normalized_lift', 'room_targets', 'gained_targets'))

    # 28번(또는 지정한 실행)과 층별로 나란히 비교합니다.
    compare = None
    if args.compare is not None and (args.compare / 'baseline_report.json').exists():
        prev = json.loads((args.compare / 'baseline_report.json').read_text(encoding='utf-8'))
        compare = {'path': str(args.compare), 'model_top5': prev.get('model_top5'),
                   'random_top5': prev.get('random_top5'), 'lift': prev.get('lift'),
                   'mean_auc': prev.get('mean_auc'),
                   'top5_by_stratum': prev.get('top5_by_stratum')}

    changed = []
    for source in sources:
        try:
            info = Path(source['path']).stat()
            if (info.st_size, info.st_mtime_ns) != (source['size_bytes'], source['mtime_ns']):
                changed.append(source['path'])
        except OSError:
            changed.append(source['path'])

    status = 'RANKING_COMPLETE' if not changed else 'CHECK_REQUIRED'
    report = {
        'version': VERSION, 'status': status, 'output_dir': str(out),
        'matrix': str(args.matrix), 'sources': sources,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'support_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (_baseline_path, baseline._support_path,
                                     roles._support_path, audit._support_path)},
        'python': sys.executable, 'model_backend': backend,
        'objective': args.objective, 'normalize': args.normalize,
        'model_params': {'n_estimators': args.n_estimators, 'max_depth': args.max_depth,
                         'learning_rate': args.learning_rate, 'seed': args.seed},
        'excluded_categories': sorted(exclude),
        'features_used': len(feats), 'features_dropped': len(dropped), 'feature_list': feats,
        'value_qa_flagged': len(qa_rows), 'dead_after_normalize': dead_after,
        'dev_poses': int(len(dev)), 'dev_systems': int(len(sizes)),
        'test_systems_untouched': test_systems,
        'eligible_systems': n_sys,
        'model_top5': round(model_rate, 4), 'random_top5': round(rnd_rate, 4),
        'lift': round(model_rate - rnd_rate, 4),
        'normalized_lift': round(norm, 4) if norm is not None else None,
        'net_targets': round(n_sys * (model_rate - rnd_rate), 1),
        'room_targets': round(sum(r['room_targets'] or 0 for r in strat_rows), 1),
        'oof_global_auc': round(oof_global_auc, 4) if oof_global_auc is not None else None,
        'oof_per_system_auc': round(oof_sys_auc, 4) if oof_sys_auc is not None else None,
        'oof_per_system_auc_n': n_oof_sys,
        'fold_metrics': fold_rows, 'top5_by_stratum': strat_rows, 'compare': compare,
        'changed_sources': changed, 'elapsed_seconds': time.perf_counter() - started,
        'policy': ['test 분할은 학습·평가에 사용하지 않았습니다.',
                   '18번 포켓 cluster 폴드를 그대로 썼고, 한 시스템이 여러 폴드에 걸치지 '
                   '않는지 확인했습니다.',
                   '랭킹 그룹은 시스템이며, 비교는 시스템 안에서만 일어납니다.',
                   '라벨 결측 pose는 학습에서 제외하고 예측·순위에는 포함했습니다.',
                   '정규화에서 결측은 NaN으로 유지했습니다(평균 대입을 하지 않았습니다).',
                   '시스템 안에서 상수인 feature는 정규화 후 0이 되어 순위에 기여하지 않습니다.',
                   '동점은 pose_id SHA1 오름차순으로 결정적으로 깼습니다.'],
        'notes': ['주지표는 oof_per_system_auc 입니다. 전역 AUC는 28번에서 시스템 난이도만 '
                  '맞혀도 높게 나오는 것이 확인됐으므로 참고용입니다.',
                  'gained_targets 는 층별 lift를 타깃 개수로 환산한 값입니다. 비율 표가 '
                  '숨기는 실제 득실(쉬운 층에서 토해내는 양)이 여기 보입니다.',
                  'eligible dev에서 무작위가 이미 0.91대이므로 남은 여백 전체가 '
                  'room_targets 개뿐입니다. 이 값이 성능 상한입니다.']}
    helper.write_json(out / 'ranking_report.json', report)
    helper.write_json(out / 'RUN_COMPLETE.json', {
        'outputs_written': True, 'ranking_finished': True, 'status': status,
        'objective': args.objective, 'normalize': args.normalize,
        'model_top5': report['model_top5'], 'random_top5': report['random_top5'],
        'lift': report['lift'], 'net_targets': report['net_targets'],
        'oof_per_system_auc': report['oof_per_system_auc'],
        'features_used': len(feats), 'sources_unchanged': not changed})

    print('\nSTATUS:', status, flush=True)
    print(f'설정: objective={args.objective} normalize={args.normalize} '
          f'features={len(feats)}', flush=True)
    print(f'ELIGIBLE dev systems: {n_sys}', flush=True)
    print(f"MODEL {report['model_top5']} | RANDOM {report['random_top5']} "
          f"| LIFT {report['lift']:+} | 정규화 {report['normalized_lift']}", flush=True)
    print(f"순득실 {report['net_targets']:+} 타깃 / 남은 여백 {report['room_targets']} 타깃",
          flush=True)
    print(f"AUC 시스템별 {report['oof_per_system_auc']} (주지표, n={n_oof_sys}) "
          f"| 전역 {report['oof_global_auc']} (참고)", flush=True)

    print('\n=== 난이도 층별 ===', flush=True)
    print(f"{'stratum':11s}{'systems':>8s}{'model':>8s}{'random':>8s}{'lift':>8s}"
          f"{'정규화':>9s}{'여백':>8s}{'획득':>8s}", flush=True)
    for r in strat_rows:
        if not r['systems']:
            continue
        print(f"{r['stratum']:11s}{r['systems']:8d}{r['model_top5']:8.3f}{r['random_top5']:8.3f}"
              f"{r['lift']:+8.3f}{r['normalized_lift']:9.3f}{r['room_targets']:8.1f}"
              f"{r['gained_targets']:+8.1f}", flush=True)

    if compare and compare['top5_by_stratum']:
        prev_by = {r['stratum']: r for r in compare['top5_by_stratum']}
        print(f"\n=== 28번 대비 (이전 전체 lift {compare['lift']}) ===", flush=True)
        print(f"{'stratum':11s}{'이전':>9s}{'현재':>9s}{'변화':>9s}", flush=True)
        for r in strat_rows:
            p = prev_by.get(r['stratum'])
            if not r['systems'] or not p or p.get('lift') is None:
                continue
            print(f"{r['stratum']:11s}{p['lift']:+9.3f}{r['lift']:+9.3f}"
                  f"{r['lift'] - p['lift']:+9.3f}", flush=True)
    return 0 if status == 'RANKING_COMPLETE' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, default=DEFAULT_MATRIX,
                        help='26번 통합 결과 폴더.')
    parser.add_argument('--objective', choices=('rank', 'binary'), default='rank',
                        help='rank=XGBRanker(rank:ndcg), binary=28번과 같은 분류기(절제용).')
    parser.add_argument('--normalize', choices=('zscore', 'rank', 'none'), default='zscore',
                        help='시스템 내 정규화 방식. none은 절제용입니다.')
    parser.add_argument('--exclude-categories', type=str, default=None,
                        help='쉼표 구분 후보군 제외. 예: cat8_ligand')
    parser.add_argument('--compare', type=Path, default=DEFAULT_COMPARE,
                        help='나란히 비교할 28번 결과 폴더(없으면 생략).')
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
    args.compare = None if args.no_compare else args.compare.absolute()
    return run(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
