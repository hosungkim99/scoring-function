#!/usr/bin/env python3
"""Process B 베이스라인(⑥). 통합 feature 행렬로 pose 스코어링 모델을 학습하고 Top-5로 평가합니다.

이 단계의 목적은 성능을 끌어올리는 것이 아니라 **접근이 작동하는지 확인**하는 것입니다.
지금까지 어떤 단일 feature도 어려운 타깃에서 Top-5 lift +0.17을 넘지 못했고 상위권이 모두
'계면 크기'라는 한 축이었습니다. 따라서 이 프로젝트의 가치는 전적으로 feature 조합에서 나와야
하는데, 그 가정은 아직 검증된 적이 없습니다. 이 스크립트가 그 첫 검증입니다.

설계
- 학습·평가는 dev(18번 분할)에서만 하고 test 923 시스템은 로드조차 하지 않습니다.
- 18번이 포켓 cluster 단위로 미리 배정한 cv_fold(0~4)를 그대로 써서 out-of-fold 점수를
  만듭니다. 같은 cluster가 학습과 검증에 동시에 들어가지 않습니다(누수 차단).
- 라벨이 결측인 pose는 학습에서 제외합니다(결측을 음성으로 쓰면 라벨 노이즈가 섞입니다).
  다만 예측·순위에는 모든 pose가 들어갑니다. 실제 사용 상황에서는 어느 pose에 라벨이 있는지
  알 수 없고, 평가도 B① 명세대로 '결측=비near-native'(보수적)로 처리하기 때문입니다.
- feature를 0으로 채우지 않습니다. 부스팅이 결측을 그대로 분기 처리합니다.
- 상수·전부널·비수치 feature는 자동 제외하고 사유를 남깁니다. 값 범위 이상도 함께 점검합니다
  (학습을 조용히 오염시키는 값을 거르기 위함이며, 경고만 하고 자동 삭제하지는 않습니다).
- 성능은 전체와 **난이도 층별**로 모두 보고합니다. eligible의 87%가 쉬운 타깃이라 전체 평균만
  보면 개선이 묻힙니다. 층마다 천장이 다르므로 raw lift와 정규화 lift를 함께 냅니다.
    정규화 lift = (모델 - 무작위) / (1 - 무작위) = 남은 여백 중 몇 %를 잡았는가

모델은 near_native 이진 분류기이고 예측 확률로 타깃 안에서 순위를 매깁니다. 순위 학습
(LambdaMART 등)은 이후 변형으로 두고, 베이스라인은 단순하고 해석 가능한 쪽을 택했습니다.

출력: out/baseline/28_run_baseline[_TAG]/ (scores.parquet은 25번 하네스 입력으로 쓸 수 있습니다)
"""
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import importlib.util
import io
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

VERSION = 'baseline_0.1.0'
OUT_KIND = 'baseline'
OUT_BASE = '28_run_baseline'
KEYS = ['system_id', 'model', 'seed', 'sample']
BASE = helper.ALLOWED_ROOT / 'Scoring_Function' / 'out'
DEFAULT_MATRIX = BASE / 'feature_matrix_all' / '26_run_integrate_all_final'
TOPK = 5
# 난이도 층: 무작위 Top-5 성공확률 구간(결과를 보기 전에 고정).
STRATA = [('very_hard', 0.0, 0.1), ('hard', 0.1, 0.3), ('medium', 0.3, 0.5),
          ('easy', 0.5, 0.8), ('very_easy', 0.8, 1.01)]
# 값 범위 점검 규칙: (이름 판별 함수, 규칙 이름, 위반 판별 함수).
QA_RULES = [
    (lambda n: n.startswith('affinity'), 'binding_energy_should_be_negative', lambda s: s > 0),
    (lambda n: 'probability' in n, 'probability_in_0_1', lambda s: (s < 0) | (s > 1)),
    (lambda n: 'fraction' in n, 'fraction_in_0_1', lambda s: (s < 0) | (s > 1)),
    (lambda n: any(k in n for k in ('_count', 'pairs', 'atoms', 'residues', 'sasa', 'dist', 'bonds')),
     'should_be_non_negative', lambda s: s < 0),
]


def tie_key(pose_id):
    """동점을 결정적으로 깨되 생성 모델과 무관한 해시 키(B① 명세)."""
    return int(hashlib.sha1(str(pose_id).encode()).hexdigest(), 16)


def random_topk(n, m, k=TOPK):
    """크기 n 풀에 near-native m개일 때 무작위 Top-k가 하나 이상 포함할 확률."""
    if m <= 0:
        return 0.0
    if n <= k:
        return 1.0
    return 1.0 - comb(n - m, k) / comb(n, k)


def normalized_lift(model_rate, random_rate):
    """남아 있던 여백 중 몇 %를 잡았는가. 층마다 천장이 달라 raw lift로는 비교가 안 됩니다."""
    room = 1.0 - random_rate
    if room <= 1e-9:
        return None
    return (model_rate - random_rate) / room


def build_model(args):
    """결측을 그대로 처리하는 부스팅 구현을 고릅니다(xgboost > lightgbm > sklearn)."""
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=args.n_estimators, max_depth=args.max_depth,
            learning_rate=args.learning_rate, subsample=0.8, colsample_bytree=0.8,
            tree_method='hist', n_jobs=args.threads, eval_metric='logloss',
            random_state=args.seed), 'xgboost'
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=args.n_estimators, max_depth=args.max_depth,
            learning_rate=args.learning_rate, subsample=0.8, colsample_bytree=0.8,
            n_jobs=args.threads, random_state=args.seed, verbose=-1), 'lightgbm'
    except ImportError:
        pass
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=args.n_estimators, max_depth=args.max_depth,
        learning_rate=args.learning_rate, random_state=args.seed), 'sklearn_hist'


def select_features(frame, catalog, exclude_categories):
    """수치형이고 정보량이 있는 feature만 고르고, 제외 사유를 모두 남깁니다."""
    import pandas as pd
    chosen, dropped = [], []
    for name, row in catalog.iterrows():
        cat = str(row['category'])
        if cat in exclude_categories:
            dropped.append({'feature': name, 'category': cat, 'reason': 'category_excluded'})
        elif name not in frame.columns:
            dropped.append({'feature': name, 'category': cat, 'reason': 'missing_column'})
        elif not pd.api.types.is_numeric_dtype(frame[name]):
            dropped.append({'feature': name, 'category': cat, 'reason': 'not_numeric'})
        elif int(frame[name].notna().sum()) == 0:
            dropped.append({'feature': name, 'category': cat, 'reason': 'all_null'})
        elif int(frame[name].dropna().nunique()) <= 1:
            dropped.append({'feature': name, 'category': cat, 'reason': 'constant'})
        else:
            chosen.append(name)
    return chosen, dropped


def value_qa(frame, feats, catalog):
    """값 범위 이상을 점검합니다(자동 삭제하지 않고 보고만 합니다)."""
    import numpy as np
    import pandas as pd
    rows = []
    for name in feats:
        # parquet에서 온 nullable 확장 dtype(Int64/boolean)은 np.isinf가 받지 않으므로
        # 먼저 평범한 수치형으로 강제 변환합니다.
        vals = pd.to_numeric(frame[name], errors='coerce').dropna()
        if vals.empty:
            continue
        n_inf = int(np.isinf(vals).sum())
        finite = vals[np.isfinite(vals)]
        n_extreme = int((finite.abs() > 1e6).sum()) if not finite.empty else 0
        rule, n_violation = None, 0
        for matches, rule_name, violates in QA_RULES:
            if matches(name):
                rule = rule_name
                n_violation = int(violates(finite).sum()) if not finite.empty else 0
                break
        if n_inf or n_extreme or n_violation:
            rows.append({'feature': name, 'category': str(catalog.loc[name, 'category']),
                         'rule': rule, 'n_inf': n_inf, 'n_extreme': n_extreme,
                         'n_rule_violation': n_violation,
                         'min': round(float(finite.min()), 4) if not finite.empty else None,
                         'max': round(float(finite.max()), 4) if not finite.empty else None})
    return rows


def topk_success(group, score_col='score', near_col='near_native'):
    """점수 내림차순 Top-5에 near-native가 있는지(동점은 해시로 결정적 처리)."""
    ordered = group.sort_values([score_col, '_tie'], ascending=[False, True],
                                na_position='last', kind='mergesort')
    return bool(ordered.head(TOPK)[near_col].to_numpy().any())


def run(args):
    """dev에서 out-of-fold 점수를 만들고 Top-5를 전체·난이도별로 평가합니다."""
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

    # test는 여기서 즉시 분리하고 이후 일절 사용하지 않습니다.
    test_systems = int(frame.loc[frame['split'] == 'test', 'system_id'].nunique())
    dev = frame[frame['split'] == 'dev'].copy()
    del frame
    dev = dev.reset_index(drop=True)
    dev['cv_fold'] = dev['cv_fold'].astype(int)
    folds = sorted(dev['cv_fold'].unique())
    print(f'DEV poses {len(dev)} | systems {dev["system_id"].nunique()} | folds {folds} '
          f'| TEST systems {test_systems} (봉인)', flush=True)

    exclude = {c.strip() for c in (args.exclude_categories or '').split(',') if c.strip()}
    feats, dropped = select_features(dev, catalog, exclude)
    audit.save_csv(out / 'dropped_features.csv',
                   dropped or [{'feature': None, 'category': None, 'reason': None}],
                   ('feature', 'category', 'reason'))
    print(f'FEATURES used {len(feats)} | dropped {len(dropped)}', flush=True)
    for reason in sorted({d['reason'] for d in dropped if d['reason']}):
        names = [d['feature'] for d in dropped if d['reason'] == reason]
        print(f'  [{reason}] {len(names)}: {names[:6]}{" ..." if len(names) > 6 else ""}', flush=True)
    if not feats:
        raise ValueError('사용할 feature가 없습니다.')

    qa_rows = value_qa(dev, feats, catalog)
    audit.save_csv(out / 'value_qa.csv',
                   qa_rows or [{'feature': None, 'category': None, 'rule': None, 'n_inf': 0,
                                'n_extreme': 0, 'n_rule_violation': 0, 'min': None, 'max': None}],
                   ('feature', 'category', 'rule', 'n_inf', 'n_extreme', 'n_rule_violation', 'min', 'max'))
    if qa_rows:
        print(f'VALUE_QA 이상 {len(qa_rows)}건 (삭제하지 않고 보고만):', flush=True)
        for r in qa_rows[:8]:
            print(f"  {r['feature']}: inf={r['n_inf']} extreme={r['n_extreme']} "
                  f"violation={r['n_rule_violation']} ({r['rule']}) range=[{r['min']}, {r['max']}]", flush=True)
    else:
        print('VALUE_QA: 이상 없음', flush=True)

    _, backend = build_model(args)
    print('MODEL backend:', backend, flush=True)

    # parquet의 nullable 확장 dtype(Int64/boolean)을 그대로 넘기면 xgboost가
    # "dtypes for data must be int, float, bool or category"로 거부합니다.
    # na_value=nan 으로 한 번에 평범한 float32 행렬을 만들어 결측 의미를 유지합니다.
    X = dev[feats].to_numpy(dtype=np.float32, na_value=np.nan)
    y = dev['near_native'].to_numpy(dtype=np.int8)
    fold_of = dev['cv_fold'].to_numpy()
    known = dev['label_known'].to_numpy(dtype=bool)
    print(f'MATRIX for fit: {X.shape} float32 '
          f'({X.nbytes / 1e6:.0f} MB) | 결측 {int(np.isnan(X).sum())} 칸', flush=True)

    # out-of-fold 점수. 18번 폴드를 그대로 써서 포켓 cluster 누수를 막습니다.
    dev['score'] = np.nan
    fold_rows = []
    importances = np.zeros(len(feats), dtype=np.float64)
    for k in folds:
        tr = (fold_of != k) & known
        va = fold_of == k
        model, _ = build_model(args)
        model.fit(X[tr], y[tr])
        pred = (model.predict_proba(X[va])[:, 1] if hasattr(model, 'predict_proba')
                else model.predict(X[va]))
        dev.loc[va, 'score'] = pred
        lab = va & known
        auc = ap = None
        if len(np.unique(y[lab])) > 1:
            auc = float(roc_auc_score(y[lab], dev.loc[lab, 'score']))
            ap = float(average_precision_score(y[lab], dev.loc[lab, 'score']))
        fold_rows.append({'cv_fold': int(k), 'train_poses': int(tr.sum()),
                          'valid_poses': int(va.sum()), 'valid_labeled': int(lab.sum()),
                          'auc': round(auc, 4) if auc is not None else None,
                          'avg_precision': round(ap, 4) if ap is not None else None})
        print(f'  fold {k}: train {int(tr.sum())} valid {int(va.sum())} '
              f'AUC {"-" if auc is None else round(auc, 4)}', flush=True)
        imp = getattr(model, 'feature_importances_', None)
        if imp is not None and len(imp) == len(feats):
            importances += np.asarray(imp, dtype=np.float64)
    audit.save_csv(out / 'fold_metrics.csv', fold_rows,
                   ('cv_fold', 'train_poses', 'valid_poses', 'valid_labeled', 'auc', 'avg_precision'))

    if importances.sum() > 0:
        order = np.argsort(-importances)
        imp_rows = [{'feature': feats[i], 'category': str(catalog.loc[feats[i], 'category']),
                     'importance': round(float(importances[i] / len(folds)), 6)} for i in order]
        audit.save_csv(out / 'feature_importance.csv', imp_rows,
                       ('feature', 'category', 'importance'))
        print('TOP FEATURES:', ', '.join(r['feature'] for r in imp_rows[:8]), flush=True)

    dev['_tie'] = dev['pose_id'].map(tie_key)
    buf = io.BytesIO()
    dev[KEYS + ['pose_id', 'score']].to_parquet(buf, engine='pyarrow', index=False,
                                                compression='snappy')
    with helper.checked_path(out / 'scores.parquet').open('xb') as handle:
        handle.write(buf.getvalue())

    # 시스템별 Top-5. eligible(near-native 보유)만 분모에 넣습니다.
    sys_rows = []
    for sid, grp in dev.groupby('system_id'):
        m = int(grp['near_native'].sum())
        if m == 0:
            continue
        sys_rows.append({'system_id': sid, 'n_pose': int(len(grp)), 'n_near_native': m,
                         'random_top5': round(random_topk(len(grp), m), 4),
                         'model_top5': bool(topk_success(grp))})
    audit.save_csv(out / 'system_top5.csv', sys_rows,
                   ('system_id', 'n_pose', 'n_near_native', 'random_top5', 'model_top5'))
    if not sys_rows:
        raise ValueError('eligible 시스템이 없습니다.')

    n_sys = len(sys_rows)
    model_rate = float(np.mean([r['model_top5'] for r in sys_rows]))
    rnd_rate = float(np.mean([r['random_top5'] for r in sys_rows]))
    norm = normalized_lift(model_rate, rnd_rate)

    strat_rows = []
    for name, lo, hi in STRATA:
        sel = [r for r in sys_rows if lo <= r['random_top5'] < hi]
        if not sel:
            strat_rows.append({'stratum': name, 'range': f'{lo}-{hi}', 'systems': 0,
                               'model_top5': None, 'random_top5': None,
                               'lift': None, 'normalized_lift': None})
            continue
        mv = float(np.mean([r['model_top5'] for r in sel]))
        rv = float(np.mean([r['random_top5'] for r in sel]))
        nl = normalized_lift(mv, rv)
        strat_rows.append({'stratum': name, 'range': f'{lo}-{hi}', 'systems': len(sel),
                           'model_top5': round(mv, 4), 'random_top5': round(rv, 4),
                           'lift': round(mv - rv, 4),
                           'normalized_lift': round(nl, 4) if nl is not None else None})
    audit.save_csv(out / 'top5_by_stratum.csv', strat_rows,
                   ('stratum', 'range', 'systems', 'model_top5', 'random_top5',
                    'lift', 'normalized_lift'))

    changed = []
    for source in sources:
        try:
            info = Path(source['path']).stat()
            if (info.st_size, info.st_mtime_ns) != (source['size_bytes'], source['mtime_ns']):
                changed.append(source['path'])
        except OSError:
            changed.append(source['path'])

    aucs = [r['auc'] for r in fold_rows if r['auc'] is not None]
    status = 'BASELINE_COMPLETE' if not changed else 'CHECK_REQUIRED'
    report = {
        'version': VERSION, 'status': status, 'output_dir': str(out),
        'matrix': str(args.matrix), 'sources': sources,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'support_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (_support_path, roles._support_path, audit._support_path)},
        'python': sys.executable, 'model_backend': backend,
        'model_params': {'n_estimators': args.n_estimators, 'max_depth': args.max_depth,
                         'learning_rate': args.learning_rate, 'seed': args.seed},
        'excluded_categories': sorted(exclude),
        'features_used': len(feats), 'features_dropped': len(dropped), 'feature_list': feats,
        'value_qa_flagged': len(qa_rows),
        'dev_poses': int(len(dev)), 'dev_systems': int(dev['system_id'].nunique()),
        'test_systems_untouched': test_systems,
        'eligible_systems': n_sys,
        'model_top5': round(model_rate, 4), 'random_top5': round(rnd_rate, 4),
        'lift': round(model_rate - rnd_rate, 4),
        'normalized_lift': round(norm, 4) if norm is not None else None,
        'mean_auc': round(float(np.mean(aucs)), 4) if aucs else None,
        'fold_metrics': fold_rows, 'top5_by_stratum': strat_rows,
        'changed_sources': changed, 'elapsed_seconds': time.perf_counter() - started,
        'policy': ['test 분할은 학습·평가에 사용하지 않았습니다.',
                   '18번 포켓 cluster 폴드를 그대로 써서 cluster 누수를 막았습니다.',
                   '라벨 결측 pose는 학습에서 제외하고 예측·순위에는 포함했습니다.',
                   'feature를 0으로 채우지 않고 결측을 부스팅에 그대로 넘겼습니다.',
                   '동점은 pose_id SHA1 오름차순으로 결정적으로 깼습니다.',
                   '값 범위 이상은 보고만 하고 자동 삭제하지 않았습니다.'],
        'notes': ['scores.parquet을 25번 하네스에 --scores로 넣어 교차 확인할 수 있습니다'
                  '(이 점수는 dev만 덮으므로 라벨을 dev로 제한해 비교하세요).',
                  'eligible 대부분이 쉬운 타깃이라 전체 평균보다 난이도 층별 lift가 실질 지표입니다.',
                  '정규화 lift = (모델-무작위)/(1-무작위), 층 간 비교를 위해 천장을 보정한 값입니다.']}
    helper.write_json(out / 'baseline_report.json', report)
    helper.write_json(out / 'RUN_COMPLETE.json', {'outputs_written': True, 'baseline_finished': True,
        'status': status, 'model_top5': report['model_top5'], 'random_top5': report['random_top5'],
        'lift': report['lift'], 'normalized_lift': report['normalized_lift'],
        'features_used': len(feats), 'sources_unchanged': not changed})

    print('\nSTATUS:', status, flush=True)
    print(f'ELIGIBLE dev systems: {n_sys}', flush=True)
    print(f"MODEL {report['model_top5']} | RANDOM {report['random_top5']} "
          f"| LIFT {report['lift']:+} | 정규화 {report['normalized_lift']}", flush=True)
    print('MEAN AUC:', report['mean_auc'], flush=True)
    print('\n=== 난이도 층별 (실질 지표) ===', flush=True)
    print(f"{'stratum':11s}{'range':10s}{'systems':>8s}{'model':>8s}{'random':>8s}"
          f"{'lift':>8s}{'정규화':>9s}", flush=True)
    for r in strat_rows:
        if not r['systems']:
            continue
        print(f"{r['stratum']:11s}{r['range']:10s}{r['systems']:8d}{r['model_top5']:8.3f}"
              f"{r['random_top5']:8.3f}{r['lift']:+8.3f}{r['normalized_lift']:9.3f}", flush=True)
    return 0 if status == 'BASELINE_COMPLETE' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, default=DEFAULT_MATRIX,
                        help='26번 통합 결과 폴더(unified_all.parquet + feature_catalog.csv).')
    parser.add_argument('--exclude-categories', type=str, default=None,
                        help='쉼표 구분 후보군 제외(증분 측정용). 예: cat2_gnina')
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
    return run(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
