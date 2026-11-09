#!/usr/bin/env python3
"""다음 사이클 1단계. "모델이 2~5등을 못 고르는 결함"의 원인을 가립니다.

**새 추출이 없습니다.** 통합 행렬과 37번이 굳힌 모델만 읽어 분석합니다.

가리려는 것 — 여백 59 타깃이 어디서 새는가

    선택 문제   근접 pose 가 점수 상위권에 있는데 중복 때문에 자리를 뺏긴다
                -> 고칠 곳은 선택 단계 (유사도 기반 규칙). 모델 재학습 불필요
    점수 문제   근접 pose 가 점수 50등쯤에 있어 어떤 선택 규칙으로도 못 찾는다
                -> 고칠 곳은 점수 (합의 feature 추가). 추출 1회 필요

이 둘은 고치는 방법이 완전히 다르므로 **먼저 가려야 한다.** 지금까지
"비슷한 포즈끼리 뭉친다"고 서술해 왔지만 그것은 추론이고 측정한 적이 없다.

재는 것 5가지

  (1) 근접 pose 의 최고 점수 순위
      타깃마다 sc_rmsd < 2A 인 pose 중 점수 순위가 가장 높은 것의 순위.
      무작위 기대값 (n+1)/(m+1) 과의 비율도 함께 낸다 — 근접 pose 가 많은
      타깃은 그냥도 1등이 나오므로 원시 순위만 보면 속는다.
  (2) success / lift 곡선 (k=1~100)
      lift 가 k=1 에서 꺾여 내려가면 모델의 우위가 맨 위에만 있다는 뜻이다.
  (3) Top-5 의 생성기 수 — 모델 vs 무작위
      무작위보다 적으면 "무작위보다 덜 다양하다"가 확정된다. k=5 에서
      모델이 무작위보다 못한 이유의 직접 증거다.
  (4) 중복성 — Top-5 안의 근접 pose 수, 그리고 1->5 증분 이득
      무작위는 자리를 늘리면 많이 벌고 모델은 적게 번다면 자리가 낭비된 것이다.
  (5) 점수 신호의 감쇠 — 시스템 내 AUC 를 1등 제외 / 상위 5개 제외로 다시 잰다
      0.5 로 떨어지면 판별력이 맨 위에만 있다는 뜻이다.

분할 방침

  **주 분할은 dev 다.** out-of-fold 2,877 타깃으로 재며, test 718 보다 분포가
  4배 촘촘하고 설계 결정을 dev 에서 끝내는 규율에도 맞는다.
  test 는 **메커니즘이 홀드아웃에서도 같은지 확인하는 데만** 쓰고, 여기서
  무엇을 고르지 않는다. test 점수는 37번이 굳힌 model.json 으로 낸다
  (그 모델의 k=1 lift 가 37번 보고서의 designated_seed 값과 맞는지 확인한다).

출력: out/diagnosis/38_run_diagnose[_TAG]/
"""
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time

# 36번을 통해 37·35·34~28번까지 불러옵니다(측정·통계 코드를 그대로 씁니다).
_t36_path = Path(__file__).with_name('36_subset_distribution.py')
_tspec = importlib.util.spec_from_file_location('subset_support', _t36_path)
subset = importlib.util.module_from_spec(_tspec)
_tspec.loader.exec_module(subset)
test_eval = subset.test_eval
union = subset.union
combination = subset.combination
selection = subset.selection
combined = subset.combined
graded = subset.graded
ranking = subset.ranking
baseline = subset.baseline
roles = subset.roles
audit = subset.audit
helper = subset.helper
rankdata = subset.rankdata

VERSION = 'rank_diagnosis_0.1.0'
OUT_KIND = 'diagnosis'
OUT_BASE = '38_run_diagnose'
KEYS = baseline.KEYS
BASE = baseline.BASE
DEFAULT_MATRIX = baseline.DEFAULT_MATRIX
RULES = combined.RULES
HEADLINE_RULE = selection.HEADLINE_RULE
DEFAULT_FREEZE = BASE / 'final_model' / '37_run_freeze_v2'
# 곡선을 낼 k. 1~20 은 촘촘히, 그 위는 띄워서 본다.
CURVE_K = tuple(list(range(1, 21)) + [25, 30, 40, 50, 75, 100])
# 판정 문턱 — 사전에 적어둔다. 실행 후 바꾸지 않는다.
SELECTION_SIDE_MEDIAN = 3.0    # 최고 순위 중앙값이 이 이하면 선택 문제 쪽
SCORE_SIDE_MEDIAN = 20.0       # 이 이상이면 점수 문제 쪽
AUC_DECAY_FLOOR = 0.53         # 1등 제외 후 AUC 가 이 아래면 신호가 맨 위에만 있다
# 규칙 A 보다 고운 다양성 축. seed·sample 열만 쓰므로 **새 추출이 필요 없다.**
# 규칙 A 는 생성기 첫 등장을 4자리 안에 반드시 담으므로, 규칙 A 가 실패하면 답의
# 생성기가 이미 틀린 pose 로 쓰였다는 뜻이다. 즉 생성기 축은 소진돼 있다.
# 같은 생성기 안을 시드/표본으로 더 쪼개면 그 자리를 비울 수 있는지 본다.
AXES = ('generator_seed', 'generator_sample')
AXIS_LABEL = {'generator_seed': '(생성기,시드)', 'generator_sample': '(생성기,표본)'}
# 더 고운 축의 값어치는 **순증**으로 본다. 구조한 수만 보면 안 되는데, 다양성을
# 강제하면 뒷자리가 점수 낮은 영역으로 밀려 **잃는 타깃**이 생기기 때문이다.
# 전체 타깃의 이 비율 이상 순증해야 값어치가 있다고 본다(dev 2,877 -> 약 14 타깃).
# 규칙 A 의 dev k=5 lift 가 +0.0255 이므로 그 1/5 규모를 기준으로 삼았다. 사전 고정.
FINER_AXIS_NET_SHARE = 0.005
PREREGISTRATION = (
    '이 실행은 분석만 하며 모델을 학습해 고르지 않는다.',
    '주 분할은 dev(out-of-fold)이고 test 는 메커니즘 확인용이다.',
    '판정 문턱(중앙 순위 3 / 20, AUC 0.53)은 실행 전에 정했다.',
    '판정은 분기 제안이며 최종 결정은 사람이 한다.',
    f'더 고운 축은 구조한 수가 아니라 순증(구조-손실)으로 판단하며 '
    f'문턱({FINER_AXIS_NET_SHARE})도 실행 전에 정했다.')


def load_final_features(freeze_dir):
    """37번이 굳힌 feature 목록과 순서를 읽습니다(손으로 적지 않습니다)."""
    path = Path(freeze_dir) / 'feature_order.json'
    if not path.exists():
        raise FileNotFoundError(f'37번 결과가 없습니다: {path}')
    data = json.loads(path.read_text(encoding='utf-8'))
    feats = list(data['features'])
    if not feats:
        raise ValueError('feature 목록이 비어 있습니다.')
    if len(feats) != int(data.get('n_features', len(feats))):
        raise ValueError('feature_order.json 의 개수가 목록과 맞지 않습니다.')
    return feats


def load_freeze_reference(freeze_dir):
    """37번이 보고한 출하 모델 성적(대조용)."""
    path = Path(freeze_dir) / 'provenance.json'
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding='utf-8'))
    return (data.get('performance') or {})


def expected_best_rank(n, m):
    """근접 pose m 개를 무작위로 흩었을 때 가장 앞선 것의 기대 순위."""
    return (n + 1.0) / (m + 1.0) if m > 0 else None


def expected_distinct(counts, k):
    """생성기별 pose 수가 counts 일 때, k개를 무작위로 뽑으면 기대 생성기 수."""
    n = sum(counts)
    if n <= 0 or k <= 0:
        return None
    k = min(k, n)
    total = 0.0
    for c in counts:
        rest = n - c
        if rest < k:
            total += 1.0
        else:
            total += 1.0 - math.comb(rest, k) / math.comb(n, k)
    return total


def diversity_order(keys, kmax):
    """키가 처음 나오면 먼저 담고 이미 쓴 키는 뒤로. 규칙 A 의 일반형입니다.

    keys 를 생성기로 주면 규칙 A 와 완전히 같습니다(점검에서 대조합니다).
    (생성기, 시드) 로 주면 더 고운 축이 되고, 추출이 필요 없습니다.
    """
    seen, first, rest = set(), [], []
    for i, key in enumerate(keys):
        if key in seen:
            rest.append(i)
        else:
            seen.add(key)
            first.append(i)
    return (first + rest)[:kmax]


def first_rank_of(keys, target_key):
    """정렬된 키 배열에서 target_key 가 처음 나오는 1-기반 순위."""
    for i, key in enumerate(keys):
        if key == target_key:
            return i + 1
    return None


def within_auc(scores, labels):
    """시스템 내 AUC. Mann-Whitney 식으로 계산하며 동점은 평균 순위로 다룹니다."""
    import numpy as np
    m = int(labels.sum())
    n = int(labels.size)
    if m == 0 or m == n:
        return None
    r = rankdata(scores)
    return float((r[labels].sum() - m * (m + 1) / 2.0) / (m * (n - m)))


def order_of(group, score):
    """점수 내림차순 정렬. 동점은 pose_id SHA1 오름차순으로 결정적으로 깹니다."""
    import numpy as np
    chunk = score[group['start']:group['start'] + group['size']]
    chunk = np.where(np.isnan(chunk), -np.inf, chunk)
    return np.lexsort((group['tie'], -chunk)), chunk


def diagnose(groups, score, split, seed, seed_arr, sample_arr,
             shuffles=5, kmax=100):
    """다섯 측정을 한 번의 순회로 모읍니다.

    best_rank 는 섞은 점수로도 같이 재서 **측정된 무작위 기준선**을 붙입니다.
    평균 비율은 이론적으로 정확히 1.0 이지만(E[best_rank] = (n+1)/(m+1)),
    중앙값 비율은 분포가 오른쪽으로 치우쳐 무작위에서도 약 0.73 이 나옵니다.
    기준선을 재지 않고 1.0 과 비교하면 오독합니다.
    """
    import numpy as np
    from collections import Counter
    rng = np.random.default_rng(seed)
    rand_best, rand_ratio = [], []
    rows = []
    # 더 고운 다양성 축. AXES 의 키 이름이 곡선·실패표의 열 이름이 됩니다.
    curve_axis = {name: {k: 0.0 for k in CURVE_K} for name in AXES}
    spread_axis = {name: 0.0 for name in AXES}
    fail_rows = []
    group_shapes = Counter()
    axis_rescued = {name: 0 for name in AXES}   # 규칙 A 실패 -> 축 성공
    axis_lost = {name: 0 for name in AXES}      # 규칙 A 성공 -> 축 실패
    curve_model = {k: 0.0 for k in CURVE_K}
    curve_rule = {k: 0.0 for k in CURVE_K}
    curve_rand = {k: 0.0 for k in CURVE_K}
    spread_model = Counter()
    spread_rand_sum = 0.0
    auc_all, auc_drop1, auc_drop5 = [], [], []
    top5_nn, top1_hit, top5_hit = [], 0, 0
    rand1_sum, rand5_sum = 0.0, 0.0
    for g in groups:
        order, chunk = order_of(g, score)
        n, m = g['size'], g['m']
        nn_ord = g['nn'][order]
        gens_ord = [str(x) for x in g['gen'][order]]

        # (1) 근접 pose 의 최고 점수 순위
        best = int(np.argmax(nn_ord)) + 1
        exp_rank = expected_best_rank(n, m)
        rows.append({'split': split, 'system_id': str(g['system_id']),
                     'poses': n, 'near_native': m,
                     'best_rank': best,
                     'expected_best_rank_random': round(exp_rank, 2),
                     'ratio_vs_random': round(best / exp_rank, 3),
                     'rand_top5': round(g['rand'][5], 4) if 5 in g['rand'] else None})
        # 같은 시스템을 섞어서도 재 둡니다(측정된 무작위 기준선).
        for _ in range(shuffles):
            shuffled = int(np.argmax(g['nn'][rng.permutation(n)])) + 1
            rand_best.append(shuffled)
            rand_ratio.append(shuffled / exp_rank)

        # (2) 곡선 — 규칙들은 접두사 성질이 있어 kmax 로 한 번 만들고 자릅니다
        sl = slice(g['start'], g['start'] + g['size'])
        seeds_ord = [str(x) for x in seed_arr[sl][order]]
        samples_ord = [str(x) for x in sample_arr[sl][order]]
        keys_of = {
            'generator': gens_ord,
            'generator_seed': [f'{a}|{b}' for a, b in zip(gens_ord, seeds_ord)],
            'generator_sample': [f'{a}|{b}' for a, b in zip(gens_ord, samples_ord)]}
        group_shapes[(len(set(gens_ord)), len(set(keys_of['generator_seed'])),
                      len(set(keys_of['generator_sample'])))] += 1
        rule_idx = combined.rule_order(gens_ord, HEADLINE_RULE, min(kmax, n))
        axis_idx = {name: diversity_order(keys_of[name], min(kmax, n)) for name in AXES}
        for k in CURVE_K:
            kk = min(k, n)
            curve_model[k] += float(nn_ord[:kk].any())
            curve_rule[k] += float(nn_ord[[rule_idx[i] for i in range(kk)]].any())
            curve_rand[k] += baseline.random_topk(n, m, kk)
            for name in AXES:
                picks = axis_idx[name][:kk]
                curve_axis[name][k] += float(nn_ord[picks].any())

        # (3) Top-5 생성기 수
        five = min(5, n)
        spread_model[len(set(gens_ord[:five]))] += 1
        for name in AXES:
            spread_axis[name] += len(set(keys_of[name][i]
                                         for i in axis_idx[name][:five]))
        counts = list(Counter(str(x) for x in g['gen']).values())
        exp_d = expected_distinct(counts, five)
        if exp_d is not None:
            spread_rand_sum += exp_d

        # (4) 중복성
        hits5 = int(nn_ord[:five].sum())
        top5_nn.append(hits5)
        top1_hit += int(bool(nn_ord[:1].any()))
        top5_hit += int(bool(hits5 > 0))
        rand1_sum += baseline.random_topk(n, m, 1)
        rand5_sum += baseline.random_topk(n, m, five)

        # (4b) 실패 타깃 조건부 — 고쳐야 할 대상은 성공한 91%가 아니라 실패한 쪽입니다.
        rule5 = [rule_idx[i] for i in range(five)]
        ok_plain = bool(nn_ord[:five].any())
        ok_rule = bool(nn_ord[rule5].any())
        ok_axis = {name: bool(nn_ord[axis_idx[name][:five]].any()) for name in AXES}
        for name in AXES:
            if ok_rule and not ok_axis[name]:
                axis_lost[name] += 1
            elif (not ok_rule) and ok_axis[name]:
                axis_rescued[name] += 1
        if not ok_rule:
            answer_gen = gens_ord[best - 1]
            answer_gs = keys_of['generator_seed'][best - 1]
            answer_gp = keys_of['generator_sample'][best - 1]
            taken_gen = set(gens_ord[i] for i in rule5)
            fail_rows.append({
                'split': split, 'system_id': str(g['system_id']),
                'near_native': m, 'best_rank': best,
                'plain_ok': ok_plain, 'ruleA_ok': ok_rule,
                'generator_seed_ok': ok_axis['generator_seed'],
                'generator_sample_ok': ok_axis['generator_sample'],
                'top5_generators_plain': len(set(gens_ord[:five])),
                'top5_generators_ruleA': len(taken_gen),
                'answer_generator': answer_gen,
                'answer_generator_first_rank': first_rank_of(gens_ord, answer_gen),
                'answer_generator_already_used': bool(answer_gen in taken_gen),
                'answer_gs_first_rank': first_rank_of(
                    keys_of['generator_seed'], answer_gs),
                'answer_gp_first_rank': first_rank_of(
                    keys_of['generator_sample'], answer_gp)})

        # (5) 점수 신호의 감쇠
        a = within_auc(chunk, g['nn'])
        if a is not None:
            auc_all.append(a)
        for drop, bucket in ((1, auc_drop1), (5, auc_drop5)):
            if n - drop < 2:
                continue
            keep = order[drop:]
            b = within_auc(chunk[keep], g['nn'][keep])
            if b is not None:
                bucket.append(b)

    n_sys = len(groups)
    n_fail = len(fail_rows)
    axis_summary = {}
    rule5_rate = curve_rule[5] / n_sys if 5 in curve_rule else None
    for name in AXES:
        at5 = curve_axis[name][5] / n_sys if 5 in curve_axis[name] else None
        net = axis_rescued[name] - axis_lost[name]
        axis_summary[name] = {
            'success_at_5': round(at5, 4) if at5 is not None else None,
            'ruleA_success_at_5': round(rule5_rate, 4) if rule5_rate is not None else None,
            'delta_vs_ruleA': (round(at5 - rule5_rate, 4)
                               if (at5 is not None and rule5_rate is not None)
                               else None),
            'top5_keys_mean': round(spread_axis[name] / n_sys, 3),
            'rescued_from_ruleA_failures': axis_rescued[name],
            'lost_from_ruleA_successes': axis_lost[name],
            'net_vs_ruleA': net,
            'net_share_of_systems': round(net / n_sys, 4),
            'ruleA_failures': n_fail,
            'note': ('구조한 수만 보면 안 됩니다. 다양성을 강제하면 뒷자리가 점수 '
                     '낮은 영역으로 밀려 잃는 타깃이 생기므로 순증으로 판단합니다.')}
    fail_summary = {
        'ruleA_failures': n_fail,
        'share': round(n_fail / n_sys, 4),
        'answer_generator_already_used': (
            round(sum(r['answer_generator_already_used'] for r in fail_rows)
                  / n_fail, 4) if n_fail else None),
        'top5_generators_plain_mean': (
            round(sum(r['top5_generators_plain'] for r in fail_rows) / n_fail, 3)
            if n_fail else None),
        'top5_generators_ruleA_mean': (
            round(sum(r['top5_generators_ruleA'] for r in fail_rows) / n_fail, 3)
            if n_fail else None),
        'best_rank_median': (round(float(np.median(
            [r['best_rank'] for r in fail_rows])), 1) if n_fail else None),
        'best_rank_within_20': (
            round(sum(r['best_rank'] <= 20 for r in fail_rows) / n_fail, 4)
            if n_fail else None),
        'note': ('규칙 A 는 생성기 첫 등장을 4자리 안에 반드시 담으므로, 실패하면 '
                 '답의 생성기가 이미 틀린 pose 로 쓰였다는 뜻입니다. '
                 'answer_generator_already_used 가 1.0 이어야 구현이 맞습니다.')}
    summary = {
        'split': split, 'systems': n_sys,
        'axes': axis_summary, 'failures': fail_summary,
        'group_shapes': {f'{a}|{b}|{c}': v
                         for (a, b, c), v in sorted(group_shapes.items())},
        'best_rank': {
            'median': float(np.median([r['best_rank'] for r in rows])),
            'q1': float(np.percentile([r['best_rank'] for r in rows], 25)),
            'q3': float(np.percentile([r['best_rank'] for r in rows], 75)),
            'mean': round(float(np.mean([r['best_rank'] for r in rows])), 2),
            'at_1': round(sum(r['best_rank'] == 1 for r in rows) / n_sys, 4),
            'within_3': round(sum(r['best_rank'] <= 3 for r in rows) / n_sys, 4),
            'within_5': round(sum(r['best_rank'] <= 5 for r in rows) / n_sys, 4),
            'within_10': round(sum(r['best_rank'] <= 10 for r in rows) / n_sys, 4),
            'within_20': round(sum(r['best_rank'] <= 20 for r in rows) / n_sys, 4),
            'beyond_50': round(sum(r['best_rank'] > 50 for r in rows) / n_sys, 4),
            'ratio_median': round(float(np.median(
                [r['ratio_vs_random'] for r in rows])), 3),
            'ratio_mean': round(float(np.mean(
                [r['ratio_vs_random'] for r in rows])), 3),
            'shuffles': shuffles,
            'random_median': round(float(np.median(rand_best)), 2),
            'random_mean': round(float(np.mean(rand_best)), 2),
            'random_ratio_median': round(float(np.median(rand_ratio)), 3),
            'random_ratio_mean': round(float(np.mean(rand_ratio)), 3),
            'median_vs_random': round(float(np.median(
                [r['best_rank'] for r in rows])) / float(np.median(rand_best)), 3),
            'note': ('ratio_mean 은 무작위에서 이론적으로 정확히 1.0 이지만 '
                     'ratio_median 은 분포가 치우쳐 1.0 이 아닙니다. '
                     'random_ratio_median 과 비교하십시오.')},
        'generator_spread': {
            'model_top5_mean': round(sum(k * v for k, v in spread_model.items())
                                     / n_sys, 3),
            'random_top5_mean': round(spread_rand_sum / n_sys, 3),
            'model_distribution': {str(k): spread_model.get(k, 0)
                                   for k in sorted(spread_model)},
            'rule_a_top5': 'min(4, 생성기 수) 보장'},
        'redundancy': {
            'model_top5_nn_mean': round(float(np.mean(top5_nn)), 3),
            'model_top5_nn_mean_given_hit': (
                round(float(np.mean([x for x in top5_nn if x > 0])), 3)
                if any(x > 0 for x in top5_nn) else None),
            'model_success_at_1': round(top1_hit / n_sys, 4),
            'model_success_at_5': round(top5_hit / n_sys, 4),
            'model_gain_1_to_5': round((top5_hit - top1_hit) / n_sys, 4),
            'random_success_at_1': round(rand1_sum / n_sys, 4),
            'random_success_at_5': round(rand5_sum / n_sys, 4),
            'random_gain_1_to_5': round((rand5_sum - rand1_sum) / n_sys, 4)},
        'score_signal': {
            'within_system_auc_all': (round(float(np.mean(auc_all)), 4)
                                      if auc_all else None),
            'n_all': len(auc_all),
            'within_system_auc_drop_top1': (round(float(np.mean(auc_drop1)), 4)
                                            if auc_drop1 else None),
            'n_drop1': len(auc_drop1),
            'within_system_auc_drop_top5': (round(float(np.mean(auc_drop5)), 4)
                                            if auc_drop5 else None),
            'n_drop5': len(auc_drop5)}}
    curve = []
    for k in CURVE_K:
        mv, rv, rr = curve_model[k] / n_sys, curve_rule[k] / n_sys, curve_rand[k] / n_sys
        row = {'split': split, 'k': k,
               'model_plain': round(mv, 4), 'model_ruleA': round(rv, 4),
               'random': round(rr, 4),
               'lift_plain': round(mv - rr, 4),
               'lift_ruleA': round(rv - rr, 4)}
        for name in AXES:
            av = curve_axis[name][k] / n_sys
            row[f'model_{name}'] = round(av, 4)
            row[f'lift_{name}'] = round(av - rr, 4)
        curve.append(row)
    return rows, summary, curve, fail_rows


def judge(dev_summary):
    """사전에 정한 문턱으로 분기를 제안합니다. 최종 결정은 사람이 합니다."""
    br = dev_summary['best_rank']
    sig = dev_summary['score_signal']
    spread = dev_summary['generator_spread']
    median = br['median']
    rand_median = br.get('random_median')
    drop1 = sig['within_system_auc_drop_top1']
    concentrated = (spread['model_top5_mean'] < spread['random_top5_mean'])
    reasons = []
    if rand_median is not None:
        reasons.append(f"섞기 기준선: 최고 순위 중앙 {rand_median} "
                       f"(모델 {median}, 비율 {br.get('median_vs_random')})")
    if median <= SELECTION_SIDE_MEDIAN:
        side = 'selection'
        reasons.append(f"최고 순위 중앙값 {median} <= {SELECTION_SIDE_MEDIAN} — "
                       f"점수는 근접 pose 를 상위권에 올린다")
    elif median >= SCORE_SIDE_MEDIAN:
        side = 'score'
        reasons.append(f"최고 순위 중앙값 {median} >= {SCORE_SIDE_MEDIAN} — "
                       f"점수가 근접 pose 를 찾지 못한다")
    else:
        side = 'both'
        reasons.append(f"최고 순위 중앙값 {median} 이 {SELECTION_SIDE_MEDIAN}~"
                       f"{SCORE_SIDE_MEDIAN} 사이 — 둘 다 작용한다")
    if drop1 is not None and drop1 < AUC_DECAY_FLOOR:
        reasons.append(f"1등 제외 후 시스템 내 AUC {drop1} < {AUC_DECAY_FLOOR} — "
                       f"판별력이 맨 위에만 있다")
        if side == 'selection':
            side = 'both'
            reasons.append('상위권에는 올리지만 2등 이하를 가리지 못하므로 '
                           '선택 규칙만으로는 한계가 있다')
    if concentrated:
        reasons.append(f"Top-5 생성기 수 모델 {spread['model_top5_mean']} < "
                       f"무작위 {spread['random_top5_mean']} — 쏠림 확정")
    else:
        reasons.append(f"Top-5 생성기 수 모델 {spread['model_top5_mean']} >= "
                       f"무작위 {spread['random_top5_mean']} — 생성기 축 쏠림은 "
                       f"아니다. 기하 유사도 축을 봐야 한다")
    recommend = {
        'selection': '선택 단계 개선 (유사도 기반 규칙). 모델 재학습 불필요.',
        'score': '합의 feature 추가 (pose 간 거리 추출 1회).',
        'both': '추출은 한 번. 합의 feature 와 선택 규칙을 함께 본다.'}[side]

    # 실패 타깃 조건부 — 생성기 축이 소진됐는지, 더 고운 축이 남았는지.
    fails = dev_summary.get('failures') or {}
    axes = dev_summary.get('axes') or {}
    n_fail = fails.get('ruleA_failures') or 0
    used = fails.get('answer_generator_already_used')
    if used is not None:
        if used >= 0.999:
            reasons.append(f'규칙 A 실패 {n_fail}개 전부에서 답의 생성기가 이미 '
                           f'쓰였다 — 생성기 축은 소진됐다(구현도 검증됨)')
        else:
            reasons.append(f'규칙 A 실패에서 답의 생성기가 이미 쓰인 비율 {used} '
                           f'— 1.0 이어야 하는데 아니다. 구현을 확인해야 한다')
    finer = []
    for name in AXES:
        got = axes.get(name) or {}
        rescued = got.get('rescued_from_ruleA_failures') or 0
        lost = got.get('lost_from_ruleA_successes') or 0
        net = got.get('net_vs_ruleA')
        net_share = got.get('net_share_of_systems') or 0.0
        reasons.append(f"{AXIS_LABEL[name]} 축: 구조 {rescued} / 손실 {lost} "
                       f"= 순증 {net} ({net_share:+.2%}), k=5 성공 "
                       f"{got.get('success_at_5')} vs 규칙A "
                       f"{got.get('ruleA_success_at_5')}")
        if net_share >= FINER_AXIS_NET_SHARE:
            finer.append((name, net, net_share))
    if finer:
        best_axis = max(finer, key=lambda x: x[1])
        recommend = (f'{AXIS_LABEL[best_axis[0]]} 다양성 규칙을 먼저 시험한다 — '
                     f'규칙 A 대비 순증 {best_axis[1]}개({best_axis[2]:+.2%})이고 '
                     f'**새 추출이 필요 없다.** 그 다음에 pose 간 거리를 걸지 '
                     f'결정한다.')
        side = 'selection_finer_axis'
    else:
        reasons.append(f'더 고운 축이 전부 문턱({FINER_AXIS_NET_SHARE:+.2%}) 미달 — '
                       f'다양성을 더 강제하면 뒷자리가 점수 낮은 영역으로 밀려 '
                       f'잃는 쪽이 크다. 생성기·시드·표본 축은 소진됐고 '
                       f'기하 유사도가 남은 유일한 축이다(pose 간 거리 추출 필요)')
    return {'side': side, 'reasons': reasons, 'recommendation': recommend,
            'finer_axis_candidates': [{'axis': a, 'rescued': r, 'share': round(s, 4)}
                                      for a, r, s in finer],
            'thresholds': {'selection_side_median': SELECTION_SIDE_MEDIAN,
                           'score_side_median': SCORE_SIDE_MEDIAN,
                           'auc_decay_floor': AUC_DECAY_FLOOR,
                           'finer_axis_net_share': FINER_AXIS_NET_SHARE},
            'decided_by_human': True}


def run(args):
    """dev out-of-fold 와 test(굳힌 모델)로 다섯 측정을 내고 분기를 제안합니다."""
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
    print('*** 사전 등록 ***', flush=True)
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

    final_features = load_final_features(args.freeze_dir)
    freeze_perf = load_freeze_reference(args.freeze_dir)
    print(f'진단 대상: 37번 출하 모델 — feature {len(final_features)}개', flush=True)
    for i, name in enumerate(final_features, start=1):
        print(f'  {i}. {name}', flush=True)

    catalog = pd.read_csv(catalog_path).set_index('feature')
    meta_cols = ['system_id', 'model', 'seed', 'sample', 'pose_id', 'sc_rmsd',
                 'near_native', 'label_known', 'split', 'cv_fold']
    frame = pd.read_parquet(matrix_path, columns=meta_cols + catalog.index.tolist())
    frame['near_native'] = frame['near_native'].fillna(False).astype(bool)
    frame['label_known'] = frame['label_known'].fillna(False).astype(bool)
    frame = frame.sort_values(KEYS, kind='mergesort').reset_index(drop=True)
    frame['_tie'] = frame['pose_id'].map(baseline.tie_key)

    feats, _ = baseline.select_features(frame, catalog, set())
    missing = sorted(set(final_features) - set(feats))
    if missing:
        raise ValueError(f'현재 행렬에 없는 feature: {missing}')
    bins = combined.BIN_SETS[args.relevance_bins]
    topk = (1, 3, 5)

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
        groups = selection.group_randoms(selection.prepare_groups(sub, codes), topk)
        parts[name] = {'frame': sub, 'codes': codes, 'rel': rel, 'known': known,
                       'X': sub[final_features].to_numpy(dtype=np.float32,
                                                         na_value=np.nan),
                       'fold_of': (sub['cv_fold'].astype(int).to_numpy()
                                   if name == 'dev' else None),
                       'seed_arr': sub['seed'].to_numpy(),
                       'sample_arr': sub['sample'].to_numpy(),
                       'groups': groups}
        print(f'{name.upper()}: poses {len(sub)} | systems '
              f'{int(np.unique(codes).size)} | 평가 가능 {len(groups)}', flush=True)

    dev, test = parts['dev'], parts['test']
    folds = sorted(int(f) for f in np.unique(dev['fold_of']))
    print('', flush=True)
    print(f'dev 점수: out-of-fold (폴드 {folds}, 시드 {args.seed})', flush=True)
    idx = np.arange(len(final_features), dtype=np.int64)
    dev_score, _, _ = selection.train_oof(dev['X'], dev['rel'], dev['codes'],
                                          dev['fold_of'], dev['known'], folds,
                                          idx, args)

    model_path = Path(args.freeze_dir) / 'model.json'
    if not model_path.exists():
        raise FileNotFoundError(f'굳힌 모델이 없습니다: {model_path}')
    booster = xgboost.Booster()
    booster.load_model(str(model_path))
    test_score = booster.predict(xgboost.DMatrix(test['X']))
    print(f'test 점수: 굳힌 모델 {model_path.name} '
          f'(sha256 {hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]}…)',
          flush=True)

    # 굳힌 모델이 37번 보고와 같은지 확인합니다.
    check = {'checked': False}
    rates, _, _ = selection.evaluate_scores(test['groups'], test_score, topk)
    controls = selection.evaluate_controls(test['groups'], topk, args.shuffles, args.seed)
    mine_k1 = rates[(HEADLINE_RULE, 1)] - controls['analytic'][1]
    want = (freeze_perf.get(f'test|lift|{HEADLINE_RULE}|k1') or {}).get('designated_seed')
    if want is not None:
        check = {'checked': True, 'reference_37': round(float(want), 4),
                 'here': round(float(mine_k1), 4),
                 'diff': round(float(mine_k1) - float(want), 4)}
        if abs(check['diff']) > 1e-3:
            raise ValueError(f'굳힌 모델의 test k=1 lift 가 37번 보고와 다릅니다: '
                             f"{check['here']} vs {check['reference_37']}")
        print(f"37번 보고와 일치 확인 — test k=1 lift {check['here']:+.4f}", flush=True)

    print('', flush=True)
    print('측정 중…', flush=True)
    dev_rows, dev_sum, dev_curve, dev_fail = diagnose(
        dev['groups'], dev_score, 'dev', args.seed,
        dev['seed_arr'], dev['sample_arr'], args.shuffles)
    test_rows, test_sum, test_curve, test_fail = diagnose(
        test['groups'], test_score, 'test', args.seed,
        test['seed_arr'], test['sample_arr'], args.shuffles)

    audit.save_csv(out / 'best_rank.csv', dev_rows + test_rows,
                   ('split', 'system_id', 'poses', 'near_native', 'best_rank',
                    'expected_best_rank_random', 'ratio_vs_random', 'rand_top5'))
    curve_header = ['split', 'k', 'random', 'model_plain', 'model_ruleA']
    curve_header += [f'model_{name}' for name in AXES]
    curve_header += ['lift_plain', 'lift_ruleA']
    curve_header += [f'lift_{name}' for name in AXES]
    audit.save_csv(out / 'success_curve.csv', dev_curve + test_curve,
                   tuple(curve_header))
    audit.save_csv(out / 'failures.csv', dev_fail + test_fail,
                   ('split', 'system_id', 'near_native', 'best_rank',
                    'plain_ok', 'ruleA_ok', 'generator_seed_ok',
                    'generator_sample_ok', 'top5_generators_plain',
                    'top5_generators_ruleA', 'answer_generator',
                    'answer_generator_first_rank', 'answer_generator_already_used',
                    'answer_gs_first_rank', 'answer_gp_first_rank'))

    flat = []
    for s in (dev_sum, test_sum):
        base = {'split': s['split'], 'systems': s['systems']}
        for block in ('best_rank', 'generator_spread', 'redundancy', 'score_signal'):
            for key, value in s[block].items():
                if isinstance(value, dict):
                    continue
                flat.append({**base, 'block': block, 'metric': key, 'value': value})
    audit.save_csv(out / 'diagnosis_summary.csv', flat,
                   ('split', 'systems', 'block', 'metric', 'value'))

    spread_rows = []
    for s in (dev_sum, test_sum):
        dist = s['generator_spread']['model_distribution']
        total = s['systems']
        for key in sorted(dist, key=lambda x: int(x)):
            spread_rows.append({'split': s['split'], 'distinct_generators': int(key),
                                'systems': dist[key],
                                'share': round(dist[key] / total, 4)})
    audit.save_csv(out / 'generator_spread.csv', spread_rows,
                   ('split', 'distinct_generators', 'systems', 'share'))

    verdict = judge(dev_sum)

    changed = []
    for source in sources:
        try:
            info = Path(source['path']).stat()
            if (info.st_size, info.st_mtime_ns) != (source['size_bytes'],
                                                    source['mtime_ns']):
                changed.append(source['path'])
        except OSError:
            changed.append(source['path'])
    status = 'DIAGNOSIS_COMPLETE' if not changed else 'CHECK_REQUIRED'

    report = {
        'version': VERSION, 'status': status, 'output_dir': str(out),
        'preregistration': list(PREREGISTRATION), 'picks_nothing': True,
        'question': '여백 59 타깃이 선택 문제인가 점수 문제인가',
        'primary_split': 'dev (out-of-fold)',
        'test_role': '메커니즘 확인용. 설계 결정에 쓰지 않는다',
        'matrix': str(args.matrix), 'freeze_dir': str(args.freeze_dir),
        'sources': sources, 'features': final_features,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'support_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (_t36_path, subset._t35_path,
                                     test_eval._union_path, union._combination_path,
                                     combination._selection_path,
                                     selection._combined_path, combined._graded_path,
                                     graded._ranking_path, ranking._baseline_path,
                                     baseline._support_path, roles._support_path,
                                     audit._support_path)},
        'python': sys.executable, 'xgboost': xgboost.__version__,
        'frozen_model_check': check,
        'curve_k': list(CURVE_K),
        'dev': dev_sum, 'test': test_sum,
        'curve': {'dev': dev_curve, 'test': test_curve},
        'axes': list(AXES), 'axis_labels': AXIS_LABEL,
        'failures': {'dev': dev_fail, 'test': test_fail},
        'verdict': verdict,
        'changed_sources': changed,
        'elapsed_seconds': time.perf_counter() - started,
        'notes': ['best_rank 의 ratio_vs_random 이 원시 순위보다 중요하다. 근접 pose 가 '
                  '많은 타깃은 무작위로도 1등이 나오므로 원시 순위만 보면 속는다.',
                  'ratio_mean 은 무작위에서 이론적으로 정확히 1.0 이다('
                  'E[best_rank] = (n+1)/(m+1) 이므로). 그러나 ratio_median 은 분포가 '
                  '오른쪽으로 치우쳐 무작위에서도 1.0 이 아니다(약 0.73). 그래서 '
                  '같은 시스템을 섞어 재어 random_median / random_ratio_median 을 '
                  '함께 적는다. 중앙값은 그쪽과 비교해야 한다.',
                  'score_signal 의 drop_top1 이 0.5 근처면 "1등은 고르지만 2등 이하는 '
                  '못 가린다"는 뜻이고, 그 경우 선택 규칙만으로는 한계가 있다.',
                  'generator_spread 에서 모델이 무작위보다 적으면 생성기 축 쏠림이고 '
                  '규칙 A 가 맞는 처방이다. 무작위와 비슷하면 쏠림의 축이 생성기가 '
                  '아니라 기하 유사도이므로 pose 간 거리가 필요하다.',
                  'verdict 는 분기 제안이며 사전에 정한 문턱을 그대로 적용한 것이다. '
                  '최종 결정은 사람이 한다.',
                  '규칙 A 는 생성기 첫 등장을 4자리 안에 반드시 담는다. 따라서 규칙 A 가 '
                  '실패하면 답의 생성기가 이미 틀린 pose 로 쓰였다는 뜻이고, '
                  'answer_generator_already_used 는 1.0 이어야 한다. 1.0 이 아니면 '
                  '구현 결함이다.',
                  'AXES 는 seed·sample 열만 쓰므로 새 추출이 필요 없다. 다만 판단은 '
                  'rescued 가 아니라 net_vs_ruleA 로 해야 한다 — 다양성을 더 강제하면 '
                  '뒷자리가 점수 낮은 영역으로 밀려 ruleA 가 맞췄던 타깃을 잃는다. '
                  '합성 점검에서 구조 6개를 얻고 전체 성공률은 0.108 -> 0.090 으로 '
                  '떨어지는 경우가 실제로 나왔다.',
                  '순증이 문턱 미달이면 생성기·시드·표본 축이 소진된 것이고, 남은 축은 '
                  '기하 유사도뿐이다(pose 간 거리 추출 1회 필요).']}
    helper.write_json(out / 'diagnosis_report.json', report)
    helper.write_json(out / 'RUN_COMPLETE.json', {
        'outputs_written': True, 'diagnosis_finished': True, 'status': status,
        'picks_nothing': True,
        'dev_best_rank_median': dev_sum['best_rank']['median'],
        'dev_best_rank_ratio_median': dev_sum['best_rank']['ratio_median'],
        'dev_auc_all': dev_sum['score_signal']['within_system_auc_all'],
        'dev_auc_drop_top1': dev_sum['score_signal']['within_system_auc_drop_top1'],
        'dev_top5_generators_model': dev_sum['generator_spread']['model_top5_mean'],
        'dev_top5_generators_random': dev_sum['generator_spread']['random_top5_mean'],
        'side': verdict['side'], 'recommendation': verdict['recommendation'],
        'finer_axis_candidates': verdict.get('finer_axis_candidates') or [],
        'dev_ruleA_failures': dev_sum['failures']['ruleA_failures'],
        'dev_answer_generator_already_used':
            dev_sum['failures']['answer_generator_already_used'],
        'dev_rescued': {name: dev_sum['axes'][name]['rescued_from_ruleA_failures']
                        for name in AXES},
        'sources_unchanged': not changed})

    print('', flush=True)
    print('STATUS:', status, flush=True)

    for s in (dev_sum, test_sum):
        tag = s['split'].upper()
        br, sig = s['best_rank'], s['score_signal']
        sp, rd = s['generator_spread'], s['redundancy']
        print('', flush=True)
        print(f"=== {tag} ({s['systems']} 타깃) ===", flush=True)
        print(f"① 근접 pose 의 최고 점수 순위", flush=True)
        print(f"     중앙 {br['median']:.0f}  사분위 {br['q1']:.0f}~{br['q3']:.0f}  "
              f"평균 {br['mean']}", flush=True)
        print(f"     1등 {br['at_1']:.1%} | 3위내 {br['within_3']:.1%} | "
              f"5위내 {br['within_5']:.1%} | 10위내 {br['within_10']:.1%} | "
              f"20위내 {br['within_20']:.1%} | 50위밖 {br['beyond_50']:.1%}", flush=True)
        print(f"     최고 순위   모델 중앙 {br['median']:.0f} vs "
              f"섞기 중앙 {br['random_median']}   "
              f"(비율 {br['median_vs_random']}, 1.0 이면 무작위와 같음)", flush=True)
        print(f"     기대 대비   모델 평균 {br['ratio_mean']} vs "
              f"섞기 평균 {br['random_ratio_mean']}   "
              f"(섞기 평균은 이론상 1.000)", flush=True)
        print(f"                 모델 중앙 {br['ratio_median']} vs "
              f"섞기 중앙 {br['random_ratio_median']}   "
              f"(중앙값은 1.0 이 기준이 아니다)", flush=True)
        print(f"③ Top-5 생성기 수   모델 {sp['model_top5_mean']} vs "
              f"무작위 {sp['random_top5_mean']}", flush=True)
        print(f"④ 중복성   Top-5 안 근접 pose {rd['model_top5_nn_mean']}개 "
              f"(맞춘 경우 {rd['model_top5_nn_mean_given_hit']}개)", flush=True)
        print(f"     1->5 증분 이득   모델 {rd['model_gain_1_to_5']:+.4f} vs "
              f"무작위 {rd['random_gain_1_to_5']:+.4f}", flush=True)
        print(f"⑤ 시스템 내 AUC   전체 {sig['within_system_auc_all']} | "
              f"1등 제외 {sig['within_system_auc_drop_top1']} | "
              f"상위5 제외 {sig['within_system_auc_drop_top5']}", flush=True)

    for s in (dev_sum, test_sum):
        fl, ax = s['failures'], s['axes']
        print('', flush=True)
        print(f"=== 실패 타깃 조건부 ({s['split'].upper()}, 규칙 A 실패 "
              f"{fl['ruleA_failures']}개 / {fl['share']:.1%}) ===", flush=True)
        print(f"  답의 생성기가 이미 쓰였다   "
              f"{(fl['answer_generator_already_used'] or 0):.1%}"
              f"   <- 1.0 이어야 구현이 맞다 (생성기 축 소진)", flush=True)
        print(f"  Top-5 생성기 수   plain {fl['top5_generators_plain_mean']} / "
              f"ruleA {fl['top5_generators_ruleA_mean']}", flush=True)
        print(f"  답의 순위   중앙 {fl['best_rank_median']} | "
              f"20위내 {(fl['best_rank_within_20'] or 0):.1%}", flush=True)
        print(f"  더 고운 축 (새 추출 불필요) — 구조한 수만 보면 안 되고 순증을 본다",
              flush=True)
        for name in AXES:
            got = ax[name]
            print(f"    {AXIS_LABEL[name]:14s} 구조 {got['rescued_from_ruleA_failures']:4d}"
                  f" / 손실 {got['lost_from_ruleA_successes']:4d}"
                  f" = 순증 {got['net_vs_ruleA']:+5d}"
                  f" ({got['net_share_of_systems']:+.2%})", flush=True)
            print(f"    {'':14s} k=5 성공 {got['success_at_5']} vs 규칙A "
                  f"{got['ruleA_success_at_5']} (차 {got['delta_vs_ruleA']:+.4f})"
                  f"  Top-5 키 수 {got['top5_keys_mean']}", flush=True)
        print(f"  pose 묶음 구조 (생성기|생성기시드|생성기표본): "
              f"{list(s['group_shapes'].items())[:4]}", flush=True)

    print('', flush=True)
    print('=== ② lift 곡선 (dev) ===', flush=True)
    head = f"{'k':>4s}{'무작위':>9s}{'plain':>9s}{'ruleA':>9s}"
    for name in AXES:
        head += f"{AXIS_LABEL[name]:>14s}"
    head += f"{'lift plain':>12s}{'lift ruleA':>12s}"
    for name in AXES:
        head += f"{'lift ' + AXIS_LABEL[name]:>16s}"
    print(head, flush=True)
    for row in dev_curve:
        if row['k'] > 20 and row['k'] not in (25, 30, 40, 50, 75, 100):
            continue
        line = (f"{row['k']:4d}{row['random']:9.4f}{row['model_plain']:9.4f}"
                f"{row['model_ruleA']:9.4f}")
        for name in AXES:
            line += f"{row[f'model_{name}']:14.4f}"
        line += f"{row['lift_plain']:+12.4f}{row['lift_ruleA']:+12.4f}"
        for name in AXES:
            line += f"{row[f'lift_{name}']:+16.4f}"
        print(line, flush=True)

    print('', flush=True)
    print('=== 분기 제안 (사전 문턱 적용) ===', flush=True)
    print(f"  판정: {verdict['side']}", flush=True)
    for line in verdict['reasons']:
        print(f"    - {line}", flush=True)
    print(f"  권고: {verdict['recommendation']}", flush=True)
    print('', flush=True)
    print('*** 이 실행은 아무것도 고르지 않았습니다. 최종 결정은 사람이 합니다. ***',
          flush=True)
    return 0 if status == 'DIAGNOSIS_COMPLETE' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, default=DEFAULT_MATRIX)
    parser.add_argument('--freeze-dir', type=Path, default=DEFAULT_FREEZE,
                        help='37번 결과 폴더(feature 목록과 굳힌 모델을 읽습니다).')
    parser.add_argument('--relevance-bins', choices=tuple(combined.BIN_SETS),
                        default='asymmetric')
    parser.add_argument('--shuffles', type=int, default=5)
    parser.add_argument('--n-estimators', type=int, default=400)
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261102)
    parser.add_argument('--tag', type=str, default=None)
    args = parser.parse_args()
    for name in ('shuffles', 'n_estimators', 'max_depth', 'threads'):
        if getattr(args, name) < 1:
            parser.error(f'--{name.replace("_", "-")} 는 1 이상이어야 합니다.')
    args.matrix = args.matrix.absolute()
    args.freeze_dir = args.freeze_dir.absolute()
    return run(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
