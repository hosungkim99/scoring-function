# -*- coding: utf-8 -*-
"""results/ 의 요약 파일만으로 README 그림을 다시 만든다.

    python figures/make_figures.py

입력은 전부 이 레포의 results/ 안에 있다. 새 계산·새 실험은 하지 않고 파일에 적힌 값만 그린다.
모든 그림은 dev(평가 대상 2,877 타깃) 기준이다.

색: 계열 1 파랑 #2a78d6, 계열 2 주황 #eb6834 (색각 이상 검증을 통과한 기본 팔레트의 앞 두 칸).
무작위 기준선·여백 같은 기준값은 계열이 아니므로 회색으로 그린다.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / 'results'
OUT = Path(__file__).resolve().parent

BLUE, ORANGE = '#2a78d6', '#eb6834'
GRAY, GRAY_LIGHT = '#898781', '#e1e0d9'
INK, INK2 = '#0b0b0b', '#52514e'
SURFACE, AXIS = '#fcfcfb', '#c3c2b7'

for name in ('Malgun Gothic', 'NanumGothic', 'AppleGothic', 'Noto Sans CJK KR'):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = name
        break
plt.rcParams.update({
    'axes.unicode_minus': False, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.grid': True, 'grid.color': GRAY_LIGHT, 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10,
    'axes.titlesize': 12, 'axes.titleweight': 'bold', 'legend.frameon': False,
    'savefig.dpi': 200, 'savefig.bbox': 'tight',
})


def rows(rel):
    with (RES / rel).open(encoding='utf-8', newline='') as fh:
        return list(csv.DictReader(fh))


def note(ax, text):
    """출처 문구를 축 제목 아래에 고정 간격으로 둔다."""
    ax.annotate(text, xy=(0, 0), xycoords='axes fraction', xytext=(-40, -48),
                textcoords='offset points', ha='left', va='top', fontsize=8, color=GRAY,
                annotation_clip=False)


# ------------------------------------------------------------------ 공통 데이터
DEV_VS_TEST = 'test_eval/35_run_test_final/dev_vs_test.csv'
anchor = next(r for r in rows(DEV_VS_TEST) if r['label'] == 'anchor_only')
K = (1, 3, 5)


def dev(metric, rule, k, stat='mean'):
    return float(anchor[f'dev|{stat}|{metric}|{rule}|k{k}'])


TEST_REPORT = 'test_eval/35_run_test_final/test_report.json'
_rand = json.loads((RES / TEST_REPORT).read_text(encoding='utf-8'))['splits']['dev']['random_analytic']
rand = {k: float(_rand[str(k)]) for k in K}   # 해석식 무작위 기준선(정확값)


# ------------------------------------------------------------------ 그림 1
def fig_headroom():
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    labels = [f'Top-{k}' for k in K]
    for i, k in enumerate(K):
        r, m = rand[k], dev('success', 'ruleA', k)
        ax.barh(i, r, color=GRAY_LIGHT, edgecolor=SURFACE, linewidth=2, height=0.55)
        ax.barh(i, m - r, left=r, color=BLUE, edgecolor=SURFACE, linewidth=2, height=0.55)
        ax.barh(i, 1 - m, left=m, color='none', edgecolor=AXIS, linewidth=1, height=0.55,
                hatch='////')
        ax.text(r / 2, i, f'무작위 {r:.4f}', ha='center', va='center', color=INK2, fontsize=9)
        ax.text(1.01, i, f"모델 {m:.4f}  (여백 {1 - r:.4f} 중 {dev('lift', 'ruleA', k):+.4f})", va='center',
                fontsize=9, color=INK)
    ax.set_yticks(range(len(K)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel('타깃 성공률 (dev, 평가 대상 2,877)')
    ax.grid(axis='y', visible=False)
    ax.set_title('무작위로 골라도 이미 높다 — 개선할 수 있는 여백은 빗금 부분뿐', loc='left', pad=26)
    handles = [plt.Rectangle((0, 0), 1, 1, color=GRAY_LIGHT),
               plt.Rectangle((0, 0), 1, 1, color=BLUE),
               plt.Rectangle((0, 0), 1, 1, facecolor='none', edgecolor=AXIS, hatch='////')]
    ax.legend(handles, ['무작위 선택', '모델이 더한 몫', '남은 여백'], ncol=3,
              loc='lower left', bbox_to_anchor=(0, 1.0), fontsize=9, borderaxespad=0.2)
    note(ax, f'출처: results/{DEV_VS_TEST} (anchor_only, ruleA, dev 열; 시드 5개 평균), '
              f'무작위 기준선은 results/{TEST_REPORT} (splits.dev.random_analytic).')
    fig.savefig(OUT / 'fig1_headroom.png')
    plt.close(fig)


# ------------------------------------------------------------------ 그림 2
def fig_lift_by_k():
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = list(K)
    ceiling = [1 - rand[k] for k in K]
    ax.plot(x, ceiling, color=GRAY, lw=1.5, ls='--', marker='o', ms=5, label='여백 상한 (1 - 무작위)')
    for rule, color, name in (('plain', ORANGE, '점수 상위 k개'), ('ruleA', BLUE, '규칙 A (생성기별 1위 먼저)')):
        y = [dev('lift', rule, k) for k in K]
        e = [dev('lift', rule, k, 'std') for k in K]
        ax.errorbar(x, y, yerr=e, color=color, lw=2, marker='o', ms=8, capsize=4, label=name)
        ax.annotate(f'{y[-1]:+.4f}', (x[-1], y[-1]), xytext=(8, 0), textcoords='offset points',
                    va='center', fontsize=9, color=INK)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xticks(x, [f'Top-{k}' for k in K])
    ax.set_xlim(0.5, 5.9)
    ax.set_ylabel('lift (모델 성공률 - 무작위)')
    ax.set_title('k가 커질수록 여백이 줄고, 상위 k개 선택은 Top-5에서 0 아래로', loc='left')
    ax.legend(loc='upper right', fontsize=9)
    note(ax, f'출처: results/{DEV_VS_TEST} (anchor_only, dev 열). 오차 막대 = 시드 5개 간 표준편차.')
    fig.savefig(OUT / 'fig2_lift_by_k.png')
    plt.close(fig)


# ------------------------------------------------------------------ 그림 3
def fig_selection_rule():
    src = 'combined/31_run_combined_repro30/selection_matrix.csv'
    data = {(r['scorer'], r['rule']): float(r['success']) for r in rows(src) if r['topk'] == '5'}
    base = next(float(r['rand_analytic']) for r in rows(src) if r['topk'] == '5')
    rules = [('plain', '점수 상위 5개'), ('ruleB', '규칙 B\n(생성기당 최대 2개)'),
             ('ruleA', '규칙 A\n(생성기별 1위 먼저)')]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for i, (rule, _) in enumerate(rules):
        c, m = data[('pseudorandom', rule)], data[('model', rule)]
        ax.plot([i, i], [c, m], color=AXIS, lw=2, zorder=1)
        ax.scatter([i], [c], s=70, color=SURFACE, edgecolor=GRAY, linewidth=2, zorder=3,
                   label='같은 규칙 + 점수 없는 순서' if i == 0 else None)
        ax.scatter([i], [m], s=80, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=4,
                   label='같은 규칙 + 모델 점수' if i == 0 else None)
        ax.annotate(f'{m:.4f}', (i, m), xytext=(10, 0), textcoords='offset points', va='center',
                    fontsize=9, color=INK)
    ax.axhline(base, color=GRAY, lw=1.5, ls='--')
    ax.text(2.45, base, f'무작위 {base:.4f}', va='bottom', ha='right', fontsize=9, color=INK2)
    ax.set_xticks(range(len(rules)), [n for _, n in rules])
    ax.set_xlim(-0.4, 2.5)
    ax.set_ylabel('Top-5 성공률')
    ax.set_title('같은 점수 파일, 고르는 방식만 바꿨다 (재학습 없음)', loc='left')
    ax.legend(loc='upper left', fontsize=9)
    note(ax, f'출처: results/{src} (k=5). 점수 상위 5개는 무작위보다 낮고, 규칙 A는 0.9347.')
    fig.savefig(OUT / 'fig3_selection_rule.png')
    plt.close(fig)


# ------------------------------------------------------------------ 그림 4
def fig_feature_ladder():
    src = 'feature_selection/32_run_selection_locked_rep5/selection_results.csv'
    lad = [r for r in rows(src) if r['kind'] in ('ladder', 'reference')]
    lad.sort(key=lambda r: int(r['n_features']))
    n = [int(r['n_features']) for r in lad]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for k, color, name in ((1, BLUE, 'Top-1'), (5, ORANGE, 'Top-5')):
        y = [float(r[f'mean|lift|ruleA|k{k}']) for r in lad]
        e = [float(r[f'std|lift|ruleA|k{k}']) for r in lad]
        ax.errorbar(n, y, yerr=e, color=color, lw=2, marker='o', ms=8, capsize=4, label=name)
        ax.annotate(name, (n[-1], y[-1]), xytext=(8, 0), textcoords='offset points',
                    va='center', fontsize=9, color=INK)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xticks(n)
    ax.set_xlabel('중요도 상위 몇 개 feature로 다시 학습했나')
    ax.set_ylabel('lift (규칙 A)')
    ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    ax.set_title('feature를 5개에서 57개로 늘려도 잡음 범위 안에서 같다', loc='left')
    ax.legend(loc='lower right', fontsize=9)
    note(ax, f'출처: results/{src} (ladder + reference 행). 오차 막대 = 시드 5개 간 표준편차.')
    fig.savefig(OUT / 'fig4_feature_ladder.png')
    plt.close(fig)


# ------------------------------------------------------------------ 그림 5
def fig_generator_spread():
    src = 'diagnosis/38_run_diagnose_second/generator_spread.csv'
    summ = 'diagnosis/38_run_diagnose_second/diagnosis_summary.csv'
    d = [r for r in rows(src) if r['split'] == 'dev']
    s = {r['metric']: float(r['value']) for r in rows(summ)
         if r['split'] == 'dev' and r['block'] == 'generator_spread'
         and r['metric'] in ('model_top5_mean', 'random_top5_mean')}
    x = [int(r['distinct_generators']) for r in d]
    y = [float(r['share']) for r in d]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar(x, y, color=BLUE, width=0.6, edgecolor=SURFACE, linewidth=2)
    for xi, yi in zip(x, y):
        ax.text(xi + 0.32, yi, f'{yi:.1%}', ha='left', va='center', fontsize=9, color=INK)
    ax.axvline(s['model_top5_mean'], color=INK, lw=1.2, ls=':')
    ax.axvline(s['random_top5_mean'], color=GRAY, lw=1.5, ls='--')
    ax.text(s['model_top5_mean'], 0.585, f"모델 평균 {s['model_top5_mean']:.3f} ", fontsize=9,
            color=INK, ha='right')
    ax.text(s['random_top5_mean'], 0.585, f" 무작위 평균 {s['random_top5_mean']:.3f}", fontsize=9,
            color=INK2)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlim(0.4, 4.9)
    ax.set_ylim(0, 0.62)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel('점수 상위 5개 안에 들어간 생성기 종류 수')
    ax.set_ylabel('타깃 비율')
    ax.grid(axis='x', visible=False)
    ax.set_title('점수만으로 고르면 Top-5가 생성기 1~2종에 쏠린다', loc='left')
    note(ax, f'출처: results/{src}, {summ} (dev). 생성기는 4종.')
    fig.savefig(OUT / 'fig5_generator_spread.png')
    plt.close(fig)


if __name__ == '__main__':
    for f in (fig_headroom, fig_lift_by_k, fig_selection_rule, fig_feature_ladder, fig_generator_spread):
        f()
        print('saved', f.__name__)
