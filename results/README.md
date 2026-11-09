# results — 요약 결과와 최종 모델

서버 실행 결과(`out/`) 중 요약 CSV·JSON과 최종 모델만 옮겼다. 폴더 구조는 원래 `out/` 아래 구조와 같다.
예를 들어 보고서 안의 `<DATA_ROOT>/Scoring_Function/out/final_model/37_run_freeze_v2`는
이 레포의 `results/final_model/37_run_freeze_v2`에 해당한다.

- 파일 내용은 원본과 같다. 다른 점은 경로 문자열의 자리표시자 치환(`../docs/ANONYMIZATION.md`)과,
  아래 "공개 범위 처리"에 적은 항목뿐이다.
- **스크립트 해시 대조**: 각 보고서의 `script_sha256`을 치환 전 원본 스크립트의 해시
  (`../pipeline/ORIGINAL_SHA256SUMS.txt`)와 맞춰 봤다.
  - **일치**: 그 결과가 현재 레포의 스크립트(경로 치환 전)로 만들어졌다는 뜻이다.
  - **불일치**: 그 실행 뒤에 스크립트가 수정됐다는 뜻이다.

## 폴더별 안내

| 폴더 | 만든 스크립트 | 무엇이 들어 있나 | 해시 |
|---|---|---|---|
| `final_model/37_run_freeze_v2/` | 37 | **최종 모델**. `model.json`(출하 모델, 시드 20261102), 시드별 모델 5개, `feature_order.json`, `spec.json`, `provenance.json`(학습 규모·입력 해시·암기 격차), `selfcheck.json`(자체 검증 4종), 독립 실행 `predict.py` | 일치 |
| `test_eval/35_run_test_final/` | 35 | **README 주지표의 출처**(`dev_vs_test.csv`의 dev 열). 집합 4종 비교, 난이도 층 규모(`strata_profile.csv`), 공헌도 분해 | 일치 |
| `subset_dist/36_run_subsets_first/` | 36 | 부분집합 35행 비교, dev↔test 전이 상관(`transfer.csv`) | 일치 |
| `diagnosis/38_run_diagnose_first/` | 38 (`first`) | 최고 순위 분포, Top-5 생성기 수, 타깃 내부 AUC, lift 곡선 | **불일치** (`second`를 위해 38번이 수정됨) |
| `diagnosis/38_run_diagnose_second/` | 38 (`second`) | 실패 조건부 측정, 더 고운 다양성 축의 구조·손실·순증, `group_shapes` | 일치 |
| `feature_selection/32_run_selection_locked_rep5/` | 32 | 중요도 사다리·범주 제외·단독 feature 비교, 시드 5회 잡음 | **불일치** (실행 후 수정, 내용 `확인 필요`) |
| `combination/33_run_combination_search/` | 33 | 조합 1,294개 탐색 결과, 기약 조합 12개, 교체 판정 | 일치 |
| `union/34_run_union_final_full/` | 34 | 합집합 비교, 최종 집합(`final_feature_set.json`) | 일치 |
| `feature_matrix_all/26_run_integrate_all_final/` | 26 | feature 63개 카탈로그·범주별 커버리지·상태 분포 | **불일치** (실행 후 수정, 내용 `확인 필요`) |
| `data_split/18_run_make_split/` | 18 | 분할 보고서(dev 3,680 / test 923, 누수 감사), 폴드 크기 | 일치 |
| `eval/25_run_evaluate/` | 25 | 모델 없는 기준선(oracle·무작위)과 시스템 분류 | 일치 |
| `labels/24_run_build_labels/` | 24 | 라벨 표 보고서, 결측 상태 | 일치 |
| `baseline/28_run_baseline_f63/` | 28 | 전역 AUC 0.8022 / Top-5 lift −0.0134 | 일치 |
| `ranking/29_run_ranking_*/` | 29 | 랭킹 목적함수 절제 3회 | 일치 |
| `graded/30_run_graded_graded/` | 30 | 등급 relevance | 일치 |
| `combined/31_run_combined_*/` | 31 | 2×2 실행 3개 + 30번 재현 1개(`repro30`). 선택 규칙별 성공률(`selection_matrix.csv`)과 공헌도 분해(`contribution.csv`) | 일치 |

## 문서 수치 → 파일

| 수치 | 파일 · 위치 |
|---|---|
| dev OOF Top-1/3/5 성공률과 lift, 어려운 층 lift | `test_eval/35_run_test_final/dev_vs_test.csv` — `anchor_only` 행, `dev|mean|…|ruleA|k*` |
| 무작위 기준선 0.7406 / 0.8735 / 0.9133 | 같은 파일의 성공률 − lift, 또는 `combined/31_run_combined_repro30/selection_matrix.csv`의 `rand_analytic` |
| 어려운 층 207개, 무작위 0.2103 | `test_eval/35_run_test_final/strata_profile.csv`, `test_report.json`의 `splits.dev.random_analytic_hard` |
| 상위 5개 0.9058 vs 규칙 A 0.9347 | `combined/31_run_combined_repro30/selection_matrix.csv` |
| 다양성 몫 +0.0128 / 모델 몫 +0.0101 | `combined/31_run_combined_repro30/contribution.csv`(`ruleA`, k=5) |
| 전역 AUC 0.8022 / lift −0.0134 | `baseline/28_run_baseline_f63/baseline_report.json` |
| 조합 1,294개, 기약 12개 | `combination/33_run_combination_search/combination_report.json` |
| 학습 시스템 2,482, 출하 모델 sha256 | `final_model/37_run_freeze_v2/provenance.json` |
| Top-5 생성기 수 1.984 vs 3.083, 타깃 내부 AUC 0.6573 | `diagnosis/38_run_diagnose_second/diagnosis_summary.csv` |
| 순증 −51 / −59, `group_shapes` `4\|20\|20` | `diagnosis/38_run_diagnose_second/diagnosis_report.json`의 `dev.axes.*`, `dev.group_shapes` |
| 평가 대상 3,595 / 450 / 558 | `eval/25_run_evaluate/eval_report.json` |

## 공개 범위 처리

- `diagnosis/38_run_diagnose_second/diagnosis_report.json`: 시스템 단위 실패 기록인 `failures` 키를 제거했다
  (dev 176건, test 58건). 나머지 키와 값은 원본과 같다.
  그래서 DEVLOG에 적은 "실패 176개의 최고 순위 분포"는 이 레포만으로는 재현되지 않는다.
- 넣지 않은 것:
  - 시스템 단위 표(`best_rank.csv`, `failures.csv`, `system_split.csv`, `system_*.csv`)
  - pose 단위 점수(`scores*.parquet`)와 전체 feature 행렬(`unified_all.parquet`, 139MB)
  - 예측 구조 파일, SLURM 로그
