# NUMBERS — README 수치 ↔ 결과 파일 대조

README 와 문서에 적힌 수치를 `results/` 파일에서 다시 읽어 대조한 결과다.
대조 스크립트가 생성한 표이며, 손으로 옮겨 적지 않았다.

**47개 항목 중 47개 일치, 0개 불일치.**

| 수치 | 파일 (results/ 기준) | 위치 | 파일 값 | 문서 값 | |
|---|---|---|---|---|---|
| 무작위 k=1 | test_eval/35_run_test_final/test_report.json | splits.dev.random_analytic.1 | 0.7406 | 0.7406 | OK |
| 무작위 k=3 | test_eval/35_run_test_final/test_report.json | splits.dev.random_analytic.3 | 0.8735 | 0.8735 | OK |
| 무작위 k=5 | test_eval/35_run_test_final/test_report.json | splits.dev.random_analytic.5 | 0.9133 | 0.9133 | OK |
| 모델 성공률 k=1 | test_eval/35_run_test_final/dev_vs_test.csv | anchor_only / dev|mean|success|ruleA|k1 | 0.79 | 0.79 | OK |
| 모델 성공률 k=3 | test_eval/35_run_test_final/dev_vs_test.csv | anchor_only / dev|mean|success|ruleA|k3 | 0.905 | 0.905 | OK |
| 모델 성공률 k=5 | test_eval/35_run_test_final/dev_vs_test.csv | anchor_only / dev|mean|success|ruleA|k5 | 0.9349 | 0.9349 | OK |
| lift k=1 | test_eval/35_run_test_final/dev_vs_test.csv | dev|mean|lift|ruleA|k1 | 0.0494 | 0.0494 | OK |
| lift k=3 | test_eval/35_run_test_final/dev_vs_test.csv | dev|mean|lift|ruleA|k3 | 0.0315 | 0.0315 | OK |
| lift k=5 | test_eval/35_run_test_final/dev_vs_test.csv | dev|mean|lift|ruleA|k5 | 0.0215 | 0.0215 | OK |
| lift 표준편차 k=1 | test_eval/35_run_test_final/dev_vs_test.csv | dev|std|lift|ruleA|k1 | 0.004 | 0.004 | OK |
| lift 표준편차 k=3 | test_eval/35_run_test_final/dev_vs_test.csv | dev|std|lift|ruleA|k3 | 0.0038 | 0.0038 | OK |
| lift 표준편차 k=5 | test_eval/35_run_test_final/dev_vs_test.csv | dev|std|lift|ruleA|k5 | 0.0034 | 0.0034 | OK |
| 어려운 층 lift k=5 | test_eval/35_run_test_final/dev_vs_test.csv | dev|mean|hard_lift|ruleA|k5 | 0.1288 | 0.1288 | OK |
| 어려운 층 lift σ | test_eval/35_run_test_final/dev_vs_test.csv | dev|std|hard_lift|ruleA|k5 | 0.025 | 0.025 | OK |
| 어려운 층 무작위 k=5 | test_eval/35_run_test_final/test_report.json | splits.dev.random_analytic_hard.5 | 0.2103 | 0.2103 | OK |
| 어려운 층 그룹 수 | strata_profile.csv | very_hard+hard+medium 행 합 | 207 | 207 | OK |
| 평가 대상 그룹 수 | strata_profile.csv | dev 전체 행 합 | 2877 | 2877 | OK |
| 전역 AUC | baseline/…/baseline_report.json | mean_auc | 0.8022 | 0.8022 | OK |
| 베이스라인 lift | baseline/…/baseline_report.json | lift | -0.0134 | -0.0134 | OK |
| 29 rank_raw lift | ranking/29_run_ranking_rank_raw/… | lift | -0.0044 | -0.0044 | OK |
| 29 rank_z lift | ranking/29_run_ranking_rank_z/… | lift | -0.0072 | -0.0072 | OK |
| 29 bin_z lift | ranking/29_run_ranking_bin_z/… | lift | -0.0145 | -0.0145 | OK |
| 상위 5개 성공률 | combined/31_run_combined_repro30/selection_matrix.csv | model/plain/k5 success | 0.9058 | 0.9058 | OK |
| 규칙 B 성공률 | combined/31_run_combined_repro30/selection_matrix.csv | model/ruleB/k5 success | 0.9246 | 0.9246 | OK |
| 규칙 A 성공률 | combined/31_run_combined_repro30/selection_matrix.csv | model/ruleA/k5 success | 0.9347 | 0.9347 | OK |
| 대조군 plain | combined/31_run_combined_repro30/selection_matrix.csv | pseudorandom/plain/k5 | 0.9107 | 0.9107 | OK |
| 대조군 ruleB | combined/31_run_combined_repro30/selection_matrix.csv | pseudorandom/ruleB/k5 | 0.9131 | 0.9131 | OK |
| 대조군 ruleA | combined/31_run_combined_repro30/selection_matrix.csv | pseudorandom/ruleA/k5 | 0.9218 | 0.9218 | OK |
| 다양성 몫 k=5 | combined/31_run_combined_repro30/contribution.csv | ruleA diversity_gain | 0.0128 | 0.0128 | OK |
| 모델 몫 k=5 | combined/31_run_combined_repro30/contribution.csv | ruleA model_gain | 0.0101 | 0.0101 | OK |
| 후보 수 | feature_matrix_all/…/integration_report.json | poses | 460300 | 460300 | OK |
| 그룹 수 | feature_matrix_all/…/integration_report.json | systems | 4603 | 4603 | OK |
| feature 수 | feature_catalog.csv | 행 수 | 63 | 63 | OK |
| feature 범주 수 | feature_catalog.csv | category 고유값 | 8 | 8 | OK |
| dev 그룹 | data_split/…/split_report.json | dev_systems | 3680 | 3680 | OK |
| test 그룹 | data_split/…/split_report.json | test_systems | 923 | 923 | OK |
| 폴드 수 | data_split/…/split_report.json | folds | 5 | 5 | OK |
| 평가 대상(전체) | eval/…/eval_report.json | category_counts.eligible | 3595 | 3595 | OK |
| 생성기 수 모델 | diagnosis/…/diagnosis_summary.csv | generator_spread.model_top5_mean | 1.984 | 1.984 | OK |
| 생성기 수 무작위 | diagnosis/…/diagnosis_summary.csv | generator_spread.random_top5_mean | 3.083 | 3.083 | OK |
| 그룹 내부 AUC | diagnosis/…/diagnosis_summary.csv | score_signal.within_system_auc_all | 0.6573 | 0.6573 | OK |
| AUC 1등 제외 | diagnosis/…/diagnosis_summary.csv | score_signal.within_system_auc_drop_top1 | 0.656 | 0.656 | OK |
| AUC 상위5 제외 | diagnosis/…/diagnosis_summary.csv | score_signal.within_system_auc_drop_top5 | 0.6528 | 0.6528 | OK |
| 순증 (생성기,시드) | diagnosis/…/diagnosis_report.json | dev.axes.generator_seed.net_vs_ruleA | -51 | -51 | OK |
| 순증 (생성기,표본) | diagnosis/…/diagnosis_report.json | dev.axes.generator_sample.net_vs_ruleA | -59 | -59 | OK |
| 학습 그룹 | final_model/…/provenance.json | models[0].train_groups | 2482 | 2482 | OK |
| 학습 후보 | final_model/…/provenance.json | models[0].train_poses | 243599 | 243599 | OK |

## 대조하지 않은 수치

아래는 결과 파일이 레포에 없어 자동 대조할 수 없다. 원본 기록(비공개)에 근거한다.

| 수치 | 근거 | 비고 |
|---|---|---|
| 조합 1,294개 탐색 / 기약 12개 | `combination/33_run_combination_search/combination_report.json` | `search.evaluated`, `irreducible` — 레포에 있음. 위 표에 넣지 않았을 뿐 |
| 라벨 결측 8.8% | 원본 개발일지 2026-10-28 | 대체 라벨 정책의 근거. 산출 과정이 결과 파일에 없음 |
| 20위 밖 76/176 | 38번 `failures` 키 (공개 범위에서 제외) | 집계값만 남기고 그룹 단위 기록은 제거 |
| 샤드 정지 사건의 수치 (결합 398개 등) | 원본 점검 기록 2026-10-29 | 일회성 진단이라 결과 파일 없음 |
| gnina 후보당 4.99초 | 원본 개발일지 2026-10-29 | smoke test 로그 기반 |

