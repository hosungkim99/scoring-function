# figures

`results/` 의 요약 파일만으로 만든 그림이다. 새 계산이나 새 실험은 하지 않는다.

```bash
python figures/make_figures.py
```

필요한 것은 matplotlib 뿐이다. 한글 폰트(Malgun Gothic / NanumGothic / AppleGothic / Noto Sans CJK KR)
중 하나가 있으면 자동으로 쓴다.

| 그림 | 보여 주는 것 | 데이터 |
|---|---|---|
| `fig1_headroom.png` | 무작위 기준선과 남은 여백. Top-5 는 무작위가 이미 0.9133 | `test_eval/35_run_test_final/dev_vs_test.csv`, `test_report.json` |
| `fig2_lift_by_k.png` | k 별 lift 와 여백 상한. 점수 상위 k 개 선택은 Top-5 에서 음수 | 같은 파일 |
| `fig3_selection_rule.png` | 선택 규칙 3종 × (모델 점수 / 점수 없는 대조군) | `combined/31_run_combined_repro30/selection_matrix.csv` |
| `fig4_feature_ladder.png` | feature 5~57 개 사다리가 평탄함 (시드 간 표준편차 포함) | `feature_selection/32_run_selection_locked_rep5/selection_results.csv` |
| `fig5_generator_spread.png` | 점수 상위 5개에 들어간 생성기 종류 수 분포 | `diagnosis/38_run_diagnose_second/generator_spread.csv`, `diagnosis_summary.csv` |

모든 그림은 dev(평가 대상 2,877 그룹) 기준이고, 캡션에 출처 파일 경로가 들어 있다.
색은 두 계열(파랑·주황)만 쓰고 기준선·대조군은 회색으로 둬서, 색각 이상에서도 계열이 구분되게 했다.

## 만들지 못한 그림

**생성기별 성공률** — 필요한 데이터가 이 저장소에 없다.
생성기(af3·bt2·of3·pt2)별로 성공률을 내려면 후보 단위 라벨과 생성기 열을 조인해야 하는데,
`unified_all.parquet` 과 `labels.parquet` 은 공개 범위에서 제외했다([DATA.md](../DATA.md)).
서버에서 집계 CSV 를 따로 뽑아야 만들 수 있다.
