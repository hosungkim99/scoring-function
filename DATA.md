# DATA — 데이터 출처와 라이선스

이 저장소의 **코드와 문서는 MIT**([LICENSE](LICENSE))다. 데이터는 그와 별개이며 아래와 같다.

---

## 1. 이 저장소에 들어 있는 것 / 들어 있지 않은 것

### 들어 있는 것

| 종류 | 위치 | 성격 |
|---|---|---|
| 요약 지표 (CSV·JSON) | `results/` | 이 프로젝트가 계산한 **집계값**. 개별 데이터 레코드가 아니다 |
| 학습된 모델 | `results/final_model/37_run_freeze_v2/*.json` | XGBoost 트리 파라미터 |
| 그림 | `figures/*.png` | 위 요약 파일로 생성 |

### 들어 있지 않은 것

| 종류 | 이유 |
|---|---|
| 구조 예측 결과 원본 | 팀이 생성한 산출물. 공개 권한이 없다 |
| 전체 feature 행렬 (`unified_all.parquet`, 139MB) | 대용량이고 위 원본에서 파생된 것 |
| 후보 단위 점수 (`scores*.parquet`) | 동일 |
| 그룹 단위 표 (`system_*.csv`, `best_rank.csv`, `failures.csv`) | 개별 시스템 식별자가 들어간 기록 |
| 입력 데이터셋 원본 | 아래 §2 참조 |

**따라서 이 저장소만으로는 파이프라인을 처음부터 재현할 수 없다.**
재현 가능한 것은 ① 요약 파일로부터의 그림 생성, ② 저장된 모델을 이용한 추론이다.

---

## 2. 입력 데이터셋 — PLINDER

이 프로젝트는 **PLINDER**(The Protein-Ligand INteraction Dataset and Evaluation Resource)에서
파생된 시스템 목록과 평가 지표를 사용했다.

**인용**

> Durairaj, J., Adeshina, Y., Cao, Z., Zhang, X., et al.
> *PLINDER: The protein-ligand interactions dataset and evaluation resource.*
> bioRxiv (2024). DOI: [10.1101/2024.07.17.603955](https://doi.org/10.1101/2024.07.17.603955)
>
> 워크숍 버전: ICML 2024 Workshop ML4LMS.

**저장소**: <https://github.com/plinder-org/plinder>

**라이선스**

| 구분 | 라이선스 |
|---|---|
| PLINDER 가 큐레이션한 데이터 | Apache License 2.0 |
| PLINDER 코드 | Apache License 2.0 |
| 그중 ChEMBL 에서 가져온 부분 | CC BY-SA 4.0 |
| 그중 BindingDB 가 큐레이션한 부분 | CC BY 4.0 |

> **`확인 필요`** — 위 구분은 PLINDER 저장소의 README 와 공개 메타데이터에 근거한다.
> 데이터를 재배포하기 전에는 **PLINDER 저장소의 LICENSE 파일과 데이터셋 문서를 직접 확인할 것.**
> 이 저장소는 PLINDER 의 데이터 레코드를 재배포하지 않고 집계값만 싣는다.

---

## 3. 사용한 외부 도구

| 도구 | 용도 | 비고 |
|---|---|---|
| **gnina** | 도킹 점수 3종 (`--score_only`) | 최종 feature 7개 중 3개의 출처 |
| **P2Rank** | 포켓 예측 점수 | 최종 feature 7개 중 1개의 출처 |
| gemmi | 구조 파일 파싱 | |
| RDKit | 분자 그래프·descriptor | |
| XGBoost 2.1.4 | 랭킹 모델 | |
| NumPy / SciPy / pandas / PyArrow / scikit-learn | 계산·입출력 | |

**각 도구의 라이선스와 인용 요건은 해당 프로젝트의 문서를 따른다 — `확인 필요`.**
버전은 XGBoost(2.1.4)만 결과 파일에 기록돼 있고 나머지는 기록되지 않았다.

> 최종 모델의 feature 7개 중 **4개가 gnina·P2Rank 의 출력**이다.
> 이 프로젝트는 그 도구들을 대체하는 것이 아니라 **그 위에 올린 재순위화기**다.

---

## 4. 재배포 시 확인할 것

이 저장소를 그대로 공개하는 경우에는 문제가 없다고 판단했다. 다만 아래를 바꾸려면 확인이 필요하다.

- `results/` 에 **그룹 단위 표나 후보 단위 데이터를 추가**하려는 경우 → 원본 데이터 권한 확인
- 구조 예측 결과를 포함하려는 경우 → 생성 주체의 동의 필요
- PLINDER 데이터 레코드 자체를 싣는 경우 → §2 의 라이선스 구분과 귀속 표기 확인
