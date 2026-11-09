# 익명화 — 경로 치환 규칙과 파일별 변경 내역

공개 레포를 만들면서 서버 환경 정보를 지웠다. 스크립트의 **동작을 바꾸는 수정은 하지 않았다.**
바꾼 것은 경로 문자열(자리표시자로 치환)과, 주석·SLURM 지시문 안의 GPU 모델명·노드명뿐이다.
한국어 주석과 docstring은 그대로 두었다.

## 1. 자리표시자

| 자리표시자 | 뜻 |
|---|---|
| `<DATA_ROOT>` | 본인 작업 루트. 그 아래에 `Scoring_Function/{pipeline,logs,out}`이 있다 |
| `<SHARED_ROOT>` | 공용 읽기 전용 루트(가상환경·외부 도구 이미지·원본 데이터) |
| `<CASP_ROOT>` | 별도 공용 데이터 루트(읽기 전용) |
| `<FS_ROOT>` | 공유 파일시스템의 마운트 지점(컨테이너 `--bind` 인자 등) |
| `<GPU_TYPE_A>` / `<GPU_TYPE_B>` | GPU 종류 두 가지(23번 GPU 러너의 지시문과 주석) |
| `<CONDA_ENV>` | 공용 conda 환경 이름. 파이썬 실행 경로의 구성요소이며 주석 2곳에도 나온다 |

환경은 "Linux GPU 서버, SLURM 작업 제출"로만 적는다. 사용자명·노드명·GPU 모델명은 남기지 않았다.

## 2. 검증 방법

- **되돌리기 일치**: 경로 자리표시자를 원래 문자열로 되돌리면 원본과 **바이트 단위로 같다**.
  다른 줄은 3절의 "경로 외 변경"에 적은 주석·지시문 줄뿐이다.
- **문법**: `.py` 39개 모두 `ast.parse` 통과, `.sh` 42개 모두 `bash -n` 통과.
- **셸 안전성**: bash에서 따옴표 밖의 `<DATA_ROOT>`는 입출력 리다이렉션으로 해석된다.
  모든 자리표시자가 따옴표 안이나 주석 안에 있음을 확인했다.
- **줄바꿈 보존**: 원본이 CRLF인 파일 5개(`26`·`35`·`36`·`37`·`38`의 `.py`)는 CRLF를 그대로 두었다.
- **해시**: 치환 때문에 사본의 sha256은 원본과 다르다. 결과 보고서의 `script_sha256`(원본 기준)과 대조하려면
  [`pipeline/ORIGINAL_SHA256SUMS.txt`](../pipeline/ORIGINAL_SHA256SUMS.txt)를 쓴다.
- **잔존 검사**: 레포 전체에서 서버 경로·계정명·노드명·GPU 모델명을 grep해 0건임을 확인했다.

## 3. 경로 외 변경 (주석·지시문)

| 파일 | 줄 | 원래 → 바꾼 것 |
|---|---|---|
| `02_build_pose_index.py` | 22 | 주석의 노드명 → "서버" |
| `03_select_representative_systems.py` | 22 | 주석의 노드명 → "서버" |
| `23_run_gnina_smoke.sh` | 7 | `#SBATCH --gres=gpu:<모델명>:1` → `gpu:<GPU_TYPE_A>:1` (**SLURM 지시문**) |
| `23_run_gnina_smoke.sh` | 14–16 | 주석의 GPU 모델명·아키텍처 → `<GPU_TYPE_A>` / `<GPU_TYPE_B>` |

### 3-1. 환경 이름 치환 (공개 전 점검에서 추가)

공개 전 점검에서 공용 conda 환경 이름이 남아 있는 것을 발견해 `<CONDA_ENV>`로 바꿨다.
`pipeline/` 63개와 `results/`의 보고서 2개, 합계 **65개 파일 66곳**이다.

- 대부분은 파이썬 실행 경로의 구성요소다 — `<SHARED_ROOT>/.micromamba/envs/<CONDA_ENV>/bin/python`
- 나머지 2곳은 같은 환경을 가리키는 주석이다.
- 바이트 단위로 치환했고 되돌리면 원본과 같다. CRLF 파일의 줄바꿈 수도 보존했다.
- 치환 후 `.py` 39개 `ast.parse`, `.sh` 42개 `bash -n`, `results/`의 모든 JSON 파싱을 다시 확인했다.

> 같은 점검에서 걸린 **"작업 ID"는 거짓 양성이다.** 러너가 실행 시 `$SLURM_JOB_ID`를 찍는 출력 라벨이며
> 하드코딩된 작업 번호는 없다(함께 걸린 6자리 숫자는 원자 수 상한값 200000·250000이었다).

> 23번 GPU 러너의 `--gres` 지시문은 실행 시 GPU 종류를 고정하는 실제 설정이다. 자리표시자 상태로는
> 제출되지 않으므로, 쓰려면 자기 클러스터의 GPU 종류로 바꿔야 한다. 이 프로젝트의 gnina 본실행은 CPU 러너를 썼다.

## 4. 파일별 치환 수 — `pipeline/`

| 파일 | `<DATA_ROOT>` | `<SHARED_ROOT>` | `<CASP_ROOT>` | `<FS_ROOT>` | 경로 외 |
|---|---|---|---|---|---|
| `01_pilot_contacts.py` | 1 |  | 1 |  |  |
| `02_build_pose_index.py` | 1 | 1 | 1 | 1 | 1 |
| `02_run_pose_index_full.sh` | 4 | 1 |  |  |  |
| `02_run_pose_index_pilot.sh` | 4 | 1 |  |  |  |
| `03_run_select_representative_systems.sh` | 4 | 1 |  |  |  |
| `03_select_representative_systems.py` | 1 | 1 |  | 1 | 1 |
| `04_run_audit_selected_structures.sh` | 4 | 1 |  |  |  |
| `05_run_audit_input_roles.sh` | 4 | 1 |  |  |  |
| `06_run_map_ligand_atoms.sh` | 4 | 1 |  |  |  |
| `07_run_contact_features.sh` | 4 | 1 |  |  |  |
| `08_run_contact_features_expanded.sh` | 4 | 1 |  |  |  |
| `09_label_exploration.py` |  | 1 |  |  |  |
| `09_run_label_exploration.sh` | 4 | 1 |  |  |  |
| `10_label_stratified.py` |  | 1 |  |  |  |
| `10_run_label_stratified.sh` | 4 | 1 |  |  |  |
| `11_run_surface_features.sh` | 4 | 1 |  |  |  |
| `12_run_ligand_descriptors.sh` | 4 | 1 |  |  |  |
| `13_run_water_metal_features.sh` | 4 | 1 |  |  |  |
| `14_pocket_features.py` |  | 1 |  |  |  |
| `14_run_pocket_features.sh` | 4 | 2 |  |  |  |
| `15_run_ligand_geometry.sh` | 4 | 1 |  |  |  |
| `16_run_interaction_features.sh` | 4 | 1 |  |  |  |
| `17_run_integrate_features.sh` | 4 | 1 |  |  |  |
| `18_make_split.py` |  | 1 |  |  |  |
| `18_run_make_split.sh` | 4 | 1 |  |  |  |
| `19_run_combined_extractor.sh` | 4 | 1 |  |  |  |
| `20_run_combined_extractor_full.sh` | 4 | 1 |  |  |  |
| `21_run_merge.sh` | 4 | 1 |  |  |  |
| `21_run_shard.sh` | 4 | 1 |  |  |  |
| `22_run_pocket_merge.sh` | 4 | 1 |  |  |  |
| `22_run_pocket_shard.sh` | 4 | 2 |  |  |  |
| `23_gnina_smoke.py` |  | 1 |  | 2 |  |
| `23_run_gnina_smoke.sh` | 4 | 2 |  | 1 | 5 |
| `23_run_gnina_smoke_cpu.sh` | 4 | 2 |  | 1 |  |
| `24_run_build_labels.sh` | 4 | 1 |  |  |  |
| `25_run_evaluate.sh` | 4 | 1 |  |  |  |
| `26_run_integrate_all.sh` | 4 | 1 |  |  |  |
| `27_gnina_shard.py` |  | 1 |  | 1 |  |
| `27_run_gnina_merge.sh` | 4 | 1 |  |  |  |
| `27_run_gnina_shard.sh` | 4 | 2 |  | 1 |  |
| `28_run_baseline.sh` | 4 | 1 |  |  |  |
| `29_run_ranking.sh` | 4 | 1 |  |  |  |
| `30_run_graded.sh` | 4 | 1 |  |  |  |
| `31_run_combined.sh` | 4 | 1 |  |  |  |
| `32_run_selection.sh` | 4 | 1 |  |  |  |
| `33_run_combination.sh` | 4 | 1 |  |  |  |
| `34_run_union.sh` | 4 | 1 |  |  |  |
| `35_run_test.sh` | 4 | 1 |  |  |  |
| `36_run_subsets.sh` | 4 | 1 |  |  |  |
| `37_run_freeze.sh` | 4 | 1 |  |  |  |
| `38_run_diagnose.sh` | 4 | 1 |  |  |  |
| **합계** | **171** | **55** | **2** | **8** | |

변경된 파일 51개 / 전체 81개. 변경 없는 파일: `04_audit_selected_structures.py`, `05_audit_input_roles.py`, `06_map_ligand_atoms.py`, `07_contact_features.py`, `08_contact_features_expanded.py`, `11_surface_features.py`, `12_ligand_descriptors.py`, `13_water_metal_features.py`, `15_ligand_geometry.py`, `16_interaction_features.py`, `17_integrate_features.py`, `19_combined_extractor.py`, `20_combined_extractor_full.py`, `21_combined_shard.py`, `22_pocket_shard.py`, `24_build_labels.py`, `25_evaluate.py`, `26_integrate_all.py`, `28_baseline_model.py`, `29_ranking_model.py`, `30_graded_model.py`, `31_combined_model.py`, `32_feature_selection.py`, `33_combination_search.py`, `34_union_test.py`, `35_test_evaluation.py`, `36_subset_distribution.py`, `37_freeze_model.py`, `38_rank_diagnosis.py`, `99_diag_mapping_hang.py`.

## 5. `results/`

결과 파일은 데이터 사본이다. 보고서 JSON 안의 경로 문자열에 같은 규칙을 적용했다(JSON 파싱 확인).
그 밖에 공개 범위 때문에 아래를 처리했다.

| 처리 | 대상 | 이유 |
|---|---|---|
| **키 제거** | `diagnosis/38_run_diagnose_second/diagnosis_report.json`의 `failures` 목록(dev 176 + test 58건) | 시스템 단위 실패 기록이라 시스템 단위 표 제외 기준과 맞춤. 집계값은 그대로 |
| 제외 | `best_rank.csv`, `failures.csv`, `system_split.csv` 등 시스템 단위 표 | 팀 내부 예측·평가에서 나온 시스템별 기록 |
| 제외 | `unified_all.parquet`(139MB, 460,300행), pose 단위 점수 `scores*.parquet` | 대용량 · pose 단위 데이터 |
| 제외 | 예측 구조 파일(CIF·PDB·SDF), SLURM 로그 | 팀 생성 구조 원본 · 로그에 노드명과 작업 번호가 남음 |

경로를 치환한 결과 파일은 24개다(전체 118개 중).

