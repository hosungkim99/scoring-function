#!/bin/bash
#SBATCH --job-name=diagnose
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --chdir=<DATA_ROOT>/Scoring_Function
#SBATCH --output=<DATA_ROOT>/Scoring_Function/logs/diagnose_%j.out
#SBATCH --error=<DATA_ROOT>/Scoring_Function/logs/diagnose_%j.err

# 다음 사이클 1단계. "모델이 2~5등을 못 고르는 결함"의 원인을 가립니다.
# GPU 불필요, CPU만 씁니다. **새 추출이 없습니다.**
#
# 가리려는 것 — 여백 59 타깃이 어디서 새는가
#   선택 문제  근접 pose 가 점수 상위권에 있는데 중복 때문에 자리를 뺏긴다
#              -> 유사도 기반 선택 규칙. 모델 재학습 불필요
#   점수 문제  근접 pose 가 점수 50등쯤에 있어 어떤 규칙으로도 못 찾는다
#              -> 합의 feature 추가. pose 간 거리 추출 1회 필요
#
# 재는 것 5가지
#   (1) 근접 pose 의 최고 점수 순위 (섞기 기준선과 함께)
#   (2) success / lift 곡선 k=1~100
#   (3) Top-5 의 생성기 수 — 모델 vs 무작위
#   (4) 중복성 — Top-5 안 근접 pose 수, 1->5 증분 이득
#   (5) 점수 신호 감쇠 — 시스템 내 AUC (전체 / 1등 제외 / 상위5 제외)
#
# 분할 방침
#   주 분할은 dev out-of-fold(2,877 타깃)입니다. test 718 보다 분포가 4배 촘촘하고,
#   설계 결정을 dev 에서 끝내는 규율에도 맞습니다. test 는 메커니즘이 홀드아웃에서도
#   같은지 확인하는 데만 쓰며 여기서 아무것도 고르지 않습니다.
#
# 본실행:
#   sbatch pipeline/38_run_diagnose.sh --tag first
#
# 37번 폴더를 직접 지정할 때:
#   sbatch pipeline/38_run_diagnose.sh --tag first \
#          --freeze-dir out/final_model/37_run_freeze_v2
#
# --tag 는 결과 폴더를 분리합니다. 기존 폴더를 덮어쓰지 않습니다.
# 이 작업은 메모리를 요청하지 않습니다(--mem 을 주면 이 클러스터에서 거부됩니다).
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf '%s\n' '이 파일은 sbatch로 제출하세요.' >&2
    exit 1
fi

project_dir='<DATA_ROOT>/Scoring_Function'
base_python='<SHARED_ROOT>/.micromamba/envs/<CONDA_ENV>/bin/python'
venv_python="$project_dir/envs/pb/bin/python"
if [[ -x "$venv_python" ]]; then
    python_bin="$venv_python"
else
    python_bin="$base_python"
fi
script_path="$project_dir/pipeline/38_rank_diagnosis.py"

[[ -x "$python_bin" ]] || { printf '%s\n' 'Python 실행 파일을 확인하세요.' >&2; exit 1; }
[[ -r "$script_path" ]] || {
    printf '%s\n' '38_rank_diagnosis.py 가 없습니다.' >&2; exit 1; }
# 38번은 36 -> 35 -> 34 -> 33 -> 32 -> 31 -> 30 -> 29 -> 28 순으로 모듈을 불러옵니다.
for dep in 36_subset_distribution.py 35_test_evaluation.py 34_union_test.py \
           33_combination_search.py 32_feature_selection.py 31_combined_model.py \
           30_graded_model.py 29_ranking_model.py 28_baseline_model.py; do
    [[ -r "$project_dir/pipeline/$dep" ]] || {
        printf '%s 가 없습니다(38번이 import합니다).\n' "$dep" >&2; exit 1; }
done

# 37번 결과 폴더 — feature 목록과 굳힌 모델을 읽습니다.
freeze_given=0
for arg in "$@"; do
    [[ "$arg" == '--freeze-dir' || "$arg" == --freeze-dir=* ]] && freeze_given=1
done
extra_args=()

if [[ "$freeze_given" -eq 0 ]]; then
    freeze_dir=''
    for name in 37_run_freeze_v2 37_run_freeze_v1; do
        if [[ -r "$project_dir/out/final_model/$name/model.json" \
           && -r "$project_dir/out/final_model/$name/feature_order.json" ]]; then
            freeze_dir="$project_dir/out/final_model/$name"
            break
        fi
    done
    if [[ -z "$freeze_dir" ]]; then
        printf '%s\n' '37번 결과(model.json + feature_order.json)를 찾지 못했습니다.' >&2
        printf '%s\n' 'out/final_model 아래에 있는 폴더:' >&2
        ls -1 "$project_dir/out/final_model" >&2 2>/dev/null || true
        printf '%s\n' '--freeze-dir 로 직접 지정해 다시 제출하세요.' >&2
        exit 1
    fi
    extra_args+=(--freeze-dir "$freeze_dir")
else
    freeze_dir='(사람이 지정)'
fi

export PYTHONDONTWRITEBYTECODE=1
ncpu="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$ncpu"
export OPENBLAS_NUM_THREADS="$ncpu"
export MKL_NUM_THREADS="$ncpu"
export NUMEXPR_NUM_THREADS="$ncpu"

printf '작업 ID: %s\n' "$SLURM_JOB_ID"
printf '할당 CPU: %s\n' "$ncpu"
printf 'Python: %s\n' "$python_bin"
printf '37번 결과: %s\n' "$freeze_dir"
printf '실행 범위: 랭킹 결함 진단 (분석만. 새 추출·선택 없음)\n'

exec "$python_bin" -B -u "$script_path" --threads "$ncpu" \
     "${extra_args[@]+"${extra_args[@]}"}" "$@"
