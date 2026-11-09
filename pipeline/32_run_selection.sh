#!/bin/bash
#SBATCH --job-name=selection
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --chdir=<DATA_ROOT>/Scoring_Function
#SBATCH --output=<DATA_ROOT>/Scoring_Function/logs/selection_%j.out
#SBATCH --error=<DATA_ROOT>/Scoring_Function/logs/selection_%j.err

# Process B 마감 2단계(7). feature 선별. GPU 불필요, CPU만 씁니다.
# 31번 2x2에서 확정한 설정(원본 feature + 비대칭 등급 경계)으로 고정되어 있습니다.
#
# 한 작업에서 설정 15종을 학습·평가합니다:
#   전체 1 + 중요도 사다리 6(top5/10/15/20/30/42) + 후보군 제외 8
# 학습이 약 75회(설정 15 x 폴드 5)라 28~31번보다 오래 걸립니다. CPU 8개를 씁니다.
#
# 본실행:
#   sbatch pipeline/32_run_selection.sh --tag locked
#
# 참고(확정 설정이 아니므로 평소에는 쓰지 않음):
#   기존 경계로 비교: sbatch pipeline/32_run_selection.sh --relevance-bins default --tag def
#   빠른 점검       : sbatch pipeline/32_run_selection.sh --n-estimators 150 --tag quick
#
# --tag 는 결과 폴더를 분리하므로 반복 실행 시 반드시 다르게 주세요.
# test 923 시스템은 사용되지 않습니다.
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
script_path="$project_dir/pipeline/32_feature_selection.py"

[[ -x "$python_bin" ]] || { printf '%s\n' 'Python 실행 파일을 확인하세요.' >&2; exit 1; }
[[ -r "$script_path" ]] || { printf '%s\n' '32_feature_selection.py 가 없습니다.' >&2; exit 1; }
# 32번은 31 -> 30 -> 29 -> 28 순으로 모듈을 불러 선택 규칙·평가·QA 코드를 공유합니다.
for dep in 31_combined_model.py 30_graded_model.py 29_ranking_model.py 28_baseline_model.py; do
    [[ -r "$project_dir/pipeline/$dep" ]] || {
        printf '%s 가 없습니다(32번이 import합니다).\n' "$dep" >&2; exit 1; }
done

export PYTHONDONTWRITEBYTECODE=1
ncpu="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$ncpu"
export OPENBLAS_NUM_THREADS="$ncpu"
export MKL_NUM_THREADS="$ncpu"
export NUMEXPR_NUM_THREADS="$ncpu"

printf '작업 ID: %s\n' "$SLURM_JOB_ID"
printf '할당 CPU: %s\n' "$ncpu"
printf 'Python: %s\n' "$python_bin"
printf '실행 범위: Process B feature 선별(사다리 + 후보군 제외, dev 전용)\n'

exec "$python_bin" -B -u "$script_path" --threads "$ncpu" "$@"
