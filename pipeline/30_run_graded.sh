#!/bin/bash
#SBATCH --job-name=graded
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --chdir=<DATA_ROOT>/Scoring_Function
#SBATCH --output=<DATA_ROOT>/Scoring_Function/logs/graded_%j.out
#SBATCH --error=<DATA_ROOT>/Scoring_Function/logs/graded_%j.err

# Process B 등급 relevance 모델(6-3). GPU 불필요, CPU만 씁니다.
# 29번 절제에서 확정된 설정(rank 목적함수 + 정규화 없음)을 고정하고 라벨 표현만 바꿉니다.
#
# 본실행:
#   sbatch pipeline/30_run_graded.sh --tag graded
#
# 절제:
#   이진 라벨로 되돌려 rank_raw 재현 확인:
#     sbatch pipeline/30_run_graded.sh --relevance binary --tag bin_check
#   gnina 제외(증분 측정):
#     sbatch pipeline/30_run_graded.sh --exclude-categories cat2_gnina --tag graded_nognina
#
# --tag 는 결과 폴더를 분리하므로 반복 실행 시 반드시 다르게 주세요.
# 기본값으로 28_run_baseline_f63 과 29_run_ranking_rank_raw 와 비교표를 출력합니다.
# test 923 시스템은 어떤 조합에서도 사용되지 않습니다.
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
script_path="$project_dir/pipeline/30_graded_model.py"

[[ -x "$python_bin" ]] || { printf '%s\n' 'Python 실행 파일을 확인하세요.' >&2; exit 1; }
[[ -r "$script_path" ]] || { printf '%s\n' '30_graded_model.py 가 없습니다.' >&2; exit 1; }
# 30번은 29번을 통해 28번까지 모듈로 불러 평가 코드를 공유합니다.
for dep in 29_ranking_model.py 28_baseline_model.py; do
    [[ -r "$project_dir/pipeline/$dep" ]] || {
        printf '%s 가 없습니다(30번이 import합니다).\n' "$dep" >&2; exit 1; }
done

export PYTHONDONTWRITEBYTECODE=1
ncpu="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$ncpu"
export OPENBLAS_NUM_THREADS="$ncpu"
export MKL_NUM_THREADS="$ncpu"
export NUMEXPR_NUM_THREADS="$ncpu"

printf '작업 ID: %s\n' "$SLURM_JOB_ID"
printf '할당 CPU: %s\n' "$ncpu"
printf 'Python: %s\n' "$python_bin"
printf '실행 범위: Process B 등급 relevance 모델 학습 + Top-5 평가(dev 전용)\n'

exec "$python_bin" -B -u "$script_path" --threads "$ncpu" "$@"
