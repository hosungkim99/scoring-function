#!/bin/bash
#SBATCH --job-name=combined
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --chdir=<DATA_ROOT>/Scoring_Function
#SBATCH --output=<DATA_ROOT>/Scoring_Function/logs/combined_%j.out
#SBATCH --error=<DATA_ROOT>/Scoring_Function/logs/combined_%j.err

# Process B 2차(6-4). GPU 불필요, CPU만 씁니다.
# 선택 규칙(plain/ruleA/ruleB) x k(1,3,5) x 채점자(모델/의사난수/셔플)를 한 실행에서 전부 냅니다.
#
# 본실행(원본+z-score 결합, 비대칭 등급 경계):
#   sbatch pipeline/31_run_combined.sh --tag both_asym
#
# 검증용 재현 — 30번과 같은 설정이므로 ruleA/k5 가 0.9347 근처여야 합니다:
#   sbatch pipeline/31_run_combined.sh --features raw --relevance-bins default --tag repro30
#
# 절제:
#   원본만            : sbatch pipeline/31_run_combined.sh --features raw     --tag raw_asym
#   z-score만         : sbatch pipeline/31_run_combined.sh --features zscore  --tag z_asym
#   경계 영향          : sbatch pipeline/31_run_combined.sh --relevance-bins default --tag both_def
#   이진 라벨          : sbatch pipeline/31_run_combined.sh --relevance binary --tag both_bin
#   gnina 제외         : sbatch pipeline/31_run_combined.sh --exclude-categories cat2_gnina --tag both_nognina
#
# --tag 는 결과 폴더를 분리하므로 반복 실행 시 반드시 다르게 주세요.
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
script_path="$project_dir/pipeline/31_combined_model.py"

[[ -x "$python_bin" ]] || { printf '%s\n' 'Python 실행 파일을 확인하세요.' >&2; exit 1; }
[[ -r "$script_path" ]] || { printf '%s\n' '31_combined_model.py 가 없습니다.' >&2; exit 1; }
# 31번은 30 -> 29 -> 28 순으로 모듈을 불러 평가·QA·정규화 코드를 공유합니다.
for dep in 30_graded_model.py 29_ranking_model.py 28_baseline_model.py; do
    [[ -r "$project_dir/pipeline/$dep" ]] || {
        printf '%s 가 없습니다(31번이 import합니다).\n' "$dep" >&2; exit 1; }
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
printf '실행 범위: Process B 결합 모델 학습 + 선택규칙·k 전조합 평가(dev 전용)\n'

exec "$python_bin" -B -u "$script_path" --threads "$ncpu" "$@"
