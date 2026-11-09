#!/bin/bash
#SBATCH --job-name=combination
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --chdir=<DATA_ROOT>/Scoring_Function
#SBATCH --output=<DATA_ROOT>/Scoring_Function/logs/combination_%j.out
#SBATCH --error=<DATA_ROOT>/Scoring_Function/logs/combination_%j.err

# Process B 조합 탐색(8). GPU 불필요, CPU만 씁니다.
# 단독으로는 무작위보다 못한 feature 31개 중, 묶으면 유의미해지는 조합을 찾습니다.
# 쌍에 한정하지 않고 크기 2는 전수(465쌍), 크기 3~6은 beam search.
# 설계와 교체 기준은 개발일지 10.15에 실행 전 등록되어 있습니다.
#
# 본실행:
#   sbatch pipeline/33_run_combination.sh --tag search
#
# 범위 조정:
#   더 넓게(느림)  : sbatch pipeline/33_run_combination.sh --beam 12 --max-size 8 --tag wide
#   더 좁게(빠름)  : sbatch pipeline/33_run_combination.sh --beam 5  --max-size 4 --tag narrow
#   빠른 점검      : sbatch pipeline/33_run_combination.sh --beam 4 --max-size 3 --repeats 2 --tag quick
#
# 후보 풀·기준 집합·현 잠정 답은 32번 결과 폴더에서 읽습니다(--selection-dir).
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
script_path="$project_dir/pipeline/33_combination_search.py"
selection_dir="$project_dir/out/feature_selection/32_run_selection_locked_rep5"

[[ -x "$python_bin" ]] || { printf '%s\n' 'Python 실행 파일을 확인하세요.' >&2; exit 1; }
[[ -r "$script_path" ]] || { printf '%s\n' '33_combination_search.py 가 없습니다.' >&2; exit 1; }
# 33번은 32 -> 31 -> 30 -> 29 -> 28 순으로 모듈을 불러 평가·대조군·학습 코드를 공유합니다.
for dep in 32_feature_selection.py 31_combined_model.py 30_graded_model.py \
           29_ranking_model.py 28_baseline_model.py; do
    [[ -r "$project_dir/pipeline/$dep" ]] || {
        printf '%s 가 없습니다(33번이 import합니다).\n' "$dep" >&2; exit 1; }
done
# 후보 풀을 32번 결과에서 읽으므로 그 파일이 반드시 있어야 합니다.
[[ -r "$selection_dir/single_feature.csv" ]] || {
    printf '32번 결과가 없습니다: %s\n' "$selection_dir/single_feature.csv" >&2; exit 1; }

export PYTHONDONTWRITEBYTECODE=1
ncpu="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$ncpu"
export OPENBLAS_NUM_THREADS="$ncpu"
export MKL_NUM_THREADS="$ncpu"
export NUMEXPR_NUM_THREADS="$ncpu"

printf '작업 ID: %s\n' "$SLURM_JOB_ID"
printf '할당 CPU: %s\n' "$ncpu"
printf 'Python: %s\n' "$python_bin"
printf '실행 범위: Process B 조합 탐색(단독 불가 feature의 유의미 조합, dev 전용)\n'

exec "$python_bin" -B -u "$script_path" --threads "$ncpu" "$@"
