#!/bin/bash
#SBATCH --job-name=union
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --chdir=<DATA_ROOT>/Scoring_Function
#SBATCH --output=<DATA_ROOT>/Scoring_Function/logs/union_%j.out
#SBATCH --error=<DATA_ROOT>/Scoring_Function/logs/union_%j.err

# Process B 최종 집합 확정(9). GPU 불필요, CPU만 씁니다.
# "조합을 여러 개 쓰면 더 좋은가"를 재서 최종 feature 집합을 확정합니다.
# 33번이 찾은 조합들을 상위 1·2·3·5개·전체로 합집합을 만들어 같은 조건으로 비교합니다.
#
# 본실행:
#   sbatch pipeline/34_run_union.sh --tag final
#
# 눈금 조정:
#   sbatch pipeline/34_run_union.sh --union-steps 1,2,4,8 --tag steps8
#
# 32·33번 결과 폴더에서 기준 집합·조합 등수표·채택 집합을 읽습니다.
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
script_path="$project_dir/pipeline/34_union_test.py"
selection_dir="$project_dir/out/feature_selection/32_run_selection_locked_rep5"
combination_dir="$project_dir/out/combination/33_run_combination_search"

[[ -x "$python_bin" ]] || { printf '%s\n' 'Python 실행 파일을 확인하세요.' >&2; exit 1; }
[[ -r "$script_path" ]] || { printf '%s\n' '34_union_test.py 가 없습니다.' >&2; exit 1; }
# 34번은 33 -> 32 -> 31 -> 30 -> 29 -> 28 순으로 모듈을 불러 코드를 공유합니다.
for dep in 33_combination_search.py 32_feature_selection.py 31_combined_model.py \
           30_graded_model.py 29_ranking_model.py 28_baseline_model.py; do
    [[ -r "$project_dir/pipeline/$dep" ]] || {
        printf '%s 가 없습니다(34번이 import합니다).\n' "$dep" >&2; exit 1; }
done
# 후보 집합을 앞선 결과에서 읽으므로 두 파일이 반드시 있어야 합니다.
[[ -r "$selection_dir/single_feature.csv" ]] || {
    printf '32번 결과가 없습니다: %s\n' "$selection_dir/single_feature.csv" >&2; exit 1; }
[[ -r "$combination_dir/ranked_combinations.csv" ]] || {
    printf '33번 결과가 없습니다: %s\n' "$combination_dir/ranked_combinations.csv" >&2; exit 1; }

export PYTHONDONTWRITEBYTECODE=1
ncpu="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$ncpu"
export OPENBLAS_NUM_THREADS="$ncpu"
export MKL_NUM_THREADS="$ncpu"
export NUMEXPR_NUM_THREADS="$ncpu"

printf '작업 ID: %s\n' "$SLURM_JOB_ID"
printf '할당 CPU: %s\n' "$ncpu"
printf 'Python: %s\n' "$python_bin"
printf '실행 범위: Process B 최종 feature 집합 확정(합집합 비교, dev 전용)\n'

exec "$python_bin" -B -u "$script_path" --threads "$ncpu" "$@"
