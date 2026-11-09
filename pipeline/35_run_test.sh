#!/bin/bash
#SBATCH --job-name=testeval
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --chdir=<DATA_ROOT>/Scoring_Function
#SBATCH --output=<DATA_ROOT>/Scoring_Function/logs/testeval_%j.out
#SBATCH --error=<DATA_ROOT>/Scoring_Function/logs/testeval_%j.err

# Process B 마지막 단계(10). test 923 타깃 평가. GPU 불필요, CPU만 씁니다.
#
# *** test 는 한 번만 쓸 수 있습니다. --open-test 없이는 실행되지 않습니다. ***
#
# 비교 설계 — feature 집합만 변인, 나머지 조건은 네 집합이 전부 동일합니다.
#   adopted_33    11개  비교 대상 A (34번이 사전 등록 규칙으로 유지한 집합)
#   union_all12   21개  비교 대상 B (기약 조합 12개 전부의 합집합)
#   anchor_only    7개  비교선 (조합의 기여가 test 에서 살아남는지)
#   top10_32      10개  비교선 (32번이 고른 집합, 참고용)
# 네 집합의 feature 목록은 34번 union_results.csv 에서 읽습니다(손으로 적지 않습니다).
#
# 본실행:
#   sbatch pipeline/35_run_test.sh --open-test --tag final
#
# 34번 결과 폴더를 직접 지정할 때:
#   sbatch pipeline/35_run_test.sh --open-test --tag final \
#          --union-dir out/union/34_run_union_final_full
#
# --tag 는 결과 폴더를 분리합니다. 기존 폴더를 덮어쓰지 않으므로 재실행 시 다르게 주세요.
# 이 작업은 메모리를 요청하지 않습니다(--mem 을 주면 이 클러스터에서 거부됩니다).
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf '%s\n' '이 파일은 sbatch로 제출하세요.' >&2
    exit 1
fi

# test 를 여는 플래그는 사람이 직접 적어야 합니다. 러너가 몰래 붙이지 않습니다.
open_test=0
for arg in "$@"; do
    [[ "$arg" == '--open-test' ]] && open_test=1
done
if [[ "$open_test" -eq 0 ]]; then
    printf '%s\n' 'test 923 타깃은 한 번만 쓸 수 있습니다.' >&2
    printf '%s\n' '열려면 --open-test 를 직접 적어 제출하세요.' >&2
    printf '%s\n' '  sbatch pipeline/35_run_test.sh --open-test --tag final' >&2
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
script_path="$project_dir/pipeline/35_test_evaluation.py"
selection_dir="$project_dir/out/feature_selection/32_run_selection_locked_rep5"

[[ -x "$python_bin" ]] || { printf '%s\n' 'Python 실행 파일을 확인하세요.' >&2; exit 1; }
[[ -r "$script_path" ]] || { printf '%s\n' '35_test_evaluation.py 가 없습니다.' >&2; exit 1; }
# 35번은 34 -> 33 -> 32 -> 31 -> 30 -> 29 -> 28 순으로 모듈을 불러 코드를 공유합니다.
for dep in 34_union_test.py 33_combination_search.py 32_feature_selection.py \
           31_combined_model.py 30_graded_model.py 29_ranking_model.py \
           28_baseline_model.py; do
    [[ -r "$project_dir/pipeline/$dep" ]] || {
        printf '%s 가 없습니다(35번이 import합니다).\n' "$dep" >&2; exit 1; }
done
[[ -r "$selection_dir/chosen_features.json" ]] || {
    printf '32번 결과가 없습니다: %s\n' "$selection_dir/chosen_features.json" >&2
    exit 1; }

# 34번 결과 폴더를 찾습니다. 사람이 --union-dir 로 준 경우에는 건드리지 않습니다.
union_given=0
for arg in "$@"; do
    [[ "$arg" == '--union-dir' || "$arg" == --union-dir=* ]] && union_given=1
done
extra_args=()
if [[ "$union_given" -eq 0 ]]; then
    union_dir=''
    for name in 34_run_union_final_full 34_run_union_final; do
        if [[ -r "$project_dir/out/union/$name/union_results.csv" ]]; then
            union_dir="$project_dir/out/union/$name"
            break
        fi
    done
    if [[ -z "$union_dir" ]]; then
        printf '%s\n' '34번 결과(union_results.csv)를 찾지 못했습니다.' >&2
        printf '%s\n' 'out/union 아래에 있는 폴더:' >&2
        ls -1 "$project_dir/out/union" >&2 2>/dev/null || true
        printf '%s\n' '--union-dir 로 직접 지정해 다시 제출하세요.' >&2
        exit 1
    fi
    extra_args+=(--union-dir "$union_dir")
else
    union_dir='(사람이 지정)'
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
printf '34번 결과: %s\n' "$union_dir"
printf '32번 결과: %s\n' "$selection_dir"
printf '실행 범위: Process B test 923 타깃 평가 (집합 4종, feature 집합만 변인)\n'
printf '*** test 를 엽니다. 한 번만 쓸 수 있습니다. ***\n'

exec "$python_bin" -B -u "$script_path" --threads "$ncpu" \
     --selection-dir "$selection_dir" "${extra_args[@]+"${extra_args[@]}"}" "$@"
