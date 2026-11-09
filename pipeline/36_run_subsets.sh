#!/bin/bash
#SBATCH --job-name=subsets
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --chdir=<DATA_ROOT>/Scoring_Function
#SBATCH --output=<DATA_ROOT>/Scoring_Function/logs/subsets_%j.out
#SBATCH --error=<DATA_ROOT>/Scoring_Function/logs/subsets_%j.err

# Process B 보강(11). "feature 집합 선택이 얼마나 중요한가"를 분포로 답합니다.
# GPU 불필요, CPU만 씁니다.
#
# *** 이 실행은 최고 집합을 고르지 않습니다. 사전 등록된 사항입니다. ***
# *** test 923 타깃을 다시 엽니다. --reuse-test 없이는 실행되지 않습니다. ***
#
# 재는 것
#   지정 3종   union_top1(13) / union_top2(14) / union_top5(18)
#              35번이 빠뜨린 집합. 시드 5회로 재어 7집합 기록을 완성합니다.
#   무작위     균일 추출 4크기 x 4개 = 16, 단독축 고정 3크기 x 4개 = 12
#              시드 2회. 분산분해로 시드 잡음을 빼냅니다.
#   재사용     adopted_33 / union_all12 / anchor_only / top10_32 는 다시 재지 않고
#              35번 dev_vs_test.csv 에서 읽어 같은 표에 올립니다.
#
# 헤드라인은 분산분해입니다(distribution.csv 의 p_value).
#   p >= 0.05  ->  집합 선택의 영향이 시드 잡음에 묻힘 = 어느 집합을 쓰든 같다
#   p <  0.05  ->  집합 간 차이가 잡음보다 큼
#
# 본실행:
#   sbatch pipeline/36_run_subsets.sh --reuse-test --tag first
#
# 표본을 늘릴 때(크기마다 8개로):
#   sbatch pipeline/36_run_subsets.sh --reuse-test --tag wide --random-per-size 8
#
# --subset-seed 를 바꿔 여러 번 돌려 좋은 집합을 찾는 것은 낚시입니다. 하지 마세요.
# --tag 는 결과 폴더를 분리합니다. 기존 폴더를 덮어쓰지 않습니다.
# 이 작업은 메모리를 요청하지 않습니다(--mem 을 주면 이 클러스터에서 거부됩니다).
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf '%s\n' '이 파일은 sbatch로 제출하세요.' >&2
    exit 1
fi

# test 를 다시 여는 플래그는 사람이 직접 적어야 합니다. 러너가 몰래 붙이지 않습니다.
reuse=0
for arg in "$@"; do
    [[ "$arg" == '--reuse-test' ]] && reuse=1
done
if [[ "$reuse" -eq 0 ]]; then
    printf '%s\n' '이 실행은 test 923 타깃을 다시 엽니다.' >&2
    printf '%s\n' '열려면 --reuse-test 를 직접 적어 제출하세요.' >&2
    printf '%s\n' '  sbatch pipeline/36_run_subsets.sh --reuse-test --tag first' >&2
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
script_path="$project_dir/pipeline/36_subset_distribution.py"

[[ -x "$python_bin" ]] || { printf '%s\n' 'Python 실행 파일을 확인하세요.' >&2; exit 1; }
[[ -r "$script_path" ]] || {
    printf '%s\n' '36_subset_distribution.py 가 없습니다.' >&2; exit 1; }
# 36번은 35 -> 34 -> 33 -> 32 -> 31 -> 30 -> 29 -> 28 순으로 모듈을 불러 코드를 공유합니다.
for dep in 35_test_evaluation.py 34_union_test.py 33_combination_search.py \
           32_feature_selection.py 31_combined_model.py 30_graded_model.py \
           29_ranking_model.py 28_baseline_model.py; do
    [[ -r "$project_dir/pipeline/$dep" ]] || {
        printf '%s 가 없습니다(36번이 import합니다).\n' "$dep" >&2; exit 1; }
done

# 34번 결과 폴더 — 지정 3종과 단독축을 읽습니다.
union_given=0
test_eval_given=0
for arg in "$@"; do
    [[ "$arg" == '--union-dir' || "$arg" == --union-dir=* ]] && union_given=1
    [[ "$arg" == '--test-eval-dir' || "$arg" == --test-eval-dir=* ]] && test_eval_given=1
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

# 35번 결과 폴더 — 이미 잰 4종의 성적을 읽습니다(재측정하지 않습니다).
if [[ "$test_eval_given" -eq 0 ]]; then
    test_eval_dir=''
    for cand in "$project_dir"/out/test_eval/35_run_test_*; do
        if [[ -r "$cand/dev_vs_test.csv" ]]; then
            test_eval_dir="$cand"
            break
        fi
    done
    if [[ -z "$test_eval_dir" ]]; then
        printf '%s\n' '35번 결과(dev_vs_test.csv)를 찾지 못했습니다.' >&2
        printf '%s\n' 'out/test_eval 아래에 있는 폴더:' >&2
        ls -1 "$project_dir/out/test_eval" >&2 2>/dev/null || true
        printf '%s\n' '--test-eval-dir 로 직접 지정해 다시 제출하세요.' >&2
        exit 1
    fi
    extra_args+=(--test-eval-dir "$test_eval_dir")
else
    test_eval_dir='(사람이 지정)'
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
printf '35번 결과: %s\n' "$test_eval_dir"
printf '실행 범위: feature 집합 선택의 영향을 분포로 측정 (아무것도 고르지 않음)\n'
printf '*** test 를 다시 엽니다. 최고 집합을 고르지 않습니다. ***\n'

exec "$python_bin" -B -u "$script_path" --threads "$ncpu" \
     "${extra_args[@]+"${extra_args[@]}"}" "$@"
