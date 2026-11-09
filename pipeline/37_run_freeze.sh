#!/bin/bash
#SBATCH --job-name=freeze
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --chdir=<DATA_ROOT>/Scoring_Function
#SBATCH --output=<DATA_ROOT>/Scoring_Function/logs/freeze_%j.out
#SBATCH --error=<DATA_ROOT>/Scoring_Function/logs/freeze_%j.err

# Process B 산출물(12). 확정된 scoring function 을 쓸 수 있는 파일로 굳힙니다.
# GPU 불필요, CPU만 씁니다.
#
# 이 실행은 설계 결정을 하나도 새로 하지 않습니다. 28~36번이 정한 레시피를 받아
# 학습된 모델과 독립 추론 스크립트로 만들 뿐입니다. 고를 것이 없으므로 test 에서
# 아무것도 선택하지 않고, test 는 "저장한 모델이 35번 수치를 재현하는가"만 봅니다.
#
# 굳히는 것
#   feature 7개   34번 union_results.csv 의 anchor_only
#                 (36번에서 이 위에 무엇을 얹어도 차이가 없음이 확인됨)
#   1단 모델      rank:ndcg, 그룹=시스템, 트리 400/깊이 6/lr 0.05
#   2단 규칙      규칙 A — 생성기별 1위 먼저, 남은 자리는 점수순
#
# 출하 모델은 기준 시드(--seed) 하나로 사전 지정합니다. 시드 5개를 전부 학습해
# 저장하고 5개 평균 성적도 적지만, 성적을 보고 출하 모델을 바꾸지 않습니다.
#
# 본실행:
#   sbatch pipeline/37_run_freeze.sh --tag v1
#
# 만들어지는 것 (out/final_model/37_run_freeze_v1/)
#   model.json  feature_order.json  spec.json  provenance.json
#   predict.py  selfcheck.json  freeze_report.json
#
# 쓰는 법 (서버에서든 받아가서든, 28~36번 없이 돌아갑니다)
#   python predict.py --poses <poses.parquet> --k 5 --out top5.csv
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
script_path="$project_dir/pipeline/37_freeze_model.py"

[[ -x "$python_bin" ]] || { printf '%s\n' 'Python 실행 파일을 확인하세요.' >&2; exit 1; }
[[ -r "$script_path" ]] || {
    printf '%s\n' '37_freeze_model.py 가 없습니다.' >&2; exit 1; }
# 37번은 35 -> 34 -> 33 -> 32 -> 31 -> 30 -> 29 -> 28 순으로 모듈을 불러 코드를 공유합니다.
for dep in 35_test_evaluation.py 34_union_test.py 33_combination_search.py \
           32_feature_selection.py 31_combined_model.py 30_graded_model.py \
           29_ranking_model.py 28_baseline_model.py; do
    [[ -r "$project_dir/pipeline/$dep" ]] || {
        printf '%s 가 없습니다(37번이 import합니다).\n' "$dep" >&2; exit 1; }
done

# 34번 결과 폴더 — 최종 집합(anchor_only)을 읽습니다.
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

# 35번 결과 폴더 — 재현 점검의 기준값을 읽습니다.
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
printf '실행 범위: 확정된 scoring function 을 파일로 굳히기 (아무것도 고르지 않음)\n'

exec "$python_bin" -B -u "$script_path" --threads "$ncpu" \
     "${extra_args[@]+"${extra_args[@]}"}" "$@"
