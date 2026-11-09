#!/bin/bash
#SBATCH --job-name=pocket_features
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --chdir=<DATA_ROOT>/Scoring_Function
#SBATCH --output=<DATA_ROOT>/Scoring_Function/logs/pocket_features_%j.out
#SBATCH --error=<DATA_ROOT>/Scoring_Function/logs/pocket_features_%j.err

# 명시적 메모리 요청을 생략합니다. P2Rank를 pose별로 1회씩 순차 실행합니다.
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf '%s\n' '이 파일은 sbatch로 제출하세요.' >&2
    exit 1
fi
if (( $# != 0 )); then
    printf '%s\n' '이 실행 파일은 인자를 받지 않습니다.' >&2
    exit 1
fi

project_dir='<DATA_ROOT>/Scoring_Function'
python_bin='<SHARED_ROOT>/.micromamba/envs/<CONDA_ENV>/bin/python'
script_path="$project_dir/pipeline/14_pocket_features.py"
prank='<SHARED_ROOT>/models/p2rank/p2rank_2.5.1/prank'

[[ -x "$python_bin" ]] || { printf '%s\n' 'Python 실행 파일을 확인하세요.' >&2; exit 1; }
[[ -x "$prank" ]] || { printf '%s\n' 'P2Rank 실행 파일을 확인하세요.' >&2; exit 1; }
# prank는 java가 필요합니다. 기본 PATH의 java를 확인합니다.
command -v java >/dev/null 2>&1 || { printf '%s\n' 'PATH에서 java를 찾을 수 없습니다.' >&2; exit 1; }
for name in 03_select_representative_systems.py 04_audit_selected_structures.py \
            05_audit_input_roles.py 07_contact_features.py 14_pocket_features.py; do
    [[ -r "$project_dir/pipeline/$name" ]] || { printf '필요한 파일 없음: %s\n' "$name" >&2; exit 1; }
done

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

printf '작업 ID: %s\n' "$SLURM_JOB_ID"
printf '실행 노드: %s\n' "${SLURMD_NODENAME:-미확인}"
printf 'java: %s\n' "$(command -v java)"
printf '실행 범위: 05번 eligible pose의 예측 포켓 적합성(category 5, P2Rank pose별 1회)\n'

exec "$python_bin" -B -u "$script_path"
