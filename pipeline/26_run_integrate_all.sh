#!/bin/bash
#SBATCH --job-name=integrate_all
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --chdir=<DATA_ROOT>/Scoring_Function
#SBATCH --output=<DATA_ROOT>/Scoring_Function/logs/integrate_all_%j.out
#SBATCH --error=<DATA_ROOT>/Scoring_Function/logs/integrate_all_%j.err

# 전체 460k pose feature 통합 + sc_rmsd 라벨 결합(Process A 마무리, ④).
# 인자 없이 제출하면 기본 경로(A6c + P2Rank + 라벨 + 분할)로 통합합니다.
# 아직 없는 추출원(P2Rank·gnina)은 경로가 없으면 자동으로 생략되므로,
# 중간 점검용으로 지금 먼저 돌려볼 수 있습니다.
#   예) 중간 통합(현재)     : sbatch pipeline/26_run_integrate_all.sh --tag nopocket
#   예) cat5 포함(P2Rank 후): sbatch pipeline/26_run_integrate_all.sh --tag withpocket
#   예) gnina까지 최종      : sbatch pipeline/26_run_integrate_all.sh --gnina <gnina.parquet> --tag final
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf '%s\n' '이 파일은 sbatch로 제출하세요.' >&2
    exit 1
fi

project_dir='<DATA_ROOT>/Scoring_Function'
python_bin='<SHARED_ROOT>/.micromamba/envs/<CONDA_ENV>/bin/python'
script_path="$project_dir/pipeline/26_integrate_all.py"

[[ -x "$python_bin" ]] || { printf '%s\n' 'Python 실행 파일을 확인하세요.' >&2; exit 1; }
# 26번은 05→04→03(경로 보호·저장) 모듈을 import합니다.
for name in 03_select_representative_systems.py 04_audit_selected_structures.py \
            05_audit_input_roles.py 26_integrate_all.py; do
    [[ -r "$project_dir/pipeline/$name" ]] || { printf '필요한 파일 없음: %s\n' "$name" >&2; exit 1; }
done

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

printf '작업 ID: %s\n' "$SLURM_JOB_ID"
printf '실행 노드: %s\n' "${SLURMD_NODENAME:-미확인}"
printf '실행 범위: 전체 460k feature 통합 + sc_rmsd 라벨 결합(④)\n'

exec "$python_bin" -B -u "$script_path" "$@"
