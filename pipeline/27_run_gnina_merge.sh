#!/bin/bash
#SBATCH --job-name=gnina_merge
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --chdir=<DATA_ROOT>/Scoring_Function
#SBATCH --output=<DATA_ROOT>/Scoring_Function/logs/gnina_merge_%j.out
#SBATCH --error=<DATA_ROOT>/Scoring_Function/logs/gnina_merge_%j.err

# gnina 샤드 병합(③). 샤드 실행과 같은 --modes·--tag 를 주어야 합니다.
#   표본 병합 : sbatch pipeline/27_run_gnina_merge.sh --modes score_only,local_only --tag sample --allow-partial
#   본실행 병합: sbatch pipeline/27_run_gnina_merge.sh --modes score_only --tag full
# 표본은 전체 pose의 일부만 처리하므로 completeness_ok가 False인 것이 정상이며,
# --allow-partial 을 주면 종료 코드를 0으로 둡니다.
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf '%s\n' '이 파일은 sbatch로 제출하세요.' >&2
    exit 1
fi

project_dir='<DATA_ROOT>/Scoring_Function'
python_bin='<SHARED_ROOT>/.micromamba/envs/<CONDA_ENV>/bin/python'
script_path="$project_dir/pipeline/27_gnina_shard.py"

[[ -x "$python_bin" ]] || { printf '%s\n' 'Python 실행 파일을 확인하세요.' >&2; exit 1; }
[[ -r "$script_path" ]] || { printf '%s\n' '27_gnina_shard.py 가 없습니다.' >&2; exit 1; }

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

printf '작업 ID: %s\n' "$SLURM_JOB_ID"
printf '실행 범위: gnina 샤드 병합\n'

exec "$python_bin" -B -u "$script_path" --merge "$@"
