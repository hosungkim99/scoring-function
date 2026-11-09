#!/bin/bash
#SBATCH --job-name=combined_merge
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --chdir=<DATA_ROOT>/Scoring_Function
#SBATCH --output=<DATA_ROOT>/Scoring_Function/logs/combined_merge_%j.out
#SBATCH --error=<DATA_ROOT>/Scoring_Function/logs/combined_merge_%j.err

# 모든 샤드가 끝난 뒤 병합합니다. array가 전부 완료된 다음 제출하세요.
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf '%s\n' '이 파일은 sbatch로 제출하세요.' >&2
    exit 1
fi

project_dir='<DATA_ROOT>/Scoring_Function'
python_bin='<SHARED_ROOT>/.micromamba/envs/<CONDA_ENV>/bin/python'
script_path="$project_dir/pipeline/21_combined_shard.py"

[[ -x "$python_bin" ]] || { printf '%s\n' 'Python 실행 파일을 확인하세요.' >&2; exit 1; }
[[ -r "$script_path" ]] || { printf '%s\n' '21_combined_shard.py가 없습니다.' >&2; exit 1; }

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

printf '작업 ID: %s\n' "$SLURM_JOB_ID"
printf '실행 범위: 전체 통합 추출 샤드 병합 + 완결성 검사 (A6c)\n'

exec "$python_bin" -B -u "$script_path" --merge --num-shards 100
