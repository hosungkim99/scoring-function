#!/bin/bash
#SBATCH --job-name=pocket_merge
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --chdir=<DATA_ROOT>/Scoring_Function
#SBATCH --output=<DATA_ROOT>/Scoring_Function/logs/pocket_merge_%j.out
#SBATCH --error=<DATA_ROOT>/Scoring_Function/logs/pocket_merge_%j.err

# 22번 포켓 샤드를 모두 합쳐 전체 포켓 feature 표(pocket_features_all.parquet)와 완결성 리포트를
# 만듭니다. 모든 shard_*.done이 생성된 뒤(즉 array가 전부 끝난 뒤) 1회 제출하세요.
set -euo pipefail

project_dir='<DATA_ROOT>/Scoring_Function'
python_bin='<SHARED_ROOT>/.micromamba/envs/<CONDA_ENV>/bin/python'
script_path="$project_dir/pipeline/22_pocket_shard.py"
NUM_SHARDS=200

[[ -x "$python_bin" ]] || { printf '%s\n' 'Python 실행 파일을 확인하세요.' >&2; exit 1; }
[[ -r "$script_path" ]] || { printf '%s\n' '22_pocket_shard.py를 확인하세요.' >&2; exit 1; }

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

printf '실행 범위: 전체 포켓 샤드 병합(A6c-2 merge)\n'
exec "$python_bin" -B -u "$script_path" --merge --num-shards "$NUM_SHARDS"
