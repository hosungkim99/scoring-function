#!/bin/bash
#SBATCH --job-name=pocket_shard
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --array=0-199%16
#SBATCH --chdir=<DATA_ROOT>/Scoring_Function
#SBATCH --output=<DATA_ROOT>/Scoring_Function/logs/pocket_shard_%A_%a.out
#SBATCH --error=<DATA_ROOT>/Scoring_Function/logs/pocket_shard_%A_%a.err

# 전체 4603 시스템을 200개 샤드로 나눠 P2Rank 포켓 feature를 추출합니다(A6c-2, category 5).
# P2Rank는 pose당 JVM을 띄우는 비싼 호출이라 샤드 수를 21번보다 늘렸습니다(샤드당 약 2300 pose).
# %16은 동시 실행 상한(공유 서버 정책에 맞게 조정). NUM_SHARDS는 --array 크기와 같아야 합니다.
# --time은 샤드당 처리 시간 추정치에 맞춰 조정하세요(pose당 P2Rank 호출 수 초~십수 초 가정).
set -euo pipefail

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    printf '%s\n' '이 파일은 sbatch(array)로 제출하세요.' >&2
    exit 1
fi

project_dir='<DATA_ROOT>/Scoring_Function'
python_bin='<SHARED_ROOT>/.micromamba/envs/<CONDA_ENV>/bin/python'
script_path="$project_dir/pipeline/22_pocket_shard.py"
prank_bin='<SHARED_ROOT>/models/p2rank/p2rank_2.5.1/prank'
NUM_SHARDS=200

[[ -x "$python_bin" ]] || { printf '%s\n' 'Python 실행 파일을 확인하세요.' >&2; exit 1; }
[[ -x "$prank_bin" ]] || { printf '%s\n' 'P2Rank(prank) 실행 파일을 확인하세요.' >&2; exit 1; }
for name in 05_audit_input_roles.py 06_map_ligand_atoms.py 07_contact_features.py \
            08_contact_features_expanded.py 14_pocket_features.py 19_combined_extractor.py \
            20_combined_extractor_full.py 22_pocket_shard.py; do
    [[ -r "$project_dir/pipeline/$name" ]] || { printf '필요한 파일 없음: %s\n' "$name" >&2; exit 1; }
done

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

printf '작업 ID: %s 배열 작업: %s\n' "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID"
printf '실행 노드: %s\n' "${SLURMD_NODENAME:-미확인}"
printf '실행 범위: 전체 포켓(P2Rank) 추출 샤드 %s/%s (A6c-2)\n' "$SLURM_ARRAY_TASK_ID" "$NUM_SHARDS"

exec "$python_bin" -B -u "$script_path" --num-shards "$NUM_SHARDS" --prank "$prank_bin"
