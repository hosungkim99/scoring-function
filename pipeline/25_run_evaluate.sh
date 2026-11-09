#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --chdir=<DATA_ROOT>/Scoring_Function
#SBATCH --output=<DATA_ROOT>/Scoring_Function/logs/evaluate_%j.out
#SBATCH --error=<DATA_ROOT>/Scoring_Function/logs/evaluate_%j.err

# 타깃별 Top-5 성공률 평가 하네스(B①). 24 라벨 테이블을 입력으로 baseline(oracle·random)과
# 함께 스코어링을 채점합니다. 인자 없이 제출하면 baseline만 산출(24 값 재현 자체검증).
# 스코어링 평가: sbatch pipeline/25_run_evaluate.sh --scores <점수.parquet> --score-col <열> --direction <asc|desc> --tag <이름>
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf '%s\n' '이 파일은 sbatch로 제출하세요.' >&2
    exit 1
fi

project_dir='<DATA_ROOT>/Scoring_Function'
python_bin='<SHARED_ROOT>/.micromamba/envs/<CONDA_ENV>/bin/python'
script_path="$project_dir/pipeline/25_evaluate.py"

[[ -x "$python_bin" ]] || { printf '%s\n' 'Python 실행 파일을 확인하세요.' >&2; exit 1; }
# 25번은 05→04→03(경로 보호·저장) 모듈을 import합니다.
for name in 03_select_representative_systems.py 04_audit_selected_structures.py \
            05_audit_input_roles.py 25_evaluate.py; do
    [[ -r "$project_dir/pipeline/$name" ]] || { printf '필요한 파일 없음: %s\n' "$name" >&2; exit 1; }
done

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

printf '작업 ID: %s\n' "$SLURM_JOB_ID"
printf '실행 노드: %s\n' "${SLURMD_NODENAME:-미확인}"
printf '실행 범위: 타깃별 Top-5 성공률 평가(GT는 채점 라벨로만 사용)\n'

# 추가 인자(--scores/--score-col/--direction/--tag 등)를 그대로 전달합니다.
exec "$python_bin" -B -u "$script_path" "$@"
