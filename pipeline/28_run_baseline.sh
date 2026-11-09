#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --chdir=<DATA_ROOT>/Scoring_Function
#SBATCH --output=<DATA_ROOT>/Scoring_Function/logs/baseline_%j.out
#SBATCH --error=<DATA_ROOT>/Scoring_Function/logs/baseline_%j.err

# Process B 베이스라인(6). GPU가 필요 없고 CPU만 씁니다.
#   기준선(63 feature 전체):
#     sbatch pipeline/28_run_baseline.sh --tag f63
#   gnina 제외(증분 측정):
#     sbatch pipeline/28_run_baseline.sh --exclude-categories cat2_gnina --tag no_gnina
#   P2Rank 제외:
#     sbatch pipeline/28_run_baseline.sh --exclude-categories cat5_pocket --tag no_pocket
#   시스템 수준 상수(cat8) 제외:
#     sbatch pipeline/28_run_baseline.sh --exclude-categories cat8_ligand --tag no_cat8
# --tag 는 결과 폴더를 분리하므로 반복 실행 시 반드시 다르게 주세요.
# test 923 시스템은 어떤 조합에서도 사용되지 않습니다.
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf '%s\n' '이 파일은 sbatch로 제출하세요.' >&2
    exit 1
fi

project_dir='<DATA_ROOT>/Scoring_Function'
base_python='<SHARED_ROOT>/.micromamba/envs/<CONDA_ENV>/bin/python'
# 부스팅 라이브러리는 공용 <CONDA_ENV> 환경이 아니라 프로젝트 안의 전용 venv에 설치합니다.
# 공용 경로(.micromamba/envs)를 수정하지 않기 위한 조치입니다. venv가 없으면 공용 환경으로
# 되돌아가지만, 그 경우 xgboost/sklearn이 없어 import 단계에서 멈춥니다.
venv_python="$project_dir/envs/pb/bin/python"
if [[ -x "$venv_python" ]]; then
    python_bin="$venv_python"
else
    python_bin="$base_python"
fi
script_path="$project_dir/pipeline/28_baseline_model.py"

[[ -x "$python_bin" ]] || { printf '%s\n' 'Python 실행 파일을 확인하세요.' >&2; exit 1; }
[[ -r "$script_path" ]] || { printf '%s\n' '28_baseline_model.py 가 없습니다.' >&2; exit 1; }

export PYTHONDONTWRITEBYTECODE=1
ncpu="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$ncpu"
export OPENBLAS_NUM_THREADS="$ncpu"
export MKL_NUM_THREADS="$ncpu"
export NUMEXPR_NUM_THREADS="$ncpu"

printf '작업 ID: %s\n' "$SLURM_JOB_ID"
printf '할당 CPU: %s\n' "$ncpu"
printf '실행 범위: Process B 베이스라인 학습 + Top-5 평가(dev 전용)\n'

printf 'Python: %s
' "$python_bin"

exec "$python_bin" -B -u "$script_path" --threads "$ncpu" "$@"
