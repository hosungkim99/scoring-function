#!/bin/bash
#SBATCH --job-name=ranking
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --chdir=<DATA_ROOT>/Scoring_Function
#SBATCH --output=<DATA_ROOT>/Scoring_Function/logs/ranking_%j.out
#SBATCH --error=<DATA_ROOT>/Scoring_Function/logs/ranking_%j.err

# Process B 랭킹 모델(6-2). GPU 불필요, CPU만 씁니다.
#
# 본실행(세 가지 변경 전부 적용 — 이걸 먼저 돌립니다):
#   sbatch pipeline/29_run_ranking.sh --tag rank_z
#
# 절제(본실행에서 효과가 확인된 뒤에만 의미가 있습니다):
#   정규화만        : sbatch pipeline/29_run_ranking.sh --objective binary --tag bin_z
#   랭킹만          : sbatch pipeline/29_run_ranking.sh --normalize none  --tag rank_raw
#   순위 정규화     : sbatch pipeline/29_run_ranking.sh --normalize rank  --tag rank_pct
#   cat8 제외       : sbatch pipeline/29_run_ranking.sh --exclude-categories cat8_ligand --tag rank_z_nocat8
#
# --tag 는 결과 폴더를 분리하므로 반복 실행 시 반드시 다르게 주세요.
# 기본값으로 28_run_baseline_f63 과 층별 비교표를 함께 출력합니다.
# test 923 시스템은 어떤 조합에서도 사용되지 않습니다.
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf '%s\n' '이 파일은 sbatch로 제출하세요.' >&2
    exit 1
fi

project_dir='<DATA_ROOT>/Scoring_Function'
base_python='<SHARED_ROOT>/.micromamba/envs/<CONDA_ENV>/bin/python'
# 28번 러너와 같은 규칙입니다. 프로젝트 전용 venv가 있으면 그걸 쓰고, 없으면 공용 환경으로
# 되돌아갑니다.
venv_python="$project_dir/envs/pb/bin/python"
if [[ -x "$venv_python" ]]; then
    python_bin="$venv_python"
else
    python_bin="$base_python"
fi
script_path="$project_dir/pipeline/29_ranking_model.py"

[[ -x "$python_bin" ]] || { printf '%s\n' 'Python 실행 파일을 확인하세요.' >&2; exit 1; }
[[ -r "$script_path" ]] || { printf '%s\n' '29_ranking_model.py 가 없습니다.' >&2; exit 1; }
# 29번은 28번을 모듈로 불러 평가 코드를 공유합니다.
[[ -r "$project_dir/pipeline/28_baseline_model.py" ]] || {
    printf '%s\n' '28_baseline_model.py 가 없습니다(29번이 import합니다).' >&2; exit 1; }

export PYTHONDONTWRITEBYTECODE=1
ncpu="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$ncpu"
export OPENBLAS_NUM_THREADS="$ncpu"
export MKL_NUM_THREADS="$ncpu"
export NUMEXPR_NUM_THREADS="$ncpu"

printf '작업 ID: %s\n' "$SLURM_JOB_ID"
printf '할당 CPU: %s\n' "$ncpu"
printf 'Python: %s\n' "$python_bin"
printf '실행 범위: Process B 랭킹 모델 학습 + Top-5 평가(dev 전용)\n'

exec "$python_bin" -B -u "$script_path" --threads "$ncpu" "$@"
