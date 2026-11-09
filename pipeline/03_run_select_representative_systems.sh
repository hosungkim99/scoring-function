#!/bin/bash
#SBATCH --job-name=select_representative_systems
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --chdir=<DATA_ROOT>/Scoring_Function
#SBATCH --output=<DATA_ROOT>/Scoring_Function/logs/select_representative_systems_%j.out
#SBATCH --error=<DATA_ROOT>/Scoring_Function/logs/select_representative_systems_%j.err

# 노드 메모리가 Slurm에 1 MiB로 등록되어 있으므로 명시적인 메모리 요청을 생략합니다.
# 이 설정은 실제 메모리 사용량의 상한을 보장하지 않습니다.
# 전체 목록에서 입력 유형이 다양한 40개 시스템을 선정하며 GPU는 요청하지 않습니다.
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf '%s\n' '이 파일은 bash로 직접 실행하지 말고 sbatch로 제출하세요.' >&2
    exit 1
fi
if (( $# != 0 )); then
    printf '%s\n' '이 파일은 인자를 받지 않는 대표 시스템 선정용입니다.' >&2
    exit 1
fi

# 공유 가상환경은 변경하지 않고 기존 Python 실행 파일만 사용합니다.
project_dir='<DATA_ROOT>/Scoring_Function'
python_bin='<SHARED_ROOT>/.micromamba/envs/<CONDA_ENV>/bin/python'
script_path="$project_dir/pipeline/03_select_representative_systems.py"

[[ -x "$python_bin" ]] || { printf '%s\n' 'Python 실행 파일을 확인하세요.' >&2; exit 1; }
[[ -r "$script_path" ]] || { printf '%s\n' '대표 시스템 선정 Python 파일을 확인하세요.' >&2; exit 1; }

# 바이트코드 생성을 막고 수치 라이브러리의 스레드를 하나로 제한합니다.
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

printf '작업 ID: %s\n' "$SLURM_JOB_ID"
printf '실행 노드: %s\n' "${SLURMD_NODENAME:-미확인}"
printf '실행 범위: 40개 시스템 선정 및 160개 pose 연결\n'

# 임시 파일과 캐시 경로는 Python 코드가 해당 실행의 out 폴더 아래로 설정합니다.
# exec를 사용하여 Python의 종료 코드가 작업 종료 코드에 반영되도록 합니다.
exec "$python_bin" -B -u "$script_path" --n-systems 40
