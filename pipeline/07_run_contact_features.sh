#!/bin/bash
#SBATCH --job-name=contact_features
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --chdir=<DATA_ROOT>/Scoring_Function
#SBATCH --output=<DATA_ROOT>/Scoring_Function/logs/contact_features_%j.out
#SBATCH --error=<DATA_ROOT>/Scoring_Function/logs/contact_features_%j.err

# 현재 Slurm의 메모리 등록값을 고려하여 명시적 메모리 요청을 생략합니다.
# 실제 메모리 상한을 보장하는 설정은 아니며 CPU 하나로 순차 처리합니다.
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
script_path="$project_dir/pipeline/07_contact_features.py"

[[ -x "$python_bin" ]] || { printf '%s\n' 'Python 실행 파일을 확인하세요.' >&2; exit 1; }
# 07번은 05→04→03 모듈을 import하므로 네 Python 파일이 모두 있어야 합니다.
for name in 03_select_representative_systems.py 04_audit_selected_structures.py \
            05_audit_input_roles.py 07_contact_features.py; do
    [[ -r "$project_dir/pipeline/$name" ]] || { printf '필요한 파일 없음: %s\n' "$name" >&2; exit 1; }
done

# 기존 가상환경을 그대로 사용하며 바이트코드 파일 생성을 막습니다.
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

printf '작업 ID: %s\n' "$SLURM_JOB_ID"
printf '실행 노드: %s\n' "${SLURMD_NODENAME:-미확인}"
printf '실행 범위: 05번 결과의 eligible pose에 대한 단백질-리간드 접촉·거리 feature(카테고리 1)\n'

# 임시 파일과 캐시는 Python 코드가 허용된 결과 폴더 아래로 설정합니다.
exec "$python_bin" -B -u "$script_path"
