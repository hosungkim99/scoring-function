#!/bin/bash
#SBATCH --job-name=ligand_geometry
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --chdir=<DATA_ROOT>/Scoring_Function
#SBATCH --output=<DATA_ROOT>/Scoring_Function/logs/ligand_geometry_%j.out
#SBATCH --error=<DATA_ROOT>/Scoring_Function/logs/ligand_geometry_%j.err

# 명시적 메모리 요청을 생략합니다. 선정 160개 pose를 CPU 하나로 순차 처리합니다.
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
script_path="$project_dir/pipeline/15_ligand_geometry.py"

[[ -x "$python_bin" ]] || { printf '%s\n' 'Python 실행 파일을 확인하세요.' >&2; exit 1; }
# 15번은 07→05→04→03 모듈을 import하고 06번 결과를 읽습니다.
for name in 03_select_representative_systems.py 04_audit_selected_structures.py \
            05_audit_input_roles.py 07_contact_features.py 15_ligand_geometry.py; do
    [[ -r "$project_dir/pipeline/$name" ]] || { printf '필요한 파일 없음: %s\n' "$name" >&2; exit 1; }
done

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

printf '작업 ID: %s\n' "$SLURM_JOB_ID"
printf '실행 노드: %s\n' "${SLURMD_NODENAME:-미확인}"
printf '실행 범위: 06 대응이 있는 pose의 리간드 내부 구조 타당성(category 4)\n'

exec "$python_bin" -B -u "$script_path"
