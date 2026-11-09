#!/bin/bash
#SBATCH --job-name=combined_extractor
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --chdir=<DATA_ROOT>/Scoring_Function
#SBATCH --output=<DATA_ROOT>/Scoring_Function/logs/combined_extractor_%j.out
#SBATCH --error=<DATA_ROOT>/Scoring_Function/logs/combined_extractor_%j.err

# 명시적 메모리 요청을 생략합니다. 선정 160개 pose를 CIF 1회 읽기로 통합 추출합니다(A6a).
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
script_path="$project_dir/pipeline/19_combined_extractor.py"

[[ -x "$python_bin" ]] || { printf '%s\n' 'Python 실행 파일을 확인하세요.' >&2; exit 1; }
# 19번은 07·11·12·13·15·16(및 05→04→03)을 import하므로 모두 있어야 합니다.
for name in 03_select_representative_systems.py 04_audit_selected_structures.py 05_audit_input_roles.py \
            07_contact_features.py 11_surface_features.py 12_ligand_descriptors.py \
            13_water_metal_features.py 15_ligand_geometry.py 16_interaction_features.py \
            19_combined_extractor.py; do
    [[ -r "$project_dir/pipeline/$name" ]] || { printf '필요한 파일 없음: %s\n' "$name" >&2; exit 1; }
done

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

printf '작업 ID: %s\n' "$SLURM_JOB_ID"
printf '실행 노드: %s\n' "${SLURMD_NODENAME:-미확인}"
printf '실행 범위: 통합 단일패스 추출(cat1·2·3·4·6/7·8) + 표본 재현 self-validation (A6a)\n'

exec "$python_bin" -B -u "$script_path"
