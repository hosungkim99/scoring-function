#!/bin/bash
#SBATCH --job-name=build_labels
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --chdir=<DATA_ROOT>/Scoring_Function
#SBATCH --output=<DATA_ROOT>/Scoring_Function/logs/build_labels_%j.out
#SBATCH --error=<DATA_ROOT>/Scoring_Function/logs/build_labels_%j.err

# 전체 460k pose의 sc_rmsd 정답 라벨 테이블을 생성합니다(Process A 라벨 단계).
# 평가 CSV(약 46만 행)의 sc_rmsd를 키로 정리만 하므로 1코어로 수 분이면 끝나는 가벼운 작업입니다.
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
script_path="$project_dir/pipeline/24_build_labels.py"

[[ -x "$python_bin" ]] || { printf '%s\n' 'Python 실행 파일을 확인하세요.' >&2; exit 1; }
# 24번은 02(pose 세계 상수)와 05→04→03(경로 보호·저장) 모듈을 import합니다.
for name in 02_build_pose_index.py 03_select_representative_systems.py \
            04_audit_selected_structures.py 05_audit_input_roles.py 24_build_labels.py; do
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
printf '실행 범위: 전체 460k pose sc_rmsd 정답 라벨 테이블 생성(GT는 정답 라벨로만 사용)\n'

# 임시 파일과 캐시는 Python 코드가 허용된 결과 폴더 아래로 설정합니다.
exec "$python_bin" -B -u "$script_path"
