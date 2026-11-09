#!/bin/bash
#SBATCH --job-name=gnina_shard
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array=0-199%16
#SBATCH --chdir=<DATA_ROOT>/Scoring_Function
#SBATCH --output=<DATA_ROOT>/Scoring_Function/logs/gnina_shard_%A_%a.out
#SBATCH --error=<DATA_ROOT>/Scoring_Function/logs/gnina_shard_%A_%a.err

# gnina docking score 전체 추출(③). CPU로 실행하며 GPU를 요청하지 않습니다
# (2026-10-29 확인: CPU 결과가 GPU와 동일, 호출당 약 5초/8코어).
#
# 1) 모드 결정용 표본 — 시스템 20개를 고르게 솎아 두 모드 모두:
#      sbatch --array=0-4 pipeline/27_run_gnina_shard.sh \
#             --max-systems 20 --num-shards 5 --modes score_only,local_only --tag sample
# 2) 본실행 — 전체, 승자 모드만:
#      sbatch pipeline/27_run_gnina_shard.sh --modes score_only --tag full
#
# --num-shards 는 --array 크기와 반드시 같게 맞추세요(기본 200 = array 0-199).
# pose 단위 체크포인트가 있어 중단돼도 끝낸 pose는 건너뜁니다(같은 명령으로 재제출).
set -euo pipefail

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    printf '%s\n' '이 파일은 sbatch(array)로 제출하세요.' >&2
    exit 1
fi

project_dir='<DATA_ROOT>/Scoring_Function'
python_bin='<SHARED_ROOT>/.micromamba/envs/<CONDA_ENV>/bin/python'
script_path="$project_dir/pipeline/27_gnina_shard.py"
sif='<SHARED_ROOT>/models/gnina/gnina.sif'
NCPU="${SLURM_CPUS_PER_TASK:-8}"
# GPU를 쓰려면 GNINA_CMD로 덮어쓰세요(예: singularity exec --nv ... gnina).
gnina_cmd="${GNINA_CMD:-singularity exec --bind <FS_ROOT> $sif gnina --no_gpu --cpu $NCPU}"

[[ -x "$python_bin" ]] || { printf '%s\n' 'Python 실행 파일을 확인하세요.' >&2; exit 1; }
[[ -r "$sif" ]] || { printf '%s\n' 'gnina 컨테이너를 찾을 수 없습니다.' >&2; exit 1; }
command -v singularity >/dev/null || { printf '%s\n' 'singularity를 찾을 수 없습니다.' >&2; exit 1; }
# 27번은 20번(역할·원자 대응)과 23번(gnina 입력·호출)을 import합니다.
for name in 03_select_representative_systems.py 04_audit_selected_structures.py \
            05_audit_input_roles.py 06_map_ligand_atoms.py \
            19_combined_extractor.py 20_combined_extractor_full.py \
            23_gnina_smoke.py 27_gnina_shard.py; do
    [[ -r "$project_dir/pipeline/$name" ]] || { printf '필요한 파일 없음: %s\n' "$name" >&2; exit 1; }
done

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$NCPU"
export OPENBLAS_NUM_THREADS="$NCPU"
export MKL_NUM_THREADS="$NCPU"
export NUMEXPR_NUM_THREADS="$NCPU"

printf '작업 ID: %s 배열 작업: %s\n' "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID"
printf '실행 노드: %s\n' "${SLURMD_NODENAME:-미확인}"
printf '실행 범위: gnina docking score 추출 샤드 %s (CPU %s코어)\n' "$SLURM_ARRAY_TASK_ID" "$NCPU"
printf 'gnina 명령: %s\n' "$gnina_cmd"

exec "$python_bin" -B -u "$script_path" --gnina "$gnina_cmd" "$@"
