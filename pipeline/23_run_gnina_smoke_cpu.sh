#!/bin/bash
#SBATCH --job-name=gnina_smoke_cpu
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --chdir=<DATA_ROOT>/Scoring_Function
#SBATCH --output=<DATA_ROOT>/Scoring_Function/logs/gnina_smoke_cpu_%j.out
#SBATCH --error=<DATA_ROOT>/Scoring_Function/logs/gnina_smoke_cpu_%j.err

# gnina smoke test의 CPU 전용 버전. GPU 포화로 23_run_gnina_smoke.sh가 며칠째 대기할 때,
# 파이프라인 검증과 모드 결정(score_only vs local_only)을 먼저 끝내기 위한 우회로다.
#
# GPU 버전과의 차이
#   - --gres 를 요청하지 않는다(즉시 할당). singularity도 --nv 없이 실행한다.
#   - gnina에 --no_gpu 와 --cpu N 을 준다. N은 --cpus-per-task 와 반드시 같게 유지한다
#     (요청 코어 수와 실제 사용 코어 수가 다르면 작업이 중지될 수 있음).
#   - CPU는 느리므로 gnina 호출당 타임아웃을 넉넉히 둔다.
#
# 주의: 23번 파이썬은 출력 폴더가 이미 있으면 덮어쓰지 않고 중단한다.
#       GPU 버전과 같은 폴더(out/gnina_smoke/23_run_gnina_smoke)를 쓰므로,
#       이 작업이 끝나면 아래처럼 이름을 바꿔 GPU 작업이 나중에 쓸 자리를 비워 둘 것.
#         mv out/gnina_smoke/23_run_gnina_smoke out/gnina_smoke/23_run_gnina_smoke_cpu
#
# 먼저 LIMIT=2 로 속도를 재고, 감당 가능하면 LIMIT=20 으로 늘릴 것.
#   LIMIT=2 sbatch pipeline/23_run_gnina_smoke_cpu.sh
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf '%s\n' '이 파일은 sbatch로 제출하세요.' >&2
    exit 1
fi

project_dir='<DATA_ROOT>/Scoring_Function'
python_bin='<SHARED_ROOT>/.micromamba/envs/<CONDA_ENV>/bin/python'
script_path="$project_dir/pipeline/23_gnina_smoke.py"
sif='<SHARED_ROOT>/models/gnina/gnina.sif'

NCPU="${SLURM_CPUS_PER_TASK:-8}"
LIMIT="${LIMIT:-20}"
TIMEOUT="${TIMEOUT:-1800}"
# --nv 없음(GPU 미사용), gnina에 --no_gpu 와 --cpu 를 전달.
gnina_cmd="${GNINA_CMD:-singularity exec --bind <FS_ROOT> $sif gnina --no_gpu --cpu $NCPU}"

[[ -x "$python_bin" ]] || { printf '%s\n' 'Python 실행 파일을 확인하세요.' >&2; exit 1; }
[[ -r "$sif" ]] || { printf '%s\n' 'gnina 컨테이너를 찾을 수 없습니다.' >&2; exit 1; }
for name in 05_audit_input_roles.py 06_map_ligand_atoms.py 23_gnina_smoke.py; do
    [[ -r "$project_dir/pipeline/$name" ]] || { printf '필요한 파일 없음: %s\n' "$name" >&2; exit 1; }
done
command -v singularity >/dev/null || { printf '%s\n' 'singularity를 찾을 수 없습니다.' >&2; exit 1; }

export PYTHONDONTWRITEBYTECODE=1
# 요청한 코어 수에 맞춰 스레드 상한을 둡니다.
export OMP_NUM_THREADS="$NCPU"
export OPENBLAS_NUM_THREADS="$NCPU"
export MKL_NUM_THREADS="$NCPU"
export NUMEXPR_NUM_THREADS="$NCPU"

printf '작업 ID: %s\n' "$SLURM_JOB_ID"
printf '실행 노드: %s\n' "${SLURMD_NODENAME:-미확인}"
printf '실행 범위: gnina smoke test (CPU 전용, score_only vs local_only), limit %s, cpu %s\n' "$LIMIT" "$NCPU"
printf 'gnina 명령: %s\n' "$gnina_cmd"

exec "$python_bin" -B -u "$script_path" \
    --gnina "$gnina_cmd" --limit "$LIMIT" --timeout "$TIMEOUT" "$@"
