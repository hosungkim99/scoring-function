#!/bin/bash
#SBATCH --job-name=gnina_smoke
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:<GPU_TYPE_A>:1
#SBATCH --time=02:00:00
#SBATCH --chdir=<DATA_ROOT>/Scoring_Function
#SBATCH --output=<DATA_ROOT>/Scoring_Function/logs/gnina_smoke_%j.out
#SBATCH --error=<DATA_ROOT>/Scoring_Function/logs/gnina_smoke_%j.err

# gnina docking score smoke test: 소수 pose에서 score_only vs local_only를 모두 추출합니다.
# gnina은 singularity 컨테이너로 실행하며(--nv로 GPU 전달). normal 파티션엔 <GPU_TYPE_A>과
# <GPU_TYPE_B>이 섞여 있는데, 컨테이너 PyTorch가 <GPU_TYPE_B>용 CUDA 커널이 없어
# 거기선 죽습니다(CUDA error: no kernel image). 그래서 --gres=gpu:<GPU_TYPE_A>:1로 GPU 종류를 고정합니다.
# 먼저 LIMIT=2로 dry run 후 out/gnina_smoke/.../raw_samples.json에서 CNNscore 파싱을 확인하고 LIMIT을
# 늘리세요. 다른 gnina 명령은 GNINA_CMD로 덮어씁니다.
set -euo pipefail

project_dir='<DATA_ROOT>/Scoring_Function'
python_bin='<SHARED_ROOT>/.micromamba/envs/<CONDA_ENV>/bin/python'
script_path="$project_dir/pipeline/23_gnina_smoke.py"
gnina_cmd="${GNINA_CMD:-singularity exec --nv --bind <FS_ROOT> <SHARED_ROOT>/models/gnina/gnina.sif gnina}"
LIMIT="${LIMIT:-20}"

[[ -x "$python_bin" ]] || { printf '%s\n' 'Python 실행 파일을 확인하세요.' >&2; exit 1; }
for name in 05_audit_input_roles.py 06_map_ligand_atoms.py 23_gnina_smoke.py; do
    [[ -r "$project_dir/pipeline/$name" ]] || { printf '필요한 파일 없음: %s\n' "$name" >&2; exit 1; }
done

command -v singularity >/dev/null || { printf '%s\n' 'singularity를 찾을 수 없습니다.' >&2; exit 1; }

export PYTHONDONTWRITEBYTECODE=1
# 요청한 코어 수(4)에 맞춰 스레드 상한을 둡니다(코어 수 불일치로 작업이 중지되지 않게).
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

printf '실행 범위: gnina smoke test(score_only vs local_only), limit %s\n' "$LIMIT"
exec "$python_bin" -B -u "$script_path" --gnina "$gnina_cmd" --limit "$LIMIT"
