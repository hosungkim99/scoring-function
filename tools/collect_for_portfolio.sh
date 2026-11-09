#!/bin/bash
#SBATCH --job-name=collect_portfolio
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=collect_portfolio_%j.out
#SBATCH --error=collect_portfolio_%j.err
#
# 포트폴리오에 필요한 결과 파일만 새 폴더로 복사한 뒤 tar.gz 로 묶습니다.
#
# 지키는 것
#   - 원본(--src)은 읽기만 합니다. 원본 트리 안에는 아무것도 쓰지 않습니다.
#   - 쓰기는 --allowed-root 안에서만 합니다. 기본값은 --src 의 상위 폴더입니다.
#   - 실행 폴더(collect_<tag>)가 이미 있으면 중단합니다. 덮어쓰지 않습니다.
#   - --limit-mb(기본 50 MiB)를 넘는 파일은 묶지 않고 LARGE_FILES.txt 에
#     경로·크기·행 수만 남깁니다.
#   - 복사한 파일마다 원본과 사본의 md5 를 대조합니다. 묶은 파일의 경로·크기·md5 는
#     MANIFEST.txt 에 남깁니다. md5sum -c 로 검증할 수 있도록 MD5SUMS 도 함께 만듭니다.
#   - MANIFEST 에는 상대경로만 적습니다. 절대경로·호스트명·사용자명은 적지 않습니다.
#
# 사용법 (몇 초면 끝나는 복사 작업입니다)
#   bash   collect_for_portfolio.sh --src <Scoring_Function 폴더> [옵션]
#   sbatch collect_for_portfolio.sh --src <Scoring_Function 폴더> [옵션]
#     sbatch 로그(collect_portfolio_<작업ID>.out)는 sbatch 를 실행한 폴더에 생깁니다.
#     원본 트리 밖, 예를 들어 --dest-root 폴더에서 제출하세요.
#
# 옵션
#   --src DIR           Scoring_Function 루트. 필수. pipeline/ 과 out/ 이 있어야 합니다
#   --dest-root DIR     결과를 둘 곳. 기본값 <src 상위>/Scoring_Function_portfolio_collect
#   --allowed-root DIR  쓰기를 허용할 루트. 기본값 <src 상위>
#   --tag NAME          실행 이름. 기본값은 UTC 시각. 결과는 <dest-root>/collect_<tag>/
#   --limit-mb N        묶을 파일의 크기 상한(MiB). 기본값 50
#   --python PATH       parquet 행 수를 셀 python(pyarrow 필요). 기본값 python3
#
# 이 작업은 메모리를 요청하지 않습니다(--mem 을 주면 이 클러스터에서 거부됩니다).
set -euo pipefail
set -o noclobber
shopt -s nullglob

die() { printf '오류: %s\n' "$*" >&2; exit 1; }

src=''; dest_root=''; allowed_root=''; tag=''; limit_mb=50; py='python3'
while (( $# )); do
    case "$1" in
        --src)          src=${2:?--src 값이 없습니다}; shift 2 ;;
        --dest-root)    dest_root=${2:?--dest-root 값이 없습니다}; shift 2 ;;
        --allowed-root) allowed_root=${2:?--allowed-root 값이 없습니다}; shift 2 ;;
        --tag)          tag=${2:?--tag 값이 없습니다}; shift 2 ;;
        --limit-mb)     limit_mb=${2:?--limit-mb 값이 없습니다}; shift 2 ;;
        --python)       py=${2:?--python 값이 없습니다}; shift 2 ;;
        -h|--help)      sed -n '2,38p' "$0"; exit 0 ;;
        *)              die "알 수 없는 인자: $1" ;;
    esac
done

# ---------------------------------------------------------------- 경로 점검
[[ -n "$src" ]] || die '--src 로 Scoring_Function 폴더를 지정하세요.'
src=$(realpath -e -- "$src") || die "--src 가 없습니다: $src"
[[ -d "$src/pipeline" && -d "$src/out" ]] || die "--src 에 pipeline/ 과 out/ 이 없습니다: $src"
[[ "$limit_mb" =~ ^[0-9]+$ ]] || die "--limit-mb 는 정수여야 합니다: $limit_mb"
[[ "$tag" =~ ^[A-Za-z0-9._-]*$ ]] || die "--tag 에는 영문·숫자·._- 만 쓸 수 있습니다: $tag"

allowed_root=$(realpath -m -- "${allowed_root:-$(dirname -- "$src")}")
dest_root=$(realpath -m -- "${dest_root:-$allowed_root/Scoring_Function_portfolio_collect}")
case "$dest_root/" in "$src/"*) die "결과 폴더가 원본 트리 안입니다: $dest_root" ;; esac
case "$dest_root/" in "$allowed_root/"*) ;; *) die "결과 폴더가 허용 루트($allowed_root) 밖입니다: $dest_root" ;; esac

tag=${tag:-$(date -u +%Y%m%dT%H%M%SZ)}
limit=$(( limit_mb * 1024 * 1024 ))
run_dir="$dest_root/collect_$tag"
bundle="portfolio_data_$tag"
staging="$run_dir/$bundle"
tar_name="$bundle.tar.gz"

mkdir -p -- "$dest_root"
mkdir -- "$run_dir" 2>/dev/null || die "이미 있는 실행 폴더입니다(덮어쓰지 않습니다): $run_dir  -> --tag 를 바꾸세요."
mkdir -- "$staging"

# ---------------------------------------------------------------- 묶을 목록
# 경로는 --src 기준 상대경로입니다. * 는 쓸 수 있고 {a,b} 는 쓸 수 없습니다.
# 50 MiB 를 넘는 파일은 목록에 있어도 묶지 않고 LARGE_FILES.txt 에만 남습니다.
PATTERNS=()
add() { local dir=$1; shift; local f; for f in "$@"; do PATTERNS+=("$dir/$f"); done; }

# [A] 인벤토리 3번 목록
add out/final_model/37_run_freeze_v2 RUN_COMPLETE.json feature_order.json freeze_report.json \
    model.json predict.py provenance.json selfcheck.json spec.json
add out/test_eval/35_run_test_final RUN_COMPLETE.json contribution.csv dev_vs_test.csv \
    head_to_head.csv strata_profile.csv test_report.json
add out/subset_dist/36_run_subsets_first RUN_COMPLETE.json distribution.csv \
    distribution_report.json subset_results.csv transfer.csv
add out/diagnosis/38_run_diagnose_first RUN_COMPLETE.json best_rank.csv diagnosis_report.json \
    diagnosis_summary.csv generator_spread.csv success_curve.csv
add out/diagnosis/38_run_diagnose_second RUN_COMPLETE.json best_rank.csv diagnosis_report.json \
    diagnosis_summary.csv failures.csv generator_spread.csv success_curve.csv
add out/feature_selection/32_run_selection_locked_rep5 RUN_COMPLETE.json chosen_features.json \
    dropped_features.csv feature_importance.csv selection_report.json selection_results.csv \
    single_feature.csv synergy.csv
add out/combination/33_run_combination_search RUN_COMPLETE.json best_by_size.csv \
    combination_report.json combination_search.csv finals.csv leave_one_out.csv \
    ranked_combinations.csv replacement_verdict.json
add out/union/34_run_union_final_full RUN_COMPLETE.json final_feature_set.json \
    member_frequency.csv union_report.json union_results.csv
add out/feature_matrix_all/26_run_integrate_all_final RUN_COMPLETE.json category_coverage.csv \
    feature_catalog.csv integration_report.json status_breakdown.csv \
    unified_all.parquet
add out/data_split/18_run_make_split RUN_COMPLETE.json fold_sizes.csv split_report.json \
    system_split.csv
add out/eval/25_run_evaluate RUN_COMPLETE.json eval_report.json overall_metrics.csv \
    split_metrics.csv

# [B] 추가 — 핵심 수치(전역 AUC 함정, 랭킹 전환, 규칙 A 부호 반전, 라벨 정책)의 출처.
#     pose 단위 점수(scores.parquet)와 시스템 단위 상세표는 넣지 않습니다.
add out/baseline/28_run_baseline_f63 RUN_COMPLETE.json baseline_report.json \
    feature_importance.csv fold_metrics.csv top5_by_stratum.csv
add 'out/ranking/29_run_ranking_*' RUN_COMPLETE.json ranking_report.json \
    feature_importance.csv fold_metrics.csv top5_by_stratum.csv
add 'out/graded/30_run_graded_*' RUN_COMPLETE.json graded_report.json \
    feature_importance.csv fold_metrics.csv top5_by_stratum.csv
add 'out/combined/31_run_combined_*' RUN_COMPLETE.json combined_report.json contribution.csv \
    selection_matrix.csv selection_by_stratum.csv feature_importance.csv fold_metrics.csv
add out/labels/24_run_build_labels RUN_COMPLETE.json labels_report.json missing_label_status.csv

# [C] 위 단계들의 Slurm 로그. 샤드 로그(수백 개)는 넣지 않습니다.
for p in testeval subsets freeze diagnose selection combination union integrate_all \
         make_split evaluate build_labels baseline ranking graded; do
    PATTERNS+=("logs/${p}_[0-9]*")
done
PATTERNS+=('logs/combined_[0-9]*')   # combined_shard_* 와 구분합니다

# ---------------------------------------------------------------- 행 수
count_rows() {
    local f=$1
    case "$f" in
        *.parquet)
            "$py" -I -c 'import sys
import pyarrow.parquet as pq
print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)' "$f" 2>/dev/null \
                || printf '확인 필요(pyarrow 로 읽지 못함)\n' ;;
        *.csv)   printf '%s\n' "$(( $(wc -l < "$f") - 1 ))" ;;
        *.jsonl) wc -l < "$f" | tr -d ' ' ;;
        *)       printf '%s\n' '-' ;;
    esac
}

# ---------------------------------------------------------------- 복사
cd -- "$src"
declare -A seen=()
packed=(); large=(); missing=(); bad=(); skipped=()
total=0

for pat in "${PATTERNS[@]}"; do
    matches=( $pat )
    if (( ${#matches[@]} == 0 )); then missing+=("$pat"); continue; fi
    for rel in "${matches[@]}"; do
        [[ -n "${seen[$rel]:-}" ]] && continue
        seen[$rel]=1
        if [[ -L "$rel" || ! -f "$rel" ]]; then skipped+=("$rel"); continue; fi
        size=$(stat -c %s -- "$rel")
        if (( size > limit )); then
            large+=("$rel"$'\t'"$size"$'\t'"$(count_rows "$rel")")
            continue
        fi
        mkdir -p -- "$staging/$(dirname -- "$rel")"
        [[ ! -e "$staging/$rel" ]] || die "사본 자리에 이미 파일이 있습니다: $staging/$rel"
        cp -p -- "$rel" "$staging/$rel"
        md5_src=$(md5sum -- "$rel" | cut -d' ' -f1)
        md5_dst=$(md5sum -- "$staging/$rel" | cut -d' ' -f1)
        if [[ "$md5_src" != "$md5_dst" ]]; then bad+=("$rel"); continue; fi
        packed+=("$md5_dst"$'\t'"$size"$'\t'"$rel")
        total=$(( total + size ))
    done
done

# ---------------------------------------------------------------- 기록
now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
{
    printf '# Scoring_Function 포트폴리오 수집 MANIFEST\n'
    printf '# 생성(UTC)   %s\n' "$now"
    printf '# 경로        Scoring_Function 루트 기준 상대경로\n'
    printf '# 크기 상한   %s bytes (%s MiB). 넘는 파일은 LARGE_FILES.txt 에만 기록\n' "$limit" "$limit_mb"
    printf '# 묶은 파일   %s개, 합계 %s bytes\n' "${#packed[@]}" "$total"
    printf '# 열          md5<TAB>size_bytes<TAB>path\n'
    for row in "${packed[@]+"${packed[@]}"}"; do printf '%s\n' "$row"; done
} > "$staging/MANIFEST.txt"

for row in "${packed[@]+"${packed[@]}"}"; do
    IFS=$'\t' read -r m _ p <<< "$row"
    printf '%s  %s\n' "$m" "$p"
done > "$staging/MD5SUMS"

{
    printf '# 크기 상한(%s MiB)을 넘어 묶지 않은 파일\n' "$limit_mb"
    printf '# 열          path<TAB>size_bytes<TAB>rows\n'
    printf '# rows        parquet 는 메타데이터의 행 수, csv 는 머리줄을 뺀 줄 수\n'
    for row in "${large[@]+"${large[@]}"}"; do printf '%s\n' "$row"; done
} > "$staging/LARGE_FILES.txt"

if (( ${#missing[@]} + ${#bad[@]} + ${#skipped[@]} )); then
    {
        printf '# 확인 필요\n'
        for x in "${missing[@]+"${missing[@]}"}"; do printf 'MISSING\t%s\n' "$x"; done
        for x in "${bad[@]+"${bad[@]}"}";         do printf 'MD5_MISMATCH\t%s\n' "$x"; done
        for x in "${skipped[@]+"${skipped[@]}"}"; do printf 'NOT_REGULAR_FILE\t%s\n' "$x"; done
    } > "$staging/CHECK.txt"
fi

# ---------------------------------------------------------------- 묶기
cd -- "$run_dir"
[[ ! -e "$tar_name" ]] || die "이미 있는 파일입니다: $run_dir/$tar_name"
tar -czf "$tar_name" "$bundle"
md5sum -- "$tar_name" > "$tar_name.md5"

printf '묶은 파일   %s개, %s bytes\n' "${#packed[@]}" "$total"
printf '큰 파일     %s개 (목록만, LARGE_FILES.txt)\n' "${#large[@]}"
for row in "${large[@]+"${large[@]}"}"; do printf '            %s\n' "$row"; done
printf '없는 패턴   %s개 | md5 불일치 %s개 | 일반 파일 아님 %s개\n' \
       "${#missing[@]}" "${#bad[@]}" "${#skipped[@]}"
for x in "${missing[@]+"${missing[@]}"}"; do printf '            MISSING %s\n' "$x"; done
printf '결과 폴더   %s\n' "$run_dir"
printf '묶음        %s (%s bytes)\n' "$tar_name" "$(stat -c %s -- "$tar_name")"
printf 'md5         %s\n' "$(cut -d' ' -f1 "$tar_name.md5")"

if (( ${#missing[@]} + ${#bad[@]} + ${#skipped[@]} )); then
    printf 'STATUS: CHECK_REQUIRED (%s/CHECK.txt)\n' "$bundle"
    exit 2
fi
printf 'STATUS: COLLECT_COMPLETE\n'
