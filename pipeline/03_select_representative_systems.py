#!/usr/bin/env python3
"""입력 처리 검증에 사용할 다양한 시스템과 네 모델의 예시 pose를 고릅니다.

전체 pose 목록 검증을 통과한 실행 결과만 입력으로 받습니다.
GT 평가 점수는 사용하지 않으며 구조 파일을 열거나 수정하지 않습니다.
선정 결과는 통계적 대표 표본이나 학습·검증·테스트 분할이 아닙니다.
"""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

# 작업 루트. 기본은 서버의 <FS_ROOT> 경로이며, 다른 노드/계정에서는 SCORING_FUNCTION_ROOT로 재정의합니다.
ALLOWED_ROOT = Path(os.environ.get('SCORING_FUNCTION_ROOT', '<DATA_ROOT>'))
# 이미 검증을 통과한 기존 전체 목록을 이동하거나 재계산하지 않고 읽습니다.
DEFAULT_INDEX = ALLOWED_ROOT / 'Scoring_Function/out/pose_index/20261019T054512Z_74cd3e83'
DEFAULT_SUMMARY = Path('<SHARED_ROOT>/distillation/data/common_4603_summary.csv')
KNOWN_SYSTEM = '1aon__1__1.D__1.BA_1.CA'
MODELS = ('af3', 'bt2', 'of3', 'pt2')
COMBOS = [(m, s, p) for m in MODELS for s in range(1, 6) for p in range(5)]
COMBO_INDEX = {key: i for i, key in enumerate(COMBOS)}
VERSION = 'representative_selection_0.1.0'
SUMMARY_FIELDS = ['system_id', 'ligand_ccd_code', 'ligand_rdkit_canonical_smiles',
                  'ligand_class', 'ligand_molecular_weight',
                  'ligand_num_rot_bonds', 'ligand_num_heavy_atoms']
METRICS = ['ligand_num_heavy_atoms', 'ligand_molecular_weight', 'ligand_num_rot_bonds']
BINS = {'ligand_num_heavy_atoms': [15, 30, 50],
        'ligand_molecular_weight': [250, 500, 750],
        'ligand_num_rot_bonds': [2, 5, 10],
        'manifest_ligand_copies': [0, 1, 2, 4]}


def checked_path(path):
    """출력 경로가 사용자 폴더 안에 있고 심볼릭 링크가 없는지 확인합니다."""
    if not ALLOWED_ROOT.is_dir() or ALLOWED_ROOT.resolve() != ALLOWED_ROOT:
        raise RuntimeError('허용된 사용자 경로가 없거나 실제 경로가 다릅니다.')
    path = Path(os.path.abspath(path))
    path.relative_to(ALLOWED_ROOT)
    path.resolve(strict=False).relative_to(ALLOWED_ROOT)
    current = ALLOWED_ROOT
    for part in path.relative_to(ALLOWED_ROOT).parts:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f'출력 경로에 심볼릭 링크가 있습니다: {current}')
    return path


def safe_mkdir(path):
    """출력 디렉터리를 허용된 범위에만 만듭니다."""
    path = checked_path(path)
    current = ALLOWED_ROOT
    for part in path.relative_to(ALLOWED_ROOT).parts:
        current = checked_path(current / part)
        if not current.exists():
            current.mkdir()
        if not current.is_dir():
            raise RuntimeError(f'디렉터리가 아닙니다: {current}')
    return path


def write_json(path, data):
    """기존 파일을 덮어쓰지 않고 JSON을 저장합니다."""
    with checked_path(path).open('x', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def write_csv(path, rows):
    """표를 UTF-8 CSV로 저장합니다."""
    if not rows:
        raise ValueError('저장할 표가 비어 있습니다.')
    with checked_path(path).open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fingerprint(path):
    """입력 파일의 경로·크기·수정 시각·해시를 기록합니다."""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        before = os.fstat(handle.fileno())
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'읽는 동안 입력이 변경됐습니다: {path}')
    return {'path': str(path), 'size_bytes': after.st_size,
            'mtime_ns': after.st_mtime_ns, 'sha256': digest.hexdigest()}


def numeric(value, integer=False):
    """결측·잘못된 값·음수를 구분하고 유효한 수를 반환합니다."""
    if value is None or str(value).strip().lower() in ('', 'nan', 'none', 'null', 'na'):
        return None, 'missing'
    try:
        value = float(value)
        if not math.isfinite(value) or value < 0 or (integer and not value.is_integer()):
            return None, 'invalid'
        return value, 'valid'
    except (TypeError, ValueError):
        return None, 'invalid'


def range_tag(field, raw):
    """명시한 구간 기준으로 선정용 태그를 만듭니다."""
    value, state = numeric(raw, integer=field != 'ligand_molecular_weight')
    if state != 'valid':
        return f'{field}:{state}'
    lower = None
    for upper in BINS[field]:
        if value <= upper:
            label = f'le_{upper}' if lower is None else f'gt_{lower}_le_{upper}'
            return f'{field}:{label}'
        lower = upper
    return f'{field}:gt_{lower}'


def tie_key(system, seed):
    """행 순서와 Python 버전에 의존하지 않는 동점 정렬 키를 만듭니다."""
    return hashlib.sha256(f'{seed}|{system}'.encode()).hexdigest()


def choose_systems(candidates, count, seed):
    """기존 예시·극단값·미포함 유형을 우선하고 나머지는 고정 해시 순서로 채웁니다."""
    selected, reasons, covered = [], {}, set()
    frequencies = Counter(tag for row in candidates.values() for tag in row['_tags'])

    def add(system, reason):
        if system not in reasons:
            selected.append(system)
            reasons[system] = []
            covered.update(candidates[system]['_tags'])
        if reason not in reasons[system]:
            reasons[system].append(reason)

    if KNOWN_SYSTEM in candidates:
        add(KNOWN_SYSTEM, '기존 네 모델 파일럿과 비교하기 위한 기준 시스템')
    for field in METRICS + ['manifest_ligand_copies']:
        values = [(system, numeric(row.get(field), integer=field != 'ligand_molecular_weight')[0])
                  for system, row in candidates.items()]
        values = [(system, value) for system, value in values if value is not None]
        if not values:
            continue
        for side in ('min', 'max'):
            extreme = (min if side == 'min' else max)(value for _, value in values)
            matches = [system for system, value in values if value == extreme]
            # 동일 극단값을 이미 포함했다면 그 시스템에 선정 이유를 추가합니다.
            existing = [system for system in selected if system in matches]
            system = existing[0] if existing else min(matches, key=lambda s: tie_key(s, seed))
            if system in reasons or len(selected) < count:
                add(system, f'{field}의 {side} 값: {extreme:g}')
    while len(selected) < count:
        remaining = [system for system in candidates if system not in reasons]
        if not remaining:
            break
        scores = {system: sum(1 / math.sqrt(frequencies[tag])
                              for tag in sorted(candidates[system]['_tags'] - covered))
                  for system in remaining}
        system = min(remaining, key=lambda s: (-scores[s], tie_key(s, seed)))
        new_tags = sorted(candidates[system]['_tags'] - covered)
        reason = '새 유형 포함: ' + ', '.join(new_tags) if new_tags else '고정 해시 순서로 추가 선정'
        add(system, reason)
    return selected, reasons, frequencies, covered


def read_candidates(summary_path, index_path):
    """요약 정보와 전체 pose 목록을 연결하고 시스템별 100개 조합을 확인합니다."""
    import pyarrow.parquet as pq
    candidates = {}
    with summary_path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not set(SUMMARY_FIELDS) <= set(reader.fieldnames or []):
            raise ValueError('요약 CSV에 선정에 필요한 열이 없습니다.')
        for row in reader:
            system = row['system_id']
            if not system or system in candidates:
                raise ValueError('요약 CSV에 빈 시스템 ID 또는 중복이 있습니다.')
            candidates[system] = {field: row[field] for field in SUMMARY_FIELDS}
    if not candidates:
        raise ValueError('요약 CSV가 비어 있습니다.')
    masks = {system: 0 for system in candidates}
    anchors = {}
    meta_by_system = {}
    columns = ['system_id', 'model', 'seed', 'sample', 'input_path', 'status',
               'file_size_bytes', 'file_mtime_ns', 'manifest_path', 'manifest_sha256',
               'ligand_ccd', 'manifest_ligand_copies']
    total = 0
    for batch in pq.ParquetFile(index_path).iter_batches(batch_size=10000, columns=columns, use_threads=False):
        for row in batch.to_pylist():
            total += 1
            system = row['system_id']
            combo = (row['model'], row['seed'], row['sample'])
            if system not in candidates or row['status'] != 'ok' or combo not in COMBO_INDEX:
                raise ValueError('목록에 알 수 없는 시스템·조합 또는 정상 외 상태가 있습니다.')
            bit = 1 << COMBO_INDEX[combo]
            if masks[system] & bit:
                raise ValueError(f'중복 pose 식별자: {system}, {combo}')
            masks[system] |= bit
            meta = (row['manifest_path'], row['manifest_sha256'], row['ligand_ccd'], row['manifest_ligand_copies'])
            if system in meta_by_system and meta_by_system[system] != meta:
                raise ValueError(f'시스템 내 manifest 정보가 일관되지 않습니다: {system}')
            meta_by_system[system] = meta
            if row['seed'] == 1 and row['sample'] == 0:
                anchors[(system, row['model'])] = row
        if total % 100000 == 0:
            print(f'목록 확인: {total}개', flush=True)
    full_mask = (1 << len(COMBOS)) - 1
    if any(mask != full_mask for mask in masks.values()):
        raise ValueError('100개 조합이 완성되지 않은 시스템이 있습니다.')
    for system, row in candidates.items():
        manifest_path, manifest_hash, ligand, copies = meta_by_system[system]
        row.update(manifest_path=manifest_path, manifest_sha256=manifest_hash,
                   manifest_ligand_ccd=ligand, manifest_ligand_copies=copies)
        ligand_class = (row['ligand_class'] or '').strip() or 'missing'
        row['_tags'] = {f'ligand_class:{ligand_class}'}
        row['_tags'].update(range_tag(field, row.get(field)) for field in BINS)
    return candidates, anchors, total


def run(args):
    """선정 결과와 선정 이유 및 데이터 범위의 차이를 기록합니다."""
    started = time.perf_counter()
    if args.n_systems < 10:
        raise ValueError('극단값을 포함하려면 최소 10개 시스템을 지정하세요.')
    report_path = args.index_run / 'run_report.json'
    complete_path = args.index_run / 'RUN_COMPLETE.json'
    index_path = args.index_run / 'pose_index.parquet'
    sources = [fingerprint(path) for path in (report_path, complete_path, index_path, args.summary_csv)]
    previous = json.loads(report_path.read_text())
    complete = json.loads(complete_path.read_text())
    if previous.get('scope') != 'full' or previous.get('validation_passed') is not True:
        raise ValueError('전체 목록 검증을 통과한 실행만 사용할 수 있습니다.')
    if complete.get('outputs_written') is not True or complete.get('validation_passed') is not True:
        raise ValueError('전체 목록의 완료 기록이 정상적이지 않습니다.')
    old_summary = [entry for entry in previous.get('sources', [])
                   if Path(entry['path']).absolute() == args.summary_csv.absolute()]
    if len(old_summary) != 1 or old_summary[0]['sha256'] != sources[-1]['sha256']:
        raise ValueError('요약 CSV가 전체 목록을 만들 때 사용한 파일과 다릅니다.')

    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    # 실행 스크립트 이름으로 고정한 폴더를 사용하고 기존 결과가 있으면 중단합니다.
    parent = safe_mkdir(ALLOWED_ROOT / 'Scoring_Function/out/representative_systems')
    out = checked_path(parent / '03_run_select_representative_systems')
    if out.exists():
        raise FileExistsError(f'기존 결과를 덮어쓰지 않습니다: {out}')
    out.mkdir()
    for name in ('tmp', 'cache'):
        safe_mkdir(out / name)
    for key in ('TMPDIR', 'TEMP', 'TMP'):
        os.environ[key] = str(out / 'tmp')
    os.environ['XDG_CACHE_HOME'] = str(out / 'cache')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    print('OUTPUT_DIR:', out, flush=True)
    candidates, anchors, total = read_candidates(args.summary_csv, index_path)
    if total != previous.get('indexed_rows') or total != complete.get('indexed_rows'):
        raise ValueError('전체 목록의 실제 행 수와 실행 기록이 다릅니다.')
    if args.n_systems > len(candidates):
        raise ValueError('요청한 시스템 수가 전체 시스템 수보다 많습니다.')
    chosen, reasons, frequencies, covered = choose_systems(candidates, args.n_systems, args.selection_seed)
    systems = []
    poses = []
    for rank, system in enumerate(chosen, start=1):
        row = {key: value for key, value in candidates[system].items() if key != '_tags'}
        row.update(selection_order=rank,
                   selection_reasons_json=json.dumps(reasons[system], ensure_ascii=False),
                   selection_tags_json=json.dumps(sorted(candidates[system]['_tags']), ensure_ascii=False))
        systems.append(row)
        for model in MODELS:
            pose = anchors[(system, model)].copy()
            pose.update(selection_order=rank, selection_reasons_json=row['selection_reasons_json'])
            poses.append(pose)
    if len(systems) != args.n_systems or len(poses) != args.n_systems * 4:
        raise RuntimeError('선정 결과의 시스템 또는 pose 수가 예상과 다릅니다.')
    selected_counts = Counter(tag for system in chosen for tag in candidates[system]['_tags'])
    coverage = [{'tag': tag, 'all_systems': count, 'selected_systems': selected_counts[tag],
                 'all_fraction': count / len(candidates),
                 'selected_fraction': selected_counts[tag] / len(chosen)}
                for tag, count in sorted(frequencies.items())]
    numeric_ranges = {}
    for field in METRICS + ['manifest_ligand_copies']:
        numeric_ranges[field] = {}
        for label, subset in [('all', candidates.keys()), ('selected', chosen)]:
            parsed = [numeric(candidates[system].get(field), integer=field != 'ligand_molecular_weight') for system in subset]
            values = [value for value, state in parsed if state == 'valid']
            numeric_ranges[field][label] = {'min': min(values) if values else None,
                                            'max': max(values) if values else None,
                                            'states': dict(Counter(state for _, state in parsed))}
    write_csv(out / 'selected_systems.csv', systems)
    write_csv(out / 'coverage_summary.csv', coverage)
    table = pa.Table.from_pylist(poses)
    with checked_path(out / 'selected_poses.parquet').open('xb') as handle:
        pq.write_table(table, handle, compression='snappy')
    restored = pq.read_table(out / 'selected_poses.parquet', use_threads=False)
    if not table.equals(restored):
        raise RuntimeError('저장 전후 선정 pose 표가 일치하지 않습니다.')
    for entry in sources:
        current = Path(entry['path']).stat()
        if (current.st_size, current.st_mtime_ns) != (entry['size_bytes'], entry['mtime_ns']):
            raise RuntimeError(f'실행 중 입력 파일이 변경됐습니다: {entry["path"]}')
    uncovered = sorted(set(frequencies) - covered)
    report = {
        'version': VERSION, 'run_id': run_id, 'output_dir': str(out), 'sources': sources,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'python_executable': sys.executable, 'pyarrow_version': pa.__version__,
        'selection_seed': args.selection_seed, 'n_systems': len(chosen), 'n_poses': len(poses),
        'all_systems': len(candidates), 'indexed_poses': total,
        'anchor_seed': 1, 'anchor_sample': 0, 'models': list(MODELS),
        'selection_algorithm': '기준 시스템과 수치 최소·최대값 우선, 새 태그에 빈도의 역제곱근 가중치를 주어 탐욕 선정, 나머지는 고정 해시 순서로 채움',
        'tag_bins_inclusive_upper': BINS, 'uncovered_tags': uncovered,
        'numeric_ranges': numeric_ranges, 'validation_passed': True,
        'gt_scores_used': False, 'cif_contents_read': False,
        'elapsed_seconds': time.perf_counter() - started,
        'limitations': ['공통 입력 처리 검증용 표본이며 통계적 대표 표본 또는 데이터 분할이 아닙니다.',
                       '구간 경계는 입력 유형을 다양하게 선택하기 위한 임의의 공학적 기준입니다.',
                       '복사본 수는 manifest의 입력 값이며 실제 예측 구조의 리간드 수가 아닙니다.',
                       '물·금속·공유결합 리간드·특수 잔기 등의 존재 여부는 이 단계에서 알 수 없습니다.',
                       '모델별 다른 seed/sample의 구조 변화는 이 160개 기준 구조에 포함되지 않습니다.',
                       'CIF 내용 점검에서 새로운 유형을 발견하면 검증 표본을 확장해야 합니다.'],
    }
    write_json(out / 'selection_report.json', report)
    write_json(out / 'RUN_COMPLETE.json', {'outputs_written': True, 'validation_passed': True,
                                         'selected_systems': len(chosen), 'selected_poses': len(poses),
                                         'uncovered_tag_count': len(uncovered)})
    print('STATUS: PASS', flush=True)
    print('SELECTED_SYSTEMS:', len(chosen), flush=True)
    print('SELECTED_POSES:', len(poses), flush=True)
    print('UNCOVERED_TAGS:', len(uncovered), flush=True)
    if uncovered:
        print('미포함 유형:', json.dumps(uncovered, ensure_ascii=False), flush=True)


def main():
    """입력 목록과 선정 개수를 지정합니다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index-run', type=Path, default=DEFAULT_INDEX, help='전체 목록 검증을 통과한 결과 폴더')
    parser.add_argument('--summary-csv', type=Path, default=DEFAULT_SUMMARY, help='시스템 요약 CSV')
    parser.add_argument('--n-systems', type=int, default=40, help='선정할 시스템 수')
    parser.add_argument('--selection-seed', type=int, default=20261019, help='동점 정렬에 사용할 고정 값')
    args = parser.parse_args()
    args.index_run = args.index_run.absolute()
    args.summary_csv = args.summary_csv.absolute()
    run(args)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
