#!/usr/bin/env python3
"""예측 구조의 식별자와 파일 경로를 연결하는 목록 작성 도구입니다.

기본 실행은 알려진 한 시스템의 100개 후보만 확인합니다.
전체 확인은 --scope full을 명시해야 합니다. CIF 내용과 GT 점수는 읽지 않습니다.
출력·임시 파일·캐시는 지정된 사용자 경로에만 생성합니다.
"""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import time
import uuid

# 작업 루트. 기본은 서버의 <FS_ROOT> 경로이며, 다른 노드/계정에서는 SCORING_FUNCTION_ROOT로 재정의합니다.
ALLOWED_ROOT = Path(os.environ.get('SCORING_FUNCTION_ROOT', '<DATA_ROOT>'))
PROJECT = Path('<SHARED_ROOT>/distillation')
SOURCE_ROOT = Path('<CASP_ROOT>/targets_ligand/distill')
PILOT_SYSTEM = '1aon__1__1.D__1.BA_1.CA'
MODELS = ('af3', 'bt2', 'of3', 'pt2')
EXPECTED = {(m, s, p) for m in MODELS for s in range(1, 6) for p in range(5)}
VERSION = 'pose_index_0.1.1'


def checked_path(path):
    """출력 경로의 범위와 심볼릭 링크를 확인합니다."""
    root = ALLOWED_ROOT
    if not root.is_dir() or root.resolve() != root:
        raise RuntimeError('허용된 사용자 경로가 없거나 실제 경로가 다릅니다.')
    path = Path(os.path.abspath(path))
    path.relative_to(root)
    path.resolve(strict=False).relative_to(root)
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f'출력 경로의 심볼릭 링크를 허용하지 않습니다: {current}')
    return path


def safe_mkdir(path):
    """허용된 경로 안에만 디렉터리를 만듭니다."""
    path = checked_path(path)
    current = ALLOWED_ROOT
    for part in path.relative_to(ALLOWED_ROOT).parts:
        current = checked_path(current / part)
        if not current.exists():
            current.mkdir()
        if not current.is_dir():
            raise RuntimeError(f'디렉터리가 아닙니다: {current}')
    return path


def write_json(path, value):
    """기존 파일을 덮어쓰지 않고 JSON을 저장합니다."""
    with checked_path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def safe_component(value):
    """경로에 포함할 단일 이름을 확인합니다."""
    if not isinstance(value, str) or not value or value in ('.', '..'):
        raise ValueError('빈 이름 또는 잘못된 경로 이름입니다.')
    if any(c in value for c in ('/', '\\', '\x00')):
        raise ValueError('경로 이름에 구분자 또는 널 문자가 있습니다.')
    return value


def snapshot(path):
    """CSV 입력의 크기·수정 시각·해시를 기록합니다."""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        initial = os.fstat(handle.fileno())
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
        final = os.fstat(handle.fileno())
    if (initial.st_size, initial.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
        raise RuntimeError(f'읽는 동안 입력이 변경됐습니다: {path}')
    return {'path': str(path), 'size_bytes': final.st_size,
            'mtime_ns': final.st_mtime_ns, 'sha256': digest.hexdigest()}


def load_systems(path):
    """시스템 목록의 고유성을 검사합니다."""
    systems = []
    seen = set()
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if 'system_id' not in (reader.fieldnames or []):
            raise ValueError('요약 CSV에 system_id 열이 없습니다.')
        for row in reader:
            system = safe_component(row['system_id'])
            if system in seen:
                raise ValueError(f'요약 CSV의 시스템 ID 중복: {system}')
            systems.append(system)
            seen.add(system)
    if not systems:
        raise ValueError('요약 CSV에 시스템이 없습니다.')
    return systems


def load_manifest(root, system):
    """GT를 사용하지 않고 예측 입력 manifest에서 경로 정보를 읽습니다."""
    path = root / system / 'manifest.json'
    raw = path.read_bytes()
    data = json.loads(raw)
    if data.get('system_id') != system:
        raise ValueError('manifest의 system_id가 폴더 이름과 다릅니다.')
    sto = safe_component(data.get('sto'))
    ligand = safe_component(data.get('ligand_ccd'))
    copies = data.get('ligand_copies')
    return {'variant': f'{sto}+{ligand}', 'manifest_path': str(path),
            'manifest_sha256': hashlib.sha256(raw).hexdigest(),
            'ligand_ccd': ligand,
            'manifest_ligand_copies': None if copies is None else str(copies)}


def run(args):
    """후보 목록을 만들고 누락·중복·입력 변경을 확인합니다."""
    # 실수로 전체 작업을 시작하지 않도록 기본 범위는 한 시스템으로 제한합니다.
    started = time.perf_counter()
    sources = [snapshot(args.summary_csv), snapshot(args.evaluation_csv)]
    all_systems = load_systems(args.summary_csv)
    if args.scope == 'pilot' and args.system_id not in all_systems:
        raise ValueError('파일럿 시스템이 요약 CSV에 없습니다.')
    selected = all_systems if args.scope == 'full' else [args.system_id]
    selected_set = set(selected)
    known_set = set(all_systems)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    # 실행 스크립트 이름으로 고정한 폴더를 사용하고 기존 결과가 있으면 중단합니다.
    parent = safe_mkdir(ALLOWED_ROOT / 'Scoring_Function' / 'out' / 'pose_index')
    out = checked_path(parent / f'02_run_pose_index_{args.scope}')
    if out.exists():
        raise FileExistsError(f'기존 결과를 덮어쓰지 않습니다: {out}')
    out.mkdir()
    for name in ('tmp', 'cache'):
        safe_mkdir(out / name)
    for name in ('TMPDIR', 'TEMP', 'TMP'):
        os.environ[name] = str(out / 'tmp')
    os.environ['XDG_CACHE_HOME'] = str(out / 'cache')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([
        ('source_csv_row', pa.int64()), ('system_id', pa.string()),
        ('model', pa.string()), ('seed', pa.int64()), ('sample', pa.int64()),
        ('input_path', pa.string()), ('file_size_bytes', pa.int64()),
        ('file_mtime_ns', pa.int64()), ('manifest_path', pa.string()),
        ('manifest_sha256', pa.string()), ('ligand_ccd', pa.string()),
        ('manifest_ligand_copies', pa.string()),
        ('status', pa.string()), ('error_message', pa.string()),
    ])
    manifests = {}
    per_system = {s: {'rows': 0, 'combos': set(), 'statuses': Counter(), 'duplicates': 0} for s in selected}
    seen = set()
    statuses = Counter()
    issues = []
    pending = []
    total_csv_rows = indexed = 0
    unknown_system_rows = 0
    unknown_examples = []
    print('OUTPUT_DIR:', out, flush=True)

    # 출력은 작은 묶음으로 저장하여 전체 목록을 메모리에 보관하지 않습니다.
    with checked_path(out / 'pose_index.parquet').open('xb') as sink:
        with pq.ParquetWriter(sink, schema, compression='snappy') as writer:
            with args.evaluation_csv.open(newline='', encoding='utf-8-sig') as handle:
                reader = csv.DictReader(handle)
                if not {'system_id', 'model', 'seed', 'sample'} <= set(reader.fieldnames or []):
                    raise ValueError('평가 CSV에 필요한 식별자 열이 없습니다.')
                for line, source in enumerate(reader, start=2):
                    total_csv_rows += 1
                    system = source['system_id']
                    if system not in known_set:
                        unknown_system_rows += 1
                        if len(unknown_examples) < 20:
                            unknown_examples.append({'row': line, 'system_id': system})
                        continue
                    if system not in selected_set:
                        continue
                    row = dict.fromkeys(schema.names)
                    row.update(source_csv_row=line, system_id=system, model=source['model'], status='ok')
                    info = per_system[system]
                    info['rows'] += 1
                    try:
                        seed, sample = int(source['seed']), int(source['sample'])
                        combo = (source['model'], seed, sample)
                        if combo not in EXPECTED:
                            raise ValueError('모델 또는 seed/sample 값이 예상 범위 밖입니다.')
                        row.update(seed=seed, sample=sample)
                        key = (system, *combo)
                        info['combos'].add(combo)
                        if key in seen:
                            row['status'] = 'duplicate_key'
                            row['error_message'] = '동일한 pose 식별자가 CSV에 반복됩니다.'
                            info['duplicates'] += 1
                        seen.add(key)
                    except (ValueError, TypeError) as exc:
                        row['status'], row['error_message'] = 'invalid_key', str(exc)

                    if row['status'] == 'ok':
                        if system not in manifests:
                            try:
                                manifests[system] = (load_manifest(args.source_root, system), None)
                            except Exception as exc:
                                manifests[system] = (None, f'{type(exc).__name__}: {exc}')
                        meta, error = manifests[system]
                        if error:
                            row['status'], row['error_message'] = 'manifest_error', error
                        else:
                            row.update({k: v for k, v in meta.items() if k != 'variant'})
                            path = (args.source_root / system / 'results' / meta['variant'] /
                                    'af3_msa' / row['model'] / f'seed_{row["seed"]}' /
                                    f'sample_{row["sample"]}' / 'model.cif')
                            row['input_path'] = str(path)
                            try:
                                file_info = path.stat()
                                row.update(file_size_bytes=file_info.st_size, file_mtime_ns=file_info.st_mtime_ns)
                                if not stat.S_ISREG(file_info.st_mode):
                                    row['status'] = 'not_regular_file'
                                elif file_info.st_size == 0:
                                    row['status'] = 'empty_file'
                            except FileNotFoundError as exc:
                                row['status'], row['error_message'] = 'missing_file', str(exc)
                            except OSError as exc:
                                row['status'], row['error_message'] = 'stat_error', str(exc)
                    indexed += 1
                    statuses[row['status']] += 1
                    info['statuses'][row['status']] += 1
                    if row['status'] != 'ok' and len(issues) < 100:
                        issues.append(row.copy())
                    pending.append(row)
                    if len(pending) == 2000:
                        writer.write_table(pa.Table.from_pylist(pending, schema=schema))
                        pending.clear()
                        print(f'진행: {indexed}개', flush=True)
                        # 공유 파일시스템에 연속으로 요청을 보내는 빈도를 조금 줄입니다.
                        time.sleep(0.1)
                if pending:
                    writer.write_table(pa.Table.from_pylist(pending, schema=schema))

    summary = []
    for system in selected:
        info = per_system[system]
        missing = EXPECTED - info['combos']
        summary.append({'system_id': system, 'expected_poses': 100,
                        'csv_rows': info['rows'], 'unique_valid_keys': len(info['combos']),
                        'missing_combinations': len(missing), 'duplicate_rows': info['duplicates'],
                        'files_ok': info['statuses']['ok'],
                        'rows_with_issue': info['rows'] - info['statuses']['ok'],
                        'missing_combinations_json': json.dumps(sorted(missing)),
                        'status_counts_json': json.dumps(dict(info['statuses']))})
    with checked_path(out / 'system_summary.csv').open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    stored_rows = pq.ParquetFile(out / 'pose_index.parquet').metadata.num_rows
    if stored_rows != indexed:
        raise RuntimeError('저장된 Parquet 행 수가 처리한 행 수와 다릅니다.')
    changed_inputs = []
    for item in sources:
        current = Path(item['path']).stat()
        if (current.st_size, current.st_mtime_ns) != (item['size_bytes'], item['mtime_ns']):
            changed_inputs.append(item['path'])
    missing_total = sum(s['missing_combinations'] for s in summary)
    expected_rows = len(selected) * 100
    passed = (statuses['ok'] == expected_rows and indexed == expected_rows and
              missing_total == 0 and not changed_inputs and
              (args.scope != 'full' or unknown_system_rows == 0))
    report = {
        'version': VERSION, 'scope': args.scope, 'run_id': run_id, 'output_dir': str(out),
        'python_executable': sys.executable, 'pyarrow_version': pa.__version__,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'sources': sources, 'source_root': str(args.source_root),
        'summary_systems': len(all_systems), 'selected_systems': len(selected),
        'expected_rows': expected_rows, 'evaluation_csv_rows': total_csv_rows,
        'indexed_rows': indexed, 'stored_rows': stored_rows, 'status_counts': dict(statuses),
        'missing_combinations': missing_total,
        'unknown_system_rows': unknown_system_rows, 'unknown_system_examples': unknown_examples,
        'issue_examples': issues, 'changed_csv_inputs': changed_inputs,
        'elapsed_seconds': time.perf_counter() - started, 'validation_passed': passed,
        'gt_values_exported': False, 'cif_contents_read': False,
        'limitations': ['파일의 존재와 크기만으로 CIF 내용의 정상성을 보장하지 않습니다.',
                       '입력 경로 규칙은 기존 평가 코드의 af3_msa 구성을 따릅니다.',
                       '시스템 누락은 system_summary.csv에 기록하며 가상의 pose 행은 만들지 않습니다.'],
    }
    write_json(out / 'run_report.json', report)
    write_json(out / 'RUN_COMPLETE.json', {'outputs_written': True, 'validation_passed': passed,
                                         'indexed_rows': indexed, 'files_ok': statuses['ok']})
    print('STATUS:', 'PASS' if passed else 'CHECK_REQUIRED', flush=True)
    print('ROWS:', indexed, '/', expected_rows, flush=True)
    print('FILE_STATUS:', json.dumps(dict(statuses)), flush=True)
    print('MISSING_COMBINATIONS:', missing_total, flush=True)
    return 0 if passed else 2


def main():
    """실행 인자를 읽고 범위를 지정합니다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=('pilot', 'full'), default='pilot', help='한 시스템 시험 또는 전체 목록')
    parser.add_argument('--system-id', default=PILOT_SYSTEM, help='파일럿에서 확인할 시스템 ID')
    parser.add_argument('--source-root', type=Path, default=SOURCE_ROOT)
    parser.add_argument('--summary-csv', type=Path, default=PROJECT / 'data/common_4603_summary.csv')
    parser.add_argument('--evaluation-csv', type=Path, default=PROJECT / 'eval_results/results_common4603_v3final_qs_filtered.csv')
    args = parser.parse_args()
    for key in ('source_root', 'summary_csv', 'evaluation_csv'):
        setattr(args, key, getattr(args, key).absolute())
    return run(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
