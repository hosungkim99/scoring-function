#!/usr/bin/env python3
"""전체 460k pose 통합 추출(A6c). 시스템을 샤드로 나눠 병렬 추출하고 재개·병합합니다.

A6b에서 검증된 자립형 엔진(20번)의 per-pose 함수를 그대로 재사용합니다. 기본 모드는 한 샤드의
시스템만 처리해 shard_NNN.parquet을 저장하며, 이미 있으면 건너뜁니다(resume). --merge 모드는 모든
샤드를 합쳐 전체 feature 표와 완결성 리포트를 만듭니다. category 5(P2Rank)는 비용이 커서 별도로
돌립니다. GT·좌표 수정 없음.

샤드 인덱스는 SLURM_ARRAY_TASK_ID 또는 --shard-index로 받습니다.
출력: out/combined_all/21_run_combined_shard/ (shards/ 아래 샤드, 병합본은 같은 폴더).
"""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import time


def _load(name):
    p = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(name.replace('.py', ''), p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

full = _load('20_combined_extractor_full.py')   # 자립형 엔진 재사용
ce = full.ce
cf1 = full.cf1
sf = full.sf
ld = full.ld
roles = full.roles
audit = full.audit
helper = full.helper

VERSION = 'combined_all_0.1.0'
OUT_KIND = 'combined_all'
OUT_NAME = '21_run_combined_shard'
BASE = helper.ALLOWED_ROOT / 'Scoring_Function' / 'out'
DEFAULT_INDEX = helper.DEFAULT_INDEX
DEFAULT_ROLES = BASE / 'input_roles' / '05_run_audit_input_roles'
KEYS = ('system_id', 'model', 'seed', 'sample')


def ensure_dir(path):
    """여러 array 작업이 동시에 만들어도 충돌하지 않게 폴더를 보장합니다(경쟁 안전)."""
    p = helper.checked_path(path)
    p.mkdir(parents=True, exist_ok=True)
    if not p.is_dir():
        raise RuntimeError(f'디렉터리가 아닙니다: {p}')
    return p


def out_dir():
    return ensure_dir(BASE / OUT_KIND / OUT_NAME)


def load_index_systems(index_run):
    """02번 전체 목록에서 ok pose와 시스템 목록을 읽습니다."""
    import pyarrow.parquet as pq
    cols = [*KEYS, 'input_path', 'file_size_bytes', 'file_mtime_ns', 'manifest_path', 'manifest_sha256', 'status']
    by_system = {}
    for batch in pq.ParquetFile(index_run / 'pose_index.parquet').iter_batches(batch_size=20000, columns=cols, use_threads=False):
        for r in batch.to_pylist():
            if r['status'] == 'ok':
                by_system.setdefault(r['system_id'], []).append(r)
    return by_system


def extract_pose(pose, entities, ligand_entity_row, unit_points, max_bytes):
    """20번과 동일한 per-pose 추출(역할·원자 대응 + cat1·2·3·4·6/7·8)."""
    system = pose['system_id']
    feature_names = []
    for _c, _p, f in ce.VALIDATE:
        feature_names += f
    row = {k: pose[k] for k in KEYS}
    row.update({'pose_id': f"{system}|{pose['model']}|{pose['seed']}|{pose['sample']}",
                'schema_version': VERSION, 'cif_sha256': None, 'ligand_typing': None,
                'cat4_available': None, 'ligand_mapping_status': None,
                'status': None, 'skip_reasons_json': '[]', 'error_message': None})
    row.update(dict.fromkeys(feature_names))
    smiles = (ligand_entity_row or {}).get('canonical_smiles') or (ligand_entity_row or {}).get('smiles')
    try:
        if smiles:
            try:
                row.update(ld.descriptors((smiles or '').strip()))
            except Exception:
                pass
        if isinstance(entities, tuple) and entities and entities[0] == 'error':
            raise ValueError(f'시스템 입력 점검 실패: {entities[1]}')
        rec_rows, lig_rows, mapping_entry, reasons, cif_sha = full.map_pose(
            pose['input_path'], pose.get('file_size_bytes'), pose.get('file_mtime_ns'),
            entities, ligand_entity_row, max_bytes)
        row['cif_sha256'] = cif_sha
        if reasons:
            row['status'] = 'skipped'
            row['skip_reasons_json'] = audit.encoded(reasons)
        else:
            row['ligand_mapping_status'] = mapping_entry['ligand_status']
            atoms, _ = cf1.read_atoms(pose['input_path'], cif_sha, max_bytes)
            rec_xyz, rec_res = cf1.pick_coords(atoms, rec_rows)
            lig_xyz, _ = cf1.pick_coords(atoms, lig_rows)
            row.update(cf1.contacts(lig_xyz, rec_xyz, rec_res))
            row.update(ce.compute_cat3(atoms, lig_rows, rec_rows, unit_points))
            c2, typing = ce.compute_cat2(atoms, lig_rows, rec_rows, mapping_entry, smiles)
            row.update(c2); row['ligand_typing'] = typing
            row.update(ce.compute_cat67(atoms, lig_rows, rec_rows))
            c4 = ce.compute_cat4(atoms, mapping_entry, smiles)
            row['cat4_available'] = c4 is not None
            if c4:
                row.update(c4)
            row['status'] = 'computed'
    except Exception as exc:
        row.update(status='error', error_message=f'{type(exc).__name__}: {exc}')
    return row


def run_shard(args):
    """한 샤드의 시스템만 처리해 shard parquet을 저장합니다."""
    import pandas as pd
    out = out_dir()
    shards = ensure_dir(out / 'shards')
    ensure_dir(out / 'tmp'); ensure_dir(out / 'cache')
    for key in ('TMPDIR', 'TMP', 'TEMP'):
        os.environ[key] = str(out / 'tmp')
    os.environ['XDG_CACHE_HOME'] = str(out / 'cache')
    shard_path = shards / f'shard_{args.shard_index:04d}.parquet'
    done_marker = shards / f'shard_{args.shard_index:04d}.done'
    if done_marker.exists() and shard_path.exists():
        print(f'SHARD {args.shard_index}: 이미 완료되어 건너뜁니다(resume).', flush=True)
        return 0
    # 완료 표식 없이 남은 불완전 샤드(중단된 작업의 잔여)는 제거 후 다시 만듭니다.
    if shard_path.exists() and not done_marker.exists():
        helper.checked_path(shard_path).unlink()
        print(f'SHARD {args.shard_index}: 불완전 샤드 제거 후 재생성합니다.', flush=True)
    started = time.perf_counter()
    unit_points = sf.fibonacci_sphere(sf.SPHERE_POINTS)
    max_bytes = args.max_cif_mib * 1024 ** 2

    by_system = load_index_systems(args.index_run)
    all_systems = sorted(by_system)
    my_systems = [s for i, s in enumerate(all_systems) if i % args.num_shards == args.shard_index]
    # 리간드 SMILES(05 input_entities, 전체 시스템 포함).
    ligand_entity = {}
    with (args.roles_run / 'input_entities.csv').open(newline='', encoding='utf-8') as handle:
        for r in csv.DictReader(handle):
            if r.get('input_kind') == 'ligand':
                ligand_entity.setdefault(r['system_id'], r)
    print(f'SHARD {args.shard_index}/{args.num_shards}: 시스템 {len(my_systems)}개', flush=True)

    rows, status_counter = [], Counter()
    entity_cache = {}
    done = 0
    for system in my_systems:
        try:
            meta_src = by_system[system][0]
            meta = {'manifest_path': meta_src['manifest_path'], 'manifest_sha256': meta_src['manifest_sha256']}
            _r, ents, _d = roles.inspect_input(system, meta, max_bytes, 30)
            entity_cache[system] = ents
        except Exception as exc:
            entity_cache[system] = ('error', f'{type(exc).__name__}: {exc}')
        for pose in by_system[system]:
            row = extract_pose(pose, entity_cache[system], ligand_entity.get(system), unit_points, max_bytes)
            status_counter[row['status']] += 1
            rows.append(row)
            done += 1
            if done % 1000 == 0:
                print(f'SHARD {args.shard_index}: {done}개', flush=True)

    frame = pd.DataFrame(rows)
    buf = io.BytesIO()
    frame.to_parquet(buf, engine='pyarrow', index=False, compression='snappy')
    with helper.checked_path(shard_path).open('xb') as handle:
        handle.write(buf.getvalue())
    meta = {'shard_index': args.shard_index, 'num_shards': args.num_shards,
            'systems': len(my_systems), 'poses': len(rows), 'status_counts': dict(status_counter),
            'errors': status_counter.get('error', 0), 'elapsed_seconds': time.perf_counter() - started}
    helper.write_json(done_marker, meta)
    print('SHARD STATUS:', 'OK' if status_counter.get('error', 0) == 0 else 'HAS_ERRORS', flush=True)
    print('SHARD POSES:', len(rows), '| COMPUTED:', status_counter.get('computed', 0),
          '| SKIPPED:', status_counter.get('skipped', 0), '| ERRORS:', status_counter.get('error', 0), flush=True)
    return 0


def run_merge(args):
    """모든 샤드를 합쳐 전체 feature 표와 완결성 리포트를 만듭니다."""
    import pandas as pd
    out = out_dir()
    shards = out / 'shards'
    files = sorted(shards.glob('shard_*.parquet'))
    if not files:
        raise FileNotFoundError('병합할 샤드 parquet이 없습니다.')
    started = time.perf_counter()
    frames = [pd.read_parquet(f) for f in files]
    merged = pd.concat(frames, ignore_index=True)
    if merged.duplicated(list(KEYS)).any():
        raise RuntimeError('샤드 병합에 pose 식별자 중복이 있습니다.')
    buf = io.BytesIO()
    merged.to_parquet(buf, engine='pyarrow', index=False, compression='snappy')
    with helper.checked_path(out / 'combined_features_all.parquet').open('xb') as handle:
        handle.write(buf.getvalue())

    # 완결성: 02번 ok pose 수와 비교.
    import pyarrow.parquet as pq
    idx_total = 0
    for batch in pq.ParquetFile(args.index_run / 'pose_index.parquet').iter_batches(batch_size=50000, columns=['status'], use_threads=False):
        idx_total += sum(1 for s in batch.column('status').to_pylist() if s == 'ok')
    status_counter = Counter(merged['status'])
    per_system = merged.groupby('system_id').size()
    report = {
        'version': VERSION, 'mode': 'merge', 'output_dir': str(out), 'shards_merged': len(files),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'poses_merged': int(len(merged)), 'index_ok_poses': int(idx_total),
        'completeness_ok': bool(len(merged) == idx_total),
        'systems': int(merged['system_id'].nunique()),
        'systems_not_100_poses': int((per_system != 100).sum()),
        'status_counts': dict(status_counter),
        'computed': int(status_counter.get('computed', 0)), 'skipped': int(status_counter.get('skipped', 0)),
        'errors': int(status_counter.get('error', 0)),
        'elapsed_seconds': time.perf_counter() - started, 'gt_used': False, 'coordinates_modified': False,
        'notes': ['cat5(P2Rank) 제외. 전체 feature 표에 cat5는 별도 추출 후 합칩니다.',
                  'completeness_ok=True면 02번 ok pose 전부 추출된 것입니다.']}
    helper.write_json(out / 'merge_report.json', report)
    helper.write_json(out / 'RUN_COMPLETE.json', {'outputs_written': True, 'merge_finished': True,
        'poses': int(len(merged)), 'completeness_ok': report['completeness_ok'], 'errors': report['errors']})
    print('MERGE poses:', len(merged), '/ index ok:', idx_total, '| completeness_ok:', report['completeness_ok'], flush=True)
    print('COMPUTED:', report['computed'], '| SKIPPED:', report['skipped'], '| ERRORS:', report['errors'], flush=True)
    print('SYSTEMS_NOT_100_POSES:', report['systems_not_100_poses'], flush=True)
    return 0 if report['completeness_ok'] and report['errors'] == 0 else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--merge', action='store_true', help='샤드 병합 모드')
    parser.add_argument('--num-shards', type=int, default=100)
    parser.add_argument('--shard-index', type=int, default=None)
    parser.add_argument('--index-run', type=Path, default=DEFAULT_INDEX)
    parser.add_argument('--roles-run', type=Path, default=DEFAULT_ROLES)
    parser.add_argument('--max-cif-mib', type=int, default=128)
    args = parser.parse_args()
    if args.merge:
        return run_merge(args)
    if args.shard_index is None:
        env = os.environ.get('SLURM_ARRAY_TASK_ID')
        if env is None:
            parser.error('샤드 인덱스가 필요합니다(--shard-index 또는 SLURM_ARRAY_TASK_ID).')
        args.shard_index = int(env)
    if not (0 <= args.shard_index < args.num_shards):
        parser.error('shard-index는 0 이상 num-shards 미만이어야 합니다.')
    return run_shard(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
