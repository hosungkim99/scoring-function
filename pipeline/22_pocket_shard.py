#!/usr/bin/env python3
"""전체 460k pose 포켓 적합성(category 5) 추출(A6c-2). P2Rank를 샤드로 나눠 병렬 실행·재개·병합합니다.

14번(표본 152 pose)의 P2Rank 호출·feature 정의를 그대로 쓰되, 선정 표본이 아니라 02번 전체 ok
pose를 대상으로 합니다. 시스템을 샤드로 나눠(21번과 동일한 i%num_shards 방식) 각 샤드가 자기 몫의
pose만 처리합니다. P2Rank는 pose당 JVM을 띄우는 비싼 호출이라, 샤드 안에서도 pose 단위 체크포인트
(.partial.jsonl)를 남겨 중단 시 이미 끝낸 pose는 건너뜁니다(resume). --merge 모드는 모든 샤드를
합쳐 전체 포켓 feature 표를 만듭니다.

20번/21번과 달리 역할 대응은 '리간드 중심 좌표'만 필요합니다(P2Rank는 리간드를 무시하고 단백질
포켓만 예측). 따라서 06 원자 대응·수용체 정합성·연결성 검토는 gate에 쓰지 않고, 리간드를 하나로
특정할 수 있는지와 구조 자체의 차단 플래그만 봅니다. 이렇게 하면 cat1~4·6~8에서 보류된 pose라도
포켓 신호는 얻을 수 있습니다. 외부 도구 출력·임시·캐시는 결과 폴더로 가두고 원본을 수정하지 않으며
GT를 쓰지 않습니다.

샤드 인덱스는 SLURM_ARRAY_TASK_ID 또는 --shard-index로 받습니다.
출력: out/pocket_all/22_run_pocket_shard/ (shards/ 아래 샤드, 병합본은 같은 폴더).
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
import shutil
import subprocess
import time


def _load(name):
    p = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(name.replace('.py', ''), p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

full = _load('20_combined_extractor_full.py')   # per-pose 역할 대응 기반 재사용
pk = _load('14_pocket_features.py')              # P2Rank 호출·feature 정의 재사용
cf1 = full.cf1
roles = full.roles
audit = full.audit
helper = full.helper
ex = full.ex

VERSION = 'pocket_all_0.1.0'
OUT_KIND = 'pocket_all'
OUT_NAME = '22_run_pocket_shard'
BASE = helper.ALLOWED_ROOT / 'Scoring_Function' / 'out'
DEFAULT_INDEX = helper.DEFAULT_INDEX
DEFAULT_ROLES = BASE / 'input_roles' / '05_run_audit_input_roles'
DEFAULT_PRANK = pk.DEFAULT_PRANK
KEYS = ('system_id', 'model', 'seed', 'sample')
FEATURE_NAMES = list(pk.feature_definitions())


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
    """02번 전체 목록에서 ok pose를 시스템별로 묶어 읽습니다."""
    import pyarrow.parquet as pq
    cols = [*KEYS, 'input_path', 'file_size_bytes', 'file_mtime_ns', 'manifest_path', 'manifest_sha256', 'status']
    by_system = {}
    for batch in pq.ParquetFile(index_run / 'pose_index.parquet').iter_batches(batch_size=20000, columns=cols, use_threads=False):
        for r in batch.to_pylist():
            if r['status'] == 'ok':
                by_system.setdefault(r['system_id'], []).append(r)
    return by_system


def resolve_ligand_rows(cif_path, size, mtime, entities, max_bytes):
    """pose의 리간드 중원자 행만 해석합니다(P2Rank는 리간드를 쓰지 않으므로 중심 좌표용)."""
    facts, e_rows, r_rows, details = audit.inspect_cif(Path(cif_path), max_bytes, full.MAX_ATOMS, size, mtime)
    observed = roles.get_observed(Path(cif_path), facts, e_rows, max_bytes)
    matches = roles.match_entities(entities, observed)
    _receptor, ligand = ex.receptor_and_ligand(matches)
    reasons = []
    if len(ligand) == 0:
        reasons.append('no_matched_ligand')
    elif len(ligand) > 1:
        reasons.append('multiple_matched_ligands')
    elif ligand[0].get('observed_residue_groups') != 1:
        reasons.append('ligand_spans_multiple_residue_groups')
    if set(details['review_flags']) & roles.BLOCK_FLAGS:
        reasons.append('cif_block_flags')
    if reasons:
        return None, sorted(set(reasons)), facts['input_sha256']
    lig_rows = json.loads(ligand[0]['heavy_atom_site_rows_1based_json'])
    return lig_rows, [], facts['input_sha256']


def pocket_features(prank, cif_path, cif_sha, lig_rows, work_dir, env, timeout, max_bytes):
    """한 pose에 P2Rank를 실행하고 7개 포켓 feature를 계산해 돌려줍니다."""
    import numpy as np
    feats = dict.fromkeys(FEATURE_NAMES)
    atoms, _ = cf1.read_atoms(cif_path, cif_sha, max_bytes)
    lig_xyz, _ = cf1.pick_coords(atoms, lig_rows)
    centroid = lig_xyz.mean(axis=0)
    pose_dir = ensure_dir(work_dir)
    try:
        pred = pk.run_p2rank(prank, cif_path, pose_dir, env, timeout)
        df = pk.parse_predictions(pred)
        feats['num_predicted_pockets'] = int(len(df))
        status = 'computed'
        if len(df) == 0:
            status = 'no_pockets_predicted'   # 포켓 기반 값은 null 유지
        else:
            top = df.sort_values('rank').iloc[0]
            feats['top_pocket_score'] = round(float(top['score']), 4)
            feats['top_pocket_probability'] = round(float(top['probability']), 4)
            centers = df[['center_x', 'center_y', 'center_z']].to_numpy(dtype=np.float64)
            dists = np.linalg.norm(centers - centroid, axis=1)
            nidx = int(dists.argmin())
            near = df.iloc[nidx]
            feats['ligand_pocket_rank'] = int(near['rank'])
            feats['ligand_pocket_score'] = round(float(near['score']), 4)
            feats['ligand_pocket_probability'] = round(float(near['probability']), 4)
            feats['ligand_to_pocket_center_dist_A'] = round(float(dists[nidx]), 4)
    finally:
        shutil.rmtree(pose_dir, ignore_errors=True)   # 460k 규모 디스크 보호: pose별 출력 즉시 정리
    return feats, status


def extract_pose(prank, pose, entities, work_root, env, timeout, max_bytes):
    """한 pose의 포켓 feature 행을 만듭니다(14번 로직을 전체 pose에 적용)."""
    system = pose['system_id']
    row = {k: pose[k] for k in KEYS}
    row.update({'pose_id': f"{system}|{pose['model']}|{pose['seed']}|{pose['sample']}",
                'schema_version': VERSION, 'cif_sha256': None,
                'status': None, 'skip_reasons_json': '[]', 'error_message': None})
    row.update(dict.fromkeys(FEATURE_NAMES))
    try:
        if isinstance(entities, tuple) and entities and entities[0] == 'error':
            raise ValueError(f'시스템 입력 점검 실패: {entities[1]}')
        lig_rows, reasons, cif_sha = resolve_ligand_rows(
            pose['input_path'], pose.get('file_size_bytes'), pose.get('file_mtime_ns'), entities, max_bytes)
        row['cif_sha256'] = cif_sha
        if reasons:
            row['status'] = 'skipped'
            row['skip_reasons_json'] = audit.encoded(reasons)
        else:
            work_dir = Path(work_root) / f"{system}_{pose['model']}_{pose['seed']}_{pose['sample']}"
            feats, status = pocket_features(prank, pose['input_path'], cif_sha, lig_rows,
                                            work_dir, env, timeout, max_bytes)
            row.update(feats)
            row['status'] = status
    except subprocess.TimeoutExpired:
        row.update(status='error', error_message='p2rank_timeout')
    except Exception as exc:
        row.update(status='error', error_message=f'{type(exc).__name__}: {exc}')
    return row


def build_env(out):
    """P2Rank(JVM)의 출력·임시·캐시를 결과 폴더로 가둡니다(14번과 동일)."""
    env = dict(os.environ)
    cache = out / 'cache'
    env.update({'TMPDIR': str(cache), 'TMP': str(cache), 'TEMP': str(cache),
                'XDG_CACHE_HOME': str(cache),
                'JAVA_TOOL_OPTIONS': f'-Djava.io.tmpdir={cache}',
                'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    return env


def load_partial(partial_path):
    """pose 단위 체크포인트를 읽어 이미 끝낸 pose_id→행 사전을 돌려줍니다(깨진 마지막 줄 무시)."""
    done = {}
    if partial_path.exists():
        with partial_path.open('r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue   # 중단으로 잘린 줄 → 해당 pose는 다시 계산
                if row.get('pose_id'):
                    done[row['pose_id']] = row
    return done


def run_shard(args):
    """한 샤드의 pose만 P2Rank로 처리해 shard parquet을 저장합니다(pose 단위 재개)."""
    import pandas as pd
    out = out_dir()
    shards = ensure_dir(out / 'shards')
    ensure_dir(out / 'tmp'); ensure_dir(out / 'cache'); work_root = ensure_dir(out / 'p2rank')
    for key in ('TMPDIR', 'TMP', 'TEMP'):
        os.environ[key] = str(out / 'tmp')
    os.environ['XDG_CACHE_HOME'] = str(out / 'cache')
    prank = args.prank
    if not Path(prank).exists():
        raise FileNotFoundError(f'P2Rank 실행 파일을 찾을 수 없습니다: {prank}')
    shard_path = shards / f'shard_{args.shard_index:04d}.parquet'
    done_marker = shards / f'shard_{args.shard_index:04d}.done'
    partial_path = shards / f'shard_{args.shard_index:04d}.partial.jsonl'
    if done_marker.exists() and shard_path.exists():
        print(f'SHARD {args.shard_index}: 이미 완료되어 건너뜁니다(resume).', flush=True)
        return 0
    if shard_path.exists() and not done_marker.exists():
        helper.checked_path(shard_path).unlink()
        print(f'SHARD {args.shard_index}: 불완전 샤드 제거 후 재생성합니다.', flush=True)
    started = time.perf_counter()
    max_bytes = args.max_cif_mib * 1024 ** 2
    env = build_env(out)

    by_system = load_index_systems(args.index_run)
    all_systems = sorted(by_system)
    my_systems = [s for i, s in enumerate(all_systems) if i % args.num_shards == args.shard_index]
    ligand_meta = {}
    with (args.roles_run / 'input_entities.csv').open(newline='', encoding='utf-8') as handle:
        for r in csv.DictReader(handle):
            if r.get('input_kind') == 'ligand':
                ligand_meta.setdefault(r['system_id'], r)

    done_rows = load_partial(partial_path)
    rows = list(done_rows.values())
    status_counter = Counter(r['status'] for r in rows)
    total_poses = sum(len(by_system[s]) for s in my_systems)
    print(f'SHARD {args.shard_index}/{args.num_shards}: 시스템 {len(my_systems)}개 pose {total_poses}개 '
          f'(이미 완료 {len(done_rows)}개)', flush=True)

    entity_cache = {}
    processed = len(done_rows)
    handle = partial_path.open('a', encoding='utf-8')
    try:
        for system in my_systems:
            if system not in entity_cache:
                try:
                    meta_src = by_system[system][0]
                    meta = {'manifest_path': meta_src['manifest_path'], 'manifest_sha256': meta_src['manifest_sha256']}
                    _r, ents, _d = roles.inspect_input(system, meta, max_bytes, 30)
                    entity_cache[system] = ents
                except Exception as exc:
                    entity_cache[system] = ('error', f'{type(exc).__name__}: {exc}')
            for pose in by_system[system]:
                pose_id = f"{system}|{pose['model']}|{pose['seed']}|{pose['sample']}"
                if pose_id in done_rows:
                    continue
                row = extract_pose(prank, pose, entity_cache[system], work_root, env, args.timeout, max_bytes)
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
                rows.append(row)
                status_counter[row['status']] += 1
                processed += 1
                if processed % 100 == 0:
                    print(f'SHARD {args.shard_index}: {processed}/{total_poses}개', flush=True)
    finally:
        handle.close()

    frame = pd.DataFrame(rows)
    if frame.duplicated(list(KEYS)).any():
        raise RuntimeError('샤드 안에 pose 식별자 중복이 있습니다.')
    buf = io.BytesIO()
    frame.to_parquet(buf, engine='pyarrow', index=False, compression='snappy')
    with helper.checked_path(shard_path).open('xb') as fh:
        fh.write(buf.getvalue())
    meta = {'shard_index': args.shard_index, 'num_shards': args.num_shards,
            'systems': len(my_systems), 'poses': len(rows), 'status_counts': dict(status_counter),
            'errors': status_counter.get('error', 0), 'elapsed_seconds': time.perf_counter() - started}
    helper.write_json(done_marker, meta)
    partial_path.unlink(missing_ok=True)
    print('SHARD STATUS:', 'OK' if status_counter.get('error', 0) == 0 else 'HAS_ERRORS', flush=True)
    print('SHARD POSES:', len(rows), '| COMPUTED:', status_counter.get('computed', 0),
          '| NO_POCKETS:', status_counter.get('no_pockets_predicted', 0),
          '| SKIPPED:', status_counter.get('skipped', 0), '| ERRORS:', status_counter.get('error', 0), flush=True)
    return 0


def run_merge(args):
    """모든 샤드를 합쳐 전체 포켓 feature 표와 완결성 리포트를 만듭니다."""
    import pandas as pd
    out = out_dir()
    shards = out / 'shards'
    files = sorted(shards.glob('shard_*.parquet'))
    if not files:
        raise FileNotFoundError('병합할 샤드 parquet이 없습니다.')
    started = time.perf_counter()
    merged = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if merged.duplicated(list(KEYS)).any():
        raise RuntimeError('샤드 병합에 pose 식별자 중복이 있습니다.')
    buf = io.BytesIO()
    merged.to_parquet(buf, engine='pyarrow', index=False, compression='snappy')
    with helper.checked_path(out / 'pocket_features_all.parquet').open('xb') as fh:
        fh.write(buf.getvalue())

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
        'status_counts': {str(k): int(v) for k, v in status_counter.items()},
        'computed': int(status_counter.get('computed', 0)),
        'no_pockets_predicted': int(status_counter.get('no_pockets_predicted', 0)),
        'skipped': int(status_counter.get('skipped', 0)), 'errors': int(status_counter.get('error', 0)),
        'feature_category': 5, 'elapsed_seconds': time.perf_counter() - started,
        'gt_used': False, 'coordinates_modified': False,
        'notes': ['category 5(P2Rank) 전체 추출. 통합 표(21번 결과)에 pose_id로 붙여 cat5 열을 채웁니다.',
                  'completeness_ok=True면 02번 ok pose 전부에 포켓 행이 있는 것입니다(computed가 아니어도 행은 존재).',
                  'gate는 리간드 특정 가능성·구조 차단 플래그만 보므로 cat1~4·6~8 보류 pose도 포함될 수 있습니다.']}
    helper.write_json(out / 'merge_report.json', report)
    helper.write_json(out / 'RUN_COMPLETE.json', {'outputs_written': True, 'merge_finished': True,
        'poses': int(len(merged)), 'completeness_ok': report['completeness_ok'], 'errors': report['errors']})
    print('MERGE poses:', len(merged), '/ index ok:', idx_total, '| completeness_ok:', report['completeness_ok'], flush=True)
    print('COMPUTED:', report['computed'], '| NO_POCKETS:', report['no_pockets_predicted'],
          '| SKIPPED:', report['skipped'], '| ERRORS:', report['errors'], flush=True)
    print('SYSTEMS_NOT_100_POSES:', report['systems_not_100_poses'], flush=True)
    return 0 if report['completeness_ok'] and report['errors'] == 0 else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--merge', action='store_true', help='샤드 병합 모드')
    parser.add_argument('--num-shards', type=int, default=200)
    parser.add_argument('--shard-index', type=int, default=None)
    parser.add_argument('--index-run', type=Path, default=DEFAULT_INDEX)
    parser.add_argument('--roles-run', type=Path, default=DEFAULT_ROLES)
    parser.add_argument('--prank', type=Path, default=DEFAULT_PRANK)
    parser.add_argument('--max-cif-mib', type=int, default=128)
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()
    if args.max_cif_mib < 1 or args.timeout < 1:
        parser.error('상한은 양수여야 합니다.')
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
