#!/usr/bin/env python3
"""gnina docking score 전체 추출(③). 샤드로 나눠 병렬 실행·pose 단위 재개·병합합니다.

23번(smoke test)에서 검증한 입력 준비·호출·파싱을 그대로 쓰되, 05/06의 파일럿 산출물이 아니라
02번 전체 ok pose를 대상으로 합니다. pose별 역할 판정과 리간드 원자 대응은 21번과 같이
20번의 map_pose로 직접 계산하므로 파일럿 범위에 묶이지 않습니다. 샤드 분할·pose 단위
체크포인트(.partial.jsonl)·merge는 22번과 동일한 구조입니다.

같은 스크립트를 두 번 씁니다.
  1) 모드 결정용 표본 : --max-systems 로 시스템을 고르게 솎아 두 모드를 모두 실행한 뒤,
                        24 라벨과 join해 25번 하네스로 within-system Top-5를 비교해 모드를 정합니다.
  2) 본실행           : --max-systems 없이 전체, 승자 모드만 실행합니다.

출력은 pose 하나당 한 행(wide)이며 모드별로 열이 갈립니다(affinity_score_only 등). 통합(26번)이
pose 키로 join하므로 행 중복이 없어야 하기 때문입니다. gnina는 CPU로도 GPU와 같은 값을 내므로
(2026-10-29 확인) --gnina 명령만 바꾸면 양쪽 모두 쓸 수 있습니다.

샤드 인덱스는 SLURM_ARRAY_TASK_ID 또는 --shard-index로 받습니다.
출력: out/gnina_all/27_run_gnina_shard[_TAG]/ (shards/ 아래 샤드, 병합본은 같은 폴더).
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
import shlex
import shutil
import subprocess
import time


def _load(name):
    p = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(name.replace('.py', ''), p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


full = _load('20_combined_extractor_full.py')   # pose별 역할·원자 대응(map_pose) 재사용
m23 = _load('23_gnina_smoke.py')                # gnina 입력 준비·호출·파싱 재사용
cf1 = full.cf1
roles = full.roles
audit = full.audit
helper = full.helper

VERSION = 'gnina_all_0.1.0'
OUT_KIND = 'gnina_all'
OUT_BASE = '27_run_gnina_shard'
BASE = helper.ALLOWED_ROOT / 'Scoring_Function' / 'out'
DEFAULT_INDEX = helper.DEFAULT_INDEX
DEFAULT_ROLES = BASE / 'input_roles' / '05_run_audit_input_roles'
KEYS = ('system_id', 'model', 'seed', 'sample')
METRICS = ('affinity', 'cnn_score', 'cnn_affinity')
MODE_ARGS = {'score_only': ['--score_only'], 'local_only': ['--local_only']}
DEFAULT_GNINA = ('singularity exec --bind <FS_ROOT> '
                 '<SHARED_ROOT>/models/gnina/gnina.sif gnina --no_gpu --cpu 8')


def feature_columns(modes):
    """모드별 점수 열 이름(wide 형식)."""
    return [f'{metric}_{mode}' for mode in modes for metric in METRICS]


def ensure_dir(path):
    """여러 array 작업이 동시에 만들어도 충돌하지 않게 폴더를 보장합니다(경쟁 안전)."""
    p = helper.checked_path(path)
    p.mkdir(parents=True, exist_ok=True)
    if not p.is_dir():
        raise RuntimeError(f'디렉터리가 아닙니다: {p}')
    return p


def out_dir(tag):
    name = OUT_BASE + (f'_{tag}' if tag else '')
    return ensure_dir(BASE / OUT_KIND / name)


def load_index_systems(index_run):
    """02번 전체 목록에서 ok pose를 시스템별로 묶어 읽습니다."""
    import pyarrow.parquet as pq
    cols = [*KEYS, 'input_path', 'file_size_bytes', 'file_mtime_ns', 'manifest_path', 'manifest_sha256', 'status']
    by_system = {}
    for batch in pq.ParquetFile(index_run / 'pose_index.parquet').iter_batches(
            batch_size=20000, columns=cols, use_threads=False):
        for r in batch.to_pylist():
            if r['status'] == 'ok':
                by_system.setdefault(r['system_id'], []).append(r)
    return by_system


def select_systems(all_systems, max_systems):
    """표본 실행용으로 시스템을 고르게 솎아냅니다(결정적, 전 구간에 퍼지도록)."""
    if not max_systems or max_systems >= len(all_systems):
        return all_systems
    step = len(all_systems) / max_systems
    picked = [all_systems[min(len(all_systems) - 1, int(i * step))] for i in range(max_systems)]
    return sorted(dict.fromkeys(picked))


def extract_pose(gnina_cmd, pose, entities, ligand_entity_row, modes, work_root, env,
                 timeout, max_bytes, keep_work):
    """한 pose의 수용체·리간드를 준비하고 모드별로 gnina를 실행해 한 행을 만듭니다."""
    system = pose['system_id']
    row = {k: pose[k] for k in KEYS}
    row.update({'pose_id': f"{system}|{pose['model']}|{pose['seed']}|{pose['sample']}",
                'schema_version': VERSION, 'cif_sha256': None, 'ligand_mapping_status': None,
                'status': None, 'skip_reasons_json': '[]', 'error_message': None})
    row.update(dict.fromkeys(feature_columns(modes)))
    for mode in modes:
        row[f'mode_status_{mode}'] = None
    work = None
    try:
        if isinstance(entities, tuple) and entities and entities[0] == 'error':
            raise ValueError(f'시스템 입력 점검 실패: {entities[1]}')
        rec_rows, lig_rows, mapping_entry, reasons, cif_sha = full.map_pose(
            pose['input_path'], pose.get('file_size_bytes'), pose.get('file_mtime_ns'),
            entities, ligand_entity_row, max_bytes)
        row['cif_sha256'] = cif_sha
        if reasons:
            row['status'] = 'skipped'
            row['skip_reasons_json'] = audit.encoded(reasons)
            return row
        row['ligand_mapping_status'] = mapping_entry.get('ligand_status')
        mapping = mapping_entry.get('atom_mapping')
        if not mapping:
            # 원자 대응이 없으면 결합 차수를 가진 리간드 SDF를 만들 수 없습니다(좌표만으로 추론하면
            # 복잡한 리간드에서 틀리므로 추측하지 않고 보류합니다).
            row['status'] = 'skipped'
            row['skip_reasons_json'] = audit.encoded(
                ['ligand_mapping_unavailable:' + str(mapping_entry.get('ligand_status'))])
            return row
        smiles = (ligand_entity_row or {}).get('canonical_smiles') or (ligand_entity_row or {}).get('smiles')
        if not (smiles or '').strip():
            raise ValueError('ligand_smiles_missing')

        atoms, _ = cf1.read_atoms(pose['input_path'], cif_sha, max_bytes)
        lig_xyz, _ = cf1.pick_coords(atoms, lig_rows)
        work = ensure_dir(Path(work_root) / f"{system}_{pose['model']}_{pose['seed']}_{pose['sample']}")
        rec_path = m23.build_receptor_pdb(pose['input_path'], work / 'receptor.pdb')
        lig_path = m23.build_ligand_sdf(smiles, {'coords': lig_xyz}, mapping, work / 'ligand.sdf')

        failed = []
        for mode, extra in modes.items():
            try:
                parsed, _raw = m23.run_gnina(gnina_cmd, rec_path, lig_path, extra, env, timeout)
                for metric in METRICS:
                    row[f'{metric}_{mode}'] = parsed.get(metric)
                ok = parsed.get('cnn_score') is not None
                row[f'mode_status_{mode}'] = 'computed' if ok else 'parsed_empty'
                if not ok:
                    failed.append(f'{mode}:parsed_empty')
            except subprocess.TimeoutExpired:
                row[f'mode_status_{mode}'] = 'timeout'
                failed.append(f'{mode}:timeout')
            except Exception as exc:
                row[f'mode_status_{mode}'] = 'error'
                failed.append(f'{mode}:{type(exc).__name__}: {exc}')
        if not failed:
            row['status'] = 'computed'
        elif len(failed) < len(modes):
            row['status'] = 'partial'
            row['error_message'] = '; '.join(failed)[:500]
        else:
            row['status'] = 'error'
            row['error_message'] = '; '.join(failed)[:500]
    except subprocess.TimeoutExpired:
        row.update(status='error', error_message='prep_timeout')
    except Exception as exc:
        row.update(status='error', error_message=f'{type(exc).__name__}: {exc}')
    finally:
        if work is not None and not keep_work:
            shutil.rmtree(work, ignore_errors=True)   # 460k 규모 디스크 보호
    return row


def build_env(out):
    """외부 도구의 임시·캐시를 결과 폴더로 가둡니다(gnina 호출 직전 23번이 일부를 다시 벗겨냄)."""
    env = dict(os.environ)
    cache = out / 'cache'
    env.update({'TMPDIR': str(cache), 'TMP': str(cache), 'TEMP': str(cache),
                'XDG_CACHE_HOME': str(cache)})
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
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue   # 중단으로 잘린 줄 → 해당 pose는 다시 계산
                if r.get('pose_id'):
                    done[r['pose_id']] = r
    return done


def run_shard(args, modes):
    """한 샤드의 pose만 gnina로 처리해 shard parquet을 저장합니다(pose 단위 재개)."""
    import pandas as pd
    out = out_dir(args.tag)
    shards = ensure_dir(out / 'shards')
    ensure_dir(out / 'tmp'); ensure_dir(out / 'cache'); work_root = ensure_dir(out / 'work')
    for key in ('TMPDIR', 'TMP', 'TEMP'):
        os.environ[key] = str(out / 'tmp')
    os.environ['XDG_CACHE_HOME'] = str(out / 'cache')

    gnina_cmd = shlex.split(args.gnina)
    if not gnina_cmd:
        raise ValueError('--gnina 명령이 비어 있습니다.')
    if shutil.which(gnina_cmd[0]) is None and not Path(gnina_cmd[0]).exists():
        raise FileNotFoundError(f'gnina 실행기를 찾을 수 없습니다: {gnina_cmd[0]}')

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
    if args.systems_file is not None:
        # 특정 시스템만 처리합니다(난이도 구간을 지정한 방법론 실험 등). --max-systems보다 우선합니다.
        wanted = {line.strip() for line in Path(args.systems_file).read_text(encoding='utf-8').splitlines()
                  if line.strip() and not line.startswith('#')}
        chosen = [s for s in all_systems if s in wanted]
        print(f'시스템 목록 지정: {len(chosen)}개 사용 (목록 {len(wanted)}개, '
              f'인덱스에 없음 {len(wanted - set(all_systems))}개)', flush=True)
        all_systems = chosen
    else:
        all_systems = select_systems(all_systems, args.max_systems)
    if not all_systems:
        raise ValueError('처리할 시스템이 없습니다(목록·max-systems 확인).')
    my_systems = [s for i, s in enumerate(all_systems) if i % args.num_shards == args.shard_index]
    ligand_entity = {}
    with (args.roles_run / 'input_entities.csv').open(newline='', encoding='utf-8') as handle:
        for r in csv.DictReader(handle):
            if r.get('input_kind') == 'ligand':
                ligand_entity.setdefault(r['system_id'], r)

    done_rows = load_partial(partial_path)
    rows = list(done_rows.values())
    status_counter = Counter(r['status'] for r in rows)
    total_poses = sum(len(by_system[s]) for s in my_systems)
    print(f'SHARD {args.shard_index}/{args.num_shards}: 시스템 {len(my_systems)}개 pose {total_poses}개 '
          f'(선택 시스템 {len(all_systems)}, 이미 완료 {len(done_rows)}개)', flush=True)
    print('MODES:', ','.join(modes), flush=True)

    entity_cache = {}
    processed = len(done_rows)
    handle = partial_path.open('a', encoding='utf-8')
    try:
        for system in my_systems:
            if system not in entity_cache:
                try:
                    meta_src = by_system[system][0]
                    meta = {'manifest_path': meta_src['manifest_path'],
                            'manifest_sha256': meta_src['manifest_sha256']}
                    _r, ents, _d = roles.inspect_input(system, meta, max_bytes, 30)
                    entity_cache[system] = ents
                except Exception as exc:
                    entity_cache[system] = ('error', f'{type(exc).__name__}: {exc}')
            for pose in by_system[system]:
                pose_id = f"{system}|{pose['model']}|{pose['seed']}|{pose['sample']}"
                if pose_id in done_rows:
                    continue
                row = extract_pose(gnina_cmd, pose, entity_cache[system], ligand_entity.get(system),
                                   modes, work_root, env, args.timeout, max_bytes, args.keep_work)
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
            'modes': list(modes), 'max_systems': args.max_systems,
            'systems_file': str(args.systems_file) if args.systems_file else None,
            'selected_systems': len(all_systems),
            'systems': len(my_systems), 'poses': len(rows),
            'status_counts': dict(status_counter), 'errors': status_counter.get('error', 0),
            'gnina': args.gnina, 'elapsed_seconds': time.perf_counter() - started}
    helper.write_json(done_marker, meta)
    partial_path.unlink(missing_ok=True)
    per_call = ((time.perf_counter() - started) / max(1, (processed - len(done_rows)) * len(modes)))
    print('SHARD STATUS:', 'OK' if status_counter.get('error', 0) == 0 else 'HAS_ERRORS', flush=True)
    print('SHARD POSES:', len(rows), '| COMPUTED:', status_counter.get('computed', 0),
          '| PARTIAL:', status_counter.get('partial', 0),
          '| SKIPPED:', status_counter.get('skipped', 0),
          '| ERRORS:', status_counter.get('error', 0), flush=True)
    print(f'호출당 평균 {per_call:.2f}초(신규 처리 기준)', flush=True)
    return 0


def run_merge(args, modes):
    """모든 샤드를 합쳐 전체 gnina feature 표와 리포트를 만듭니다."""
    import pandas as pd
    out = out_dir(args.tag)
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
    with helper.checked_path(out / 'gnina_features_all.parquet').open('xb') as fh:
        fh.write(buf.getvalue())

    import pyarrow.parquet as pq
    idx_total = 0
    for batch in pq.ParquetFile(args.index_run / 'pose_index.parquet').iter_batches(
            batch_size=50000, columns=['status'], use_threads=False):
        idx_total += sum(1 for s in batch.column('status').to_pylist() if s == 'ok')
    status_counter = Counter(merged['status'])
    cols = [c for c in feature_columns(modes) if c in merged.columns]
    coverage = {c: int(merged[c].notna().sum()) for c in cols}
    complete = bool(len(merged) == idx_total)
    report = {
        'version': VERSION, 'mode': 'merge', 'output_dir': str(out), 'shards_merged': len(files),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'modes': list(modes), 'poses_merged': int(len(merged)), 'index_ok_poses': int(idx_total),
        'completeness_ok': complete, 'allow_partial': bool(args.allow_partial),
        'systems': int(merged['system_id'].nunique()),
        'status_counts': {str(k): int(v) for k, v in status_counter.items()},
        'computed': int(status_counter.get('computed', 0)),
        'partial': int(status_counter.get('partial', 0)),
        'skipped': int(status_counter.get('skipped', 0)),
        'errors': int(status_counter.get('error', 0)),
        'feature_coverage': coverage, 'feature_category': '2(gnina docking score)',
        'elapsed_seconds': time.perf_counter() - started,
        'gt_used': False, 'coordinates_modified': False,
        'notes': ['26번 통합의 --gnina 입력으로 씁니다(pose 키로 join, 행은 pose당 하나).',
                  '표본 실행(--max-systems)에서는 completeness_ok가 False인 것이 정상이며 '
                  '--allow-partial로 종료 코드를 0으로 둡니다.',
                  'local_only는 점수 feature로만 쓰고 제출 pose·기하 feature를 교체하지 않습니다.']}
    helper.write_json(out / 'merge_report.json', report)
    helper.write_json(out / 'RUN_COMPLETE.json', {'outputs_written': True, 'merge_finished': True,
        'poses': int(len(merged)), 'completeness_ok': complete, 'errors': report['errors']})
    print('MERGE poses:', len(merged), '/ index ok:', idx_total, '| completeness_ok:', complete, flush=True)
    print('COMPUTED:', report['computed'], '| PARTIAL:', report['partial'],
          '| SKIPPED:', report['skipped'], '| ERRORS:', report['errors'], flush=True)
    print('FEATURE_COVERAGE:', roles.j(coverage), flush=True)
    ok = report['errors'] == 0 and (complete or args.allow_partial)
    return 0 if ok else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--merge', action='store_true', help='샤드 병합 모드')
    parser.add_argument('--num-shards', type=int, default=200)
    parser.add_argument('--shard-index', type=int, default=None)
    parser.add_argument('--index-run', type=Path, default=DEFAULT_INDEX)
    parser.add_argument('--roles-run', type=Path, default=DEFAULT_ROLES)
    parser.add_argument('--gnina', type=str, default=DEFAULT_GNINA,
                        help='gnina 실행 명령 전체(공백 구분). 기본은 CPU 실행(--no_gpu --cpu 8).')
    parser.add_argument('--modes', type=str, default='score_only',
                        help='쉼표 구분. score_only,local_only 중 선택(표본 실행은 둘 다).')
    parser.add_argument('--max-systems', type=int, default=None,
                        help='표본 실행용. 전체에서 고르게 솎아 이 개수만 처리합니다.')
    parser.add_argument('--systems-file', type=Path, default=None,
                        help='처리할 system_id 목록 파일(한 줄에 하나, # 주석 가능). --max-systems보다 우선합니다.')
    parser.add_argument('--allow-partial', action='store_true',
                        help='merge에서 전체 미만이어도 종료 코드 0(표본 병합용).')
    parser.add_argument('--max-cif-mib', type=int, default=128)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--keep-work', action='store_true', help='pose별 수용체/리간드 파일을 남깁니다(디버그).')
    parser.add_argument('--tag', type=str, default=None, help='출력 폴더 접미사(표본/본실행 분리용).')
    args = parser.parse_args()
    if args.max_cif_mib < 1 or args.timeout < 1:
        parser.error('상한은 양수여야 합니다.')
    names = [m.strip() for m in args.modes.split(',') if m.strip()]
    unknown = [m for m in names if m not in MODE_ARGS]
    if not names or unknown:
        parser.error(f'알 수 없는 모드: {unknown or "(비어 있음)"}. 가능: {list(MODE_ARGS)}')
    modes = {m: MODE_ARGS[m] for m in names}
    if args.merge:
        return run_merge(args, modes)
    if args.shard_index is None:
        env = os.environ.get('SLURM_ARRAY_TASK_ID')
        if env is None:
            parser.error('샤드 인덱스가 필요합니다(--shard-index 또는 SLURM_ARRAY_TASK_ID).')
        args.shard_index = int(env)
    if not (0 <= args.shard_index < args.num_shards):
        parser.error('shard-index는 0 이상 num-shards 미만이어야 합니다.')
    return run_shard(args, modes)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
