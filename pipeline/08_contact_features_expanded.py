#!/usr/bin/env python3
"""선정 40시스템의 전체 100 pose(약 4,000개)로 접촉·거리 feature(카테고리 1)를 확대 추출합니다.

같은 타깃 안 seed/sample 변동을 보기 위한 확대 검증이며, 추출 자체의 안정성(성공/실패·사유·
처리 시간·최대 메모리)을 집계하는 것이 목표입니다. sc_rmsd 등 GT 라벨은 읽지 않으며, 라벨과의
관계 분석은 이후 별도 탐색 단계에서 수행합니다. 원본 구조와 좌표를 수정하지 않습니다.

07번을 import해 contacts·pick_coords·feature 정의를 재사용하고, 07이 import하는 05→04→03의
입력 역할 점검·CIF 대응 함수를 그대로 씁니다(각 모듈 main은 실행하지 않음). 입력은 02번 전체
pose 목록(권위 있는 20261019T054512Z_74cd3e83)과 03번 선정 시스템, 그리고 원본 CIF입니다.
출력: out/contact_features_expanded/08_run_contact_features_expanded/ 아래 고정 폴더.
"""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import time

# 07번을 재사용하면 07→05→04→03 체인이 함께 로드됩니다.
_support_path = Path(__file__).with_name('07_contact_features.py')
_spec = importlib.util.spec_from_file_location('contact_support', _support_path)
cf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cf)
roles = cf.roles       # 05 모듈
audit = cf.audit       # 04 모듈
helper = cf.helper     # 03 모듈

VERSION = 'contact_features_expanded_0.1.0'
OUT_KIND = 'contact_features_expanded'
OUT_NAME = '08_run_contact_features_expanded'
DEFAULT_INDEX = helper.DEFAULT_INDEX                 # 02번 전체 목록(권위)
DEFAULT_SELECTION = audit.DEFAULT_SELECTION          # 03번 선정 결과
KEYS = ('system_id', 'model', 'seed', 'sample')
MAX_ATOMS = 250000


def specification():
    """확대 추출의 feature 정의와 선택 정책을 기록합니다(07번과 동일한 13개 feature)."""
    return {
        'schema_version': VERSION,
        'scope': '선정 40시스템의 전체 100 pose에 대한 수용체-리간드 접촉·거리(카테고리 1) 확대 추출.',
        'features': cf.feature_definitions(),
        'selection': {
            'poses': '02번 전체 목록에서 선정 시스템의 status==ok 행(시스템당 100개).',
            'receptor': '각 pose CIF에서 05 대응으로 얻은 매칭 단백질 체인(id_sequence_consistent, 역할검토 없음)의 중원자 합집합.',
            'ligand': '각 pose CIF에서 매칭된 비폴리머(id_composition_consistent, 잔기 그룹 1개)의 중원자.',
            'eligibility': '리간드 1개·수용체 존재·치명적 CIF 플래그 없음·출력 연결 기록 없음일 때만 계산.',
            'distance': '원본 공통 좌표계 유클리드 거리, strict < cutoff. H/D 제외.'},
        'stability_goal': '성공/실패·사유·pose당 처리 시간·최대 RSS를 집계해 전체 추출 확장성을 점검.',
        'labels': 'sc_rmsd 등 GT 라벨은 읽지 않으며 분별력 분석은 별도 단계에서 수행.',
        'notes': ['05번을 재실행하지 않고 선정 시스템 입력 역할을 pose와 독립적으로 1회 계산합니다.',
                  '같은 모델이라도 seed/sample마다 출력 CIF를 개별 대응합니다.',
                  '카테고리 1은 원자 단위 화학 대응이 필요 없어 06번의 대칭·mismatch와 무관합니다.']}


def load_selected_systems(selection_dir):
    """03번 선정 결과에서 시스템 ID 집합을 읽습니다."""
    path = selection_dir / 'selected_systems.csv'
    with path.open(newline='', encoding='utf-8-sig') as handle:
        systems = {r['system_id'] for r in csv.DictReader(handle)}
    if not systems:
        raise ValueError('선정 시스템 목록이 비어 있습니다.')
    return systems


def load_selected_poses(index_dir, selected):
    """02번 전체 목록에서 선정 시스템의 ok pose 행만 읽습니다."""
    import pyarrow.parquet as pq
    columns = [*KEYS, 'input_path', 'file_size_bytes', 'file_mtime_ns',
               'manifest_path', 'manifest_sha256', 'status']
    poses = []
    for batch in pq.ParquetFile(index_dir / 'pose_index.parquet').iter_batches(
            batch_size=20000, columns=columns, use_threads=False):
        for row in batch.to_pylist():
            if row['system_id'] in selected and row['status'] == 'ok':
                poses.append(row)
    return poses


def receptor_and_ligand(matches):
    """매칭 결과에서 깨끗한 수용체 체인과 단일 리간드를 고릅니다."""
    receptor, ligand = [], []
    for m in matches:
        rows_json = m.get('heavy_atom_site_rows_1based_json')
        if not rows_json:
            continue
        flags = json.loads(m.get('role_review_flags_json') or '[]')
        if m['input_kind'] == 'protein' and m['match_status'] == 'id_sequence_consistent' and not flags:
            receptor.append(m)
        elif m['input_kind'] == 'ligand' and m['match_status'] == 'id_composition_consistent':
            ligand.append(m)
    return receptor, ligand


def run(args):
    """선정 시스템의 전체 pose에 접촉 feature를 확대 추출하고 안정성을 집계합니다."""
    started = time.perf_counter()
    parent = helper.safe_mkdir(helper.ALLOWED_ROOT / 'Scoring_Function' / 'out' / OUT_KIND)
    out = helper.checked_path(parent / OUT_NAME)
    if out.exists():
        raise FileExistsError(f'기존 결과를 덮어쓰지 않습니다: {out}')
    out.mkdir()
    for name in ('tmp', 'cache'):
        helper.safe_mkdir(out / name)
    for key in ('TMPDIR', 'TMP', 'TEMP'):
        os.environ[key] = str(out / 'tmp')
    os.environ['XDG_CACHE_HOME'] = str(out / 'cache')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    print('OUTPUT_DIR:', out, flush=True)

    import gemmi
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)

    spec = specification()
    spec_hash = hashlib.sha256((json.dumps(spec, ensure_ascii=False, sort_keys=True,
                                           allow_nan=False)).encode()).hexdigest()
    helper.write_json(out / 'feature_spec.json', spec)

    sources = [helper.fingerprint(p) for p in (
        args.index_run / 'RUN_COMPLETE.json', args.index_run / 'pose_index.parquet',
        args.selection_run / 'selected_systems.csv')]
    index_complete = json.loads((args.index_run / 'RUN_COMPLETE.json').read_text(encoding='utf-8'))
    if index_complete.get('validation_passed') is not True:
        raise ValueError('02번 전체 pose 목록이 정상 완료된 결과가 아닙니다.')

    selected = load_selected_systems(args.selection_run)
    poses = load_selected_poses(args.index_run, selected)
    got_systems = {p['system_id'] for p in poses}
    if got_systems != selected or len(poses) != len(selected) * 100:
        raise ValueError(f'선정 시스템의 100개 pose 구성이 예상과 다릅니다: 시스템 {len(got_systems)}/{len(selected)}, pose {len(poses)}')

    feature_names = list(cf.feature_definitions())
    max_bytes = args.max_cif_mib * 1024 ** 2
    entity_cache = {}      # system_id -> entities 또는 ('error', message)
    rows_out = []
    status_counter = Counter()
    skip_counter = Counter()
    system_stat = defaultdict(lambda: Counter())
    poses = sorted(poses, key=lambda p: tuple(p[k] for k in KEYS))

    for number, pose in enumerate(poses, 1):
        system = pose['system_id']
        begin = time.perf_counter()
        row = {k: pose[k] for k in KEYS}
        row.update({'pose_id': f"{system}|{pose['model']}|{pose['seed']}|{pose['sample']}",
                    'input_path': pose['input_path'], 'schema_version': VERSION,
                    'config_sha256': spec_hash, 'cif_sha256': None,
                    'receptor_chain_ids_json': None, 'ligand_chain': None,
                    'receptor_heavy_atoms': None, 'ligand_heavy_atoms': None,
                    'status': None, 'skip_reasons_json': '[]', 'error_message': None,
                    'elapsed_seconds': None})
        row.update(dict.fromkeys(feature_names))
        try:
            if system not in entity_cache:
                try:
                    meta = {'manifest_path': pose['manifest_path'], 'manifest_sha256': pose['manifest_sha256']}
                    _, entities, _ = roles.inspect_input(system, meta, max_bytes, args.short_protein_limit)
                    entity_cache[system] = entities
                except Exception as exc:
                    entity_cache[system] = ('error', f'{type(exc).__name__}: {exc}')
            cached = entity_cache[system]
            if isinstance(cached, tuple) and cached and cached[0] == 'error':
                raise ValueError(f'시스템 입력 점검 실패: {cached[1]}')
            entities = cached
            facts, e_rows, r_rows, details = audit.inspect_cif(
                Path(pose['input_path']), max_bytes, MAX_ATOMS,
                pose.get('file_size_bytes'), pose.get('file_mtime_ns'))
            row['cif_sha256'] = facts['input_sha256']
            observed = roles.get_observed(Path(pose['input_path']), facts, e_rows, max_bytes)
            matches = roles.match_entities(entities, observed)
            receptor, ligand = receptor_and_ligand(matches)
            reasons = []
            if len(ligand) == 0:
                reasons.append('no_matched_ligand')
            elif len(ligand) > 1:
                reasons.append('multiple_matched_ligands')
            elif ligand[0].get('observed_residue_groups') != 1:
                reasons.append('ligand_spans_multiple_residue_groups')
            if not receptor:
                reasons.append('no_clean_receptor')
            # 역할검토가 있으면 짧은 체인 등을 자동으로 넣거나 빼지 않고 보류합니다(05/07과 동일).
            if any(json.loads(m.get('role_review_flags_json') or '[]') for m in matches):
                reasons.append('receptor_role_review')
            if set(details['review_flags']) & roles.BLOCK_FLAGS:
                reasons.append('cif_block_flags')
            if details['struct_conn']:
                reasons.append('output_connections_need_review')
            if reasons:
                row['status'] = 'skipped'
                row['skip_reasons_json'] = audit.encoded(sorted(set(reasons)))
            else:
                atoms, _ = cf.read_atoms(pose['input_path'], facts['input_sha256'], max_bytes)
                receptor_rows, chains = [], []
                for m in receptor:
                    receptor_rows.extend(json.loads(m['heavy_atom_site_rows_1based_json']))
                    chains.append(m['observed_chain_id'])
                ligand_rows = json.loads(ligand[0]['heavy_atom_site_rows_1based_json'])
                if len(set(receptor_rows)) != len(receptor_rows):
                    raise ValueError('receptor_rows_overlap')
                if set(receptor_rows) & set(ligand_rows):
                    raise ValueError('receptor_ligand_rows_overlap')
                rec_xyz, rec_res = cf.pick_coords(atoms, receptor_rows)
                lig_xyz, _ = cf.pick_coords(atoms, ligand_rows)
                if not len(rec_xyz) or not len(lig_xyz):
                    raise ValueError('empty_receptor_or_ligand')
                row.update(cf.contacts(lig_xyz, rec_xyz, rec_res))
                row.update(status='computed',
                           receptor_chain_ids_json=audit.encoded(sorted(c for c in chains if c)),
                           ligand_chain=ligand[0]['observed_chain_id'],
                           receptor_heavy_atoms=len(rec_xyz), ligand_heavy_atoms=len(lig_xyz))
        except Exception as exc:
            row.update(status='error', error_message=f'{type(exc).__name__}: {exc}')
        row['elapsed_seconds'] = time.perf_counter() - begin
        status_counter[row['status']] += 1
        system_stat[system][row['status']] += 1
        if row['status'] == 'skipped':
            for reason in json.loads(row['skip_reasons_json']):
                skip_counter[reason] += 1
        rows_out.append(row)
        if number % 200 == 0 or number == len(poses):
            print(f'확대 접촉 feature: {number}/{len(poses)}개', flush=True)

    frame = pd.DataFrame(rows_out)
    if frame.duplicated(list(KEYS)).any():
        raise RuntimeError('pose 식별자 중복이 있습니다.')
    buf = io.BytesIO()
    frame.to_parquet(buf, engine='pyarrow', index=False, compression='snappy')
    with helper.checked_path(out / 'contact_features.parquet').open('xb') as handle:
        handle.write(buf.getvalue())
    restored = pd.read_parquet(out / 'contact_features.parquet', engine='pyarrow')
    pd.testing.assert_frame_equal(restored, frame, check_dtype=False)

    status_rows = [{'status': s, 'count': c} for s, c in sorted(status_counter.items())]
    audit.save_csv(out / 'pose_status.csv', status_rows, ('status', 'count'))
    skip_rows = [{'skip_reason': s, 'count': c} for s, c in sorted(skip_counter.items())]
    audit.save_csv(out / 'skip_reasons.csv', skip_rows or [{'skip_reason': None, 'count': 0}],
                   ('skip_reason', 'count'))
    system_rows = [{'system_id': s, **{k: stat.get(k, 0) for k in ('computed', 'skipped', 'error')}}
                   for s, stat in sorted(system_stat.items())]
    audit.save_csv(out / 'system_summary.csv', system_rows, ('system_id', 'computed', 'skipped', 'error'))

    changed = []
    for source in sources:
        try:
            info = Path(source['path']).stat()
            if (info.st_size, info.st_mtime_ns) != (source['size_bytes'], source['mtime_ns']):
                changed.append(source['path'])
        except OSError:
            changed.append(source['path'])
    elapsed = [r['elapsed_seconds'] for r in rows_out if r['elapsed_seconds'] is not None]
    try:
        import resource
        max_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        max_rss_mib = None
    computed = status_counter.get('computed', 0)
    errors = status_counter.get('error', 0)
    status = 'FEATURES_COMPLETE' if not errors and not changed else 'CHECK_REQUIRED'
    report = {
        'version': VERSION, 'status': status, 'output_dir': str(out),
        'index_run': str(args.index_run), 'selection_run': str(args.selection_run),
        'sources': sources, 'config_sha256': spec_hash,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'support_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (_support_path, cf._support_path, roles._support_path, audit._support_path)},
        'python': sys.executable, 'gemmi_version': gemmi.__version__, 'numpy_version': np.__version__,
        'pyarrow_version': pa.__version__, 'selected_systems': len(selected), 'poses': len(rows_out),
        'computed': computed, 'skipped': status_counter.get('skipped', 0), 'errors': errors,
        'status_counts': dict(status_counter), 'skip_reason_counts': dict(skip_counter),
        'changed_sources': changed, 'feature_category': 1,
        'elapsed_seconds': time.perf_counter() - started,
        'per_pose_seconds': {'mean': (sum(elapsed) / len(elapsed)) if elapsed else None,
                             'max': max(elapsed) if elapsed else None,
                             'min': min(elapsed) if elapsed else None},
        'max_rss_mib': max_rss_mib,
        'gt_used': False, 'coordinates_modified': False,
        'limitations': [
            '추출 안정성과 타깃 내 변동을 보기 위한 확대 검증이며 pose 품질·scoring 성능이 아닙니다.',
            '수용체는 역할검토 없는 매칭 단백질 체인만 포함하며 짧은·겹침 체인은 제외합니다.',
            '리간드 1개·수용체 존재·치명 플래그 없음이 아닌 pose는 계산하지 않고 사유만 기록합니다.',
            'sc_rmsd 등 GT 라벨은 읽지 않으며 분별력 분석은 별도 단계에서 합니다.',
            '최대 RSS는 참고값이며 공유 서버의 실제 자원 상한을 보장하지 않습니다.']}
    helper.write_json(out / 'audit_report.json', report)
    helper.write_json(out / 'RUN_COMPLETE.json', {'outputs_written': True, 'features_finished': True,
        'status': status, 'poses': len(rows_out), 'computed': computed, 'errors': errors,
        'sources_unchanged': not changed})
    print('STATUS:', status, flush=True)
    print('POSES:', len(rows_out), flush=True)
    print('COMPUTED:', computed, flush=True)
    print('SKIPPED:', status_counter.get('skipped', 0), flush=True)
    print('ERRORS:', errors, flush=True)
    print('SKIP_REASONS:', roles.j(dict(skip_counter)), flush=True)
    print('PER_POSE_SECONDS_MEAN:', report['per_pose_seconds']['mean'], flush=True)
    print('MAX_RSS_MIB:', max_rss_mib, flush=True)
    return 0 if status == 'FEATURES_COMPLETE' else 2


def main():
    """입력 위치와 상한을 지정하며 출력 위치는 사용자 경로로 고정합니다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index-run', type=Path, default=DEFAULT_INDEX)
    parser.add_argument('--selection-run', type=Path, default=DEFAULT_SELECTION)
    parser.add_argument('--max-cif-mib', type=int, default=128)
    parser.add_argument('--short-protein-limit', type=int, default=30)
    args = parser.parse_args()
    if args.max_cif_mib < 1 or args.short_protein_limit < 1:
        parser.error('상한은 양수여야 합니다.')
    return run(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
