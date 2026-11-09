#!/usr/bin/env python3
"""자립형 통합 추출기(A6b·A6c 엔진). 저장된 대응 없이 pose마다 역할·원자 대응을 직접 계산합니다.

08번의 per-pose 역할 대응(inspect_cif→get_observed→match_entities)과 06번의 per-pose 원자 대응
(SMILES 템플릿↔출력 연결성 매칭)을 결합하고, 19번에서 표본 재현이 검증된 추출 조립부(cat1·2·3·4·
6/7·8)를 그대로 호출합니다. 따라서 선정에 들어가지 않은 임의 pose도 처리할 수 있습니다. category 5
(P2Rank)는 외부 바이너리라 제외하고 14번으로 별도 유지합니다. GT·좌표 수정 없음.

A6b 범위: 02번 전체 목록에서 지정 시스템(기본: 선정 40, 4000 pose)만 추출하여 비용·안정성·실패
패턴을 측정하고, 겹치는 seed1/sample0에서 19번 결과와 값 일치를 확인합니다.
출력: out/combined_full/20_run_combined_extractor_full/ 아래 고정 폴더.
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


def _load(name):
    p = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(name.replace('.py', ''), p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

ce = _load('19_combined_extractor.py')     # 추출 조립부 재사용
mp = _load('06_map_ligand_atoms.py')        # 원자 대응(연결성)
ex = _load('08_contact_features_expanded.py')  # per-pose 역할 대응 보조(receptor_and_ligand)
cf1 = ce.cf1
sf = ce.sf
it = ce.it
wm = ce.wm
lg = ce.lg
ld = ce.ld
roles = ce.roles
audit = ce.audit
helper = ce.helper

VERSION = 'combined_full_0.1.0'
OUT_KIND = 'combined_full'
OUT_NAME = '20_run_combined_extractor_full'
BASE = helper.ALLOWED_ROOT / 'Scoring_Function' / 'out'
DEFAULT_INDEX = helper.DEFAULT_INDEX               # 02번 전체 목록
DEFAULT_SELECTION = audit.DEFAULT_SELECTION        # 03번 선정(기본 대상 시스템)
DEFAULT_VALIDATE = BASE / 'combined_features' / '19_run_combined_extractor' / 'combined_features.parquet'
KEYS = ('system_id', 'model', 'seed', 'sample')
MAX_ATOMS = 250000


def map_pose(cif_path, size, mtime, entities, ligand_entity, max_bytes):
    """한 pose의 역할 대응·원자 대응을 계산해 수용체/리간드 행과 06 매핑을 돌려줍니다."""
    facts, e_rows, r_rows, details = audit.inspect_cif(Path(cif_path), max_bytes, MAX_ATOMS, size, mtime)
    observed = roles.get_observed(Path(cif_path), facts, e_rows, max_bytes)
    matches = roles.match_entities(entities, observed)
    receptor, ligand = ex.receptor_and_ligand(matches)
    reasons = []
    if len(ligand) == 0:
        reasons.append('no_matched_ligand')
    elif len(ligand) > 1:
        reasons.append('multiple_matched_ligands')
    elif ligand[0].get('observed_residue_groups') != 1:
        reasons.append('ligand_spans_multiple_residue_groups')
    if not receptor:
        reasons.append('no_clean_receptor')
    if any(json.loads(m.get('role_review_flags_json') or '[]') for m in matches):
        reasons.append('receptor_role_review')
    if set(details['review_flags']) & roles.BLOCK_FLAGS:
        reasons.append('cif_block_flags')
    if details['struct_conn']:
        reasons.append('output_connections_need_review')
    if reasons:
        return None, None, None, sorted(set(reasons)), facts['input_sha256']
    receptor_rows = []
    for m in receptor:
        receptor_rows.extend(json.loads(m['heavy_atom_site_rows_1based_json']))
    ligand_rows = json.loads(ligand[0]['heavy_atom_site_rows_1based_json'])
    # 06 원자 대응.
    mapping_entry = {'ligand_status': 'no_template', 'atom_mapping': None}
    smiles = ligand_entity.get('canonical_smiles') or ligand_entity.get('smiles') if ligand_entity else None
    if smiles:
        try:
            extracted = mp.extract_ligand_atoms(cif_path, facts['input_sha256'], ligand_rows,
                                                ligand[0].get('observed_chain_id'), max_bytes)
            extracted['rows_1based'] = ligand_rows
            # 대칭 심한 리간드의 조합 폭발을 막기 위해 매칭 열거 상한을 둡니다(값·분류 불변).
            status06, _detail06, atom_mapping = mp.map_one({'canonical_smiles': smiles}, extracted, max_matches=64)
            mapping_entry = {'ligand_status': status06.split(':')[0], 'atom_mapping': atom_mapping}
        except Exception as exc:
            mapping_entry = {'ligand_status': f'map_error:{type(exc).__name__}', 'atom_mapping': None}
    return receptor_rows, ligand_rows, mapping_entry, [], facts['input_sha256']


def run(args):
    """지정 시스템의 전체 pose에 자립형 통합 추출을 적용합니다."""
    started = time.perf_counter()
    parent = helper.safe_mkdir(BASE / OUT_KIND)
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

    import numpy as np
    import pandas as pd
    import pyarrow as pa
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    unit_points = sf.fibonacci_sphere(sf.SPHERE_POINTS)

    index_complete = json.loads((args.index_run / 'RUN_COMPLETE.json').read_text(encoding='utf-8'))
    if index_complete.get('validation_passed') is not True:
        raise ValueError('02번 전체 pose 목록이 정상 완료된 결과가 아닙니다.')
    with (args.selection_run / 'selected_systems.csv').open(newline='', encoding='utf-8-sig') as handle:
        target_systems = {r['system_id'] for r in csv.DictReader(handle)}
    # 02번에서 지정 시스템의 ok pose만 읽습니다.
    import pyarrow.parquet as pq
    cols = [*KEYS, 'input_path', 'file_size_bytes', 'file_mtime_ns', 'manifest_path', 'manifest_sha256', 'status']
    poses = []
    for batch in pq.ParquetFile(args.index_run / 'pose_index.parquet').iter_batches(batch_size=20000, columns=cols, use_threads=False):
        for r in batch.to_pylist():
            if r['system_id'] in target_systems and r['status'] == 'ok':
                poses.append(r)
    poses.sort(key=lambda p: tuple(p[k] for k in KEYS))

    # 리간드 입력 엔티티(05 input_entities에서 시스템별 SMILES). 없으면 06 매핑은 생략됩니다.
    ligand_entity = {}
    ie = args.roles_run / 'input_entities.csv'
    if ie.exists():
        with ie.open(newline='', encoding='utf-8') as handle:
            for r in csv.DictReader(handle):
                if r.get('input_kind') == 'ligand':
                    ligand_entity.setdefault(r['system_id'], r)

    feature_names = []
    for _c, _p, f in ce.VALIDATE:
        feature_names += f
    max_bytes = args.max_cif_mib * 1024 ** 2
    entity_cache = {}
    rows_out = []
    status_counter = Counter()
    skip_counter = Counter()
    for number, pose in enumerate(poses, 1):
        system = pose['system_id']
        begin = time.perf_counter()
        row = {k: pose[k] for k in KEYS}
        row.update({'pose_id': f"{system}|{pose['model']}|{pose['seed']}|{pose['sample']}",
                    'schema_version': VERSION, 'cif_sha256': None, 'ligand_typing': None,
                    'cat4_available': None, 'ligand_mapping_status': None,
                    'status': None, 'skip_reasons_json': '[]', 'error_message': None, 'elapsed_seconds': None})
        row.update(dict.fromkeys(feature_names))
        try:
            # 시스템 입력 엔티티(역할 대응용) 1회 계산.
            if system not in entity_cache:
                try:
                    meta = {'manifest_path': pose['manifest_path'], 'manifest_sha256': pose['manifest_sha256']}
                    _rowx, ents, _detx = roles.inspect_input(system, meta, max_bytes, 30)
                    entity_cache[system] = ents
                except Exception as exc:
                    entity_cache[system] = ('error', f'{type(exc).__name__}: {exc}')
            ents = entity_cache[system]
            smiles = (ligand_entity.get(system) or {}).get('canonical_smiles') or (ligand_entity.get(system) or {}).get('smiles')
            if smiles:
                try:
                    row.update(ld.descriptors((smiles or '').strip()))
                except Exception:
                    pass
            if isinstance(ents, tuple) and ents and ents[0] == 'error':
                raise ValueError(f'시스템 입력 점검 실패: {ents[1]}')
            rec_rows, lig_rows, mapping_entry, reasons, cif_sha = map_pose(
                pose['input_path'], pose.get('file_size_bytes'), pose.get('file_mtime_ns'),
                ents, ligand_entity.get(system), max_bytes)
            row['cif_sha256'] = cif_sha
            if reasons:
                row['status'] = 'skipped'
                row['skip_reasons_json'] = audit.encoded(reasons)
                for rsn in reasons:
                    skip_counter[rsn] += 1
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
        row['elapsed_seconds'] = time.perf_counter() - begin
        status_counter[row['status']] += 1
        rows_out.append(row)
        if number % 200 == 0 or number == len(poses):
            print(f'자립형 통합 추출: {number}/{len(poses)}개', flush=True)

    frame = pd.DataFrame(rows_out)
    buf = io.BytesIO()
    frame.to_parquet(buf, engine='pyarrow', index=False, compression='snappy')
    with helper.checked_path(out / 'combined_features.parquet').open('xb') as handle:
        handle.write(buf.getvalue())

    # 겹치는 seed1/sample0에서 19번(A6a)과 값 일치 확인.
    val = {'checked': 0, 'mismatches': 0, 'available': False}
    if Path(args.validate_against).exists():
        ref = pd.read_parquet(args.validate_against)
        for k in ('seed', 'sample'):
            ref[k] = ref[k].astype(int)
        ref['system_id'] = ref['system_id'].astype(str); ref['model'] = ref['model'].astype(str)
        ref = ref.set_index(list(KEYS))
        cur = frame.copy()
        cur['system_id'] = cur['system_id'].astype(str); cur['model'] = cur['model'].astype(str)
        cur = cur.set_index(list(KEYS))
        common = cur.index.intersection(ref.index)
        val['available'] = True
        val['checked'] = int(len(common))
        for f in feature_names:
            if f in cur.columns and f in ref.columns:
                a = cur.loc[common, f]; b = ref.loc[common, f]
                both_null = a.isna() & b.isna()
                if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                    bad = int((((a.fillna(0) - b.fillna(0)).abs() > 1e-6) & ~both_null).sum())
                else:
                    bad = int(((a.astype('object') != b.astype('object')) & ~both_null).sum())
                val['mismatches'] += bad

    status_rows = [{'status': s, 'count': c} for s, c in sorted(status_counter.items())]
    audit.save_csv(out / 'pose_status.csv', status_rows, ('status', 'count'))
    skip_rows = [{'skip_reason': s, 'count': c} for s, c in sorted(skip_counter.items())]
    audit.save_csv(out / 'skip_reasons.csv', skip_rows or [{'skip_reason': None, 'count': 0}], ('skip_reason', 'count'))

    elapsed = [r['elapsed_seconds'] for r in rows_out if r['elapsed_seconds'] is not None]
    try:
        import resource
        max_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        max_rss_mib = None
    computed = status_counter.get('computed', 0)
    errors = status_counter.get('error', 0)
    status = 'SCALE_TEST_COMPLETE' if not errors else 'CHECK_REQUIRED'
    report = {
        'version': VERSION, 'status': status, 'output_dir': str(out),
        'index_run': str(args.index_run), 'selection_run': str(args.selection_run), 'roles_run': str(args.roles_run),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'python': sys.executable, 'target_systems': len(target_systems), 'poses': len(rows_out),
        'computed': computed, 'skipped': status_counter.get('skipped', 0), 'errors': errors,
        'status_counts': dict(status_counter), 'skip_reason_counts': dict(skip_counter),
        'validation_vs_19': val,
        'per_pose_seconds': {'mean': (sum(elapsed) / len(elapsed)) if elapsed else None,
                             'max': max(elapsed) if elapsed else None, 'min': min(elapsed) if elapsed else None},
        'max_rss_mib': max_rss_mib, 'elapsed_seconds': time.perf_counter() - started,
        'gt_used': False, 'coordinates_modified': False,
        'notes': ['저장 대응 없이 pose마다 역할·원자 대응을 직접 계산합니다(A6c 엔진).',
                  'CIF 읽기는 pose당 inspect_cif·get_observed·원자추출·read_atoms로 여러 번이며, 전체 확대 시 최적화 여지가 있습니다.',
                  'cat5(P2Rank) 제외. 전체 확대(A6c)는 이 스크립트를 array sharding+resume으로 돌립니다.']}
    helper.write_json(out / 'combined_report.json', report)
    helper.write_json(out / 'RUN_COMPLETE.json', {'outputs_written': True, 'extraction_finished': True,
        'status': status, 'poses': len(rows_out), 'computed': computed, 'errors': errors,
        'validation_vs_19_mismatches': val['mismatches']})
    print('STATUS:', status, flush=True)
    print('POSES:', len(rows_out), '| COMPUTED:', computed, '| SKIPPED:', status_counter.get('skipped', 0), '| ERRORS:', errors, flush=True)
    print('SKIP_REASONS:', roles.j(dict(skip_counter)), flush=True)
    print('VALIDATION_vs_19: checked', val['checked'], 'mismatches', val['mismatches'], flush=True)
    print('PER_POSE_SEC_MEAN:', report['per_pose_seconds']['mean'], '| MAX_RSS_MIB:', max_rss_mib, flush=True)
    return 0 if status == 'SCALE_TEST_COMPLETE' and val['mismatches'] == 0 else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index-run', type=Path, default=DEFAULT_INDEX)
    parser.add_argument('--selection-run', type=Path, default=DEFAULT_SELECTION)
    parser.add_argument('--roles-run', type=Path, default=helper.ALLOWED_ROOT / 'Scoring_Function' / 'out' / 'input_roles' / '05_run_audit_input_roles')
    parser.add_argument('--validate-against', type=Path, default=DEFAULT_VALIDATE)
    parser.add_argument('--max-cif-mib', type=int, default=128)
    args = parser.parse_args()
    if args.max_cif_mib < 1:
        parser.error('크기 상한은 양수여야 합니다.')
    return run(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
