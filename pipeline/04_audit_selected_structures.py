#!/usr/bin/env python3
"""선정한 예측 CIF의 입력 표현을 읽기 전용으로 점검합니다.

단백질 수용체와 펩타이드 리간드의 역할이나 리간드 복사본 수를 확정하지 않습니다.
구조를 수정하지 않고 좌표·식별자·entity·잔기·명시적 결합 정보의 상태를 기록합니다.
같은 폴더의 03_select_representative_systems.py에서 경로 보호·저장 함수를 재사용합니다.
"""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time

# 이전 단계의 경로 보호 함수를 재사용하며 해당 파일의 main은 실행하지 않습니다.
_support_path = Path(__file__).with_name('03_select_representative_systems.py')
_spec = importlib.util.spec_from_file_location('selection_support', _support_path)
helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helper)
DEFAULT_SELECTION = helper.ALLOWED_ROOT / 'Scoring_Function/out/representative_systems/03_run_select_representative_systems'
VERSION = 'structure_audit_0.1.0'
WATER_NAMES = {'HOH', 'DOD', 'WAT', 'H2O'}
IDENTIFIERS = ('system_id', 'model', 'seed', 'sample')
RESIDUE_FIELDS = ('pdbx_PDB_model_num', 'label_asym_id', 'auth_asym_id', 'label_entity_id',
                  'label_seq_id', 'auth_seq_id', 'pdbx_PDB_ins_code', 'label_comp_id')


def clean(value):
    """Gemmi가 해석한 두 종류의 CIF 결측값을 보고서에서 null로 정규화합니다."""
    return None if value is None or value is False else value


def rows(block, prefix, max_rows=None):
    """CIF 범주를 읽으며 열 길이와 행 수 상한을 확인합니다."""
    data = block.get_mmcif_category(prefix)
    if not data:
        return []
    sizes = {len(value) for value in data.values()}
    if len(sizes) != 1:
        raise ValueError(f'CIF 범주의 열 길이가 다릅니다: {prefix}')
    count = sizes.pop()
    if max_rows is not None and count > max_rows:
        raise ValueError(f'점검용 행 수 상한을 초과했습니다: {prefix}, {count}')
    return [{key: clean(value[i]) for key, value in data.items()} for i in range(count)]


def finite_number(value):
    """유효한 유한 실수만 반환합니다."""
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def encoded(value):
    """복합 정보를 CSV의 JSON 문자열 열로 저장합니다."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def entity_kind(entity_type, polymer_type, comp_types):
    """메타데이터가 지시하는 화학적 유형만 분류하며 생물학적 역할은 확정하지 않습니다."""
    if entity_type == 'polymer':
        if polymer_type and polymer_type.lower().startswith('polypeptide'):
            return 'polypeptide_declared'
        if polymer_type:
            return 'other_polymer_declared'
        if comp_types and all('peptide linking' in value for value in comp_types):
            return 'peptide_like_inferred'
        return 'polymer_type_unknown'
    return {'non-polymer': 'non_polymer', 'water': 'water_declared',
            'branched': 'branched'}.get(entity_type, 'entity_type_unknown')


def inspect_cif(path, max_bytes, max_atoms, expected_size=None, expected_mtime=None):
    """좌표와 식별자를 변경하지 않고 한 CIF의 구조적 구성을 요약합니다."""
    import gemmi
    with path.open('rb') as handle:
        before = os.fstat(handle.fileno())
        if expected_size is not None and before.st_size != expected_size:
            raise ValueError('목록 작성 이후 구조 파일 크기가 변경됐습니다.')
        if expected_mtime is not None and before.st_mtime_ns != expected_mtime:
            raise ValueError('목록 작성 이후 구조 파일 수정 시각이 변경됐습니다.')
        if before.st_size > max_bytes:
            raise ValueError('점검용 파일 크기 상한을 초과했습니다.')
        raw = handle.read(max_bytes + 1)
        after = os.fstat(handle.fileno())
    if len(raw) > max_bytes or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('읽는 동안 구조 파일이 변경됐거나 크기 상한을 초과했습니다.')
    block = gemmi.cif.read_string(raw.decode('utf-8')).sole_block()
    atoms = rows(block, '_atom_site.', max_atoms)
    if not atoms:
        raise ValueError('원자 좌표 표가 없습니다.')
    required = ('type_symbol', 'label_comp_id', 'label_atom_id', 'Cartn_x', 'Cartn_y', 'Cartn_z')
    if any(field not in atoms[0] for field in required):
        raise ValueError('좌표 분석에 필요한 원자 열이 없습니다.')
    categories = {name: rows(block, name) for name in (
        '_entity.', '_entity_poly.', '_struct_asym.', '_chem_comp.',
        '_chem_comp_atom.', '_chem_comp_bond.', '_struct_conn.')}
    entity_map = {r.get('id'): r for r in categories['_entity.']}
    polymer_map = {r.get('entity_id'): r for r in categories['_entity_poly.']}
    comp_map = {r.get('id'): r for r in categories['_chem_comp.']}
    asym_map = {r.get('id'): r.get('entity_id') for r in categories['_struct_asym.']}
    bond_counts = Counter(r.get('comp_id') for r in categories['_chem_comp_bond.'])
    comp_atom_counts = Counter(r.get('comp_id') for r in categories['_chem_comp_atom.'])
    element_cache = {}
    counters = Counter()
    groups = defaultdict(list)
    residue_groups = defaultdict(list)
    identities = Counter()
    for atom in atoms:
        symbol = str(atom.get('type_symbol') or '').upper()
        if symbol not in element_cache:
            element = gemmi.Element(symbol)
            element_cache[symbol] = (element.atomic_number, element.is_metal)
        atomic_number, metal = element_cache[symbol]
        atom['_heavy'] = atomic_number > 1
        atom['_metal'] = bool(metal)
        counters['unknown_element_atoms'] += atomic_number == 0
        counters['hydrogen_atoms'] += atomic_number == 1
        counters['heavy_atoms'] += atomic_number > 1
        counters['metal_atoms'] += bool(metal)
        counters['invalid_coordinate_atoms'] += any(finite_number(atom.get(k)) is None for k in ('Cartn_x', 'Cartn_y', 'Cartn_z'))
        occupancy = finite_number(atom.get('occupancy'))
        counters['missing_or_invalid_occupancy_atoms'] += occupancy is None
        counters['nonunit_occupancy_atoms'] += occupancy is not None and abs(occupancy - 1) > 1e-8
        counters['altloc_atoms'] += atom.get('label_alt_id') is not None
        counters['missing_model_id_atoms'] += atom.get('pdbx_PDB_model_num') is None
        counters['missing_chain_id_atoms'] += atom.get('label_asym_id') is None
        counters['missing_atom_name_atoms'] += atom.get('label_atom_id') is None
        counters['missing_comp_id_atoms'] += atom.get('label_comp_id') is None
        counters['unknown_entity_atoms'] += atom.get('label_entity_id') not in entity_map
        chain = atom.get('label_asym_id')
        counters['asym_entity_mismatch_atoms'] += chain in asym_map and asym_map[chain] != atom.get('label_entity_id')
        key = tuple(atom.get(k) for k in RESIDUE_FIELDS)
        identities[key + (atom.get('label_atom_id'), atom.get('label_alt_id'))] += 1
        residue_groups[key].append(atom)
        group = tuple(atom.get(k) for k in ('pdbx_PDB_model_num', 'label_asym_id', 'auth_asym_id', 'label_entity_id'))
        groups[group].append(atom)
    duplicate_rows = sum(count - 1 for count in identities.values() if count > 1)
    entity_records = []
    kinds = {}
    for key, group_atoms in groups.items():
        model, chain, auth_chain, entity_id = key
        entity = entity_map.get(entity_id, {})
        etype = str(entity.get('type') or '').lower()
        ptype = polymer_map.get(entity_id, {}).get('type')
        components = sorted({a.get('label_comp_id') for a in group_atoms if a.get('label_comp_id')})
        types = [str(comp_map.get(comp, {}).get('type') or '').lower() for comp in components]
        kind = entity_kind(etype, ptype, types)
        kinds[key] = kind
        record = {'coordinate_model': model, 'label_chain': chain, 'auth_chain': auth_chain,
                  'entity_id': entity_id, 'entity_type': etype, 'polymer_type': ptype,
                  'entity_kind': kind, 'atom_rows': len(group_atoms),
                  'heavy_atoms': sum(a['_heavy'] for a in group_atoms),
                  'metal_atoms': sum(a['_metal'] for a in group_atoms),
                  'observed_residue_groups': len({tuple(a.get(k) for k in RESIDUE_FIELDS) for a in group_atoms}),
                  'component_ids_json': encoded(components),
                  'biological_role': 'unresolved'}
        entity_records.append(record)
    nonpolymer_records = []
    water_groups = metal_only_groups = 0
    for number, (key, group_atoms) in enumerate(residue_groups.items(), start=1):
        first = group_atoms[0]
        group_key = tuple(first.get(k) for k in ('pdbx_PDB_model_num', 'label_asym_id', 'auth_asym_id', 'label_entity_id'))
        kind = kinds[group_key]
        comp = first.get('label_comp_id')
        heavy = [a for a in group_atoms if a['_heavy']]
        named_water = str(comp).upper() in WATER_NAMES and len(heavy) == 1 and str(heavy[0]['type_symbol']).upper() == 'O'
        is_water = kind == 'water_declared' or named_water
        # 금속 원소를 포함한 유기 리간드는 단원자 금속과 구분합니다.
        single_metal = kind == 'non_polymer' and len(heavy) == 1 and heavy[0]['_metal']
        water_groups += is_water
        metal_only_groups += single_metal
        if kind in ('polypeptide_declared', 'peptide_like_inferred', 'other_polymer_declared', 'polymer_type_unknown'):
            continue
        names = [a.get('label_atom_id') for a in group_atoms]
        elements = Counter(str(a.get('type_symbol') or '').upper() for a in group_atoms)
        nonpolymer_records.append({
            'residue_group_index': number, 'coordinate_model': first.get('pdbx_PDB_model_num'),
            'label_chain': first.get('label_asym_id'), 'auth_chain': first.get('auth_asym_id'),
            'entity_id': first.get('label_entity_id'), 'entity_kind': kind,
            'label_seq_id': first.get('label_seq_id'), 'auth_seq_id': first.get('auth_seq_id'),
            'insertion_code': first.get('pdbx_PDB_ins_code'), 'component_id': comp,
            'atom_rows': len(group_atoms), 'heavy_atoms': len(heavy),
            'metal_atoms': sum(a['_metal'] for a in group_atoms),
            'water_candidate': is_water, 'monatomic_metal_candidate': single_metal,
            'both_residue_numbers_missing': first.get('label_seq_id') is None and first.get('auth_seq_id') is None,
            'chem_comp_atom_rows': comp_atom_counts[comp], 'chem_comp_bond_rows': bond_counts[comp],
            'element_counts_json': encoded(dict(elements)), 'atom_names_json': encoded(names),
            'ligand_identity': 'unresolved'})
    missing_number_groups = sum(r['both_residue_numbers_missing'] for r in nonpolymer_records)
    models = {a.get('pdbx_PDB_model_num') for a in atoms}
    inferred_groups = sum(r['entity_kind'] == 'peptide_like_inferred' for r in entity_records)
    flags = []
    for name in ('unknown_element_atoms', 'invalid_coordinate_atoms', 'missing_or_invalid_occupancy_atoms',
                 'nonunit_occupancy_atoms', 'altloc_atoms', 'missing_model_id_atoms', 'missing_chain_id_atoms',
                 'missing_atom_name_atoms', 'missing_comp_id_atoms', 'unknown_entity_atoms', 'asym_entity_mismatch_atoms'):
        if counters[name]:
            flags.append(name)
    if len(models) != 1:
        flags.append('multiple_coordinate_models')
    if duplicate_rows:
        flags.append('duplicate_atom_identity_rows')
    if missing_number_groups:
        flags.append('nonpolymer_residue_numbers_missing')
    if inferred_groups:
        flags.append('polymer_type_inferred_from_components')
    if any(r['entity_kind'] in ('entity_type_unknown', 'polymer_type_unknown') for r in entity_records):
        flags.append('unresolved_entity_type')
    if not categories['_chem_comp_bond.']:
        flags.append('no_explicit_chem_comp_bonds')
    else:
        no_bonds = [r for r in nonpolymer_records if r['heavy_atoms'] > 1 and not r['water_candidate'] and not r['chem_comp_bond_rows']]
        if no_bonds:
            flags.append('some_nonpolymer_components_without_explicit_bonds')
    if any(r['heavy_atoms'] == 0 for r in nonpolymer_records):
        flags.append('nonpolymer_group_without_known_heavy_atoms')
    facts = {'input_sha256': hashlib.sha256(raw).hexdigest(), 'input_size_bytes': len(raw),
             'atom_rows': len(atoms), 'coordinate_model_count': len(models),
             'entity_chain_groups': len(entity_records), 'observed_residue_groups': len(residue_groups),
             'nonpolymer_residue_groups': len(nonpolymer_records), 'water_candidate_groups': water_groups,
             'monatomic_metal_candidate_groups': metal_only_groups,
             'duplicate_atom_identity_rows': duplicate_rows,
             'nonpolymer_groups_missing_residue_numbers': missing_number_groups,
             'peptide_like_inferred_groups': inferred_groups,
             'chem_comp_bond_rows': len(categories['_chem_comp_bond.']),
             'struct_conn_rows': len(categories['_struct_conn.']),
             'review_flags_json': encoded(flags), **dict(counters)}
    # 개별 화학 성분의 메타데이터와 명시적 연결 기록을 보존합니다.
    details = {'atom_site_columns': sorted(k for k in atoms[0] if not k.startswith('_')),
               'entities': categories['_entity.'], 'entity_polymers': categories['_entity_poly.'],
               'struct_asym': categories['_struct_asym.'], 'chemical_components': categories['_chem_comp.'],
               'chem_comp_bond_counts': {str(k): v for k, v in bond_counts.items()},
               'struct_conn': categories['_struct_conn.'], 'review_flags': flags,
               'notes': ['잔기 그룹은 식별자별 묶음이며 리간드 분자 또는 복사본 수를 확정한 값이 아닙니다.',
                         'polypeptide 유형만으로 수용체 단백질과 펩타이드 리간드를 구분하지 않습니다.',
                         '명시적 결합 기록이 없다고 실제 화학 결합이 없다는 뜻은 아닙니다.']}
    return facts, entity_records, nonpolymer_records, details


def save_csv(path, data, empty_fields):
    """오류 행에서 일부 열이 없더라도 공통 헤더를 만들어 저장합니다."""
    fields = list(dict.fromkeys(key for row in data for key in row)) or list(empty_fields)
    with helper.checked_path(path).open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)


def run(args):
    """선정 목록을 검증한 뒤 구조를 하나씩 읽고 점검 결과를 저장합니다."""
    import gemmi
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    started = time.perf_counter()
    paths = [args.selection_run / name for name in ('RUN_COMPLETE.json', 'selection_report.json',
                                                   'selected_systems.csv', 'selected_poses.parquet')]
    sources = [helper.fingerprint(path) for path in paths]
    complete = json.loads(paths[0].read_text())
    selection = json.loads(paths[1].read_text())
    if complete.get('validation_passed') is not True or complete.get('outputs_written') is not True:
        raise ValueError('선정 단계의 정상 완료 기록이 없습니다.')
    with paths[2].open(newline='', encoding='utf-8-sig') as handle:
        system_rows = list(csv.DictReader(handle))
    systems = {r['system_id']: r for r in system_rows}
    poses = pq.read_table(paths[3], use_threads=False).to_pylist()
    if len(systems) != len(system_rows) or len(systems) != complete.get('selected_systems') or len(poses) != complete.get('selected_poses'):
        raise ValueError('선정 결과의 시스템·pose 수가 완료 기록과 다릅니다.')
    if selection.get('validation_passed') is not True or len(poses) != 4 * len(systems):
        raise ValueError('시스템당 네 모델의 선정 기록이 필요합니다.')
    seen = set()
    for pose in poses:
        key = tuple(pose[k] for k in IDENTIFIERS)
        if key in seen or pose['system_id'] not in systems or pose.get('status') != 'ok':
            raise ValueError('선정 pose에 중복·알 수 없는 시스템·오류 상태가 있습니다.')
        if pose['model'] not in helper.MODELS or pose['seed'] != 1 or pose['sample'] != 0:
            raise ValueError('이번 점검은 모델별 seed 1, sample 0 기준 목록을 사용합니다.')
        seen.add(key)
    parent = helper.safe_mkdir(helper.ALLOWED_ROOT / 'Scoring_Function/out/structure_audit')
    out = helper.checked_path(parent / '04_run_audit_selected_structures')
    if out.exists():
        raise FileExistsError(f'기존 결과를 덮어쓰지 않습니다: {out}')
    out.mkdir()
    for name in ('tmp', 'cache'):
        helper.safe_mkdir(out / name)
    for key in ('TMPDIR', 'TMP', 'TEMP'):
        os.environ[key] = str(out / 'tmp')
    os.environ['XDG_CACHE_HOME'] = str(out / 'cache')
    print('OUTPUT_DIR:', out, flush=True)
    pose_rows, entity_rows, residue_rows = [], [], []
    manifest_cache = {}
    flags_by_model = Counter()
    with helper.checked_path(out / 'structure_details.jsonl').open('x', encoding='utf-8') as details_file:
        for position, pose in enumerate(poses, start=1):
            begin = time.perf_counter()
            ids = {key: pose[key] for key in IDENTIFIERS}
            meta = systems[pose['system_id']]
            record = {**ids, 'input_path': pose['input_path'], 'ligand_class': meta['ligand_class'],
                      'manifest_ligand_ccd': meta['manifest_ligand_ccd'],
                      'manifest_ligand_copies': meta['manifest_ligand_copies'],
                      'status': 'parsed', 'error_type': None, 'error_message': None}
            try:
                manifest_path = Path(pose['manifest_path'])
                if str(manifest_path) not in manifest_cache:
                    raw_manifest = manifest_path.read_bytes()
                    manifest_cache[str(manifest_path)] = (hashlib.sha256(raw_manifest).hexdigest(), json.loads(raw_manifest))
                digest, manifest = manifest_cache[str(manifest_path)]
                if digest != pose['manifest_sha256'] or digest != meta['manifest_sha256']:
                    raise ValueError('선정 이후 manifest 내용이 변경됐습니다.')
                if manifest.get('system_id') != pose['system_id']:
                    raise ValueError('manifest의 시스템 ID가 목록과 다릅니다.')
                facts, entities, residues, details = inspect_cif(Path(pose['input_path']),
                    args.max_input_mib * 1024 * 1024, args.max_atoms,
                    pose.get('file_size_bytes'), pose.get('file_mtime_ns'))
                record.update(facts)
                entity_rows.extend({**ids, **row} for row in entities)
                residue_rows.extend({**ids, **row} for row in residues)
                for flag in details['review_flags']:
                    flags_by_model[(pose['model'], flag)] += 1
                details_file.write(encoded({**ids, 'input_path': pose['input_path'], 'details': details}) + '\n')
            except Exception as exc:
                record.update(status='error', error_type=type(exc).__name__, error_message=str(exc))
                details_file.write(encoded({**ids, 'status': 'error', 'error_type': type(exc).__name__, 'error_message': str(exc)}) + '\n')
            record['elapsed_seconds'] = time.perf_counter() - begin
            pose_rows.append(record)
            if position % 20 == 0 or position == len(poses):
                print(f'구조 점검: {position}/{len(poses)}개', flush=True)
    save_csv(out / 'pose_audit.csv', pose_rows, IDENTIFIERS)
    save_csv(out / 'entity_audit.csv', entity_rows, IDENTIFIERS)
    save_csv(out / 'nonpolymer_residue_audit.csv', residue_rows, IDENTIFIERS)
    flag_rows = [{'model': model, 'review_flag': flag, 'pose_count': count}
                 for (model, flag), count in sorted(flags_by_model.items())]
    save_csv(out / 'review_summary.csv', flag_rows, ('model', 'review_flag', 'pose_count'))
    changed = []
    for source in sources:
        now = Path(source['path']).stat()
        if (now.st_size, now.st_mtime_ns) != (source['size_bytes'], source['mtime_ns']):
            changed.append(source['path'])
    errors = [row for row in pose_rows if row['status'] == 'error']
    review_count = sum(bool(json.loads(row.get('review_flags_json', '[]'))) for row in pose_rows)
    report = {'version': VERSION, 'sources': sources, 'output_dir': str(out),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'support_script_sha256': hashlib.sha256(_support_path.read_bytes()).hexdigest(),
              'python_executable': sys.executable, 'gemmi_version': gemmi.__version__, 'pyarrow_version': pa.__version__,
              'pose_count': len(poses), 'parsed': len(poses) - len(errors), 'errors': len(errors),
              'review_pose_count': review_count, 'review_summary': flag_rows, 'error_rows': errors,
              'selected_inputs_changed': changed, 'elapsed_seconds': time.perf_counter() - started,
              'max_input_mib': args.max_input_mib, 'max_atoms': args.max_atoms,
              'gt_used': False, 'coordinates_modified': False, 'ligand_assignment_completed': False,
              'notes': ['parsed는 파일을 읽고 요약했다는 뜻이며 구조의 타당성 또는 feature 준비 완료를 뜻하지 않습니다.',
                        '검토 표시는 입력 처리 정책을 정하기 위한 정보이며 곧바로 나쁜 pose라는 뜻이 아닙니다.',
                        '수소·대체 위치·여러 좌표 모델은 삭제하거나 선택하지 않고 그대로 집계합니다.',
                        '실제 리간드 분자와 복사본 수는 이 단계에서 확정하지 않습니다.']}
    helper.write_json(out / 'audit_report.json', report)
    helper.write_json(out / 'RUN_COMPLETE.json', {'outputs_written': True, 'audit_finished': True,
                      'parsed': report['parsed'], 'errors': len(errors), 'selected_inputs_unchanged': not changed})
    print('STATUS:', 'AUDIT_COMPLETE' if not errors and not changed else 'CHECK_REQUIRED', flush=True)
    print('PARSED:', report['parsed'], '/', len(poses), flush=True)
    print('ERRORS:', len(errors), flush=True)
    print('REVIEW_POSES:', review_count, flush=True)
    print('LIGAND_ASSIGNMENT: NOT_YET_DETERMINED', flush=True)
    return 0 if not errors and not changed else 2


def main():
    """점검 범위와 파일 크기 상한을 지정합니다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-run', type=Path, default=DEFAULT_SELECTION)
    parser.add_argument('--max-input-mib', type=int, default=128)
    parser.add_argument('--max-atoms', type=int, default=250000)
    args = parser.parse_args()
    if args.max_input_mib < 1 or args.max_atoms < 1:
        parser.error('크기 상한은 양수여야 합니다.')
    return run(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
