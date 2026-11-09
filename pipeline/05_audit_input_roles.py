#!/usr/bin/env python3
"""전체 생성 입력과 선정된 출력 구조의 역할 대응 근거를 기록합니다.

원본 구조·공유 환경·GT 평가 파일은 수정하거나 실행하지 않습니다.
화학 결합의 원자 대응, 생물학적 역할, 최종 feature 준비 완료를 확정하지 않습니다.
같은 폴더에 있는 03번과 04번 Python 파일의 읽기·경로 보호 함수를 재사용합니다.
"""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

_support_path = Path(__file__).with_name('04_audit_selected_structures.py')
_spec = importlib.util.spec_from_file_location('audit_support', _support_path)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)
helper = audit.helper
VERSION = 'input_roles_0.1.0'
OUT_NAME = '05_run_audit_input_roles'
KEYS = ('system_id', 'model', 'seed', 'sample')
AA = dict(zip(
    'ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split(),
    'ARNDCQEGHILKMFPSTWYV'))
# 이 표는 화학적 동일성 판정이 아니라 추가 검토 후보를 찾는 데만 사용합니다.
REVIEW_AA = {**AA, 'TPO': 'T', 'SEP': 'S', 'PTR': 'Y', 'MSE': 'M'}
BLOCK_FLAGS = {
    'unknown_element_atoms', 'invalid_coordinate_atoms',
    'missing_or_invalid_occupancy_atoms', 'nonunit_occupancy_atoms', 'altloc_atoms',
    'missing_model_id_atoms', 'missing_chain_id_atoms', 'missing_atom_name_atoms',
    'missing_comp_id_atoms', 'unknown_entity_atoms', 'asym_entity_mismatch_atoms',
    'multiple_coordinate_models', 'duplicate_atom_identity_rows', 'unresolved_entity_type',
}


def j(value):
    """복합 값을 CSV 안에 저장하기 위해 JSON 문자열로 변환합니다."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def read_bytes(path, limit):
    """상한 이하의 파일만 읽고 읽는 동안 변경되지 않았는지 확인합니다."""
    with Path(path).open('rb') as handle:
        before = os.fstat(handle.fileno())
        if before.st_size > limit:
            raise ValueError(f'읽기 크기 상한 초과: {path}')
        raw = handle.read(limit + 1)
        after = os.fstat(handle.fileno())
    if len(raw) > limit or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'읽는 동안 파일 변경 또는 크기 상한 초과: {path}')
    return raw, {'path': str(path), 'size_bytes': after.st_size,
                 'mtime_ns': after.st_mtime_ns, 'sha256': hashlib.sha256(raw).hexdigest()}


def safe_component(value):
    """메타데이터 값으로 상위 디렉터리나 다른 파일을 탐색하지 못하게 합니다."""
    if not isinstance(value, str) or not value or value in ('.', '..') or '/' in value or '\\' in value:
        raise ValueError(f'경로 구성에 사용할 수 없는 값: {value!r}')
    return value


def smiles_info(smiles):
    """입력 SMILES의 조성만 계산하며 좌표를 만들거나 결합을 출력 원자에 배정하지 않습니다."""
    from rdkit import Chem
    if not isinstance(smiles, str) or not smiles:
        return {'chemistry_status': 'smiles_unavailable', 'heavy_elements_json': None,
                'heavy_atoms': None, 'fragment_count': None, 'canonical_smiles': None}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        return {'chemistry_status': 'smiles_invalid_or_unspecified_atoms',
                'heavy_elements_json': None, 'heavy_atoms': None,
                'fragment_count': None, 'canonical_smiles': None}
    counts = Counter(a.GetSymbol().upper() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    return {'chemistry_status': 'input_graph_parsed', 'heavy_elements_json': j(dict(counts)),
            'heavy_atoms': sum(counts.values()), 'fragment_count': len(Chem.GetMolFrags(mol)),
            'canonical_smiles': Chem.MolToSmiles(mol)}


def parse_input(data, system):
    """AF3 형식의 단일·복수 ID를 인스턴스별로 펼치고 지원 밖 구문은 거부합니다."""
    if not isinstance(data, dict) or data.get('dialect') != 'alphafold3':
        raise ValueError('지원하는 입력은 dialect=alphafold3인 객체입니다.')
    sequences = data.get('sequences')
    if not isinstance(sequences, list) or not sequences:
        raise ValueError('sequences 목록이 없거나 비어 있습니다.')
    entities, seen = [], set()
    for number, item in enumerate(sequences):
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError('sequences 항목에 분자 종류가 하나만 있어야 합니다.')
        kind, value = next(iter(item.items()))
        if kind not in ('protein', 'ligand', 'dna', 'rna') or not isinstance(value, dict):
            raise ValueError(f'지원하지 않는 입력 분자 구문: {kind}')
        ids = value.get('id')
        ids = [ids] if isinstance(ids, str) else ids
        if not isinstance(ids, list) or not ids or any(not isinstance(i, str) or not i for i in ids):
            raise ValueError('입력 ID는 문자열 또는 비어 있지 않은 문자열 목록이어야 합니다.')
        # 별도 개수 필드의 의미를 추측하여 복사본 수를 확정하지 않습니다.
        if any(k in value for k in ('count', 'copies', 'numCopies')):
            raise ValueError('ID 외 복사본 수 지정 구문은 별도 확인이 필요합니다.')
        seq = value.get('sequence', '')
        if kind in ('protein', 'dna', 'rna') and (not isinstance(seq, str) or not seq):
            raise ValueError('폴리머 입력 서열이 없습니다.')
        chem = smiles_info(value.get('smiles')) if kind == 'ligand' else {}
        if kind == 'ligand' and not value.get('smiles') and not value.get('ccdCodes'):
            raise ValueError('리간드 SMILES와 CCD 정의가 모두 없습니다.')
        for chain in ids:
            if chain in seen:
                raise ValueError(f'입력 체인 ID 중복: {chain}')
            seen.add(chain)
            entities.append({
                'system_id': system, 'input_entry_index': number, 'input_chain_id': chain,
                'input_kind': kind, 'sequence': seq, 'sequence_length': len(seq),
                'smiles': value.get('smiles'), 'ccd_codes_json': j(value.get('ccdCodes', [])),
                'description': value.get('description', ''),
                'modifications_json': j(value.get('modifications', [])),
                'input_fields_json': j(sorted(value)), 'role_review_flags_json': '[]', **chem})
    return entities


def overlap_candidate(sequence, description):
    """리간드 설명과 서열의 겹침 가능성만 찾고 동일 분자라고 확정하지 않습니다."""
    if not isinstance(description, str) or '-' not in description:
        return False
    tokens = description.upper().split('-')
    if len(tokens) != len(sequence):
        return False
    template = ''.join(REVIEW_AA.get(token, 'X') for token in tokens)
    known = [(a, b) for a, b in zip(sequence.upper(), template) if a != 'X' and b != 'X']
    return len(known) >= 3 and all(a == b for a, b in known)


def inspect_input(system, meta, max_bytes, short_limit):
    """manifest와 생성 입력을 비교하고 모호한 역할을 별도로 표시합니다."""
    manifest_path = Path(meta['manifest_path'])
    raw, manifest_fp = read_bytes(manifest_path, max_bytes)
    if manifest_fp['sha256'] != meta['manifest_sha256']:
        raise ValueError('전체 목록 작성 이후 manifest 내용이 변경됐습니다.')
    manifest = json.loads(raw)
    if manifest.get('system_id') != system:
        raise ValueError('manifest의 system_id가 전체 목록과 다릅니다.')
    variant = safe_component(manifest['sto']) + '+' + safe_component(manifest['ligand_ccd'])
    input_path = manifest_path.parent / 'inputs' / variant / 'af3_msa' / 'input.json'
    raw, input_fp = read_bytes(input_path, max_bytes)
    data = json.loads(raw)
    entities = parse_input(data, system)
    proteins = [e for e in entities if e['input_kind'] == 'protein']
    ligands = [e for e in entities if e['input_kind'] == 'ligand']
    flags = []
    count = manifest.get('ligand_copies')
    try:
        if isinstance(count, bool) or float(count) < 0 or not float(count).is_integer():
            raise ValueError()
        count = int(count)
    except (ValueError, TypeError, OverflowError):
        count = None
        flags.append('manifest_copy_count_invalid')
    if count is not None and count != len(ligands):
        flags.append('manifest_input_copy_mismatch')
    if not ligands:
        flags.append('no_input_ligand')
    if len(ligands) > 1:
        flags.append('multiple_input_ligands_need_instance_policy')
    if not proteins:
        flags.append('no_input_protein')
    if any(e['input_kind'] not in ('protein', 'ligand') for e in entities):
        flags.append('other_input_polymer_role_review')
    for protein in proteins:
        pflags = []
        # 길이 기준은 검토 대상을 넓게 찾기 위한 값이며 리간드 판정 기준이 아닙니다.
        if protein['sequence_length'] <= short_limit:
            pflags.append('short_protein_role_review')
        descriptions = [e['description'] or manifest.get('ligand_ccd', '') for e in ligands]
        if any(overlap_candidate(protein['sequence'], desc) for desc in descriptions):
            pflags.append('protein_ligand_sequence_overlap_candidate')
        if json.loads(protein['modifications_json']):
            pflags.append('input_protein_modifications_need_review')
        protein['role_review_flags_json'] = j(pflags)
        flags.extend(pflags)
    if any(e.get('chemistry_status') != 'input_graph_parsed' for e in ligands):
        flags.append('ligand_composition_unavailable')
    if any(e.get('fragment_count', 0) and e['fragment_count'] > 1 for e in ligands):
        flags.append('disconnected_ligand_input')
    if data.get('bondedAtomPairs'):
        flags.append('explicit_input_crosslinks_need_review')
    row = {'system_id': system, 'status': 'parsed', 'manifest_path': str(manifest_path),
           'manifest_sha256': manifest_fp['sha256'], 'input_json_path': str(input_path),
           'input_json_sha256': input_fp['sha256'], 'input_version': data.get('version'),
           'manifest_ligand_copies': count, 'input_ligand_copies': len(ligands),
           'input_protein_copies': len(proteins), 'manifest_ligand_ccd': manifest.get('ligand_ccd'),
           'copy_count_match': count == len(ligands) if count is not None else None,
           'role_review_protein_ids_json': j([e['input_chain_id'] for e in proteins if json.loads(e['role_review_flags_json'])]),
           'review_flags_json': j(sorted(set(flags))), 'error_message': None}
    detail = {'system_id': system, 'manifest_source': manifest_fp, 'input_source': input_fp,
              'input_dialect': data.get('dialect'), 'input_version': data.get('version'),
              'input_top_level_keys': sorted(data), 'input_model_seeds': data.get('modelSeeds'),
              'bonded_atom_pairs': data.get('bondedAtomPairs'),
              'gen_input_flags': manifest.get('gen_input_flags'),
              'input_role_definition': 'protein/ligand 등 입력 구분이며 생물학적 역할을 확정하지 않음'}
    return row, entities, detail


def get_observed(path, facts, entity_rows, max_bytes):
    """출력 원자의 행 순서와 원래 식별자를 보존하며 체인별 좌표 집합을 요약합니다."""
    import gemmi
    raw, fp = read_bytes(path, max_bytes)
    if fp['sha256'] != facts['input_sha256']:
        raise ValueError('CIF 점검 중 파일 내용이 변경됐습니다.')
    block = gemmi.cif.read_string(raw.decode('utf-8')).sole_block()
    atoms = audit.rows(block, '_atom_site.', 250000)
    groups = defaultdict(list)
    for i, atom in enumerate(atoms):
        key = tuple(atom.get(k) for k in ('pdbx_PDB_model_num', 'label_asym_id', 'auth_asym_id', 'label_entity_id'))
        groups[key].append((i + 1, atom))
    result = []
    for entity in entity_rows:
        key = tuple(entity[k] for k in ('coordinate_model', 'label_chain', 'auth_chain', 'entity_id'))
        indexed = groups[key]
        residues = {}
        heavy = Counter()
        atom_indices = []
        for i, atom in indexed:
            residues.setdefault(tuple(atom.get(k) for k in audit.RESIDUE_FIELDS), atom.get('label_comp_id'))
            element = gemmi.Element(str(atom.get('type_symbol') or ''))
            if element.atomic_number > 1:
                heavy[element.name.upper()] += 1
                atom_indices.append(i)
        seq = ''.join(AA.get(str(comp).upper(), 'X') for comp in residues.values())
        result.append({**entity, 'observed_sequence': seq,
                       'heavy_elements_json': j(dict(heavy)),
                       'heavy_atom_site_rows_1based_json': j(atom_indices),
                       'nonstandard_residue_count': sum(str(comp).upper() not in AA for comp in residues.values())})
    return result


def match_entities(entities, observed):
    """ID와 서열 또는 조성의 일치 근거를 기록하며 화학적 원자 대응은 완료하지 않습니다."""
    matches = []
    for entity in entities:
        kind = entity['input_kind']
        same_id = [o for o in observed if o['label_chain'] == entity['input_chain_id']]
        compatible = []
        for o in observed:
            if kind == 'protein':
                good = o['entity_kind'] in ('polypeptide_declared', 'peptide_like_inferred')
                good = good and o['observed_sequence'] == entity['sequence']
            elif kind == 'ligand' and entity.get('heavy_elements_json'):
                good = (o['entity_kind'] == 'non_polymer'
                        and o['heavy_elements_json'] == entity['heavy_elements_json'])
            else:
                good = False
            if good:
                compatible.append(o)
        direct = [o for o in compatible if o in same_id]
        chosen = direct[0] if len(direct) == 1 else None
        if chosen:
            state = 'id_sequence_consistent' if kind == 'protein' else 'id_composition_consistent'
        elif len(compatible) == 1:
            state = 'unique_candidate_id_unconfirmed'
        elif len(compatible) > 1:
            state = 'ambiguous_candidates'
        elif same_id:
            state = 'id_found_content_unconfirmed'
        else:
            state = 'no_supported_candidate'
        matches.append({
            'input_chain_id': entity['input_chain_id'], 'input_kind': kind,
            'match_status': state, 'observed_chain_id': chosen['label_chain'] if chosen else None,
            'compatible_chain_ids_json': j([o['label_chain'] for o in compatible]),
            'same_id_found': bool(same_id),
            'observed_residue_groups': chosen['observed_residue_groups'] if chosen else None,
            'observed_heavy_atoms': chosen['heavy_atoms'] if chosen else None,
            'heavy_atom_site_rows_1based_json': chosen['heavy_atom_site_rows_1based_json'] if chosen else None,
            'chemical_atom_mapping_status': 'not_performed',
            'role_review_flags_json': entity['role_review_flags_json']})
    # 동일 출력 체인이 둘 이상의 입력에 배정되는 경우에는 어느 쪽도 확정하지 않습니다.
    assigned = Counter(m['observed_chain_id'] for m in matches if m['observed_chain_id'])
    for m in matches:
        if m['observed_chain_id'] and assigned[m['observed_chain_id']] > 1:
            m.update(match_status='ambiguous_reused_output_chain', observed_chain_id=None,
                     heavy_atom_site_rows_1based_json=None)
    return matches


def load_index(index_dir, expected):
    """전체 목록에서 식별자와 manifest 정보만 읽고 100개 조합을 재확인합니다."""
    import pyarrow.parquet as pq
    complete = json.loads((index_dir / 'RUN_COMPLETE.json').read_text())
    report = json.loads((index_dir / 'run_report.json').read_text())
    if complete.get('validation_passed') is not True or report.get('scope') != 'full':
        raise ValueError('정상 완료된 전체 pose 목록이 필요합니다.')
    metadata, masks = {}, defaultdict(int)
    count = 0
    columns = [*KEYS, 'manifest_path', 'manifest_sha256', 'status']
    for batch in pq.ParquetFile(index_dir / 'pose_index.parquet').iter_batches(
            batch_size=10000, columns=columns, use_threads=False):
        for row in batch.to_pylist():
            system = row['system_id']
            safe_component(system)
            combo = (row['model'], row['seed'], row['sample'])
            if row['status'] != 'ok' or combo not in helper.COMBO_INDEX:
                raise ValueError('전체 목록에 잘못된 조합 또는 오류 상태가 있습니다.')
            bit = 1 << helper.COMBO_INDEX[combo]
            if masks[system] & bit:
                raise ValueError('전체 목록에 pose 식별자 중복이 있습니다.')
            masks[system] |= bit
            meta = {k: row[k] for k in ('manifest_path', 'manifest_sha256')}
            if system in metadata and metadata[system] != meta:
                raise ValueError('동일 시스템의 manifest 정보가 일관되지 않습니다.')
            metadata[system] = meta
            count += 1
    if (len(metadata) != expected or count != expected * 100
            or count != complete.get('indexed_rows') or count != report.get('indexed_rows')
            or any(v != (1 << 100) - 1 for v in masks.values())):
        raise ValueError('전체 시스템 수 또는 시스템당 100개 pose 조합이 예상과 다릅니다.')
    return metadata


def run(args):
    """전체 입력 목록과 선정 pose의 대응 근거를 고정된 결과 폴더에 저장합니다."""
    started = time.perf_counter()
    out = helper.checked_path(helper.ALLOWED_ROOT / 'Scoring_Function/out/input_roles' / OUT_NAME)
    if out.exists():
        raise FileExistsError(f'기존 결과를 덮어쓰지 않습니다: {out}')
    helper.safe_mkdir(out)
    for name in ('tmp', 'cache'):
        helper.safe_mkdir(out / name)
    for key in ('TMPDIR', 'TMP', 'TEMP'):
        os.environ[key] = str(out / 'tmp')
    os.environ['XDG_CACHE_HOME'] = str(out / 'cache')
    print('OUTPUT_DIR:', out, flush=True)
    import gemmi
    import rdkit
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    sources = [helper.fingerprint(path) for path in (
        args.index_run / 'RUN_COMPLETE.json', args.index_run / 'run_report.json',
        args.index_run / 'pose_index.parquet', args.selection_run / 'RUN_COMPLETE.json',
        args.selection_run / 'selected_poses.parquet')]
    metadata = load_index(args.index_run, args.expected_systems)
    complete = json.loads((args.selection_run / 'RUN_COMPLETE.json').read_text())
    poses = pq.read_table(args.selection_run / 'selected_poses.parquet', use_threads=False).to_pylist()
    seen = {tuple(p[k] for k in KEYS) for p in poses}
    selected = {p['system_id'] for p in poses}
    expected_keys = {(s, m, 1, 0) for s in selected for m in helper.MODELS}
    if (complete.get('validation_passed') is not True or complete.get('outputs_written') is not True
            or len(selected) != args.expected_selected or len(poses) != len(seen)
            or seen != expected_keys or not selected <= metadata.keys()
            or complete.get('selected_systems') != len(selected)
            or complete.get('selected_poses') != len(poses)):
        raise ValueError('선정 목록의 시스템 수·중복·네 모델 조합을 확인하세요.')
    for p in poses:
        if p.get('status') != 'ok' or any(p[k] != metadata[p['system_id']][k] for k in ('manifest_path', 'manifest_sha256')):
            raise ValueError('선정 목록과 전체 목록의 manifest 정보 또는 상태가 다릅니다.')
    system_rows, entity_rows, cache, details, input_sources = [], [], {}, [], []
    for number, (system, meta) in enumerate(sorted(metadata.items()), 1):
        try:
            row, entities, detail = inspect_input(system, meta, args.max_input_mib * 1024**2, args.short_protein_limit)
            entity_rows.extend(entities)
            details.append(detail)
            input_sources.extend([detail['manifest_source'], detail['input_source']])
            if system in selected:
                cache[system] = (row, entities)
        except Exception as exc:
            row = {'system_id': system, 'status': 'error', 'review_flags_json': '[]',
                   'error_message': f'{type(exc).__name__}: {exc}'}
        system_rows.append(row)
        if number % 250 == 0 or number == len(metadata):
            print(f'입력 점검: {number}/{len(metadata)}개', flush=True)
    pose_rows, mapping_rows, observed_rows = [], [], []
    for number, pose in enumerate(poses, 1):
        ids = {k: pose[k] for k in KEYS}
        record = {**ids, 'input_path': pose['input_path'], 'status': 'audited',
                  'observed_ligand_copies': None, 'ligand_copy_status': 'not_chemically_resolved',
                  'chemical_atom_mapping_status': 'not_performed',
                  'pair_geometry_status': 'hold', 'error_message': None}
        try:
            if pose['system_id'] not in cache:
                raise ValueError('이 시스템의 입력 정의 점검에 실패했습니다.')
            system_row, entities = cache[pose['system_id']]
            facts, e_rows, r_rows, cif_details = audit.inspect_cif(
                Path(pose['input_path']), args.max_cif_mib * 1024**2, 250000,
                pose.get('file_size_bytes'), pose.get('file_mtime_ns'))
            observed = get_observed(Path(pose['input_path']), facts, e_rows, args.max_cif_mib * 1024**2)
            matches = match_entities(entities, observed)
            input_flags = json.loads(system_row['review_flags_json'])
            blockers = set(cif_details['review_flags']) & BLOCK_FLAGS
            if any(m['match_status'] not in ('id_sequence_consistent', 'id_composition_consistent') for m in matches):
                blockers.add('input_output_mapping_unresolved')
            if any(json.loads(e['role_review_flags_json']) for e in entities):
                blockers.add('receptor_role_review')
            for f in ('no_input_ligand', 'no_input_protein', 'multiple_input_ligands_need_instance_policy',
                      'other_input_polymer_role_review', 'disconnected_ligand_input',
                      'explicit_input_crosslinks_need_review'):
                if f in input_flags:
                    blockers.add(f)
            matched_ids = {m['observed_chain_id'] for m in matches if m['observed_chain_id']}
            extra = [o['label_chain'] for o in observed if o['label_chain'] not in matched_ids]
            if extra:
                blockers.add('unassigned_output_entities')
            if cif_details['struct_conn']:
                blockers.add('output_connections_need_review')
            matched_ligands = [m for m in matches if m['input_kind'] == 'ligand' and m['match_status'] == 'id_composition_consistent']
            if any(m['observed_residue_groups'] != 1 for m in matched_ligands):
                blockers.add('ligand_spans_multiple_residue_groups')
            record.update(
                cif_sha256=facts['input_sha256'], manifest_ligand_copies=system_row['manifest_ligand_copies'],
                input_ligand_copies=system_row['input_ligand_copies'],
                matched_ligand_chain_groups=len(matched_ligands),
                observed_nonpolymer_residue_groups=facts['nonpolymer_residue_groups'],
                water_candidate_groups=facts['water_candidate_groups'],
                monatomic_metal_candidate_groups=facts['monatomic_metal_candidate_groups'],
                water_feature_status='explicit_water_not_observed' if not facts['water_candidate_groups'] else 'requires_extractor',
                input_review_flags_json=system_row['review_flags_json'],
                cif_review_flags_json=facts['review_flags_json'],
                unassigned_output_chain_ids_json=j(extra), blocking_reasons_json=j(sorted(blockers)),
                pair_geometry_status='hold' if blockers else 'eligible_after_policy_review')
            provenance = {**ids, 'cif_path': pose['input_path'], 'cif_sha256': facts['input_sha256']}
            mapping_rows.extend({**provenance, **m} for m in matches)
            observed_rows.extend({**provenance, **o} for o in observed)
        except Exception as exc:
            record.update(status='error', error_message=f'{type(exc).__name__}: {exc}')
        pose_rows.append(record)
        if number % 20 == 0 or number == len(poses):
            print(f'선정 구조 대응: {number}/{len(poses)}개', flush=True)
    audit.save_csv(out / 'system_input_audit.csv', system_rows, ('system_id', 'status'))
    audit.save_csv(out / 'input_entities.csv', entity_rows, ('system_id', 'input_chain_id'))
    audit.save_csv(out / 'selected_pose_roles.csv', pose_rows, KEYS)
    # 원자 행 번호 목록은 큰 CSV 셀을 피하기 위해 JSONL에만 보존합니다.
    compact = [{k: v for k, v in r.items() if k != 'heavy_atom_site_rows_1based_json'} for r in mapping_rows]
    audit.save_csv(out / 'selected_entity_mapping.csv', compact, (*KEYS, 'input_chain_id'))
    for filename, rows in [('input_details.jsonl', details), ('selected_atom_sets.jsonl', mapping_rows),
                           ('selected_observed_entities.jsonl', observed_rows)]:
        with helper.checked_path(out / filename).open('x', encoding='utf-8') as handle:
            for row in rows:
                handle.write(j(row) + '\n')
    counts = Counter()
    for row in system_rows:
        for flag in set(json.loads(row.get('review_flags_json', '[]'))):
            counts[('system', 'all', flag)] += 1
    for row in pose_rows:
        for flag in set(json.loads(row.get('blocking_reasons_json', '[]'))):
            counts[('selected_pose', row['model'], flag)] += 1
    review = [{'scope': scope, 'model': model, 'reason': flag, 'count': count}
              for (scope, model, flag), count in sorted(counts.items())]
    audit.save_csv(out / 'review_summary.csv', review, ('scope', 'model', 'reason', 'count'))
    changed = []
    for source in sources + input_sources:
        try:
            info = Path(source['path']).stat()
            if (info.st_size, info.st_mtime_ns) != (source['size_bytes'], source['mtime_ns']):
                changed.append(source['path'])
        except OSError:
            changed.append(source['path'])
    system_errors = sum(r['status'] == 'error' for r in system_rows)
    pose_errors = sum(r['status'] == 'error' for r in pose_rows)
    status = 'AUDIT_COMPLETE' if not system_errors and not pose_errors and not changed else 'CHECK_REQUIRED'
    report = {
        'version': VERSION, 'status': status, 'output_dir': str(out), 'sources': sources,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'support_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (_support_path, audit._support_path)},
        'python': sys.executable, 'gemmi_version': gemmi.__version__, 'rdkit_version': rdkit.__version__,
        'pyarrow_version': pa.__version__, 'systems': len(system_rows), 'system_errors': system_errors,
        'selected_poses': len(pose_rows), 'pose_errors': pose_errors,
        'manifest_input_copy_mismatches': sum(r.get('copy_count_match') is False for r in system_rows),
        'systems_with_role_review': sum(bool(json.loads(r.get('role_review_protein_ids_json', '[]'))) for r in system_rows),
        'pair_geometry_states': dict(Counter(r['pair_geometry_status'] for r in pose_rows)),
        'changed_sources': changed, 'review_summary': review, 'elapsed_seconds': time.perf_counter() - started,
        'gt_used': False, 'coordinates_modified': False, 'chemical_atom_mapping_performed': False,
        'all_pose_features_extracted': False, 'short_protein_review_limit': args.short_protein_limit,
        'limitations': [
            '현재 저장된 생성 입력을 조사하며 실제 실행 당시 최종 모델 입력임을 증명하지 않습니다.',
            '전체 시스템 입력을 조사하지만 출력 구조 대응은 선정된 160개에 한정합니다.',
            '리간드 조성 일치는 결합 차수·입체화학 또는 생물학적 역할의 검증이 아닙니다.',
            '짧은 서열 기준과 설명 문자열 겹침은 검토 후보를 찾는 규칙이며 삭제·제외 판정이 아닙니다.',
            'eligible_after_policy_review는 좌표 기반 계산 후보이며 feature 준비 완료 또는 좋은 pose라는 뜻이 아닙니다.',
            '원자 집합은 CIF SHA256과 1부터 시작하는 atom_site 행 번호로 식별합니다.',
            '화학적 분자 경계를 확정하지 않아 observed_ligand_copies는 비워 두고 대응 체인 그룹 수를 별도로 기록합니다.',
            '런타임 MSA 기록과 GT 평가 점수는 역할 판정에 사용하지 않습니다.']}
    helper.write_json(out / 'audit_report.json', report)
    helper.write_json(out / 'RUN_COMPLETE.json', {'outputs_written': True, 'audit_finished': True,
        'status': status, 'system_errors': system_errors, 'pose_errors': pose_errors,
        'all_features_ready': False, 'sources_unchanged': not changed})
    print('STATUS:', status, flush=True)
    print('SYSTEMS:', len(system_rows), flush=True)
    print('SYSTEM_ERRORS:', system_errors, flush=True)
    print('SELECTED_POSES:', len(pose_rows), flush=True)
    print('POSE_ERRORS:', pose_errors, flush=True)
    print('MANIFEST_INPUT_COPY_MISMATCHES:', report['manifest_input_copy_mismatches'], flush=True)
    print('SYSTEMS_WITH_ROLE_REVIEW:', report['systems_with_role_review'], flush=True)
    print('PAIR_GEOMETRY:', j(report['pair_geometry_states']), flush=True)
    print('CHEMICAL_ATOM_MAPPING: NOT_PERFORMED', flush=True)
    return 0 if status == 'AUDIT_COMPLETE' else 2


def main():
    """입력 위치와 읽기 상한을 지정하며 출력 위치는 사용자 경로로 고정합니다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index-run', type=Path, default=helper.DEFAULT_INDEX)
    parser.add_argument('--selection-run', type=Path, default=audit.DEFAULT_SELECTION)
    parser.add_argument('--expected-systems', type=int, default=4603)
    parser.add_argument('--expected-selected', type=int, default=40)
    parser.add_argument('--max-input-mib', type=int, default=64)
    parser.add_argument('--max-cif-mib', type=int, default=128)
    parser.add_argument('--short-protein-limit', type=int, default=30)
    args = parser.parse_args()
    if any(getattr(args, k) < 1 for k in ('expected_systems', 'expected_selected', 'max_input_mib', 'max_cif_mib', 'short_protein_limit')):
        parser.error('개수·크기·검토 길이 상한은 양수여야 합니다.')
    return run(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
