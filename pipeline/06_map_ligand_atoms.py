#!/usr/bin/env python3
"""깨끗한 단일 유기 비폴리머 리간드에 한정하여 입력 SMILES 그래프와 출력 원자의 대응을 기록합니다.

GT를 사용하지 않고 원본 구조·좌표·공유 환경을 수정하지 않습니다. SMILES에서 새 3D 좌표를
만들거나 출력 원자에 결합 차수를 강제 배정하지 않으며, 템플릿은 원소·연결성 참조로만 씁니다.
이 단계는 '화학적 원자 대응'의 1차 구현으로, 연결성(그래프 동형) 수준의 대응만 확립하고
결합 차수·입체화학·형식 전하는 확정하지 않습니다. 다성분·펩타이드·금속·물·역할 모호(hold)
케이스는 대응을 시도하지 않고 사유만 기록합니다.

같은 폴더의 03/04/05 Python 파일에서 경로 보호·CIF 파싱·저장 함수를 재사용하며 각 main은
실행하지 않습니다. 05번의 완료된 결과(selected_atom_sets.jsonl 등)와 원본 CIF만 읽습니다.
출력: out/ligand_mapping/06_run_map_ligand_atoms/ 아래 고정 폴더.
"""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

# 이전 단계(05→04→03)의 함수를 재사용하며 해당 파일의 main은 실행하지 않습니다.
_support_path = Path(__file__).with_name('05_audit_input_roles.py')
_spec = importlib.util.spec_from_file_location('roles_support', _support_path)
roles = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(roles)
audit = roles.audit        # 04_audit_selected_structures 모듈
helper = roles.helper      # 03_select_representative_systems 모듈

VERSION = 'ligand_mapping_0.1.0'
OUT_KIND = 'ligand_mapping'
OUT_NAME = '06_run_map_ligand_atoms'
DEFAULT_ROLES = helper.ALLOWED_ROOT / 'Scoring_Function' / 'out' / 'input_roles' / '05_run_audit_input_roles'
KEYS = ('system_id', 'model', 'seed', 'sample')
# 05번이 읽기·점검용으로 적용한 것과 같은 원자 표 행 수 상한을 사용합니다.
MAX_ATOMS = 250000
# 출력 연결성이 템플릿보다 이 배수를 넘게 조밀하면 예측 pose의 기하가 붕괴된 것으로 봅니다.
# 실제 사례: 38원자 리간드에 결합 398개(원자당 평균 차수 20.9)가 추론되어 동형 탐색이
# 끝나지 않았습니다. 탄소의 최대 결합 수가 4이므로 화학적으로 불가능한 그래프입니다.
MAX_BOND_RATIO = 2.0


def read_csv_rows(path):
    """05번이 UTF-8로 저장한 CSV를 사전 행 목록으로 읽습니다."""
    with Path(path).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def read_jsonl_rows(path):
    """JSONL을 행별 사전으로 읽습니다."""
    out = []
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def as_int(value):
    """CSV 문자열 수치를 정수로 바꾸되 결측·실패는 None으로 둡니다."""
    if value is None or str(value).strip() == '' or str(value).strip().lower() in ('nan', 'none', 'null'):
        return None
    try:
        f = float(value)
        return int(f) if f.is_integer() else None
    except (TypeError, ValueError):
        return None


def key_of(row):
    """식별자 네 개를 (str, str, int, int)로 정규화합니다."""
    return (str(row['system_id']), str(row['model']), as_int(row['seed']), as_int(row['sample']))


def template_from_smiles(smiles):
    """입력 SMILES로 중원자 템플릿 분자를 만들며 좌표는 생성하지 않습니다."""
    from rdkit import Chem
    if not isinstance(smiles, str) or not smiles:
        raise ValueError('template_smiles_missing')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('template_smiles_unparseable')
    if any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        raise ValueError('template_has_unspecified_atoms')
    # MolFromSmiles는 명시적 수소를 두지 않으므로 중원자 그래프만 남습니다.
    counts = Counter(a.GetSymbol().upper() for a in mol.GetAtoms())
    return mol, counts


def flatten_graph(mol):
    """결합 차수·방향성·전하를 지우고 연결성만 남긴 분자를 돌려줍니다."""
    from rdkit import Chem
    rw = Chem.RWMol(mol)
    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    for atom in rw.GetAtoms():
        atom.SetIsAromatic(False)
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
    flat = rw.GetMol()
    Chem.FastFindRings(flat)
    return flat


def build_output_mol(elements, coords):
    """출력 중원자의 원소와 좌표만으로 결합 없는 분자와 conformer를 만듭니다."""
    from rdkit import Chem
    from rdkit.Geometry import Point3D
    rw = Chem.RWMol()
    conf = Chem.Conformer(len(elements))
    for i, (symbol, (x, y, z)) in enumerate(zip(elements, coords)):
        idx = rw.AddAtom(Chem.Atom(symbol))
        conf.SetAtomPosition(idx, Point3D(float(x), float(y), float(z)))
    mol = rw.GetMol()
    mol.AddConformer(conf, assignId=True)
    return mol


def connectivity_from_bond_table(mol, names, bonds, comp_id):
    """CIF의 명시적 chem_comp_bond로 중원자 사이 연결만 추가합니다(결합 차수는 무시)."""
    from rdkit import Chem
    index = {}
    for i, name in enumerate(names):
        if name in index:
            raise ValueError('duplicate_atom_name_in_bond_mapping')
        index[name] = i
    rw = Chem.RWMol(mol)
    used = 0
    for row in bonds:
        if str(row.get('comp_id')) != str(comp_id):
            continue
        a, b = row.get('atom_id_1'), row.get('atom_id_2')
        if a in index and b in index:
            rw.AddBond(index[a], index[b], Chem.BondType.SINGLE)
            used += 1
    return rw.GetMol(), used


def connectivity_from_distance(mol):
    """좌표만으로 연결성을 추론합니다. 결합 차수는 판정하지 않습니다."""
    from rdkit.Chem import rdDetermineBonds
    rdDetermineBonds.DetermineConnectivity(mol)
    return mol


def graph_matches(template, output_mol, max_matches=200000):
    """연결성 수준에서 템플릿과 출력의 동형 대응을 열거합니다(상한까지).

    max_matches는 열거 상한일 뿐이며 첫 대응과 대칭 여부(1개 초과) 판정에는 영향을 주지 않습니다.
    대칭이 심한 리간드에서 조합 폭발을 막기 위해 호출 측이 작은 상한을 줄 수 있습니다."""
    t = flatten_graph(template)
    o = flatten_graph(output_mol)
    if t.GetNumAtoms() != o.GetNumAtoms():
        return []
    return list(o.GetSubstructMatches(t, uniquify=False, useChirality=False, maxMatches=max_matches))


def clean_single_reasons(pose_role, ligand_entity, ligand_atomset):
    """깨끗한 단일 유기 리간드 조건을 점검하고 통과하지 못한 사유를 모읍니다."""
    reasons = []
    if pose_role is None:
        return ['missing_pose_role_record']
    if pose_role.get('pair_geometry_status') != 'eligible_after_policy_review':
        reasons.append('pair_geometry_' + str(pose_role.get('pair_geometry_status')))
    if as_int(pose_role.get('matched_ligand_chain_groups')) != 1:
        reasons.append('matched_ligand_chain_groups_not_1')
    if as_int(pose_role.get('observed_nonpolymer_residue_groups')) != 1:
        reasons.append('nonpolymer_residue_groups_not_1')
    if as_int(pose_role.get('water_candidate_groups')) not in (0, None):
        reasons.append('water_candidate_present')
    if as_int(pose_role.get('monatomic_metal_candidate_groups')) not in (0, None):
        reasons.append('monatomic_metal_candidate_present')
    if ligand_atomset is None:
        reasons.append('missing_ligand_atom_set')
    elif ligand_atomset.get('match_status') != 'id_composition_consistent':
        reasons.append('ligand_match_' + str(ligand_atomset.get('match_status')))
    if ligand_entity is None:
        reasons.append('missing_ligand_input_entity')
    else:
        if ligand_entity.get('chemistry_status') != 'input_graph_parsed':
            reasons.append('ligand_chemistry_' + str(ligand_entity.get('chemistry_status')))
        frag = as_int(ligand_entity.get('fragment_count'))
        if frag is not None and frag > 1:
            reasons.append('disconnected_ligand_input')
        if not (ligand_entity.get('canonical_smiles') or '').strip():
            reasons.append('canonical_smiles_missing')
    return reasons


def extract_ligand_atoms(cif_path, expected_sha, rows_1based, observed_chain, max_bytes):
    """저장된 행 번호로 리간드 중원자의 원소·좌표·이름·comp_id와 결합표를 읽습니다."""
    import gemmi
    raw, fp = roles.read_bytes(cif_path, max_bytes)
    if fp['sha256'] != expected_sha:
        raise ValueError('cif_changed_since_audit')
    block = gemmi.cif.read_string(raw.decode('utf-8')).sole_block()
    atoms = audit.rows(block, '_atom_site.', MAX_ATOMS)
    total = len(atoms)
    elements, coords, names, comps, chains = [], [], [], [], []
    for r in rows_1based:
        if not isinstance(r, int) or r < 1 or r > total:
            raise ValueError('atom_site_row_out_of_range')
        a = atoms[r - 1]
        symbol = str(a.get('type_symbol') or '').upper()
        element = gemmi.Element(symbol)
        if element.atomic_number <= 1:
            raise ValueError('non_heavy_atom_in_ligand_rows')
        elements.append(element.name)
        coords.append((audit.finite_number(a.get('Cartn_x')),
                       audit.finite_number(a.get('Cartn_y')),
                       audit.finite_number(a.get('Cartn_z'))))
        names.append(a.get('label_atom_id'))
        comps.append(a.get('label_comp_id'))
        chains.append(a.get('label_asym_id'))
    if any(None in c for c in coords):
        raise ValueError('invalid_ligand_coordinates')
    if observed_chain is not None and any(c != observed_chain for c in chains):
        raise ValueError('ligand_rows_span_other_chains')
    comp_set = {c for c in comps if c is not None}
    if len(comp_set) != 1:
        raise ValueError('ligand_rows_span_multiple_components')
    comp_id = comp_set.pop()
    if len(set(names)) != len(names):
        raise ValueError('duplicate_ligand_atom_names')
    bonds = audit.rows(block, '_chem_comp_bond.')
    return {'elements': elements, 'coords': coords, 'names': names,
            'comp_id': comp_id, 'bonds': bonds, 'fingerprint': fp}


def map_one(ligand_entity, extracted, max_matches=200000):
    """추출한 출력 리간드와 입력 SMILES 템플릿 사이 연결성 대응을 확립합니다."""
    template, template_counts = template_from_smiles(ligand_entity.get('canonical_smiles')
                                                     or ligand_entity.get('smiles'))
    output_counts = Counter(e.upper() for e in extracted['elements'])
    composition_match = (dict(template_counts) == dict(output_counts))
    detail = {'bond_source': None, 'bonds_used': None, 'template_heavy_atoms': template.GetNumAtoms(),
              'template_bonds': template.GetNumBonds(),
              'output_heavy_atoms': len(extracted['elements']), 'composition_match': composition_match,
              'match_multiplicity': None, 'ligand_comp_id': extracted['comp_id'],
              'template_element_counts_json': audit.encoded(dict(template_counts)),
              'output_element_counts_json': audit.encoded(dict(output_counts))}
    if template.GetNumAtoms() != len(extracted['elements']):
        return 'atom_count_mismatch', detail, None
    # 출력 분자의 연결성: 명시적 결합표 우선, 없으면 좌표 기반 추론.
    base = build_output_mol(extracted['elements'], extracted['coords'])
    bonds_present = any(str(b.get('comp_id')) == str(extracted['comp_id']) for b in extracted['bonds'])
    if bonds_present:
        try:
            output_mol, used = connectivity_from_bond_table(base, extracted['names'],
                                                            extracted['bonds'], extracted['comp_id'])
            detail['bond_source'] = 'cif_bond_table'
            detail['bonds_used'] = used
        except Exception as exc:
            return 'bond_table_failed:' + type(exc).__name__, detail, None
    else:
        try:
            output_mol = connectivity_from_distance(base)
            detail['bond_source'] = 'distance_inferred'
            detail['bonds_used'] = output_mol.GetNumBonds()
        except Exception as exc:
            return 'bond_inference_failed:' + type(exc).__name__, detail, None
    # 추론된 연결성이 템플릿보다 터무니없이 조밀하면 예측 pose의 기하가 붕괴된 것입니다.
    # 그대로 동형 탐색에 넘기면 경우의 수가 폭발해 끝나지 않으므로, 시도하지 않고
    # 사유만 남깁니다. 이 pose는 원자 대응이 불가능할 뿐이며 대응이 필요 없는
    # feature(접촉·표면·물/금속·descriptor)는 호출 측에서 그대로 계산됩니다.
    if (detail['bonds_used'] or 0) > template.GetNumBonds() * MAX_BOND_RATIO:
        return 'degenerate_inferred_connectivity', detail, None
    matches = graph_matches(template, output_mol, max_matches)
    detail['match_multiplicity'] = len(matches)
    if not matches:
        return 'connectivity_mismatch', detail, None
    m0 = matches[0]
    mapping = [{'template_atom_index': i, 'output_atom_order': int(m0[i]),
                'output_atom_site_row': int(extracted['rows_1based'][m0[i]]),
                'output_element': extracted['elements'][m0[i]],
                'output_atom_name': extracted['names'][m0[i]],
                'template_element': template.GetAtomWithIdx(i).GetSymbol()}
               for i in range(len(m0))]
    status = 'clean_single_full_mapping' if len(matches) == 1 else 'clean_single_symmetric_mapping'
    return status, detail, mapping


def mapping_spec():
    """대응 방법과 상태 어휘를 저장해 재현과 해석의 근거로 남깁니다."""
    return {
        'schema_version': VERSION,
        'scope': '깨끗한 단일 유기 비폴리머 리간드 pose의 입력-출력 연결성 대응 1차 구현.',
        'method': {
            'template': '입력 canonical SMILES의 중원자 그래프(좌표 미생성).',
            'output_connectivity': 'CIF _chem_comp_bond가 있으면 우선 사용, 없으면 좌표 기반 연결성 추론.',
            'matching': '결합 차수·방향성·전하를 제거한 연결성 그래프의 동형(부분구조) 대응을 열거.',
            'symmetry': 'uniquify=False로 대응 다중도를 기록하여 대칭 원자의 모호성을 드러냄.'},
        'not_performed': ['결합 차수 확정', '입체화학/형식 전하 확정', '생물학적 역할 확정',
                          'SMILES 3D 좌표 생성', '출력 원자 좌표 수정', 'GT 사용'],
        'status_values': {
            'clean_single_full_mapping': '단일하게 결정된 연결성 동형 대응.',
            'clean_single_symmetric_mapping': '동형 대응 성립, 대칭으로 원자 단위 식별은 다중.',
            'connectivity_mismatch': '템플릿과 출력 연결성 사이 동형 대응 없음.',
            'degenerate_inferred_connectivity':
                f'출력 연결성이 템플릿 결합 수의 {MAX_BOND_RATIO}배를 초과해 조밀함'
                '(예측 pose 기하 붕괴). 조합 폭발을 피해 대응을 시도하지 않음.',
            'atom_count_mismatch': '템플릿과 출력 중원자 수 불일치.',
            'bond_inference_failed': '좌표 기반 연결성 추론 실패.',
            'bond_table_failed': '명시적 결합표 처리 실패.',
            'template_build_failed': 'SMILES 템플릿 생성 실패.',
            'extract_failed': '출력 원자 추출 또는 CIF 재검증 실패.',
            'skipped_non_clean': '깨끗한 단일 유기 리간드 조건 미충족(사유 기록).',
            'error': '예상치 못한 처리 오류.'}}


def run(args):
    """05번 결과와 원본 CIF를 읽어 깨끗한 단일 리간드의 대응을 고정 폴더에 저장합니다."""
    started = time.perf_counter()
    roles_dir = args.roles_run
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
    import rdkit

    # 05번 완료 기록과 입력 파일의 출처를 확인합니다.
    source_names = ('RUN_COMPLETE.json', 'audit_report.json', 'selected_pose_roles.csv',
                    'input_entities.csv', 'selected_atom_sets.jsonl')
    sources = [helper.fingerprint(roles_dir / name) for name in source_names]
    complete = json.loads((roles_dir / 'RUN_COMPLETE.json').read_text(encoding='utf-8'))
    if complete.get('status') != 'AUDIT_COMPLETE' or complete.get('outputs_written') is not True:
        raise ValueError('05번 입력 역할 점검이 정상 완료된 결과가 아닙니다.')

    pose_roles = {key_of(r): r for r in read_csv_rows(roles_dir / 'selected_pose_roles.csv')}
    ligand_entities = {}
    for r in read_csv_rows(roles_dir / 'input_entities.csv'):
        if r.get('input_kind') == 'ligand':
            ligand_entities[(str(r['system_id']), str(r['input_chain_id']))] = r
    atom_sets = read_jsonl_rows(roles_dir / 'selected_atom_sets.jsonl')

    # pose별 리간드 원자 집합 행을 모읍니다(입력 kind가 ligand인 대응 행).
    ligand_sets = {}
    for row in atom_sets:
        if row.get('input_kind') != 'ligand':
            continue
        k = key_of(row)
        if k in ligand_sets:
            # 한 pose에 리간드 입력이 둘 이상이면 이번 단계의 단일 리간드 전제를 벗어납니다.
            ligand_sets[k] = 'multiple'
        else:
            ligand_sets[k] = row
    pose_keys = sorted(pose_roles)

    rows_out, jsonl_out = [], []
    status_counter = Counter()
    spec = mapping_spec()
    helper.write_json(out / 'mapping_spec.json', spec)

    with helper.checked_path(out / 'ligand_atom_mapping.jsonl').open('x', encoding='utf-8') as detail_file:
        for number, k in enumerate(pose_keys, 1):
            pose_role = pose_roles[k]
            system_id, model, seed, sample = k
            ident = {'system_id': system_id, 'model': model, 'seed': seed, 'sample': sample}
            lig_set = ligand_sets.get(k)
            lig_entity = None
            if isinstance(lig_set, dict):
                lig_entity = ligand_entities.get((system_id, str(lig_set.get('input_chain_id'))))
            record = {**ident, 'ligand_status': None, 'skip_reasons_json': '[]',
                      'bond_source': None, 'bonds_used': None, 'template_heavy_atoms': None,
                      'output_heavy_atoms': None, 'composition_match': None,
                      'match_multiplicity': None, 'ligand_comp_id': None,
                      'input_chain_id': lig_set.get('input_chain_id') if isinstance(lig_set, dict) else None,
                      'observed_chain_id': lig_set.get('observed_chain_id') if isinstance(lig_set, dict) else None,
                      'cif_sha256': lig_set.get('cif_sha256') if isinstance(lig_set, dict) else None,
                      'canonical_smiles': (lig_entity or {}).get('canonical_smiles'),
                      'error_message': None}
            try:
                if lig_set == 'multiple':
                    reasons = ['multiple_ligand_atom_sets']
                else:
                    reasons = clean_single_reasons(pose_role, lig_entity, lig_set)
                if reasons:
                    record['ligand_status'] = 'skipped_non_clean'
                    record['skip_reasons_json'] = audit.encoded(sorted(set(reasons)))
                else:
                    rows_1based = json.loads(lig_set['heavy_atom_site_rows_1based_json'])
                    extracted = extract_ligand_atoms(
                        lig_set['cif_path'], lig_set['cif_sha256'], rows_1based,
                        lig_set.get('observed_chain_id'), args.max_cif_mib * 1024 ** 2)
                    extracted['rows_1based'] = rows_1based
                    status, detail, mapping = map_one(lig_entity, extracted)
                    record.update({'ligand_status': status.split(':')[0], 'bond_source': detail['bond_source'],
                                   'bonds_used': detail['bonds_used'],
                                   'template_heavy_atoms': detail['template_heavy_atoms'],
                                   'output_heavy_atoms': detail['output_heavy_atoms'],
                                   'composition_match': detail['composition_match'],
                                   'match_multiplicity': detail['match_multiplicity'],
                                   'ligand_comp_id': detail['ligand_comp_id']})
                    if status.startswith('bond_inference_failed') or status.startswith('bond_table_failed'):
                        record['error_message'] = status
                    detail_file.write(audit.encoded({
                        **ident, 'cif_path': lig_set['cif_path'], 'cif_sha256': lig_set['cif_sha256'],
                        'ligand_comp_id': detail['ligand_comp_id'], 'ligand_status': record['ligand_status'],
                        'bond_source': detail['bond_source'], 'bonds_used': detail['bonds_used'],
                        'template_bonds': detail['template_bonds'], 'match_multiplicity': detail['match_multiplicity'],
                        'template_element_counts_json': detail['template_element_counts_json'],
                        'output_element_counts_json': detail['output_element_counts_json'],
                        'atom_mapping': mapping}) + '\n')
            except Exception as exc:
                status = 'template_build_failed' if 'template' in str(exc) else 'extract_failed'
                record['ligand_status'] = status
                record['error_message'] = f'{type(exc).__name__}: {exc}'
            status_counter[record['ligand_status']] += 1
            rows_out.append(record)
            if number % 20 == 0 or number == len(pose_keys):
                print(f'리간드 대응: {number}/{len(pose_keys)}개', flush=True)

    audit.save_csv(out / 'ligand_atom_mapping.csv', rows_out, KEYS)
    summary = [{'ligand_status': status, 'count': count} for status, count in sorted(status_counter.items())]
    audit.save_csv(out / 'mapping_summary.csv', summary, ('ligand_status', 'count'))

    changed = []
    for source in sources:
        try:
            info = Path(source['path']).stat()
            if (info.st_size, info.st_mtime_ns) != (source['size_bytes'], source['mtime_ns']):
                changed.append(source['path'])
        except OSError:
            changed.append(source['path'])
    errors = sum(r['ligand_status'] in ('extract_failed', 'template_build_failed', 'error') for r in rows_out)
    mapped = sum(r['ligand_status'] in ('clean_single_full_mapping', 'clean_single_symmetric_mapping') for r in rows_out)
    skipped = status_counter.get('skipped_non_clean', 0)
    status = 'MAPPING_COMPLETE' if not errors and not changed else 'CHECK_REQUIRED'
    report = {
        'version': VERSION, 'status': status, 'output_dir': str(out), 'roles_run': str(roles_dir),
        'sources': sources, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'support_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (_support_path, roles._support_path, audit._support_path)},
        'python': sys.executable, 'gemmi_version': gemmi.__version__, 'rdkit_version': rdkit.__version__,
        'poses': len(rows_out), 'clean_single_mapped': mapped, 'skipped_non_clean': skipped,
        'errors': errors, 'status_counts': dict(status_counter), 'changed_sources': changed,
        'elapsed_seconds': time.perf_counter() - started,
        'gt_used': False, 'coordinates_modified': False, 'bond_orders_assigned': False,
        'chemical_atom_mapping_scope': 'connectivity_only',
        'limitations': [
            '연결성 수준의 동형 대응만 확립하며 결합 차수·입체화학·형식 전하는 확정하지 않습니다.',
            'clean_single_symmetric_mapping은 대칭 원자에서 원자 단위 식별이 하나로 결정되지 않습니다.',
            '깨끗한 단일 유기 리간드만 대상으로 하며 다성분·펩타이드·금속·물·hold는 사유만 기록합니다.',
            '좌표 기반 연결성 추론은 좌표 품질에 의존하며 실패를 상태로 남깁니다.',
            '입력 SMILES로 3D 좌표를 만들거나 출력 원자 좌표를 수정하지 않습니다.',
            'GT 구조와 평가 점수를 대응·선택에 사용하지 않습니다.']}
    helper.write_json(out / 'audit_report.json', report)
    helper.write_json(out / 'RUN_COMPLETE.json', {'outputs_written': True, 'mapping_finished': True,
        'status': status, 'poses': len(rows_out), 'clean_single_mapped': mapped,
        'errors': errors, 'sources_unchanged': not changed})
    print('STATUS:', status, flush=True)
    print('POSES:', len(rows_out), flush=True)
    print('CLEAN_SINGLE_MAPPED:', mapped, flush=True)
    print('SKIPPED_NON_CLEAN:', skipped, flush=True)
    print('ERRORS:', errors, flush=True)
    print('STATUS_COUNTS:', roles.j(dict(status_counter)), flush=True)
    print('CHEMICAL_ATOM_MAPPING: CONNECTIVITY_ONLY', flush=True)
    return 0 if status == 'MAPPING_COMPLETE' else 2


def main():
    """05번 결과 위치와 CIF 읽기 상한을 지정하며 출력 위치는 사용자 경로로 고정합니다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roles-run', type=Path, default=DEFAULT_ROLES)
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
