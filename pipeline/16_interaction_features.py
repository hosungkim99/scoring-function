#!/usr/bin/env python3
"""단백질-리간드 상호작용·반발(category 2) feature를 추출합니다.

수소가 예측 구조에 없으므로 수소결합은 중원자 거리 기반의 'potential' 지표로 정직하게 명명합니다.
리간드 원자유형(극성·전하·방향족)은 06번 원자 대응으로 입력 SMILES 템플릿에서 전사하고(06 보강),
수용체 원자유형은 표준 단백질 잔기·원자명으로 분류합니다. 극성 접촉·소수성·충돌은 원소와 좌표만
있으면 되므로 eligible pose 전체에 계산하고, 염다리·방향족은 리간드 전하·방향족 정보(06 필요)가
있는 pose에만 계산하며 없으면 null로 둡니다(관찰된 0과 구분). GT·좌표 수정 없음.

07번을 import해 CIF 읽기·원자 집합을 재사용하고, 06번 매핑과 05번 리간드 SMILES를 읽습니다.
출력: out/interaction_features/16_run_interaction_features/ 아래 고정 폴더.
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

_support_path = Path(__file__).with_name('07_contact_features.py')
_spec = importlib.util.spec_from_file_location('contact_support', _support_path)
cf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cf)
roles = cf.roles
audit = cf.audit
helper = cf.helper

VERSION = 'interaction_features_0.1.0'
OUT_KIND = 'interaction_features'
OUT_NAME = '16_run_interaction_features'
DEFAULT_ROLES = helper.ALLOWED_ROOT / 'Scoring_Function' / 'out' / 'input_roles' / '05_run_audit_input_roles'
DEFAULT_MAPPING = helper.ALLOWED_ROOT / 'Scoring_Function' / 'out' / 'ligand_mapping' / '06_run_map_ligand_atoms'
KEYS = ('system_id', 'model', 'seed', 'sample')
MAPPED_STATUSES = {'clean_single_full_mapping', 'clean_single_symmetric_mapping'}
HBOND_DIST = 3.5         # 중원자 극성-극성 거리(수소 없음, potential)
SALT_DIST = 4.0
HYDROPHOBIC_DIST = 4.0
AROMATIC_DIST = 4.5
CLASH_DIST = 2.2         # 단백질-리간드 입체 충돌(Å)
POLAR = {'N', 'O'}
# 전하 잔기·원자명(근사).
POS_ATOMS = {('ARG', 'NH1'), ('ARG', 'NH2'), ('ARG', 'NE'), ('LYS', 'NZ'), ('HIS', 'ND1'), ('HIS', 'NE2')}
NEG_ATOMS = {('ASP', 'OD1'), ('ASP', 'OD2'), ('GLU', 'OE1'), ('GLU', 'OE2')}
AROMATIC_RES = {'PHE', 'TYR', 'TRP', 'HIS'}
BACKBONE = {'N', 'CA', 'C', 'O', 'OXT'}


def feature_definitions():
    """상호작용 feature의 정의와 단위, 결측 의미를 고정합니다."""
    return {
        'potential_hbond_pairs': {'unit': 'count', 'definition': f'리간드 N/O와 수용체 N/O가 {HBOND_DIST}Å 이내인 원자쌍 수. 수소 없음→potential.'},
        'hydrophobic_pairs': {'unit': 'count', 'definition': f'리간드 C와 수용체 C가 {HYDROPHOBIC_DIST}Å 이내인 원자쌍 수.'},
        'receptor_clash_pairs': {'unit': 'count', 'definition': f'리간드-수용체 중원자쌍이 {CLASH_DIST}Å보다 가까운 수(입체 충돌).'},
        'salt_bridge_pairs': {'unit': 'count', 'definition': f'리간드 전하 원자와 반대 전하 수용체 원자가 {SALT_DIST}Å 이내인 쌍 수. 리간드 전하 미상이면 null.'},
        'aromatic_contact_pairs': {'unit': 'count', 'definition': f'리간드 방향족 원자와 수용체 방향족 잔기 원자가 {AROMATIC_DIST}Å 이내인 쌍 수. 리간드 방향족 미상이면 null.'},
    }


def specification():
    return {
        'schema_version': VERSION,
        'scope': '단백질-리간드 상호작용·반발(category 2). 수소 없는 중원자 기반.',
        'features': feature_definitions(),
        'typing': {
            'ligand_basic': '원소 기반(N/O 극성, C 소수성) — eligible 전체.',
            'ligand_template': '06 대응으로 전사한 전하·방향족 — 매핑된 pose에만.',
            'receptor': '표준 단백질 잔기·원자명(전하·방향족·소수성).'},
        'distances': {'hbond': HBOND_DIST, 'salt': SALT_DIST, 'hydrophobic': HYDROPHOBIC_DIST,
                      'aromatic': AROMATIC_DIST, 'clash': CLASH_DIST},
        'absence_policy': '리간드 전하·방향족을 모르는 pose(06 미매핑)는 salt/aromatic을 null로 둡니다(0 아님).',
        'notes': ['수소가 없어 수소결합은 기하 각도 없이 거리 기반 potential로만 셉니다.',
                  '전하·방향족 수용체 분류는 잔기·원자명 근사이며 pKa·양성자화 상태를 확정하지 않습니다.',
                  'GT·좌표 수정 없음.']}


def extract_atoms(atoms, rows_1based):
    """행 번호로 중원자의 원소·좌표·comp·atom_id를 추출합니다."""
    import gemmi
    import numpy as np
    total = len(atoms)
    elems, coords, comps, aids = [], [], [], []
    for r in rows_1based:
        if not isinstance(r, int) or r < 1 or r > total:
            raise ValueError('atom_site_row_out_of_range')
        a = atoms[r - 1]
        sym = str(a.get('type_symbol') or '').upper()
        if gemmi.Element(sym).atomic_number <= 1:
            raise ValueError('non_heavy_atom_in_selected_rows')
        xyz = [audit.finite_number(a.get(k)) for k in ('Cartn_x', 'Cartn_y', 'Cartn_z')]
        if any(v is None for v in xyz):
            raise ValueError('invalid_coordinate')
        elems.append(sym)
        coords.append(xyz)
        comps.append(str(a.get('label_comp_id') or '').upper())
        aids.append(str(a.get('label_atom_id') or '').upper())
    return elems, np.asarray(coords, dtype=np.float64), comps, aids


def ligand_template_typing(smiles, atom_mapping):
    """06 대응으로 출력 리간드 행별 전하·방향족 여부를 돌려줍니다."""
    from rdkit import Chem
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise ValueError('template_build_failed')
    arom_by_row, charge_by_row = {}, {}
    for m in atom_mapping:
        ti = int(m['template_atom_index'])
        row = int(m['output_atom_site_row'])
        a = template.GetAtomWithIdx(ti)
        arom_by_row[row] = bool(a.GetIsAromatic())
        charge_by_row[row] = int(a.GetFormalCharge())
    return arom_by_row, charge_by_row


def count_pairs(a_xyz, a_mask, b_xyz, b_mask, cutoff):
    """두 좌표 집합의 선택된 원자쌍 중 거리 < cutoff인 쌍 수를 셉니다."""
    import numpy as np
    from scipy.spatial.distance import cdist
    ai = np.where(a_mask)[0]
    bi = np.where(b_mask)[0]
    if len(ai) == 0 or len(bi) == 0:
        return 0
    d = cdist(a_xyz[ai], b_xyz[bi])
    return int((d < cutoff).sum())


def run(args):
    """eligible pose의 상호작용 feature를 계산해 저장합니다."""
    started = time.perf_counter()
    roles_dir = args.roles_run
    mapping_dir = args.mapping_run
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

    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import rdkit
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)

    spec = specification()
    spec_hash = hashlib.sha256((json.dumps(spec, ensure_ascii=False, sort_keys=True,
                                           allow_nan=False)).encode()).hexdigest()
    helper.write_json(out / 'feature_spec.json', spec)

    sources = [helper.fingerprint(p) for p in (
        roles_dir / 'RUN_COMPLETE.json', roles_dir / 'selected_pose_roles.csv', roles_dir / 'input_entities.csv',
        mapping_dir / 'RUN_COMPLETE.json', mapping_dir / 'ligand_atom_mapping.jsonl')]
    complete = json.loads((roles_dir / 'RUN_COMPLETE.json').read_text(encoding='utf-8'))
    if complete.get('status') != 'AUDIT_COMPLETE' or complete.get('outputs_written') is not True:
        raise ValueError('05번 입력 역할 점검이 정상 완료된 결과가 아닙니다.')

    with (roles_dir / 'selected_pose_roles.csv').open(newline='', encoding='utf-8') as handle:
        pose_roles = {(str(r['system_id']), str(r['model']), int(float(r['seed'])), int(float(r['sample']))): r
                      for r in csv.DictReader(handle)}
    receptor_sets, ligand_sets, cif_meta = cf.load_atom_sets(roles_dir)
    system_smiles = {}
    with (roles_dir / 'input_entities.csv').open(newline='', encoding='utf-8') as handle:
        for r in csv.DictReader(handle):
            if r.get('input_kind') == 'ligand':
                smi = (r.get('canonical_smiles') or '').strip() or (r.get('smiles') or '').strip()
                system_smiles.setdefault(r['system_id'], smi)
    mapping = {}
    with (mapping_dir / 'ligand_atom_mapping.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            mapping[(str(o['system_id']), str(o['model']), int(o['seed']), int(o['sample']))] = o

    feature_names = list(feature_definitions())
    max_bytes = args.max_cif_mib * 1024 ** 2
    rows_out = []
    status_counter = Counter()
    for number, key in enumerate(sorted(pose_roles), 1):
        system_id, model, seed, sample = key
        pose_role = pose_roles[key]
        cif_path, cif_sha = cif_meta.get(key, (None, None))
        begin = time.perf_counter()
        row = {'system_id': system_id, 'model': model, 'seed': seed, 'sample': sample,
               'pose_id': f'{system_id}|{model}|{seed}|{sample}', 'schema_version': VERSION,
               'config_sha256': spec_hash, 'cif_sha256': cif_sha, 'ligand_typing': None,
               'status': None, 'skip_reasons_json': '[]', 'error_message': None, 'elapsed_seconds': None}
        row.update(dict.fromkeys(feature_names))
        try:
            if pose_role.get('pair_geometry_status') != 'eligible_after_policy_review':
                row['status'] = 'skipped_not_eligible'
            elif ligand_sets.get(key) == 'multiple' or key not in ligand_sets:
                row['status'] = 'skipped_no_ligand_atom_set'
            elif key not in receptor_sets or not receptor_sets[key]:
                row['status'] = 'skipped_no_clean_receptor'
            else:
                atoms, _ = cf.read_atoms(cif_path, cif_sha, max_bytes)
                lig_rows = ligand_sets[key]['rows']
                rec_rows = []
                for _c, rr in receptor_sets[key]:
                    rec_rows.extend(rr)
                lig_el, lig_xyz, _lc, _la = extract_atoms(atoms, lig_rows)
                rec_el, rec_xyz, rec_comp, rec_aid = extract_atoms(atoms, rec_rows)
                lig_el = np.asarray(lig_el)
                rec_el = np.asarray(rec_el)
                # 수용체 원자유형(표준 단백질).
                rec_polar = np.asarray([e in POLAR for e in rec_el])
                rec_carbon = rec_el == 'C'
                rec_pos = np.asarray([(c, a) in POS_ATOMS for c, a in zip(rec_comp, rec_aid)])
                rec_neg = np.asarray([(c, a) in NEG_ATOMS for c, a in zip(rec_comp, rec_aid)])
                rec_arom = np.asarray([(c in AROMATIC_RES and a not in BACKBONE and e in ('C', 'N'))
                                       for c, a, e in zip(rec_comp, rec_aid, rec_el)])
                # 리간드 기본 유형(원소).
                lig_polar = np.asarray([e in POLAR for e in lig_el])
                lig_carbon = lig_el == 'C'
                # 원소·좌표만으로 가능한 세 feature(전체 eligible).
                row['potential_hbond_pairs'] = count_pairs(lig_xyz, lig_polar, rec_xyz, rec_polar, HBOND_DIST)
                row['hydrophobic_pairs'] = count_pairs(lig_xyz, lig_carbon, rec_xyz, rec_carbon, HYDROPHOBIC_DIST)
                row['receptor_clash_pairs'] = count_pairs(lig_xyz, np.ones(len(lig_el), bool),
                                                          rec_xyz, np.ones(len(rec_el), bool), CLASH_DIST)
                # 전하·방향족은 06 매핑이 있어야 리간드 유형을 알 수 있습니다.
                entry = mapping.get(key)
                if entry and entry.get('ligand_status') in MAPPED_STATUSES and entry.get('atom_mapping') \
                        and system_smiles.get(system_id):
                    arom_by_row, charge_by_row = ligand_template_typing(system_smiles[system_id], entry['atom_mapping'])
                    lig_arom = np.asarray([bool(arom_by_row.get(r, False)) for r in lig_rows])
                    lig_pos = np.asarray([charge_by_row.get(r, 0) > 0 for r in lig_rows])
                    lig_neg = np.asarray([charge_by_row.get(r, 0) < 0 for r in lig_rows])
                    salt = (count_pairs(lig_xyz, lig_pos, rec_xyz, rec_neg, SALT_DIST)
                            + count_pairs(lig_xyz, lig_neg, rec_xyz, rec_pos, SALT_DIST))
                    row['salt_bridge_pairs'] = int(salt)
                    row['aromatic_contact_pairs'] = count_pairs(lig_xyz, lig_arom, rec_xyz, rec_arom, AROMATIC_DIST)
                    row['ligand_typing'] = 'template_typed'
                else:
                    # salt/aromatic은 null 유지(정보 부재).
                    row['ligand_typing'] = 'element_only'
                row['status'] = 'computed'
        except Exception as exc:
            row.update(status='error', error_message=f'{type(exc).__name__}: {exc}')
        row['elapsed_seconds'] = time.perf_counter() - begin
        status_counter[row['status']] += 1
        rows_out.append(row)
        if number % 20 == 0 or number == len(pose_roles):
            print(f'상호작용 feature: {number}/{len(pose_roles)}개', flush=True)

    frame = pd.DataFrame(rows_out)
    if frame.duplicated(list(KEYS)).any():
        raise RuntimeError('pose 식별자 중복이 있습니다.')
    buf = io.BytesIO()
    frame.to_parquet(buf, engine='pyarrow', index=False, compression='snappy')
    with helper.checked_path(out / 'interaction_features.parquet').open('xb') as handle:
        handle.write(buf.getvalue())
    restored = pd.read_parquet(out / 'interaction_features.parquet', engine='pyarrow')
    pd.testing.assert_frame_equal(restored, frame, check_dtype=False)

    status_rows = [{'status': s, 'count': c} for s, c in sorted(status_counter.items())]
    audit.save_csv(out / 'pose_status.csv', status_rows, ('status', 'count'))
    typing_counter = Counter(r['ligand_typing'] for r in rows_out if r['status'] == 'computed')

    changed = []
    for source in sources:
        try:
            info = Path(source['path']).stat()
            if (info.st_size, info.st_mtime_ns) != (source['size_bytes'], source['mtime_ns']):
                changed.append(source['path'])
        except OSError:
            changed.append(source['path'])
    computed = status_counter.get('computed', 0)
    errors = status_counter.get('error', 0)
    comp = frame[frame['status'] == 'computed']
    poses_with_clash = int((comp['receptor_clash_pairs'] > 0).sum()) if len(comp) else 0
    status = 'FEATURES_COMPLETE' if not errors and not changed else 'CHECK_REQUIRED'
    report = {
        'version': VERSION, 'status': status, 'output_dir': str(out), 'roles_run': str(roles_dir),
        'mapping_run': str(mapping_dir), 'sources': sources, 'config_sha256': spec_hash,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'support_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (_support_path, cf._support_path, roles._support_path, audit._support_path)},
        'python': sys.executable, 'rdkit_version': rdkit.__version__, 'numpy_version': np.__version__,
        'pyarrow_version': pa.__version__, 'poses': len(rows_out), 'computed': computed, 'errors': errors,
        'status_counts': dict(status_counter), 'ligand_typing_counts': dict(typing_counter),
        'poses_with_receptor_clash': poses_with_clash, 'changed_sources': changed, 'feature_category': 2,
        'elapsed_seconds': time.perf_counter() - started, 'gt_used': False, 'coordinates_modified': False,
        'limitations': [
            '수소가 없어 수소결합은 거리 기반 potential이며 각도 기하를 포함하지 않습니다.',
            '염다리·방향족은 리간드 전하·방향족을 아는 06 매핑 pose에만 계산하고 나머지는 null입니다.',
            '수용체 전하·방향족은 잔기·원자명 근사이며 양성자화 상태를 확정하지 않습니다.',
            'eligible이 아닌 pose는 계산하지 않고 상태만 남깁니다.',
            'GT·좌표 수정 없음.']}
    helper.write_json(out / 'audit_report.json', report)
    helper.write_json(out / 'RUN_COMPLETE.json', {'outputs_written': True, 'features_finished': True,
        'status': status, 'poses': len(rows_out), 'computed': computed, 'errors': errors,
        'ligand_typing_counts': dict(typing_counter), 'sources_unchanged': not changed})
    print('STATUS:', status, flush=True)
    print('POSES:', len(rows_out), '| COMPUTED:', computed, '| ERRORS:', errors, flush=True)
    print('LIGAND_TYPING:', roles.j(dict(typing_counter)), flush=True)
    print('STATUS_COUNTS:', roles.j(dict(status_counter)), flush=True)
    return 0 if status == 'FEATURES_COMPLETE' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roles-run', type=Path, default=DEFAULT_ROLES)
    parser.add_argument('--mapping-run', type=Path, default=DEFAULT_MAPPING)
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
