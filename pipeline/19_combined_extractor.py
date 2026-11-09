#!/usr/bin/env python3
"""통합 단일패스 feature 추출기(A6a). CIF를 pose당 한 번만 읽고 cat1·2·3·4·6/7·8을 모두 계산합니다.

기존 per-category 스크립트(07·11·12·13·15·16)의 핵심 함수를 그대로 재사용해 값이 달라지지 않게
하고, 계산 결과를 기존 per-category 산출물과 자동 대조(self-validation)하여 표본(152) 재현을
확인합니다. category 5(P2Rank)는 외부 바이너리라 통합에서 제외하고 14번으로 따로 둡니다. 전체
460k 확대(A6b·A6c)에서는 입력(역할·원자 대응)만 pose별 계산으로 바꾸면 되도록 조립부를 한 곳에
모읍니다. GT·좌표 수정 없음. 05→04→03 함수를 재사용합니다.

A6a 범위: 05 selected_atom_sets·06 ligand_atom_mapping을 입력으로 선정 160 pose에 적용.
출력: out/combined_features/19_run_combined_extractor/ 아래 고정 폴더.
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

cf1 = _load('07_contact_features.py')       # 접촉(cat1)
sf = _load('11_surface_features.py')        # 표면(cat3)
ld = _load('12_ligand_descriptors.py')      # 리간드 고유(cat8)
wm = _load('13_water_metal_features.py')     # 물·금속(cat6/7)
lg = _load('15_ligand_geometry.py')          # 내부구조(cat4)
it = _load('16_interaction_features.py')     # 상호작용(cat2)
roles = cf1.roles
audit = cf1.audit
helper = cf1.helper

VERSION = 'combined_features_0.1.0'
OUT_KIND = 'combined_features'
OUT_NAME = '19_run_combined_extractor'
DEFAULT_ROLES = helper.ALLOWED_ROOT / 'Scoring_Function' / 'out' / 'input_roles' / '05_run_audit_input_roles'
DEFAULT_MAPPING = helper.ALLOWED_ROOT / 'Scoring_Function' / 'out' / 'ligand_mapping' / '06_run_map_ligand_atoms'
BASE = helper.ALLOWED_ROOT / 'Scoring_Function' / 'out'
KEYS = ('system_id', 'model', 'seed', 'sample')
MAX_ATOMS = 250000

# self-validation 대상: (feature 접두사, 기존 parquet, feature 열 목록)
VALIDATE = [
    ('cat1', BASE / 'contact_features/07_run_contact_features/contact_features.parquet', list(cf1.feature_definitions())),
    ('cat2', BASE / 'interaction_features/16_run_interaction_features/interaction_features.parquet', list(it.feature_definitions())),
    ('cat3', BASE / 'surface_features/11_run_surface_features/surface_features.parquet', list(sf.feature_definitions())),
    ('cat4', BASE / 'ligand_geometry/15_run_ligand_geometry/ligand_geometry.parquet', list(lg.feature_definitions())),
    ('cat67', BASE / 'water_metal_features/13_run_water_metal_features/water_metal_features.parquet', list(wm.feature_definitions())),
    ('cat8', BASE / 'ligand_descriptors/12_run_ligand_descriptors/ligand_descriptors.parquet', list(ld.feature_definitions())),
]


def compute_cat3(atoms, lig_rows, rec_rows, unit_points):
    """11번과 동일한 조립으로 SASA feature를 계산합니다."""
    import numpy as np
    lig_el, lig_xyz = sf.extract(atoms, lig_rows)
    rec_el, rec_xyz = sf.extract(atoms, rec_rows)
    lig_r = sf.radii_for(lig_el)
    rec_r = sf.radii_for(rec_el)
    iso = sf.per_atom_sasa(lig_xyz, lig_r, lig_xyz, lig_r, unit_points)
    comp_occ_xyz = np.vstack([lig_xyz, rec_xyz])
    comp_occ_r = np.concatenate([lig_r, rec_r])
    cpx = sf.per_atom_sasa(lig_xyz, lig_r, comp_occ_xyz, comp_occ_r, unit_points)
    polar_mask = np.asarray([e in sf.POLAR_ELEMENTS for e in lig_el])
    iso_total = float(iso.sum())
    cpx_total = float(cpx.sum())
    buried = iso - cpx
    return {
        'ligand_sasa_isolated_A2': round(iso_total, 4),
        'ligand_sasa_complex_A2': round(cpx_total, 4),
        'ligand_buried_sasa_A2': round(iso_total - cpx_total, 4),
        'ligand_buried_fraction': round((iso_total - cpx_total) / iso_total, 4) if iso_total > 0 else None,
        'ligand_sasa_isolated_polar_A2': round(float(iso[polar_mask].sum()), 4),
        'ligand_sasa_isolated_nonpolar_A2': round(float(iso[~polar_mask].sum()), 4),
        'ligand_buried_polar_A2': round(float(buried[polar_mask].sum()), 4),
    }


def compute_cat2(atoms, lig_rows, rec_rows, mapping_entry, smiles):
    """16번과 동일한 조립으로 상호작용 feature와 ligand_typing을 계산합니다."""
    import numpy as np
    lig_el, lig_xyz, _lc, _la = it.extract_atoms(atoms, lig_rows)
    rec_el, rec_xyz, rec_comp, rec_aid = it.extract_atoms(atoms, rec_rows)
    lig_el = np.asarray(lig_el); rec_el = np.asarray(rec_el)
    rec_polar = np.asarray([e in it.POLAR for e in rec_el])
    rec_carbon = rec_el == 'C'
    rec_pos = np.asarray([(c, a) in it.POS_ATOMS for c, a in zip(rec_comp, rec_aid)])
    rec_neg = np.asarray([(c, a) in it.NEG_ATOMS for c, a in zip(rec_comp, rec_aid)])
    rec_arom = np.asarray([(c in it.AROMATIC_RES and a not in it.BACKBONE and e in ('C', 'N'))
                           for c, a, e in zip(rec_comp, rec_aid, rec_el)])
    lig_polar = np.asarray([e in it.POLAR for e in lig_el])
    lig_carbon = lig_el == 'C'
    out = {'potential_hbond_pairs': it.count_pairs(lig_xyz, lig_polar, rec_xyz, rec_polar, it.HBOND_DIST),
           'hydrophobic_pairs': it.count_pairs(lig_xyz, lig_carbon, rec_xyz, rec_carbon, it.HYDROPHOBIC_DIST),
           'receptor_clash_pairs': it.count_pairs(lig_xyz, np.ones(len(lig_el), bool), rec_xyz, np.ones(len(rec_el), bool), it.CLASH_DIST),
           'salt_bridge_pairs': None, 'aromatic_contact_pairs': None}
    typing = 'element_only'
    if mapping_entry and mapping_entry.get('ligand_status') in it.MAPPED_STATUSES \
            and mapping_entry.get('atom_mapping') and smiles:
        arom_by_row, charge_by_row = it.ligand_template_typing(smiles, mapping_entry['atom_mapping'])
        lig_arom = np.asarray([bool(arom_by_row.get(r, False)) for r in lig_rows])
        lig_pos = np.asarray([charge_by_row.get(r, 0) > 0 for r in lig_rows])
        lig_neg = np.asarray([charge_by_row.get(r, 0) < 0 for r in lig_rows])
        out['salt_bridge_pairs'] = int(it.count_pairs(lig_xyz, lig_pos, rec_xyz, rec_neg, it.SALT_DIST)
                                       + it.count_pairs(lig_xyz, lig_neg, rec_xyz, rec_pos, it.SALT_DIST))
        out['aromatic_contact_pairs'] = it.count_pairs(lig_xyz, lig_arom, rec_xyz, rec_arom, it.AROMATIC_DIST)
        typing = 'template_typed'
    return out, typing


def compute_cat67(atoms, lig_rows, rec_rows):
    """13번과 동일한 조립으로 물·금속 feature를 계산합니다."""
    import numpy as np
    from scipy.spatial.distance import cdist
    lig_xyz, _ = cf1.pick_coords(atoms, lig_rows)
    rec_xyz, _ = cf1.pick_coords(atoms, rec_rows)
    water_count, metals = wm.scan_atoms(atoms, lig_rows, rec_rows)
    out = dict.fromkeys(wm.feature_definitions())
    out['water_observed_count'] = int(water_count)
    if water_count > 0:
        water_xyz = []
        for a in atoms:
            if str(a.get('label_comp_id') or '').upper() in wm.WATER_NAMES and str(a.get('type_symbol') or '').upper() == 'O':
                xyz = [audit.finite_number(a.get(k)) for k in ('Cartn_x', 'Cartn_y', 'Cartn_z')]
                if all(v is not None for v in xyz):
                    water_xyz.append(xyz)
        water_xyz = np.asarray(water_xyz, dtype=np.float64)
        near_lig = (cdist(water_xyz, lig_xyz) < wm.WATER_BRIDGE_DIST).any(axis=1)
        near_rec = (cdist(water_xyz, rec_xyz) < wm.WATER_BRIDGE_DIST).any(axis=1)
        out['water_bridge_count'] = int((near_lig & near_rec).sum())
    out['metal_observed_count'] = int(len(metals))
    out['metal_in_ligand_count'] = int(sum(m[2] == 'ligand' for m in metals))
    out['metal_in_receptor_count'] = int(sum(m[2] == 'receptor' for m in metals))
    out['metal_independent_count'] = int(sum(m[2] == 'independent' for m in metals))
    if metals:
        metal_xyz = np.asarray([m[1] for m in metals], dtype=np.float64)
        d_lig = cdist(metal_xyz, lig_xyz)
        d_rec = cdist(metal_xyz, rec_xyz)
        out['min_metal_ligand_distance_A'] = round(float(d_lig.min()), 4)
        out['metal_ligand_receptor_bridge'] = bool(((d_lig < wm.METAL_COORD_DIST).any(axis=1)
                                                    & (d_rec < wm.METAL_COORD_DIST).any(axis=1)).any())
    return out


def compute_cat4(atoms, mapping_entry, smiles):
    """15번과 동일한 조립으로 리간드 내부 기하를 계산합니다(06 매핑 필요)."""
    from rdkit import Chem
    if not mapping_entry or mapping_entry.get('ligand_status') not in lg.MAPPED_STATUSES \
            or not mapping_entry.get('atom_mapping') or not smiles:
        return None
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise ValueError('template_build_failed')
    amap = mapping_entry['atom_mapping']
    nt = template.GetNumAtoms()
    if len(amap) != nt:
        raise ValueError('mapping_atom_count_mismatch')
    rows_by_template = [None] * nt
    for m in amap:
        rows_by_template[int(m['template_atom_index'])] = int(m['output_atom_site_row'])
    if any(r is None for r in rows_by_template):
        raise ValueError('incomplete_template_mapping')
    coords, _ = cf1.pick_coords(atoms, rows_by_template)
    return lg.geometry(template, coords)


def run(args):
    """선정 pose에 통합 추출을 적용하고 기존 산출물과 대조합니다."""
    started = time.perf_counter()
    roles_dir = args.roles_run
    mapping_dir = args.mapping_run
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

    with (roles_dir / 'selected_pose_roles.csv').open(newline='', encoding='utf-8') as handle:
        pose_roles = {(str(r['system_id']), str(r['model']), int(float(r['seed'])), int(float(r['sample']))): r
                      for r in csv.DictReader(handle)}
    receptor_sets, ligand_sets, cif_meta = cf1.load_atom_sets(roles_dir)
    system_smiles = {}
    with (roles_dir / 'input_entities.csv').open(newline='', encoding='utf-8') as handle:
        for r in csv.DictReader(handle):
            if r.get('input_kind') == 'ligand':
                system_smiles.setdefault(r['system_id'], (r.get('canonical_smiles') or '').strip() or (r.get('smiles') or '').strip())
    mapping = {}
    with (mapping_dir / 'ligand_atom_mapping.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                o = json.loads(line)
                mapping[(str(o['system_id']), str(o['model']), int(o['seed']), int(o['sample']))] = o

    all_feat = []
    for _c, _p, feats in VALIDATE:
        all_feat += feats
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
               'cif_sha256': cif_sha, 'ligand_typing': None, 'cat4_available': None,
               'status': None, 'error_message': None, 'elapsed_seconds': None}
        row.update(dict.fromkeys(all_feat))
        try:
            smiles = system_smiles.get(system_id)
            entry = mapping.get(key)
            # cat8은 입력 전용이라 eligible 여부와 무관하게 계산(12번과 동일).
            if smiles:
                try:
                    row.update(ld.descriptors(smiles))
                except Exception:
                    pass
            if pose_role.get('pair_geometry_status') != 'eligible_after_policy_review':
                row['status'] = 'skipped_not_eligible'
            elif ligand_sets.get(key) == 'multiple' or key not in ligand_sets:
                row['status'] = 'skipped_no_ligand_atom_set'
            elif key not in receptor_sets or not receptor_sets[key]:
                row['status'] = 'skipped_no_clean_receptor'
            else:
                atoms, _ = cf1.read_atoms(cif_path, cif_sha, max_bytes)   # CIF 1회 읽기
                lig_rows = ligand_sets[key]['rows']
                rec_rows = []
                for _ch, rr in receptor_sets[key]:
                    rec_rows.extend(rr)
                # cat1 접촉
                rec_xyz, rec_res = cf1.pick_coords(atoms, rec_rows)
                lig_xyz, _ = cf1.pick_coords(atoms, lig_rows)
                row.update(cf1.contacts(lig_xyz, rec_xyz, rec_res))
                # cat3 표면
                row.update(compute_cat3(atoms, lig_rows, rec_rows, unit_points))
                # cat2 상호작용
                c2, typing = compute_cat2(atoms, lig_rows, rec_rows, entry, smiles)
                row.update(c2); row['ligand_typing'] = typing
                # cat6/7 물·금속
                row.update(compute_cat67(atoms, lig_rows, rec_rows))
                # cat4 내부구조(06 매핑 있을 때만)
                c4 = compute_cat4(atoms, entry, smiles)
                row['cat4_available'] = c4 is not None
                if c4:
                    row.update(c4)
                row['status'] = 'computed'
        except Exception as exc:
            row.update(status='error', error_message=f'{type(exc).__name__}: {exc}')
        row['elapsed_seconds'] = time.perf_counter() - begin
        status_counter[row['status']] += 1
        rows_out.append(row)
        if number % 20 == 0 or number == len(pose_roles):
            print(f'통합 추출: {number}/{len(pose_roles)}개', flush=True)

    frame = pd.DataFrame(rows_out)
    buf = io.BytesIO()
    frame.to_parquet(buf, engine='pyarrow', index=False, compression='snappy')
    with helper.checked_path(out / 'combined_features.parquet').open('xb') as handle:
        handle.write(buf.getvalue())

    # self-validation: 기존 per-category 산출물과 값 대조.
    val_rows = []
    mismatches_total = 0
    merged = frame.set_index(list(KEYS))
    for cat, path, feats in VALIDATE:
        ref = pd.read_parquet(path)
        for k in ('seed', 'sample'):
            ref[k] = ref[k].astype(int)
        ref['system_id'] = ref['system_id'].astype(str); ref['model'] = ref['model'].astype(str)
        ref = ref.set_index(list(KEYS))
        for f in feats:
            a = merged[f]
            b = ref[f].reindex(merged.index)
            # 둘 다 null인 자리는 일치로 간주. 수치는 미세 오차 허용.
            both_null = a.isna() & b.isna()
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                diff = (a.fillna(0) - b.fillna(0)).abs()
                bad = int(((diff > 1e-6) & ~both_null).sum())
            else:
                bad = int(((a.astype('object') != b.astype('object')) & ~both_null).sum())
            mismatches_total += bad
            val_rows.append({'category': cat, 'feature': f, 'mismatches': bad})
    audit.save_csv(out / 'validation.csv', val_rows, ('category', 'feature', 'mismatches'))

    computed = status_counter.get('computed', 0)
    errors = status_counter.get('error', 0)
    validated = mismatches_total == 0
    status = 'COMBINED_VALIDATED' if validated and not errors else 'CHECK_REQUIRED'
    report = {
        'version': VERSION, 'status': status, 'output_dir': str(out), 'roles_run': str(roles_dir),
        'mapping_run': str(mapping_dir),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'support_sha256': {n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                           for n in ('07_contact_features.py', '11_surface_features.py', '12_ligand_descriptors.py',
                                     '13_water_metal_features.py', '15_ligand_geometry.py', '16_interaction_features.py')},
        'python': sys.executable, 'poses': len(rows_out), 'computed': computed, 'errors': errors,
        'status_counts': dict(status_counter), 'feature_count': len(all_feat),
        'validation_mismatches_total': mismatches_total, 'validated_against_per_category': validated,
        'excluded': 'cat5 (P2Rank)은 외부 바이너리라 통합 제외, 14번으로 별도 유지.',
        'elapsed_seconds': time.perf_counter() - started, 'gt_used': False, 'coordinates_modified': False,
        'notes': ['CIF를 pose당 1회만 읽어 cat1·2·3·4·6/7·8을 계산합니다.',
                  '기존 per-category 산출물과 전 값 대조로 표본 재현을 검증합니다(A6a).',
                  '전체 확대(A6b·A6c)에서는 입력(역할·원자 대응)을 pose별 계산으로 교체합니다.']}
    helper.write_json(out / 'combined_report.json', report)
    helper.write_json(out / 'RUN_COMPLETE.json', {'outputs_written': True, 'extraction_finished': True,
        'status': status, 'poses': len(rows_out), 'computed': computed, 'errors': errors,
        'validated': validated})
    print('STATUS:', status, flush=True)
    print('POSES:', len(rows_out), '| COMPUTED:', computed, '| ERRORS:', errors, flush=True)
    print('VALIDATION_MISMATCHES:', mismatches_total, '(0이면 표본 재현 성공)', flush=True)
    print('STATUS_COUNTS:', roles.j(dict(status_counter)), flush=True)
    return 0 if status == 'COMBINED_VALIDATED' else 2


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
