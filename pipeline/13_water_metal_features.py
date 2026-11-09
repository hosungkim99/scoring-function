#!/usr/bin/env python3
"""물 매개(category 6)와 금속 관련(category 7) feature를 추출합니다.

관찰된 0과 정보 부재(null)를 엄격히 구분합니다. 예측 구조에 명시적 물이 없으면 물 bridge는 0이
아니라 null(계산 불가)로 두고, 물 분자 수 0은 '관찰된 사실'로 기록합니다. 금속도 마찬가지로 금속
수 0은 관찰이지만 배위 기하는 금속이 없으면 null입니다. 좌표와 원소만 사용하므로 06번 원자 대응이
필요 없습니다. 금속은 리간드 내부/수용체 내부/독립 이온을 구분합니다. GT를 사용하지 않고 원본
좌표를 수정하지 않습니다. 07번 import로 원자 집합 로딩·CIF 읽기를 재사용하고 05번 eligible pose
(152개)를 대상으로 합니다.

출력: out/water_metal_features/13_run_water_metal_features/ 아래 고정 폴더.
"""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import Counter
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

VERSION = 'water_metal_features_0.1.0'
OUT_KIND = 'water_metal_features'
OUT_NAME = '13_run_water_metal_features'
DEFAULT_ROLES = helper.ALLOWED_ROOT / 'Scoring_Function' / 'out' / 'input_roles' / '05_run_audit_input_roles'
KEYS = ('system_id', 'model', 'seed', 'sample')
MAX_ATOMS = 250000
WATER_NAMES = {'HOH', 'DOD', 'WAT', 'H2O'}
WATER_BRIDGE_DIST = 3.5   # 물-단백질/리간드 수소결합 가능 거리(Å)
METAL_COORD_DIST = 2.8    # 금속 배위로 볼 수 있는 거리(Å)
METAL_LIGAND_DIST = 3.0   # 금속이 리간드와 연관됐다고 볼 거리(Å)


def feature_definitions():
    """물·금속 feature의 정의와 단위, 그리고 결측 의미를 고정합니다."""
    return {
        # category 6: 물 매개
        'water_observed_count': {'unit': 'count', 'definition': '예측 구조에서 관찰된 물 분자 수. 0도 관찰된 사실로 기록.'},
        'water_bridge_count': {'unit': 'count', 'definition': '리간드와 수용체 모두와 수소결합 거리 내인 물 수. 물이 없으면 null(계산 불가).'},
        # category 7: 금속
        'metal_observed_count': {'unit': 'count', 'definition': '구조에서 관찰된 금속 원자 수(관찰된 사실, 0 포함).'},
        'metal_in_ligand_count': {'unit': 'count', 'definition': '리간드 원자 집합에 포함된 금속 수.'},
        'metal_in_receptor_count': {'unit': 'count', 'definition': '수용체 원자 집합에 포함된 금속 수.'},
        'metal_independent_count': {'unit': 'count', 'definition': '리간드·수용체 어느 쪽도 아닌 독립 금속 원자 수.'},
        'min_metal_ligand_distance_A': {'unit': 'angstrom', 'definition': '금속과 리간드 중원자 사이 최소 거리. 금속이 없으면 null.'},
        'metal_ligand_receptor_bridge': {'unit': 'bool', 'definition': '리간드와 수용체 모두를 배위 거리 내에 두는 금속 존재 여부. 금속이 없으면 null.'},
    }


def specification():
    return {
        'schema_version': VERSION,
        'scope': '물 매개(cat 6)·금속(cat 7)의 관찰과 기하. 관찰된 0과 정보 부재를 구분.',
        'features': feature_definitions(),
        'absence_policy': {
            'water': '명시적 물 미관찰 시 water_observed_count=0(관찰), water_bridge_count=null(계산 불가).',
            'metal': '금속 미관찰 시 금속 수=0(관찰), 거리·bridge=null(계산 불가).',
            'rule': '계산 불가·정보 부재를 0으로 채우지 않습니다.'},
        'distances': {'water_bridge': WATER_BRIDGE_DIST, 'metal_coordination': METAL_COORD_DIST,
                      'metal_ligand_association': METAL_LIGAND_DIST},
        'method': {'metal_detection': 'gemmi Element.is_metal, 중원자.',
                   'water_detection': 'comp_id가 물 이름이거나 단일 산소 잔기.',
                   'classification': '금속을 리간드/수용체/독립으로 05 원자 집합 기준 구분.'},
        'notes': ['co-folding 예측은 보통 명시적 물을 출력하지 않아 물 매개는 대체로 null이 될 수 있습니다.',
                  '리간드 내부 금속(예: 유기 수은)과 독립 금속 이온을 구분합니다.',
                  'GT·좌표 수정 없음.']}


def scan_atoms(atoms, lig_rows, rec_rows):
    """물 분자와 금속 원자를 찾고 금속을 리간드/수용체/독립으로 분류합니다."""
    import gemmi
    import numpy as np
    lig_set, rec_set = set(lig_rows), set(rec_rows)
    water_residues = set()
    metals = []          # (row, xyz, scope)
    for i, a in enumerate(atoms, start=1):
        sym = str(a.get('type_symbol') or '').upper()
        el = gemmi.Element(sym)
        if el.atomic_number <= 1:
            continue
        comp = str(a.get('label_comp_id') or '').upper()
        if comp in WATER_NAMES and sym == 'O':
            water_residues.add(tuple(a.get(k) for k in audit.RESIDUE_FIELDS))
        if el.is_metal:
            xyz = [audit.finite_number(a.get(k)) for k in ('Cartn_x', 'Cartn_y', 'Cartn_z')]
            if any(v is None for v in xyz):
                continue
            scope = 'ligand' if i in lig_set else ('receptor' if i in rec_set else 'independent')
            metals.append((i, np.asarray(xyz, dtype=np.float64), scope))
    return len(water_residues), metals


def run(args):
    """eligible pose의 물·금속 feature를 관찰/부재 구분과 함께 저장합니다."""
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
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    from scipy.spatial.distance import cdist
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)

    spec = specification()
    spec_hash = hashlib.sha256((json.dumps(spec, ensure_ascii=False, sort_keys=True,
                                           allow_nan=False)).encode()).hexdigest()
    helper.write_json(out / 'feature_spec.json', spec)

    source_names = ('RUN_COMPLETE.json', 'selected_pose_roles.csv', 'selected_atom_sets.jsonl')
    sources = [helper.fingerprint(roles_dir / n) for n in source_names]
    complete = json.loads((roles_dir / 'RUN_COMPLETE.json').read_text(encoding='utf-8'))
    if complete.get('status') != 'AUDIT_COMPLETE' or complete.get('outputs_written') is not True:
        raise ValueError('05번 입력 역할 점검이 정상 완료된 결과가 아닙니다.')

    import csv
    with (roles_dir / 'selected_pose_roles.csv').open(newline='', encoding='utf-8') as handle:
        pose_roles = {(str(r['system_id']), str(r['model']), int(float(r['seed'])), int(float(r['sample']))): r
                      for r in csv.DictReader(handle)}
    receptor_sets, ligand_sets, cif_meta = cf.load_atom_sets(roles_dir)

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
               'config_sha256': spec_hash, 'cif_sha256': cif_sha,
               'water_status': None, 'metal_status': None, 'status': None,
               'skip_reasons_json': '[]', 'error_message': None, 'elapsed_seconds': None}
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
                lig = ligand_sets[key]
                lig_rows = lig['rows']
                rec_rows = []
                for _c, rr in receptor_sets[key]:
                    rec_rows.extend(rr)
                lig_xyz, _ = cf.pick_coords(atoms, lig_rows)
                rec_xyz, _ = cf.pick_coords(atoms, rec_rows)
                water_count, metals = scan_atoms(atoms, lig_rows, rec_rows)

                # --- category 6: 물 매개 (관찰 vs 부재 구분) ---
                row['water_observed_count'] = int(water_count)
                if water_count == 0:
                    row['water_status'] = 'observed_absent'   # water_bridge_count는 null 유지
                else:
                    # 물 산소 좌표로 bridge 계산(리간드·수용체 모두에 근접한 물).
                    water_xyz = []
                    for a in atoms:
                        if str(a.get('label_comp_id') or '').upper() in WATER_NAMES and str(a.get('type_symbol') or '').upper() == 'O':
                            xyz = [audit.finite_number(a.get(k)) for k in ('Cartn_x', 'Cartn_y', 'Cartn_z')]
                            if all(v is not None for v in xyz):
                                water_xyz.append(xyz)
                    water_xyz = np.asarray(water_xyz, dtype=np.float64)
                    near_lig = (cdist(water_xyz, lig_xyz) < WATER_BRIDGE_DIST).any(axis=1)
                    near_rec = (cdist(water_xyz, rec_xyz) < WATER_BRIDGE_DIST).any(axis=1)
                    row['water_bridge_count'] = int((near_lig & near_rec).sum())
                    row['water_status'] = 'computed'

                # --- category 7: 금속 (관찰 vs 부재 구분) ---
                row['metal_observed_count'] = int(len(metals))
                row['metal_in_ligand_count'] = int(sum(m[2] == 'ligand' for m in metals))
                row['metal_in_receptor_count'] = int(sum(m[2] == 'receptor' for m in metals))
                row['metal_independent_count'] = int(sum(m[2] == 'independent' for m in metals))
                if not metals:
                    row['metal_status'] = 'observed_absent'   # 거리·bridge는 null 유지
                else:
                    metal_xyz = np.asarray([m[1] for m in metals], dtype=np.float64)
                    d_lig = cdist(metal_xyz, lig_xyz)
                    d_rec = cdist(metal_xyz, rec_xyz)
                    row['min_metal_ligand_distance_A'] = round(float(d_lig.min()), 4)
                    # 리간드·수용체 모두를 배위 거리 내에 두는 금속이 하나라도 있으면 bridge.
                    bridge = bool(((d_lig < METAL_COORD_DIST).any(axis=1) & (d_rec < METAL_COORD_DIST).any(axis=1)).any())
                    row['metal_ligand_receptor_bridge'] = bool(bridge)
                    row['metal_status'] = 'computed'

                row['status'] = 'computed'
        except Exception as exc:
            row.update(status='error', error_message=f'{type(exc).__name__}: {exc}')
        row['elapsed_seconds'] = time.perf_counter() - begin
        status_counter[row['status']] += 1
        rows_out.append(row)
        if number % 20 == 0 or number == len(pose_roles):
            print(f'물·금속 feature: {number}/{len(pose_roles)}개', flush=True)

    frame = pd.DataFrame(rows_out)
    if frame.duplicated(list(KEYS)).any():
        raise RuntimeError('pose 식별자 중복이 있습니다.')
    buf = io.BytesIO()
    frame.to_parquet(buf, engine='pyarrow', index=False, compression='snappy')
    with helper.checked_path(out / 'water_metal_features.parquet').open('xb') as handle:
        handle.write(buf.getvalue())
    restored = pd.read_parquet(out / 'water_metal_features.parquet', engine='pyarrow')
    pd.testing.assert_frame_equal(restored, frame, check_dtype=False)

    status_rows = [{'status': s, 'count': c} for s, c in sorted(status_counter.items())]
    audit.save_csv(out / 'pose_status.csv', status_rows, ('status', 'count'))

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
    water_present = int((comp['water_observed_count'] > 0).sum()) if len(comp) else 0
    metal_present = int((comp['metal_observed_count'] > 0).sum()) if len(comp) else 0
    status = 'FEATURES_COMPLETE' if not errors and not changed else 'CHECK_REQUIRED'
    report = {
        'version': VERSION, 'status': status, 'output_dir': str(out), 'roles_run': str(roles_dir),
        'sources': sources, 'config_sha256': spec_hash,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'support_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (_support_path, cf._support_path, roles._support_path, audit._support_path)},
        'python': sys.executable, 'gemmi_version': gemmi.__version__, 'numpy_version': np.__version__,
        'pyarrow_version': pa.__version__, 'poses': len(rows_out), 'computed': computed, 'errors': errors,
        'status_counts': dict(status_counter), 'changed_sources': changed, 'feature_category': '6,7',
        'poses_with_water_observed': water_present, 'poses_with_metal_observed': metal_present,
        'elapsed_seconds': time.perf_counter() - started, 'gt_used': False, 'coordinates_modified': False,
        'limitations': [
            '명시적 물 미관찰 시 bridge는 null(계산 불가)이며 물리적 0이 아닙니다.',
            '금속 수 0은 관찰이지만 배위 기하는 금속이 없으면 null입니다.',
            '리간드 내부 금속과 독립 금속 이온을 구분하되 배위 화학 전체를 확정하지 않습니다.',
            'eligible이 아닌 pose는 계산하지 않고 상태만 남깁니다.',
            'GT·좌표 수정 없음.']}
    helper.write_json(out / 'audit_report.json', report)
    helper.write_json(out / 'RUN_COMPLETE.json', {'outputs_written': True, 'features_finished': True,
        'status': status, 'poses': len(rows_out), 'computed': computed, 'errors': errors,
        'poses_with_water_observed': water_present, 'poses_with_metal_observed': metal_present,
        'sources_unchanged': not changed})
    print('STATUS:', status, flush=True)
    print('POSES:', len(rows_out), '| COMPUTED:', computed, '| ERRORS:', errors, flush=True)
    print('WATER_OBSERVED_POSES:', water_present, '| METAL_OBSERVED_POSES:', metal_present, flush=True)
    print('STATUS_COUNTS:', roles.j(dict(status_counter)), flush=True)
    return 0 if status == 'FEATURES_COMPLETE' else 2


def main():
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
