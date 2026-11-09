#!/usr/bin/env python3
"""선정 pose의 리간드 표면적·매몰 feature(category 3)를 추출합니다.

FreeSASA가 설치되어 있지 않아 Shrake-Rupley 알고리즘을 numpy/scipy로 직접 구현합니다. 원소별
van der Waals 반경과 좌표만 사용하므로 06번의 원자 단위 화학 대응이 필요 없습니다(category 1과
같은 수준). 수소는 예측 구조에 없으므로 중원자 기준 SASA 근사이며, 이를 명시합니다. 원본 좌표를
수정하지 않고 GT를 사용하지 않습니다. 05번이 검증한 수용체·리간드 원자 집합과 07번과 동일한
eligible pose(152개)를 대상으로 하며, 07번을 import해 원자 집합 로딩·CIF 읽기 함수를 재사용합니다.

출력: out/surface_features/11_run_surface_features/ 아래 고정 폴더.
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

VERSION = 'surface_features_0.1.0'
OUT_KIND = 'surface_features'
OUT_NAME = '11_run_surface_features'
DEFAULT_ROLES = helper.ALLOWED_ROOT / 'Scoring_Function' / 'out' / 'input_roles' / '05_run_audit_input_roles'
KEYS = ('system_id', 'model', 'seed', 'sample')
MAX_ATOMS = 250000
PROBE_RADIUS = 1.4        # 물 분자 probe 반경(Å)
SPHERE_POINTS = 256       # 원자당 표면 샘플 점 수
POLAR_ELEMENTS = {'N', 'O'}
# Bondi 등에서 가져온 원소별 van der Waals 반경(Å). 미정의 원소는 계산 보류 사유로 남깁니다.
VDW_RADII = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'P': 1.80, 'S': 1.80,
    'CL': 1.75, 'BR': 1.85, 'I': 1.98, 'SE': 1.90, 'B': 1.92, 'SI': 2.10,
    'NA': 2.27, 'MG': 1.73, 'K': 2.75, 'CA': 2.31, 'MN': 2.05, 'FE': 2.05,
    'CO': 2.00, 'NI': 1.97, 'CU': 2.00, 'ZN': 2.10, 'HG': 1.55, 'CD': 2.18,
}


def feature_definitions():
    """추출하는 표면적·매몰 feature의 정의와 단위를 고정합니다."""
    return {
        'ligand_sasa_isolated_A2': {'unit': 'angstrom^2', 'definition': '수용체 없이 리간드 중원자만의 용매접근표면적(SASA).'},
        'ligand_sasa_complex_A2': {'unit': 'angstrom^2', 'definition': '수용체와 함께일 때 리간드 중원자의 노출 SASA.'},
        'ligand_buried_sasa_A2': {'unit': 'angstrom^2', 'definition': 'isolated - complex. 결합으로 매몰된 리간드 표면적.'},
        'ligand_buried_fraction': {'unit': 'ratio', 'definition': 'buried / isolated. isolated=0이면 null.'},
        'ligand_sasa_isolated_polar_A2': {'unit': 'angstrom^2', 'definition': 'isolated SASA 중 N·O 원자 기여.'},
        'ligand_sasa_isolated_nonpolar_A2': {'unit': 'angstrom^2', 'definition': 'isolated SASA 중 N·O 외 원자 기여.'},
        'ligand_buried_polar_A2': {'unit': 'angstrom^2', 'definition': 'N·O 원자의 매몰 SASA.'},
    }


def specification():
    """feature 정의와 방법·정책을 기록합니다."""
    return {
        'schema_version': VERSION,
        'scope': '05번 eligible pose의 리간드 표면적·매몰(category 3). Shrake-Rupley 직접 구현.',
        'features': feature_definitions(),
        'method': {
            'algorithm': 'Shrake-Rupley, 원자당 구 표면 샘플점으로 SASA 추정.',
            'sphere_points': SPHERE_POINTS, 'probe_radius': PROBE_RADIUS,
            'vdw_source': 'Bondi 등 표준 반경표(코드에 고정). 미정의 원소는 pose 보류.',
            'hydrogens': '예측 구조에 수소가 없어 중원자 기준 SASA 근사입니다.',
            'isolated': '가림 원자 = 리간드 중원자만.',
            'complex': '가림 원자 = 리간드 + 수용체 중원자.',
            'receptor': '역할검토 없는 매칭 단백질 체인(07번과 동일).',
            'ligand': '매칭된 비폴리머 중원자(05번 원자 집합).'},
        'not_performed': ['수소 복원', '결합 차수·전하 기반 반경 보정', 'GT 사용', '좌표 수정'],
        'notes': ['buried는 샘플 점 추정 노이즈로 소폭 음수가 될 수 있어 원값과 함께 상태를 남깁니다.',
                  'category 1의 접촉 수와 상보적인 매몰 면적 관점의 feature입니다.']}


def fibonacci_sphere(n):
    """단위 구에 고르게 분포한 n개의 점을 만듭니다."""
    import numpy as np
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    golden = np.pi * (1.0 + 5.0 ** 0.5)
    theta = golden * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1)


def extract(atoms, rows_1based):
    """행 번호로 중원자의 원소 기호와 좌표를 추출합니다."""
    import gemmi
    import numpy as np
    total = len(atoms)
    elems, coords = [], []
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
    return elems, np.asarray(coords, dtype=np.float64)


def radii_for(elements):
    """원소 기호 목록을 van der Waals 반경 배열로 바꾸며 미정의 원소를 보고합니다."""
    import numpy as np
    unknown = sorted({e for e in elements if e not in VDW_RADII})
    if unknown:
        raise ValueError('unknown_vdw_element:' + ','.join(unknown))
    return np.asarray([VDW_RADII[e] for e in elements], dtype=np.float64)


def per_atom_sasa(target_xyz, target_r, occ_xyz, occ_r, unit_points):
    """대상 원자들의 SASA를 가림 원자 집합 아래에서 원자별로 계산합니다."""
    import numpy as np
    from scipy.spatial import cKDTree
    probe_t = target_r + PROBE_RADIUS
    probe_o = occ_r + PROBE_RADIUS
    tree = cKDTree(occ_xyz)
    max_o = float(probe_o.max()) if len(probe_o) else 0.0
    out = np.zeros(len(target_xyz), dtype=np.float64)
    n = len(unit_points)
    for i in range(len(target_xyz)):
        center = target_xyz[i]
        ri = probe_t[i]
        surface = center + unit_points * ri
        # 이 원자의 표면점을 가릴 수 있는 이웃만 고려합니다.
        idx = tree.query_ball_point(center, ri + max_o)
        accessible = np.ones(n, dtype=bool)
        for j in idx:
            d = occ_xyz[j] - center
            if abs(d[0]) < 1e-9 and abs(d[1]) < 1e-9 and abs(d[2]) < 1e-9 and probe_o[j] == ri:
                continue  # 자기 자신(동일 좌표·반경)은 건너뜁니다.
            diff = surface - occ_xyz[j]
            inside = np.einsum('ij,ij->i', diff, diff) < probe_o[j] ** 2
            accessible &= ~inside
            if not accessible.any():
                break
        out[i] = 4.0 * np.pi * ri * ri * accessible.sum() / n
    return out


def run(args):
    """05번 eligible pose에 대해 리간드 SASA·매몰 feature를 계산해 저장합니다."""
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
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)

    spec = specification()
    spec_hash = hashlib.sha256((json.dumps(spec, ensure_ascii=False, sort_keys=True,
                                           allow_nan=False)).encode()).hexdigest()
    helper.write_json(out / 'feature_spec.json', spec)
    unit_points = fibonacci_sphere(SPHERE_POINTS)

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
               'ligand_heavy_atoms': None, 'receptor_heavy_atoms': None,
               'status': None, 'skip_reasons_json': '[]', 'error_message': None, 'elapsed_seconds': None}
        row.update(dict.fromkeys(feature_names))
        try:
            if pose_role.get('pair_geometry_status') != 'eligible_after_policy_review':
                row['status'] = 'skipped_not_eligible'
            elif ligand_sets.get(key) == 'multiple':
                row['status'] = 'skipped_multiple_ligand_sets'
            elif key not in ligand_sets:
                row['status'] = 'skipped_no_ligand_atom_set'
            elif key not in receptor_sets or not receptor_sets[key]:
                row['status'] = 'skipped_no_clean_receptor'
            else:
                atoms, _ = cf.read_atoms(cif_path, cif_sha, max_bytes)
                receptor_rows = []
                for _chain, rr in receptor_sets[key]:
                    receptor_rows.extend(rr)
                lig = ligand_sets[key]
                lig_elems, lig_xyz = extract(atoms, lig['rows'])
                rec_elems, rec_xyz = extract(atoms, receptor_rows)
                try:
                    lig_r = radii_for(lig_elems)
                    rec_r = radii_for(rec_elems)
                except ValueError as exc:
                    row['status'] = 'skipped_unknown_vdw'
                    row['skip_reasons_json'] = audit.encoded([str(exc)])
                    raise _Skip()
                # isolated: 가림 = 리간드만. complex: 가림 = 리간드 + 수용체.
                iso = per_atom_sasa(lig_xyz, lig_r, lig_xyz, lig_r, unit_points)
                comp_occ_xyz = np.vstack([lig_xyz, rec_xyz])
                comp_occ_r = np.concatenate([lig_r, rec_r])
                cpx = per_atom_sasa(lig_xyz, lig_r, comp_occ_xyz, comp_occ_r, unit_points)
                polar_mask = np.asarray([e in POLAR_ELEMENTS for e in lig_elems])
                iso_total = float(iso.sum())
                cpx_total = float(cpx.sum())
                buried = iso - cpx
                row.update({
                    'ligand_sasa_isolated_A2': round(iso_total, 4),
                    'ligand_sasa_complex_A2': round(cpx_total, 4),
                    'ligand_buried_sasa_A2': round(iso_total - cpx_total, 4),
                    'ligand_buried_fraction': round((iso_total - cpx_total) / iso_total, 4) if iso_total > 0 else None,
                    'ligand_sasa_isolated_polar_A2': round(float(iso[polar_mask].sum()), 4),
                    'ligand_sasa_isolated_nonpolar_A2': round(float(iso[~polar_mask].sum()), 4),
                    'ligand_buried_polar_A2': round(float(buried[polar_mask].sum()), 4),
                    'status': 'computed', 'ligand_heavy_atoms': len(lig_elems),
                    'receptor_heavy_atoms': len(rec_elems)})
        except _Skip:
            pass
        except Exception as exc:
            row.update(status='error', error_message=f'{type(exc).__name__}: {exc}')
        row['elapsed_seconds'] = time.perf_counter() - begin
        status_counter[row['status']] += 1
        rows_out.append(row)
        if number % 20 == 0 or number == len(pose_roles):
            print(f'표면 feature: {number}/{len(pose_roles)}개', flush=True)

    frame = pd.DataFrame(rows_out)
    if frame.duplicated(list(KEYS)).any():
        raise RuntimeError('pose 식별자 중복이 있습니다.')
    buf = io.BytesIO()
    frame.to_parquet(buf, engine='pyarrow', index=False, compression='snappy')
    with helper.checked_path(out / 'surface_features.parquet').open('xb') as handle:
        handle.write(buf.getvalue())
    restored = pd.read_parquet(out / 'surface_features.parquet', engine='pyarrow')
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
    neg_buried = int((comp['ligand_buried_sasa_A2'] < -1.0).sum()) if len(comp) else 0
    status = 'FEATURES_COMPLETE' if not errors and not changed else 'CHECK_REQUIRED'
    report = {
        'version': VERSION, 'status': status, 'output_dir': str(out), 'roles_run': str(roles_dir),
        'sources': sources, 'config_sha256': spec_hash,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'support_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (_support_path, cf._support_path, roles._support_path, audit._support_path)},
        'python': sys.executable, 'gemmi_version': gemmi.__version__, 'numpy_version': np.__version__,
        'pyarrow_version': pa.__version__, 'poses': len(rows_out), 'computed': computed, 'errors': errors,
        'status_counts': dict(status_counter), 'changed_sources': changed, 'feature_category': 3,
        'sphere_points': SPHERE_POINTS, 'probe_radius': PROBE_RADIUS,
        'buried_negative_gt1A2': neg_buried,
        'elapsed_seconds': time.perf_counter() - started,
        'gt_used': False, 'coordinates_modified': False,
        'limitations': [
            '예측 구조에 수소가 없어 중원자 기준 SASA 근사입니다.',
            'Shrake-Rupley 샘플점 추정이라 buried가 소폭 음수가 될 수 있습니다(원값 보존).',
            '미정의 van der Waals 반경 원소는 계산하지 않고 보류합니다.',
            'eligible이 아닌 pose(hold 포함)와 집합이 없는 pose는 계산하지 않고 상태만 남깁니다.',
            'GT 구조·평가 점수를 feature 계산에 사용하지 않습니다.']}
    helper.write_json(out / 'audit_report.json', report)
    helper.write_json(out / 'RUN_COMPLETE.json', {'outputs_written': True, 'features_finished': True,
        'status': status, 'poses': len(rows_out), 'computed': computed, 'errors': errors,
        'sources_unchanged': not changed})
    print('STATUS:', status, flush=True)
    print('POSES:', len(rows_out), '| COMPUTED:', computed, '| ERRORS:', errors, flush=True)
    print('STATUS_COUNTS:', roles.j(dict(status_counter)), flush=True)
    print('BURIED_NEGATIVE(<-1A2):', neg_buried, flush=True)
    return 0 if status == 'FEATURES_COMPLETE' else 2


class _Skip(Exception):
    """정상 보류를 오류와 구분하기 위한 내부 신호입니다."""


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
