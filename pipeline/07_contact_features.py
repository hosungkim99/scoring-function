#!/usr/bin/env python3
"""선정된 pose의 단백질-리간드 접촉·거리 feature(카테고리 1)를 추출합니다.

01번 네 구조 파일럿의 13개 feature를, 05번이 검증한 수용체·리간드 원자 집합을 이용해
선정 160개 pose로 일반화합니다. 원본 좌표를 수정하지 않고 H/D를 제외한 중원자만 씁니다.
수용체는 역할검토 플래그가 없는 매칭 단백질 체인의 합집합, 리간드는 매칭된 비폴리머 원자
집합으로 정의합니다(핸드오프 8.1: 모든 polymer를 조건 없이 합치지 않음). GT를 사용하지 않고
좌표 기반 접촉만 계산하며, 역할이 모호하거나 eligible이 아닌 pose는 계산하지 않고 상태만
남깁니다. 같은 폴더의 03/04/05 Python 파일에서 경로 보호·CIF 파싱·저장 함수를 재사용합니다.

출력: out/contact_features/07_run_contact_features/ 아래 고정 폴더.
"""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import time

_support_path = Path(__file__).with_name('05_audit_input_roles.py')
_spec = importlib.util.spec_from_file_location('roles_support', _support_path)
roles = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(roles)
audit = roles.audit        # 04 모듈
helper = roles.helper      # 03 모듈

VERSION = 'contact_features_0.1.0'
OUT_KIND = 'contact_features'
OUT_NAME = '07_run_contact_features'
DEFAULT_ROLES = helper.ALLOWED_ROOT / 'Scoring_Function' / 'out' / 'input_roles' / '05_run_audit_input_roles'
DEFAULT_MAPPING = helper.ALLOWED_ROOT / 'Scoring_Function' / 'out' / 'ligand_mapping' / '06_run_map_ligand_atoms'
KEYS = ('system_id', 'model', 'seed', 'sample')
MAX_ATOMS = 250000
CUTOFFS = (3, 4, 5)


def feature_definitions():
    """추출하는 feature 이름과 정의를 고정합니다(01번과 동일한 13개)."""
    result = {'pl_min_distance_A': {'unit': 'angstrom',
              'definition': '선택한 단백질·리간드 중원자 사이 최소 거리.'}}
    for c in CUTOFFS:
        result.update({
            f'pl_pairs_lt_{c}A': {'unit': 'count', 'definition': f'거리 < {c}Å인 단백질-리간드 원자쌍 수.'},
            f'ligand_atoms_lt_{c}A': {'unit': 'count', 'definition': f'어떤 단백질 원자와 거리 < {c}Å인 리간드 원자 수.'},
            f'protein_atoms_lt_{c}A': {'unit': 'count', 'definition': f'어떤 리간드 원자와 거리 < {c}Å인 단백질 원자 수.'},
            f'protein_residues_lt_{c}A': {'unit': 'count', 'definition': f'어떤 리간드 원자와 거리 < {c}Å인 단백질 원자를 가진 잔기 수.'},
        })
    return result


def specification():
    """feature 정의와 입력 선택 정책을 기록합니다."""
    return {
        'schema_version': VERSION,
        'scope': '05번이 검증한 eligible pose의 수용체-리간드 접촉·거리(카테고리 1).',
        'features': feature_definitions(),
        'selection': {
            'receptor': '매칭된 단백질 입력 체인(id_sequence_consistent) 중 역할검토 플래그가 없는 체인의 중원자 합집합.',
            'ligand': '매칭된 비폴리머(id_composition_consistent)의 중원자 집합. 05번의 1부터 시작하는 행 번호 사용.',
            'eligibility': 'pair_geometry_status == eligible_after_policy_review 인 pose만 계산.',
            'hydrogens': 'H/D 제외(05번이 저장한 중원자 행만 사용).',
            'distance': '원본 공통 좌표계의 유클리드 거리, strict < cutoff.',
            'held_out': 'hold·역할모호·수용체 없음·리간드 없음은 계산하지 않고 상태만 기록.'},
        'provenance': 'CIF SHA256과 원자 행 번호, 06번 리간드 대응 상태를 함께 기록.',
        'notes': ['pose 품질 순위나 최종 scoring 성능이 아니라 pose별 접촉 기하 feature입니다.',
                  '원자 단위 화학 대응이 필요 없는 카테고리이므로 06번의 대칭·mismatch와 무관합니다.',
                  '수용체에 역할검토 단백질 체인을 포함하지 않아 과다 집계를 피합니다.']}


def load_atom_sets(roles_dir):
    """05번의 selected_atom_sets.jsonl을 pose별 수용체·리간드 원자 집합으로 정리합니다."""
    receptor = defaultdict(list)   # key -> [(chain, rows)]
    ligand = {}                    # key -> dict
    cif_meta = {}                  # key -> (cif_path, cif_sha256)
    with (roles_dir / 'selected_atom_sets.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            key = (str(o['system_id']), str(o['model']), int(o['seed']), int(o['sample']))
            cif_meta[key] = (o.get('cif_path'), o.get('cif_sha256'))
            rows_json = o.get('heavy_atom_site_rows_1based_json')
            rows = json.loads(rows_json) if rows_json else None
            flags = json.loads(o.get('role_review_flags_json') or '[]')
            if o.get('input_kind') == 'protein' and o.get('match_status') == 'id_sequence_consistent' \
                    and not flags and rows:
                receptor[key].append((o.get('observed_chain_id'), rows))
            elif o.get('input_kind') == 'ligand' and o.get('match_status') == 'id_composition_consistent' and rows:
                # eligible pose에서는 리간드 대응 체인 그룹이 하나이므로 마지막 값으로 덮어쓰지 않습니다.
                if key in ligand:
                    ligand[key] = 'multiple'
                else:
                    ligand[key] = {'chain': o.get('observed_chain_id'), 'rows': rows}
    return receptor, ligand, cif_meta


def load_mapping_status(mapping_dir):
    """있으면 06번 리간드 대응 상태를 provenance로 읽습니다(없어도 진행)."""
    path = mapping_dir / 'ligand_atom_mapping.csv'
    status = {}
    try:
        import csv
        with path.open(newline='', encoding='utf-8') as handle:
            for r in csv.DictReader(handle):
                key = (str(r['system_id']), str(r['model']), int(float(r['seed'])), int(float(r['sample'])))
                status[key] = r.get('ligand_status')
    except OSError:
        return None
    return status


def read_atoms(cif_path, expected_sha, max_bytes):
    """CIF를 재검증 후 원자 표를 돌려줍니다. 원본은 수정하지 않습니다."""
    import gemmi
    raw, fp = roles.read_bytes(cif_path, max_bytes)
    if fp['sha256'] != expected_sha:
        raise ValueError('cif_changed_since_audit')
    block = gemmi.cif.read_string(raw.decode('utf-8')).sole_block()
    return audit.rows(block, '_atom_site.', MAX_ATOMS), fp


def pick_coords(atoms, rows_1based):
    """행 번호로 중원자의 좌표와 잔기 키를 추출합니다."""
    import gemmi
    import numpy as np
    total = len(atoms)
    coords, res_keys = [], []
    for r in rows_1based:
        if not isinstance(r, int) or r < 1 or r > total:
            raise ValueError('atom_site_row_out_of_range')
        a = atoms[r - 1]
        if gemmi.Element(str(a.get('type_symbol') or '').upper()).atomic_number <= 1:
            raise ValueError('non_heavy_atom_in_selected_rows')
        xyz = [audit.finite_number(a.get(k)) for k in ('Cartn_x', 'Cartn_y', 'Cartn_z')]
        if any(v is None for v in xyz):
            raise ValueError('invalid_coordinate')
        coords.append(xyz)
        res_keys.append(tuple(a.get(k) for k in audit.RESIDUE_FIELDS))
    return np.asarray(coords, dtype=np.float64), res_keys


def contacts(ligand_xyz, receptor_xyz, receptor_res_keys):
    """리간드-수용체 거리에서 13개 접촉 feature를 계산합니다."""
    import numpy as np
    from scipy.spatial.distance import cdist
    if not (np.isfinite(ligand_xyz).all() and np.isfinite(receptor_xyz).all()):
        raise ValueError('non_finite_coordinates')
    distances = cdist(ligand_xyz, receptor_xyz)
    values = {'pl_min_distance_A': float(distances.min())}
    for c in CUTOFFS:
        mask = distances < c
        near_receptor = mask.any(axis=0)
        values[f'pl_pairs_lt_{c}A'] = int(mask.sum())
        values[f'ligand_atoms_lt_{c}A'] = int(mask.any(axis=1).sum())
        values[f'protein_atoms_lt_{c}A'] = int(near_receptor.sum())
        values[f'protein_residues_lt_{c}A'] = len({receptor_res_keys[i]
                                                   for i, near in enumerate(near_receptor) if near})
    return values


def run(args):
    """05번 결과와 원본 CIF로 접촉 feature를 계산해 고정 폴더에 저장합니다."""
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

    # 05번 완료 기록과 입력 파일의 출처를 확인합니다.
    source_names = ('RUN_COMPLETE.json', 'selected_pose_roles.csv', 'selected_atom_sets.jsonl')
    sources = [helper.fingerprint(roles_dir / name) for name in source_names]
    complete = json.loads((roles_dir / 'RUN_COMPLETE.json').read_text(encoding='utf-8'))
    if complete.get('status') != 'AUDIT_COMPLETE' or complete.get('outputs_written') is not True:
        raise ValueError('05번 입력 역할 점검이 정상 완료된 결과가 아닙니다.')

    import csv
    with (roles_dir / 'selected_pose_roles.csv').open(newline='', encoding='utf-8') as handle:
        pose_roles = {(str(r['system_id']), str(r['model']), int(float(r['seed'])), int(float(r['sample']))): r
                      for r in csv.DictReader(handle)}
    receptor_sets, ligand_sets, cif_meta = load_atom_sets(roles_dir)
    mapping_status = load_mapping_status(args.mapping_run)

    feature_names = list(feature_definitions())
    rows_out = []
    status_counter = Counter()
    pose_keys = sorted(pose_roles)
    for number, key in enumerate(pose_keys, 1):
        system_id, model, seed, sample = key
        pose_role = pose_roles[key]
        cif_path, cif_sha = cif_meta.get(key, (None, None))
        row = {'system_id': system_id, 'model': model, 'seed': seed, 'sample': sample,
               'pose_id': f'{system_id}|{model}|{seed}|{sample}',
               'input_path': pose_role.get('input_path'), 'schema_version': VERSION,
               'config_sha256': spec_hash, 'cif_sha256': cif_sha,
               'ligand_mapping_status': (mapping_status or {}).get(key),
               'receptor_chain_ids_json': None, 'ligand_chain': None,
               'receptor_heavy_atoms': None, 'ligand_heavy_atoms': None,
               'status': None, 'error_message': None}
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
                atoms, fp = read_atoms(cif_path, cif_sha, args.max_cif_mib * 1024 ** 2)
                receptor_rows, chains = [], []
                for chain, rows in receptor_sets[key]:
                    receptor_rows.extend(rows)
                    chains.append(chain)
                if len(set(receptor_rows)) != len(receptor_rows):
                    raise ValueError('receptor_rows_overlap')
                lig = ligand_sets[key]
                if set(receptor_rows) & set(lig['rows']):
                    raise ValueError('receptor_ligand_rows_overlap')
                rec_xyz, rec_res = pick_coords(atoms, receptor_rows)
                lig_xyz, _ = pick_coords(atoms, lig['rows'])
                if not len(rec_xyz) or not len(lig_xyz):
                    raise ValueError('empty_receptor_or_ligand')
                row.update(contacts(lig_xyz, rec_xyz, rec_res))
                row.update(status='computed',
                           receptor_chain_ids_json=audit.encoded(sorted(c for c in chains if c)),
                           ligand_chain=lig['chain'], receptor_heavy_atoms=len(rec_xyz),
                           ligand_heavy_atoms=len(lig_xyz))
        except Exception as exc:
            row.update(status='error', error_message=f'{type(exc).__name__}: {exc}')
        status_counter[row['status']] += 1
        rows_out.append(row)
        if number % 20 == 0 or number == len(pose_keys):
            print(f'접촉 feature: {number}/{len(pose_keys)}개', flush=True)

    frame = pd.DataFrame(rows_out)
    if frame.duplicated(list(KEYS)).any():
        raise RuntimeError('pose 식별자 중복이 있습니다.')
    buf = io.BytesIO()
    frame.to_parquet(buf, engine='pyarrow', index=False, compression='snappy')
    with helper.checked_path(out / 'contact_features.parquet').open('xb') as handle:
        handle.write(buf.getvalue())
    restored = pd.read_parquet(out / 'contact_features.parquet', engine='pyarrow')
    pd.testing.assert_frame_equal(restored, frame, check_dtype=False)

    status_rows = [{'status': status, 'count': count} for status, count in sorted(status_counter.items())]
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
    status = 'FEATURES_COMPLETE' if not errors and not changed else 'CHECK_REQUIRED'
    report = {
        'version': VERSION, 'status': status, 'output_dir': str(out), 'roles_run': str(roles_dir),
        'mapping_run': str(args.mapping_run) if mapping_status is not None else None,
        'sources': sources, 'config_sha256': spec_hash,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'support_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (_support_path, roles._support_path, audit._support_path)},
        'python': sys.executable, 'gemmi_version': gemmi.__version__, 'numpy_version': np.__version__,
        'pyarrow_version': pa.__version__, 'poses': len(rows_out), 'computed': computed,
        'errors': errors, 'status_counts': dict(status_counter), 'changed_sources': changed,
        'elapsed_seconds': time.perf_counter() - started,
        'gt_used': False, 'coordinates_modified': False, 'feature_category': 1,
        'limitations': [
            '같은 시스템 pose 사이의 접촉 기하만 계산하며 pose 품질이나 scoring 성능을 뜻하지 않습니다.',
            '수용체는 역할검토 없는 매칭 단백질 체인만 포함하고 짧은·겹침 체인은 제외합니다.',
            'eligible이 아닌 pose(hold 포함)와 수용체·리간드 집합이 없는 pose는 계산하지 않고 상태만 남깁니다.',
            '거리는 원본 공통 좌표계에서 계산하며 좌표를 수정하지 않습니다.',
            'GT 구조와 평가 점수를 feature 계산에 사용하지 않습니다.']}
    helper.write_json(out / 'audit_report.json', report)
    helper.write_json(out / 'RUN_COMPLETE.json', {'outputs_written': True, 'features_finished': True,
        'status': status, 'poses': len(rows_out), 'computed': computed, 'errors': errors,
        'sources_unchanged': not changed})
    print('STATUS:', status, flush=True)
    print('POSES:', len(rows_out), flush=True)
    print('COMPUTED:', computed, flush=True)
    print('ERRORS:', errors, flush=True)
    print('STATUS_COUNTS:', roles.j(dict(status_counter)), flush=True)
    return 0 if status == 'FEATURES_COMPLETE' else 2


def main():
    """입력 위치와 CIF 읽기 상한을 지정하며 출력 위치는 사용자 경로로 고정합니다."""
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
