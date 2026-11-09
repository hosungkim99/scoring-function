#!/usr/bin/env python3
"""리간드 내부 구조 타당성(category 4) feature를 06번 원자 대응으로 추출합니다.

06번이 확립한 입력 SMILES 템플릿↔출력 원자 대응을 이용해, 템플릿의 결합 그래프를 출력 좌표에
전사합니다(06 보강). 이로써 어떤 원자쌍이 결합인지 알 수 있어 결합 길이 분포와 비결합 내부 충돌을
계산합니다. 이상 기하 기준표(이상 결합 길이·각)를 쓰지 않고, 관찰된 결합 길이 분포와 위상 거리 기반
충돌만 기록하여 가정을 최소화합니다. 결합 차수 자체가 확정된 full/symmetric 매핑(06)만 대상으로 하며,
연결성 불일치(유기 수은 등)·역할 보류 pose는 사유를 남기고 계산하지 않습니다. GT·좌표 수정 없음.

07번을 import해 CIF 읽기·좌표 추출을 재사용하고, 06번 ligand_atom_mapping.jsonl과 05번 리간드
SMILES를 읽습니다.

출력: out/ligand_geometry/15_run_ligand_geometry/ 아래 고정 폴더.
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

VERSION = 'ligand_geometry_0.1.0'
OUT_KIND = 'ligand_geometry'
OUT_NAME = '15_run_ligand_geometry'
DEFAULT_ROLES = helper.ALLOWED_ROOT / 'Scoring_Function' / 'out' / 'input_roles' / '05_run_audit_input_roles'
DEFAULT_MAPPING = helper.ALLOWED_ROOT / 'Scoring_Function' / 'out' / 'ligand_mapping' / '06_run_map_ligand_atoms'
KEYS = ('system_id', 'model', 'seed', 'sample')
MAPPED_STATUSES = {'clean_single_full_mapping', 'clean_single_symmetric_mapping'}
LONG_BOND = 1.9      # 중원자-중원자 결합으로는 비정상적으로 긴 길이(Å)
SHORT_BOND = 1.0     # 비정상적으로 짧은 길이(Å)
CLASH_TOPO_SEP = 4   # 이 이상 떨어진(결합 경로) 원자쌍만 충돌 후보로 봄
CLASH_DIST = 2.5     # 그런 쌍이 이 거리보다 가까우면 내부 충돌(Å)


def feature_definitions():
    """리간드 내부 기하 feature의 정의와 단위를 고정합니다."""
    return {
        'lig_bond_count': {'unit': 'count', 'definition': '템플릿 결합 수(전사된 결합 그래프).'},
        'lig_bond_len_min_A': {'unit': 'angstrom', 'definition': '관찰된 결합 길이 최소.'},
        'lig_bond_len_max_A': {'unit': 'angstrom', 'definition': '관찰된 결합 길이 최대.'},
        'lig_bond_len_mean_A': {'unit': 'angstrom', 'definition': '관찰된 결합 길이 평균.'},
        'lig_long_bonds': {'unit': 'count', 'definition': f'길이 > {LONG_BOND}Å인 결합 수.'},
        'lig_short_bonds': {'unit': 'count', 'definition': f'길이 < {SHORT_BOND}Å인 결합 수.'},
        'lig_internal_clashes': {'unit': 'count', 'definition': f'결합 경로 {CLASH_TOPO_SEP}이상 떨어졌는데 3D 거리 < {CLASH_DIST}Å인 비결합 원자쌍 수.'},
        'lig_min_nonbonded_dist_A': {'unit': 'angstrom', 'definition': f'결합 경로 {CLASH_TOPO_SEP}이상 쌍의 최소 3D 거리. 그런 쌍이 없으면 null.'},
    }


def specification():
    return {
        'schema_version': VERSION,
        'scope': '06 대응으로 결합 그래프를 전사한 리간드의 내부 기하(category 4).',
        'features': feature_definitions(),
        'method': {
            'bond_graph': '입력 SMILES 템플릿의 결합을 06 원자 대응으로 출력 좌표에 전사.',
            'targets': '06의 full/symmetric 매핑 pose만. 대칭 매핑은 결합 집합이 동일하여 길이 분포가 보존됨.',
            'no_ideal_table': '이상 결합 길이·각 기준표를 쓰지 않고 관찰 분포와 위상 거리 충돌만 기록.',
            'clash': f'위상 거리 ≥{CLASH_TOPO_SEP}인 비결합 쌍이 3D로 {CLASH_DIST}Å보다 가까우면 충돌.'},
        'excluded': ['연결성 불일치(예: 유기 수은)·역할 보류 pose는 결합 차수 전사 불가로 계산하지 않음.'],
        'notes': ['수소가 없어 수소 관련 기하는 다루지 않습니다.',
                  '대칭 매핑에서 개별 원자 식별은 모호하나 결합 길이 집합과 충돌 수는 잘 정의됩니다.',
                  'GT·좌표 수정 없음.']}


def geometry(template, coords):
    """전사된 결합 그래프와 출력 좌표로 결합 길이와 내부 충돌을 계산합니다."""
    from rdkit import Chem
    import numpy as np
    n = template.GetNumAtoms()
    topo = Chem.GetDistanceMatrix(template)   # 위상(결합 경로) 거리 행렬
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in template.GetBonds()]
    bond_lengths = [float(np.linalg.norm(coords[a] - coords[b])) for a, b in bonds]
    # 비결합(위상 거리 ≥ 기준) 쌍의 3D 거리.
    far_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            if topo[i][j] >= CLASH_TOPO_SEP:
                far_dists.append(float(np.linalg.norm(coords[i] - coords[j])))
    values = {
        'lig_bond_count': len(bonds),
        'lig_bond_len_min_A': round(min(bond_lengths), 4) if bond_lengths else None,
        'lig_bond_len_max_A': round(max(bond_lengths), 4) if bond_lengths else None,
        'lig_bond_len_mean_A': round(sum(bond_lengths) / len(bond_lengths), 4) if bond_lengths else None,
        'lig_long_bonds': int(sum(d > LONG_BOND for d in bond_lengths)),
        'lig_short_bonds': int(sum(d < SHORT_BOND for d in bond_lengths)),
        'lig_internal_clashes': int(sum(d < CLASH_DIST for d in far_dists)),
        'lig_min_nonbonded_dist_A': round(min(far_dists), 4) if far_dists else None,
    }
    return values


def run(args):
    """06 대응이 있는 pose의 리간드 내부 기하를 계산해 저장합니다."""
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
    from rdkit import Chem
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
    map_complete = json.loads((mapping_dir / 'RUN_COMPLETE.json').read_text(encoding='utf-8'))
    if map_complete.get('status') != 'MAPPING_COMPLETE' or map_complete.get('outputs_written') is not True:
        raise ValueError('06번 원자 대응이 정상 완료된 결과가 아닙니다.')

    with (roles_dir / 'selected_pose_roles.csv').open(newline='', encoding='utf-8') as handle:
        pose_keys = [(str(r['system_id']), str(r['model']), int(float(r['seed'])), int(float(r['sample'])))
                     for r in csv.DictReader(handle)]
    # 리간드 SMILES(시스템별).
    system_smiles = {}
    with (roles_dir / 'input_entities.csv').open(newline='', encoding='utf-8') as handle:
        for r in csv.DictReader(handle):
            if r.get('input_kind') == 'ligand':
                smi = (r.get('canonical_smiles') or '').strip() or (r.get('smiles') or '').strip()
                system_smiles.setdefault(r['system_id'], smi)
    # 06 대응(pose별).
    mapping = {}
    with (mapping_dir / 'ligand_atom_mapping.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            key = (str(o['system_id']), str(o['model']), int(o['seed']), int(o['sample']))
            mapping[key] = o

    feature_names = list(feature_definitions())
    max_bytes = args.max_cif_mib * 1024 ** 2
    rows_out = []
    status_counter = Counter()
    for number, key in enumerate(sorted(pose_keys), 1):
        system_id, model, seed, sample = key
        begin = time.perf_counter()
        row = {'system_id': system_id, 'model': model, 'seed': seed, 'sample': sample,
               'pose_id': f'{system_id}|{model}|{seed}|{sample}', 'schema_version': VERSION,
               'config_sha256': spec_hash, 'cif_sha256': None, 'ligand_mapping_status': None,
               'status': None, 'skip_reasons_json': '[]', 'error_message': None, 'elapsed_seconds': None}
        row.update(dict.fromkeys(feature_names))
        try:
            entry = mapping.get(key)
            if entry is None:
                row['status'] = 'skipped_no_mapping_record'
            else:
                row['ligand_mapping_status'] = entry.get('ligand_status')
                row['cif_sha256'] = entry.get('cif_sha256')
                if entry.get('ligand_status') not in MAPPED_STATUSES or not entry.get('atom_mapping'):
                    row['status'] = 'skipped_no_bondorder_mapping'
                    row['skip_reasons_json'] = audit.encoded([str(entry.get('ligand_status'))])
                else:
                    smi = system_smiles.get(system_id)
                    template = Chem.MolFromSmiles(smi) if smi else None
                    if template is None:
                        raise ValueError('template_build_failed')
                    amap = entry['atom_mapping']
                    nt = template.GetNumAtoms()
                    if len(amap) != nt:
                        raise ValueError('mapping_atom_count_mismatch')
                    # 템플릿 원자 index → 출력 행 번호 정렬.
                    rows_by_template = [None] * nt
                    for m in amap:
                        rows_by_template[int(m['template_atom_index'])] = int(m['output_atom_site_row'])
                    if any(r is None for r in rows_by_template):
                        raise ValueError('incomplete_template_mapping')
                    atoms, _ = cf.read_atoms(entry['cif_path'], entry['cif_sha256'], max_bytes)
                    coords, _ = cf.pick_coords(atoms, rows_by_template)  # 템플릿 index 순서의 좌표
                    row.update(geometry(template, coords))
                    row['status'] = 'computed'
        except Exception as exc:
            row.update(status='error', error_message=f'{type(exc).__name__}: {exc}')
        row['elapsed_seconds'] = time.perf_counter() - begin
        status_counter[row['status']] += 1
        rows_out.append(row)
        if number % 20 == 0 or number == len(pose_keys):
            print(f'리간드 내부 기하: {number}/{len(pose_keys)}개', flush=True)

    frame = pd.DataFrame(rows_out)
    if frame.duplicated(list(KEYS)).any():
        raise RuntimeError('pose 식별자 중복이 있습니다.')
    buf = io.BytesIO()
    frame.to_parquet(buf, engine='pyarrow', index=False, compression='snappy')
    with helper.checked_path(out / 'ligand_geometry.parquet').open('xb') as handle:
        handle.write(buf.getvalue())
    restored = pd.read_parquet(out / 'ligand_geometry.parquet', engine='pyarrow')
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
    poses_with_clash = int((comp['lig_internal_clashes'] > 0).sum()) if len(comp) else 0
    poses_with_long_bond = int((comp['lig_long_bonds'] > 0).sum()) if len(comp) else 0
    status = 'FEATURES_COMPLETE' if not errors and not changed else 'CHECK_REQUIRED'
    report = {
        'version': VERSION, 'status': status, 'output_dir': str(out), 'roles_run': str(roles_dir),
        'mapping_run': str(mapping_dir), 'sources': sources, 'config_sha256': spec_hash,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'support_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (_support_path, cf._support_path, roles._support_path, audit._support_path)},
        'python': sys.executable, 'rdkit_version': rdkit.__version__, 'numpy_version': np.__version__,
        'pyarrow_version': pa.__version__, 'poses': len(rows_out), 'computed': computed, 'errors': errors,
        'status_counts': dict(status_counter), 'changed_sources': changed, 'feature_category': 4,
        'poses_with_internal_clash': poses_with_clash, 'poses_with_long_bond': poses_with_long_bond,
        'elapsed_seconds': time.perf_counter() - started, 'gt_used': False, 'coordinates_modified': False,
        'limitations': [
            '06 결합차수 전사가 된 full/symmetric 매핑 pose만 대상이며 수은 mismatch·보류는 계산하지 않습니다.',
            '이상 기하 기준표 없이 관찰 결합 길이 분포와 위상 거리 충돌만 기록합니다.',
            '수소가 없어 수소 관련 기하는 다루지 않습니다.',
            'GT·좌표 수정 없음.']}
    helper.write_json(out / 'audit_report.json', report)
    helper.write_json(out / 'RUN_COMPLETE.json', {'outputs_written': True, 'features_finished': True,
        'status': status, 'poses': len(rows_out), 'computed': computed, 'errors': errors,
        'sources_unchanged': not changed})
    print('STATUS:', status, flush=True)
    print('POSES:', len(rows_out), '| COMPUTED:', computed, '| ERRORS:', errors, flush=True)
    print('POSES_WITH_CLASH:', poses_with_clash, '| POSES_WITH_LONG_BOND:', poses_with_long_bond, flush=True)
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
