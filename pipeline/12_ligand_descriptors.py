#!/usr/bin/env python3
"""리간드 고유 특성(category 8) feature를 입력 SMILES에서 추출합니다.

입력 SMILES와 RDKit만 사용하므로 예측 좌표·CIF·06번 원자 대응이 필요 없습니다. 같은 시스템의
100개 pose는 입력 리간드가 동일하여 이 feature는 타깃 내에서 상수입니다. 단독 pose 순위화에는
도움이 되지 않지만 다른 feature를 해석하는 조건 정보로 보존합니다(일괄 제외하지 않음). GT를
사용하지 않습니다. 05번 input_entities.csv의 리간드 SMILES와 selected_pose_roles.csv의 pose
식별자를 읽어, 시스템별로 계산한 값을 선정 pose로 펼쳐 기록합니다. 05→04→03 함수를 재사용합니다.

출력: out/ligand_descriptors/12_run_ligand_descriptors/ 아래 고정 폴더.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import csv
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
audit = roles.audit
helper = roles.helper

VERSION = 'ligand_descriptors_0.1.0'
OUT_KIND = 'ligand_descriptors'
OUT_NAME = '12_run_ligand_descriptors'
DEFAULT_ROLES = helper.ALLOWED_ROOT / 'Scoring_Function' / 'out' / 'input_roles' / '05_run_audit_input_roles'
KEYS = ('system_id', 'model', 'seed', 'sample')


def feature_definitions():
    """RDKit로 계산하는 리간드 고유 특성 feature의 정의와 단위를 고정합니다."""
    return {
        'lig_mol_weight': {'unit': 'g/mol', 'definition': '리간드 분자량(RDKit MolWt).'},
        'lig_heavy_atoms': {'unit': 'count', 'definition': '중원자 수.'},
        'lig_rotatable_bonds': {'unit': 'count', 'definition': '회전 가능 결합 수(CalcNumRotatableBonds).'},
        'lig_hbd': {'unit': 'count', 'definition': '수소 결합 주개 수(CalcNumHBD).'},
        'lig_hba': {'unit': 'count', 'definition': '수소 결합 받개 수(CalcNumHBA).'},
        'lig_tpsa': {'unit': 'angstrom^2', 'definition': '위상학적 극성 표면적(CalcTPSA).'},
        'lig_logp': {'unit': 'dimensionless', 'definition': 'Crippen 계산 logP(MolLogP).'},
        'lig_ring_count': {'unit': 'count', 'definition': '고리 수(CalcNumRings).'},
        'lig_aromatic_rings': {'unit': 'count', 'definition': '방향족 고리 수(CalcNumAromaticRings).'},
        'lig_fraction_csp3': {'unit': 'ratio', 'definition': 'sp3 탄소 비율(CalcFractionCSP3).'},
        'lig_num_heteroatoms': {'unit': 'count', 'definition': '헤테로 원자 수(CalcNumHeteroatoms).'},
        'lig_formal_charge': {'unit': 'charge', 'definition': '분자 형식 전하(GetFormalCharge).'},
    }


def specification():
    """feature 정의와 계산 정책을 기록합니다."""
    return {
        'schema_version': VERSION,
        'scope': '입력 SMILES 기반 리간드 고유 특성(category 8). 좌표·CIF·원자 대응 불필요.',
        'features': feature_definitions(),
        'method': {
            'input': '05번 input_entities.csv의 리간드 canonical_smiles(없으면 smiles).',
            'tool': 'RDKit 2D 그래프 descriptor(좌표 미생성).',
            'granularity': '시스템별 1회 계산 후 같은 시스템의 선정 pose로 펼침(타깃 내 상수).',
            'gt': 'GT 미사용.'},
        'interpretation': ['타깃 내 상수이므로 단독 Top-k 순위화에는 기여하지 않습니다.',
                           '다른 feature를 해석하는 조건 정보(리간드 크기·유연성 등)로 보존합니다.'],
        'status_values': {
            'computed': '입력 SMILES에서 정상 계산.',
            'smiles_unavailable': '리간드 입력 SMILES가 없음.',
            'rdkit_parse_failed': 'SMILES를 RDKit이 해석하지 못함.'}}


def descriptors(smiles):
    """입력 SMILES에서 RDKit 2D descriptor를 계산합니다. 좌표를 만들지 않습니다."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('rdkit_parse_failed')
    return {
        'lig_mol_weight': round(float(Descriptors.MolWt(mol)), 4),
        'lig_heavy_atoms': int(mol.GetNumHeavyAtoms()),
        'lig_rotatable_bonds': int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        'lig_hbd': int(rdMolDescriptors.CalcNumHBD(mol)),
        'lig_hba': int(rdMolDescriptors.CalcNumHBA(mol)),
        'lig_tpsa': round(float(rdMolDescriptors.CalcTPSA(mol)), 4),
        'lig_logp': round(float(Crippen.MolLogP(mol)), 4),
        'lig_ring_count': int(rdMolDescriptors.CalcNumRings(mol)),
        'lig_aromatic_rings': int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        'lig_fraction_csp3': round(float(rdMolDescriptors.CalcFractionCSP3(mol)), 4),
        'lig_num_heteroatoms': int(rdMolDescriptors.CalcNumHeteroatoms(mol)),
        'lig_formal_charge': int(Chem.GetFormalCharge(mol)),
    }


def run(args):
    """선정 시스템 리간드의 고유 특성을 계산해 선정 pose로 펼쳐 저장합니다."""
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

    import rdkit
    import pandas as pd
    import pyarrow as pa
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)

    spec = specification()
    spec_hash = hashlib.sha256((json.dumps(spec, ensure_ascii=False, sort_keys=True,
                                           allow_nan=False)).encode()).hexdigest()
    helper.write_json(out / 'feature_spec.json', spec)

    sources = [helper.fingerprint(roles_dir / n) for n in ('RUN_COMPLETE.json', 'selected_pose_roles.csv', 'input_entities.csv')]
    complete = json.loads((roles_dir / 'RUN_COMPLETE.json').read_text(encoding='utf-8'))
    if complete.get('status') != 'AUDIT_COMPLETE' or complete.get('outputs_written') is not True:
        raise ValueError('05번 입력 역할 점검이 정상 완료된 결과가 아닙니다.')

    with (roles_dir / 'selected_pose_roles.csv').open(newline='', encoding='utf-8') as handle:
        pose_rows = [{k: r[k] for k in KEYS} for r in csv.DictReader(handle)]
    selected_systems = sorted({r['system_id'] for r in pose_rows})

    # 선정 시스템의 리간드 입력 SMILES를 모읍니다.
    ligand_smiles = {}
    with (roles_dir / 'input_entities.csv').open(newline='', encoding='utf-8') as handle:
        for r in csv.DictReader(handle):
            if r.get('input_kind') == 'ligand' and r['system_id'] in set(selected_systems):
                smi = (r.get('canonical_smiles') or '').strip() or (r.get('smiles') or '').strip()
                chain = r.get('input_chain_id')
                prev = ligand_smiles.get(r['system_id'])
                if prev is None:
                    ligand_smiles[r['system_id']] = {'smiles': smi, 'chain': chain}
                else:
                    prev['multiple'] = True  # 리간드 입력이 둘 이상이면 표시만 남깁니다.

    feature_names = list(feature_definitions())
    system_results = {}
    for sid in selected_systems:
        entry = ligand_smiles.get(sid)
        rec = {'system_id': sid, 'ligand_input_chain': entry.get('chain') if entry else None,
               'canonical_smiles': entry.get('smiles') if entry else None,
               'multiple_ligand_inputs': bool(entry and entry.get('multiple')),
               'status': None, 'error_message': None}
        rec.update(dict.fromkeys(feature_names))
        try:
            if not entry or not entry.get('smiles'):
                rec['status'] = 'smiles_unavailable'
            else:
                rec.update(descriptors(entry['smiles']))
                rec['status'] = 'computed'
        except Exception as exc:
            rec['status'] = 'rdkit_parse_failed' if 'rdkit_parse_failed' in str(exc) else 'error'
            rec['error_message'] = f'{type(exc).__name__}: {exc}'
        system_results[sid] = rec

    # 시스템 값을 선정 pose로 펼칩니다(타깃 내 상수).
    pose_out = []
    for r in pose_rows:
        sid = r['system_id']
        base = system_results[sid]
        row = {**r, 'pose_id': f"{sid}|{r['model']}|{r['seed']}|{r['sample']}",
               'schema_version': VERSION, 'config_sha256': spec_hash,
               'ligand_input_chain': base['ligand_input_chain'], 'canonical_smiles': base['canonical_smiles'],
               'pose_independent': True, 'status': base['status'], 'error_message': base['error_message']}
        for f in feature_names:
            row[f] = base[f]
        pose_out.append(row)

    frame = pd.DataFrame(pose_out)
    buf = io.BytesIO()
    frame.to_parquet(buf, engine='pyarrow', index=False, compression='snappy')
    with helper.checked_path(out / 'ligand_descriptors.parquet').open('xb') as handle:
        handle.write(buf.getvalue())
    restored = pd.read_parquet(out / 'ligand_descriptors.parquet', engine='pyarrow')
    pd.testing.assert_frame_equal(restored, frame, check_dtype=False)

    audit.save_csv(out / 'ligand_descriptors_by_system.csv', list(system_results.values()),
                   ('system_id', 'ligand_input_chain', 'status'))
    from collections import Counter
    status_counter = Counter(v['status'] for v in system_results.values())
    audit.save_csv(out / 'system_status.csv', [{'status': s, 'count': c} for s, c in sorted(status_counter.items())],
                   ('status', 'count'))

    changed = []
    for source in sources:
        try:
            info = Path(source['path']).stat()
            if (info.st_size, info.st_mtime_ns) != (source['size_bytes'], source['mtime_ns']):
                changed.append(source['path'])
        except OSError:
            changed.append(source['path'])
    computed = status_counter.get('computed', 0)
    errors = status_counter.get('error', 0) + status_counter.get('rdkit_parse_failed', 0)
    status = 'FEATURES_COMPLETE' if not errors and not changed else 'CHECK_REQUIRED'
    report = {
        'version': VERSION, 'status': status, 'output_dir': str(out), 'roles_run': str(roles_dir),
        'sources': sources, 'config_sha256': spec_hash,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'support_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (_support_path, roles._support_path, audit._support_path)},
        'python': sys.executable, 'rdkit_version': rdkit.__version__, 'pyarrow_version': pa.__version__,
        'selected_systems': len(selected_systems), 'computed_systems': computed,
        'poses': len(pose_out), 'system_status_counts': dict(status_counter), 'errors': errors,
        'changed_sources': changed, 'feature_category': 8, 'elapsed_seconds': time.perf_counter() - started,
        'gt_used': False, 'coordinates_modified': False, 'pose_independent': True,
        'limitations': [
            '타깃 내 상수 feature이므로 단독 순위화에는 기여하지 않습니다(조건 정보로 보존).',
            '입력 SMILES 2D 그래프 기반이며 예측 좌표의 3D conformer를 쓰지 않습니다.',
            'SMILES 부재·RDKit 해석 실패를 상태로 구분하며 값을 0으로 채우지 않습니다.',
            'GT 구조·평가 점수를 사용하지 않습니다.']}
    helper.write_json(out / 'audit_report.json', report)
    helper.write_json(out / 'RUN_COMPLETE.json', {'outputs_written': True, 'features_finished': True,
        'status': status, 'selected_systems': len(selected_systems), 'computed_systems': computed,
        'errors': errors, 'sources_unchanged': not changed})
    print('STATUS:', status, flush=True)
    print('SELECTED_SYSTEMS:', len(selected_systems), '| COMPUTED:', computed, '| ERRORS:', errors, flush=True)
    print('SYSTEM_STATUS:', roles.j(dict(status_counter)), flush=True)
    return 0 if status == 'FEATURES_COMPLETE' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roles-run', type=Path, default=DEFAULT_ROLES)
    args = parser.parse_args()
    return run(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
