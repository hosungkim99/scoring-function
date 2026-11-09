#!/usr/bin/env python3
"""gnina docking score smoke test(과정 B 준비). score_only vs local_only 변별력 비교용 소규모 추출.

후보군 2(상호작용·반발)에 gnina CNN score를 feature로 넣기 전에, 소수 pose에서 두 모드를 모두
돌려 어느 쪽이 near-native pose를 더 잘 가려내는지 확인하기 위한 실험용 스크립트다. 전체 460k
추출은 이 smoke test로 플래그·출력 파싱·비용을 확정한 뒤 별도 스크립트로 만든다.

입력 준비:
- 수용체 PDB: 원본 pose CIF에서 리간드·물을 떼어낸 단백질만(gemmi). 원본은 수정하지 않고 결과
  폴더에 파생 파일로 쓴다.
- 리간드 SDF: 입력 SMILES 템플릿(결합 차수·방향성 보유)에 pose의 중원자 좌표를 06 연결성 대응으로
  얹고 수소를 좌표와 함께 추가(RDKit). 좌표만으로 결합을 추론하면 복잡한 리간드에서 틀리기 쉬우므로
  템플릿 화학을 쓴다.

대상: 05/06 표본(선정 152 pose) 중 '깨끗한 단일 리간드' 조건을 통과하고 06 연결성 대응이 성립한
pose에서 --limit개를 고른다. GT는 쓰지 않는다(sc_rmsd와의 변별력 비교는 이 표를 라벨에 join하는
후속 분석 단계에서 한다).

주의(첫 실행 시 확인 필요): gnina 실행 파일 경로·플래그·stdout 형식은 환경마다 다를 수 있다.
--gnina로 경로를 주고, 1~2 pose dry run으로 CNNscore/CNNaffinity/Affinity 파싱이 맞는지 먼저
확인한 뒤 --limit을 늘릴 것.
출력: out/gnina_smoke/23_run_gnina_smoke/ 아래 고정 폴더.
"""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import Counter
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

import importlib.util


def _load(name):
    p = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(name.replace('.py', ''), p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

m06 = _load('06_map_ligand_atoms.py')     # 리간드 추출·연결성 대응 재사용
roles = m06.roles
audit = m06.audit
helper = m06.helper

VERSION = 'gnina_smoke_0.1.0'
OUT_KIND = 'gnina_smoke'
OUT_NAME = '23_run_gnina_smoke'
BASE = helper.ALLOWED_ROOT / 'Scoring_Function' / 'out'
DEFAULT_ROLES = BASE / 'input_roles' / '05_run_audit_input_roles'
DEFAULT_MAP = BASE / 'ligand_mapping' / '06_run_map_ligand_atoms'
KEYS = ('system_id', 'model', 'seed', 'sample')

SCORE_PATTERNS = {
    'affinity': re.compile(r'Affinity:\s*(-?\d+(?:\.\d+)?)'),
    'cnn_score': re.compile(r'CNNscore:\s*(-?\d+(?:\.\d+)?)'),
    'cnn_affinity': re.compile(r'CNNaffinity:\s*(-?\d+(?:\.\d+)?)'),
}


def build_receptor_pdb(cif_path, rec_path):
    """pose CIF에서 리간드·물을 떼어낸 단백질만 PDB로 씁니다(원본 미수정)."""
    import gemmi
    st = gemmi.read_structure(str(cif_path))
    st.setup_entities()
    st.remove_ligands_and_waters()
    st.remove_empty_chains()
    if sum(1 for _ in st[0].all()) == 0:
        raise ValueError('receptor_empty_after_strip')
    st.write_pdb(str(rec_path))
    return rec_path


def build_ligand_sdf(smiles, extracted, mapping, lig_path):
    """템플릿 SMILES 분자에 pose 좌표를 얹고 수소를 추가해 리간드 SDF를 씁니다."""
    from rdkit import Chem
    from rdkit.Geometry import Point3D
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('ligand_smiles_unparseable')
    if mol.GetNumAtoms() != len(mapping):
        raise ValueError('mapping_atom_count_mismatch')
    conf = Chem.Conformer(mol.GetNumAtoms())
    for entry in mapping:
        i = entry['template_atom_index']
        order = entry['output_atom_order']
        x, y, z = extracted['coords'][order]
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)
    Chem.SanitizeMol(mol)
    molH = Chem.AddHs(mol, addCoords=True)   # 좌표 기반 수소 배치(극성 수소 포함)
    writer = Chem.SDWriter(str(lig_path))
    writer.write(molH)
    writer.close()
    return lig_path


def parse_gnina(stdout):
    """gnina stdout에서 Affinity·CNNscore·CNNaffinity를 파싱합니다(없으면 None)."""
    out = {}
    for key, pat in SCORE_PATTERNS.items():
        found = pat.findall(stdout)
        out[key] = float(found[-1]) if found else None   # 마지막(최종 pose) 값
    return out


def run_gnina(gnina_cmd, rec_path, lig_path, extra_args, env, timeout):
    """gnina를 한 pose에 실행하고 파싱 결과와 raw stdout(앞부분)을 돌려줍니다."""
    # gnina(PyTorch/CUDA CNN)는 TMPDIR/XDG_CACHE_HOME가 프로젝트(<FS_ROOT>) 폴더로 바뀌면
    # CUDA 커널 캐시(~/.cache)를 못 찾아 죽습니다(CUDA error: no kernel image). 이 변수들을
    # 제거해 기본값(공유 홈 ~/.cache)을 쓰게 하여 정상 GPU 실행을 보장합니다.
    run_env = {k: v for k, v in env.items() if k not in ('TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME')}
    cmd = [*gnina_cmd, '-r', str(rec_path), '-l', str(lig_path), *extra_args]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=run_env)
    stdout = proc.stdout or ''
    if proc.returncode != 0:
        err = proc.stderr or stdout or ''
        msg = (err[:400] + ' … ' + err[-200:]) if len(err) > 600 else err
        raise RuntimeError(f'gnina_failed(rc={proc.returncode}): {msg}')
    parsed = parse_gnina(stdout)
    return parsed, stdout[:1500]


def collect_targets(roles_dir, map_dir, limit, max_bytes):
    """05/06 표본에서 깨끗한 단일 리간드 + 연결성 대응 성립 pose를 골라 대상 목록을 만듭니다."""
    pose_roles = {m06.key_of(r): r for r in m06.read_csv_rows(roles_dir / 'selected_pose_roles.csv')}
    ligand_entities = {}
    for r in m06.read_csv_rows(roles_dir / 'input_entities.csv'):
        if r.get('input_kind') == 'ligand':
            ligand_entities[(str(r['system_id']), str(r['input_chain_id']))] = r
    ligand_sets = {}
    for row in m06.read_jsonl_rows(roles_dir / 'selected_atom_sets.jsonl'):
        if row.get('input_kind') != 'ligand':
            continue
        k = m06.key_of(row)
        ligand_sets[k] = 'multiple' if k in ligand_sets else row
    # 06 대응이 성립한 pose만(연결성 대응 상태 기준).
    mapped_ok = set()
    for r in m06.read_csv_rows(map_dir / 'ligand_atom_mapping.csv'):
        if r.get('ligand_status') in ('clean_single_full_mapping', 'clean_single_symmetric_mapping'):
            mapped_ok.add(m06.key_of(r))

    targets = []
    for k in sorted(pose_roles):
        if k not in mapped_ok:
            continue
        lig_set = ligand_sets.get(k)
        if not isinstance(lig_set, dict):
            continue
        lig_entity = ligand_entities.get((k[0], str(lig_set.get('input_chain_id'))))
        if m06.clean_single_reasons(pose_roles[k], lig_entity, lig_set):
            continue
        targets.append((k, lig_set, lig_entity))
        if len(targets) >= limit:
            break
    return targets


def run(args):
    """소수 pose에 대해 입력을 준비하고 gnina score_only·local_only를 모두 실행합니다."""
    started = time.perf_counter()
    parent = helper.safe_mkdir(BASE / OUT_KIND)
    out = helper.checked_path(parent / OUT_NAME)
    if out.exists():
        raise FileExistsError(f'기존 결과를 덮어쓰지 않습니다: {out}')
    out.mkdir()
    for name in ('tmp', 'cache', 'work'):
        helper.safe_mkdir(out / name)
    for key in ('TMPDIR', 'TMP', 'TEMP'):
        os.environ[key] = str(out / 'tmp')
    os.environ['XDG_CACHE_HOME'] = str(out / 'cache')
    env = dict(os.environ)
    print('OUTPUT_DIR:', out, flush=True)

    import pandas as pd
    gnina_cmd = shlex.split(args.gnina)
    if not gnina_cmd:
        raise ValueError('--gnina 명령이 비어 있습니다.')
    if shutil.which(gnina_cmd[0]) is None and not Path(gnina_cmd[0]).exists():
        raise FileNotFoundError(f'gnina 실행기를 찾을 수 없습니다: {gnina_cmd[0]}')

    max_bytes = args.max_cif_mib * 1024 ** 2
    modes = {
        'score_only': args.score_only_args.split(),
        'local_only': args.local_only_args.split(),
    }
    targets = collect_targets(args.roles_run, args.map_run, args.limit, max_bytes)
    print(f'대상 pose: {len(targets)}개 (limit {args.limit})', flush=True)

    rows = []
    status_counter = Counter()
    raw_samples = []
    for number, (k, lig_set, lig_entity) in enumerate(targets, 1):
        system_id, model, seed, sample = k
        pose_id = f'{system_id}|{model}|{seed}|{sample}'
        work = helper.safe_mkdir(out / 'work' / f'{system_id}_{model}_{seed}_{sample}')
        base = {'system_id': system_id, 'model': model, 'seed': seed, 'sample': sample, 'pose_id': pose_id}
        try:
            rows_1based = json.loads(lig_set['heavy_atom_site_rows_1based_json'])
            extracted = m06.extract_ligand_atoms(lig_set['cif_path'], lig_set['cif_sha256'], rows_1based,
                                                 lig_set.get('observed_chain_id'), max_bytes)
            extracted['rows_1based'] = rows_1based
            smiles = (lig_entity or {}).get('canonical_smiles') or (lig_entity or {}).get('smiles')
            status06, _detail, mapping = m06.map_one({'canonical_smiles': smiles}, extracted, max_matches=64)
            if not mapping:
                raise ValueError(f'mapping_unavailable:{status06}')
            rec_path = build_receptor_pdb(lig_set['cif_path'], work / 'receptor.pdb')
            lig_path = build_ligand_sdf(smiles, extracted, mapping, work / 'ligand.sdf')
        except subprocess.TimeoutExpired:
            for mode in modes:
                rows.append({**base, 'mode': mode, 'status': 'error', 'error_message': 'prep_timeout',
                             'affinity': None, 'cnn_score': None, 'cnn_affinity': None})
                status_counter[('prep', 'error')] += 1
            continue
        except Exception as exc:
            for mode in modes:
                rows.append({**base, 'mode': mode, 'status': 'prep_error',
                             'error_message': f'{type(exc).__name__}: {exc}',
                             'affinity': None, 'cnn_score': None, 'cnn_affinity': None})
                status_counter[('prep', 'error')] += 1
            continue

        for mode, extra in modes.items():
            row = {**base, 'mode': mode, 'status': None, 'error_message': None,
                   'affinity': None, 'cnn_score': None, 'cnn_affinity': None}
            try:
                parsed, raw = run_gnina(gnina_cmd, rec_path, lig_path, extra, env, args.timeout)
                row.update(parsed)
                row['status'] = 'computed' if parsed.get('cnn_score') is not None else 'parsed_empty'
                if len(raw_samples) < 6:
                    raw_samples.append({'pose_id': pose_id, 'mode': mode, 'stdout_head': raw})
            except subprocess.TimeoutExpired:
                row.update(status='error', error_message='gnina_timeout')
            except Exception as exc:
                row.update(status='error', error_message=f'{type(exc).__name__}: {exc}')
            status_counter[(mode, row['status'])] += 1
            rows.append(row)
        if not args.keep_work:
            shutil.rmtree(work, ignore_errors=True)
        print(f'gnina smoke: {number}/{len(targets)}개', flush=True)

    frame = pd.DataFrame(rows)
    buf = io.BytesIO()
    frame.to_parquet(buf, engine='pyarrow', index=False, compression='snappy')
    with helper.checked_path(out / 'gnina_smoke.parquet').open('xb') as fh:
        fh.write(buf.getvalue())
    frame.to_csv(out / 'gnina_smoke.csv', index=False, encoding='utf-8')
    helper.write_json(out / 'raw_samples.json', raw_samples)

    report = {
        'version': VERSION, 'output_dir': str(out), 'roles_run': str(args.roles_run),
        'map_run': str(args.map_run), 'gnina': str(args.gnina),
        'modes': {m: a for m, a in modes.items()},
        'targets': len(targets), 'rows': len(rows),
        'status_counts': {f'{m}:{s}': c for (m, s), c in sorted(status_counter.items())},
        'elapsed_seconds': time.perf_counter() - started, 'gt_used': False, 'coordinates_modified': False,
        'notes': ['smoke test: 소수 pose에서 score_only vs local_only를 모두 추출합니다.',
                  'CNNscore/CNNaffinity/Affinity 파싱은 gnina 출력 형식에 의존하므로 raw_samples.json로 1차 확인하세요.',
                  'sc_rmsd와의 변별력 비교는 이 표를 라벨에 join하는 후속 분석에서 수행합니다.',
                  'local_only는 pose를 국소 최소화하므로 점수는 좋아질 수 있으나, 기하 feature·제출 pose를 '
                  '교체하면 안 됩니다(점수 feature로만 사용).']}
    helper.write_json(out / 'smoke_report.json', report)
    helper.write_json(out / 'RUN_COMPLETE.json', {'outputs_written': True, 'targets': len(targets),
        'rows': len(rows), 'status_counts': report['status_counts']})
    print('TARGETS:', len(targets), '| ROWS:', len(rows), flush=True)
    print('STATUS_COUNTS:', roles.j(report['status_counts']), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roles-run', type=Path, default=DEFAULT_ROLES)
    parser.add_argument('--map-run', type=Path, default=DEFAULT_MAP)
    parser.add_argument('--gnina', type=str,
                        default='singularity exec --nv --bind <FS_ROOT> '
                                '<SHARED_ROOT>/models/gnina/gnina.sif gnina',
                        help='gnina 실행 명령 전체(공백 구분, shlex로 분리). 기본은 CASP17 singularity 컨테이너.')
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--score-only-args', type=str, default='--score_only',
                        help='score_only 모드에 넘길 gnina 인자(공백 구분).')
    parser.add_argument('--local-only-args', type=str, default='--local_only',
                        help='local_only 모드에 넘길 gnina 인자(공백 구분).')
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--max-cif-mib', type=int, default=128)
    parser.add_argument('--keep-work', action='store_true', help='pose별 수용체/리간드 파일을 남깁니다(디버그).')
    args = parser.parse_args()
    if args.limit < 1 or args.timeout < 1:
        parser.error('limit·timeout은 양수여야 합니다.')
    return run(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
