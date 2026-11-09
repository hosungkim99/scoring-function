#!/usr/bin/env python3
"""예측 포켓 적합성(category 5) feature를 P2Rank로 추출합니다.

외부 도구 P2Rank(prank) CLI를 pose별로 1회 실행합니다. co-folding은 pose마다 수용체 구조가 달라
포켓 예측도 pose마다 달라집니다. 원본 model.cif를 읽기 전용으로 넘기며 P2Rank가 리간드를 무시하고
단백질 포켓만 예측하므로 파생 구조 파일을 만들지 않습니다. 리간드와 포켓의 관계는 05번 리간드
좌표로 계산합니다. GT를 사용하지 않습니다.

외부 도구의 출력·임시·캐시를 모두 허용된 결과 폴더 안으로 제한합니다(설치 폴더·시스템 임시 미사용).
시각화는 끄고(-visualizations 0, 원본 복사 방지), chemcomp 캐시는 결과 폴더에 모아 재사용합니다.
custom 리간드명 다운로드 실패 경고는 비치명적입니다. 07번을 import해 원자 집합·CIF 읽기를 재사용하고
05번 eligible pose(152개)를 대상으로 합니다.

출력: out/pocket_features/14_run_pocket_features/ 아래 고정 폴더.
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
import subprocess
import time

_support_path = Path(__file__).with_name('07_contact_features.py')
_spec = importlib.util.spec_from_file_location('contact_support', _support_path)
cf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cf)
roles = cf.roles
audit = cf.audit
helper = cf.helper

VERSION = 'pocket_features_0.1.0'
OUT_KIND = 'pocket_features'
OUT_NAME = '14_run_pocket_features'
DEFAULT_ROLES = helper.ALLOWED_ROOT / 'Scoring_Function' / 'out' / 'input_roles' / '05_run_audit_input_roles'
DEFAULT_PRANK = Path('<SHARED_ROOT>/models/p2rank/p2rank_2.5.1/prank')
KEYS = ('system_id', 'model', 'seed', 'sample')


def feature_definitions():
    """포켓 적합성 feature의 정의와 단위를 고정합니다."""
    return {
        'num_predicted_pockets': {'unit': 'count', 'definition': 'P2Rank가 예측한 포켓 수(0도 관찰로 기록).'},
        'top_pocket_score': {'unit': 'score', 'definition': '순위 1 포켓의 P2Rank score. 포켓이 없으면 null.'},
        'top_pocket_probability': {'unit': 'probability', 'definition': '순위 1 포켓의 확률. 포켓이 없으면 null.'},
        'ligand_pocket_rank': {'unit': 'rank', 'definition': '리간드 중심에 가장 가까운 포켓의 순위(1이면 최상위 포켓). 포켓이 없으면 null.'},
        'ligand_pocket_score': {'unit': 'score', 'definition': '그 포켓의 score. 포켓이 없으면 null.'},
        'ligand_pocket_probability': {'unit': 'probability', 'definition': '그 포켓의 확률. 포켓이 없으면 null.'},
        'ligand_to_pocket_center_dist_A': {'unit': 'angstrom', 'definition': '리간드 중심과 가장 가까운 포켓 중심 사이 거리. 포켓이 없으면 null.'},
    }


def specification():
    return {
        'schema_version': VERSION,
        'scope': 'P2Rank 예측 포켓과 리간드의 적합성(category 5). pose별 수용체로 1회 예측.',
        'features': feature_definitions(),
        'tool': {'name': 'P2Rank', 'binary': str(DEFAULT_PRANK), 'mode': 'predict -f <cif> -o <dir> -visualizations 0'},
        'method': {'receptor': '원본 model.cif를 읽기 전용으로 입력(P2Rank가 리간드 무시).',
                   'ligand': '05번 리간드 중원자 좌표로 중심을 구해 포켓과의 거리·순위 계산.',
                   'pose_specific': 'pose마다 수용체 구조가 달라 포켓 예측을 개별 수행.'},
        'absence_policy': '포켓 0개면 개수=0(관찰)이고 포켓 기반 값은 null(계산 불가).',
        'containment': '출력·임시·chemcomp 캐시를 결과 폴더로 제한하고 시각화를 끕니다. 원본 미수정.',
        'notes': ['custom 리간드명 CCD 다운로드 실패 경고는 비치명적입니다.',
                  '전체 확대 시 P2Rank batch/병렬과 네트워크 의존성 점검이 필요합니다.',
                  'GT 미사용.']}


def parse_predictions(path):
    """P2Rank predictions.csv를 열 이름 공백을 정리해 읽습니다."""
    import pandas as pd
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    return df


def run_p2rank(prank, cif_path, pose_dir, env, timeout):
    """P2Rank를 한 pose에 실행하고 predictions.csv 경로를 돌려줍니다."""
    cmd = [str(prank), 'predict', '-f', str(cif_path), '-o', str(pose_dir), '-visualizations', '0']
    proc = subprocess.run(cmd, cwd=str(pose_dir.parent), env=env, capture_output=True, text=True, timeout=timeout)
    pred = pose_dir / f'{Path(cif_path).name}_predictions.csv'
    if proc.returncode != 0 or not pred.exists():
        tail = (proc.stderr or proc.stdout or '')[-500:]
        raise RuntimeError(f'p2rank_failed(rc={proc.returncode}): {tail}')
    return pred


def run(args):
    """eligible pose에 P2Rank 포켓 feature를 계산해 저장합니다."""
    started = time.perf_counter()
    roles_dir = args.roles_run
    prank = args.prank
    if not Path(prank).exists():
        raise FileNotFoundError(f'P2Rank 실행 파일을 찾을 수 없습니다: {prank}')
    parent = helper.safe_mkdir(helper.ALLOWED_ROOT / 'Scoring_Function' / 'out' / OUT_KIND)
    out = helper.checked_path(parent / OUT_NAME)
    if out.exists():
        raise FileExistsError(f'기존 결과를 덮어쓰지 않습니다: {out}')
    out.mkdir()
    for name in ('tmp', 'cache', 'p2rank'):
        helper.safe_mkdir(out / name)
    # 외부 도구의 임시·캐시를 허용 폴더로 가둡니다.
    env = dict(os.environ)
    env.update({'TMPDIR': str(out / 'cache'), 'TMP': str(out / 'cache'), 'TEMP': str(out / 'cache'),
                'XDG_CACHE_HOME': str(out / 'cache'),
                'JAVA_TOOL_OPTIONS': f"-Djava.io.tmpdir={out / 'cache'}",
                'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    for key in ('TMPDIR', 'TMP', 'TEMP'):
        os.environ[key] = str(out / 'tmp')
    os.environ['XDG_CACHE_HOME'] = str(out / 'cache')
    print('OUTPUT_DIR:', out, flush=True)

    import numpy as np
    import pandas as pd
    import pyarrow as pa
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)

    spec = specification()
    spec_hash = hashlib.sha256((json.dumps(spec, ensure_ascii=False, sort_keys=True,
                                           allow_nan=False)).encode()).hexdigest()
    helper.write_json(out / 'feature_spec.json', spec)

    sources = [helper.fingerprint(roles_dir / n) for n in ('RUN_COMPLETE.json', 'selected_pose_roles.csv', 'selected_atom_sets.jsonl')]
    complete = json.loads((roles_dir / 'RUN_COMPLETE.json').read_text(encoding='utf-8'))
    if complete.get('status') != 'AUDIT_COMPLETE' or complete.get('outputs_written') is not True:
        raise ValueError('05번 입력 역할 점검이 정상 완료된 결과가 아닙니다.')

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
               'status': None, 'skip_reasons_json': '[]', 'error_message': None, 'elapsed_seconds': None}
        row.update(dict.fromkeys(feature_names))
        try:
            if pose_role.get('pair_geometry_status') != 'eligible_after_policy_review':
                row['status'] = 'skipped_not_eligible'
            elif ligand_sets.get(key) == 'multiple' or key not in ligand_sets:
                row['status'] = 'skipped_no_ligand_atom_set'
            else:
                # 리간드 중심 좌표(05 원자 집합, 원본 CIF 해시 재검증).
                atoms, _ = cf.read_atoms(cif_path, cif_sha, max_bytes)
                lig_xyz, _ = cf.pick_coords(atoms, ligand_sets[key]['rows'])
                centroid = lig_xyz.mean(axis=0)
                pose_dir = helper.safe_mkdir(out / 'p2rank' / f'{system_id}_{model}_{seed}_{sample}')
                pred = run_p2rank(prank, cif_path, pose_dir, env, args.timeout)
                df = parse_predictions(pred)
                row['num_predicted_pockets'] = int(len(df))
                if len(df) == 0:
                    row['status'] = 'no_pockets_predicted'   # 포켓 기반 값은 null 유지
                else:
                    top = df.sort_values('rank').iloc[0]
                    row['top_pocket_score'] = round(float(top['score']), 4)
                    row['top_pocket_probability'] = round(float(top['probability']), 4)
                    centers = df[['center_x', 'center_y', 'center_z']].to_numpy(dtype=np.float64)
                    dists = np.linalg.norm(centers - centroid, axis=1)
                    nidx = int(dists.argmin())
                    near = df.iloc[nidx]
                    row['ligand_pocket_rank'] = int(near['rank'])
                    row['ligand_pocket_score'] = round(float(near['score']), 4)
                    row['ligand_pocket_probability'] = round(float(near['probability']), 4)
                    row['ligand_to_pocket_center_dist_A'] = round(float(dists[nidx]), 4)
                    row['status'] = 'computed'
        except subprocess.TimeoutExpired:
            row.update(status='error', error_message='p2rank_timeout')
        except Exception as exc:
            row.update(status='error', error_message=f'{type(exc).__name__}: {exc}')
        row['elapsed_seconds'] = time.perf_counter() - begin
        status_counter[row['status']] += 1
        rows_out.append(row)
        if number % 10 == 0 or number == len(pose_roles):
            print(f'포켓 feature: {number}/{len(pose_roles)}개', flush=True)

    frame = pd.DataFrame(rows_out)
    if frame.duplicated(list(KEYS)).any():
        raise RuntimeError('pose 식별자 중복이 있습니다.')
    buf = io.BytesIO()
    frame.to_parquet(buf, engine='pyarrow', index=False, compression='snappy')
    with helper.checked_path(out / 'pocket_features.parquet').open('xb') as handle:
        handle.write(buf.getvalue())
    restored = pd.read_parquet(out / 'pocket_features.parquet', engine='pyarrow')
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
    status = 'FEATURES_COMPLETE' if not errors and not changed else 'CHECK_REQUIRED'
    try:
        import subprocess as _sp
        pv = _sp.run([str(prank)], capture_output=True, text=True, timeout=30)
        prank_banner = (pv.stdout or pv.stderr or '').splitlines()[:3]
    except Exception:
        prank_banner = None
    report = {
        'version': VERSION, 'status': status, 'output_dir': str(out), 'roles_run': str(roles_dir),
        'prank_binary': str(prank), 'prank_banner': prank_banner, 'sources': sources, 'config_sha256': spec_hash,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'support_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (_support_path, cf._support_path, roles._support_path, audit._support_path)},
        'python': sys.executable, 'pyarrow_version': pa.__version__, 'numpy_version': np.__version__,
        'poses': len(rows_out), 'computed': computed, 'errors': errors, 'status_counts': dict(status_counter),
        'no_pockets_predicted': status_counter.get('no_pockets_predicted', 0),
        'changed_sources': changed, 'feature_category': 5, 'elapsed_seconds': time.perf_counter() - started,
        'gt_used': False, 'coordinates_modified': False,
        'limitations': [
            'pose별 P2Rank 1회 실행이며 전체 확대 시 batch·병렬·네트워크 의존성을 점검해야 합니다.',
            '포켓 0개면 개수=0(관찰)이고 포켓 기반 값은 null입니다.',
            '리간드-포켓 관계는 중심 거리·순위로 요약하며 잔기 단위 겹침은 포함하지 않았습니다.',
            '외부 도구 출력·임시·캐시를 결과 폴더로 제한하고 원본을 수정하지 않습니다.',
            'GT·평가 점수를 사용하지 않습니다.']}
    helper.write_json(out / 'audit_report.json', report)
    helper.write_json(out / 'RUN_COMPLETE.json', {'outputs_written': True, 'features_finished': True,
        'status': status, 'poses': len(rows_out), 'computed': computed, 'errors': errors,
        'sources_unchanged': not changed})
    print('STATUS:', status, flush=True)
    print('POSES:', len(rows_out), '| COMPUTED:', computed, '| ERRORS:', errors, flush=True)
    print('STATUS_COUNTS:', roles.j(dict(status_counter)), flush=True)
    return 0 if status == 'FEATURES_COMPLETE' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roles-run', type=Path, default=DEFAULT_ROLES)
    parser.add_argument('--prank', type=Path, default=DEFAULT_PRANK)
    parser.add_argument('--max-cif-mib', type=int, default=128)
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()
    if args.max_cif_mib < 1 or args.timeout < 1:
        parser.error('상한은 양수여야 합니다.')
    return run(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
