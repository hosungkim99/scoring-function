#!/usr/bin/env python3
"""A6c 샤드 hang의 원인을 조회만 합니다(읽기 전용, 아무것도 저장하지 않음).

21번이 멈춘 지점(샤드별 마지막 진행 로그)으로부터 그때 처리 중이던 시스템을 역산하고,
(1) 그 리간드의 크기·대칭성, (2) 그 시스템의 구조 파일 크기(= 수용체 규모 대리지표)를 봅니다.
pose 하나를 처리할 때 리간드 매핑뿐 아니라 수용체 전체 원자에 대한 SASA·접촉 계산을 하므로,
리간드가 평범해도 '시스템이 너무 크면' 한 pose에서 사실상 멈출 수 있습니다. 전체 분포와 비교해
용의 시스템이 바깥값인지 판단합니다.

인덱스·역할 경로는 03/05 모듈의 기본값을 그대로 가져와 하드코딩하지 않습니다.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import csv
import importlib.util
from pathlib import Path

_support_path = Path(__file__).with_name('05_audit_input_roles.py')
_spec = importlib.util.spec_from_file_location('roles_support', _support_path)
roles = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(roles)
helper = roles.helper      # 03 모듈(ALLOWED_ROOT, DEFAULT_INDEX)

BASE = helper.ALLOWED_ROOT / 'Scoring_Function' / 'out'
DEFAULT_ROLES = BASE / 'input_roles' / '05_run_audit_input_roles'
NUM_SHARDS = 100
# 샤드별 '마지막으로 찍힌 진행 수' → hang은 그 다음 1000개 구간 안에서 발생.
LAST_LOG = {15: 2000, 18: 3000, 40: 3000, 62: 3000, 71: 3000, 87: 2000}
MIB = 1024 ** 2


def ligand_metrics(smiles):
    """리간드 크기와 대칭성. sym_ratio(원자수/대칭클래스)가 클수록 대칭이 심합니다."""
    from rdkit import Chem
    if not smiles:
        return {'heavy': None, 'sym_ratio': None, 'parse': 'missing'}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'heavy': None, 'sym_ratio': None, 'parse': 'failed'}
    heavy = mol.GetNumHeavyAtoms()
    classes = len(set(Chem.CanonicalRankAtoms(mol, breakTies=False)))
    return {'heavy': heavy, 'sym_ratio': round(heavy / classes, 2) if classes else None, 'parse': 'ok'}


def load_system_info(index_run):
    """02 인덱스에서 시스템별 ok pose 수와 구조 파일 크기(최대)를 읽습니다."""
    import pyarrow.parquet as pq
    path = index_run / 'pose_index.parquet'
    if not path.exists():
        raise FileNotFoundError(f'pose 인덱스가 없습니다: {path}')
    info = {}
    cols = ['system_id', 'status', 'file_size_bytes']
    for batch in pq.ParquetFile(path).iter_batches(batch_size=50000, columns=cols, use_threads=False):
        d = batch.to_pydict()
        for sid, st, sz in zip(d['system_id'], d['status'], d['file_size_bytes']):
            if st != 'ok':
                continue
            rec = info.setdefault(sid, {'poses': 0, 'size': 0})
            rec['poses'] += 1
            if sz and sz > rec['size']:
                rec['size'] = sz
    return info


def load_ligand_smiles(roles_run):
    """05 input_entities.csv에서 시스템별 리간드 SMILES를 읽습니다."""
    out = {}
    with (roles_run / 'input_entities.csv').open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            if row.get('input_kind') != 'ligand' or row['system_id'] in out:
                continue
            out[row['system_id']] = (row.get('canonical_smiles') or row.get('smiles') or '').strip()
    return out


def pct(values, p):
    """정렬된 목록의 백분위 값."""
    return values[min(len(values) - 1, int(len(values) * p))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index-run', type=Path, default=helper.DEFAULT_INDEX)
    parser.add_argument('--roles-run', type=Path, default=DEFAULT_ROLES)
    parser.add_argument('--top', type=int, default=25)
    args = parser.parse_args()

    print('INDEX_RUN:', args.index_run, flush=True)
    info = load_system_info(args.index_run)
    all_sys = sorted(info)
    smiles = load_ligand_smiles(args.roles_run)
    print('ok 시스템:', len(all_sys), '| ok pose:', sum(v['poses'] for v in info.values()), flush=True)

    # 시스템별 샤드 번호와 '그 샤드 안에서의 pose 구간'을 미리 계산합니다.
    shard_of, pose_window = {}, {}
    cum = {}
    for i, sid in enumerate(all_sys):
        sh = i % NUM_SHARDS
        shard_of[sid] = sh
        start = cum.get(sh, 0)
        end = start + info[sid]['poses']
        pose_window[sid] = (start + 1, end)
        cum[sh] = end

    # ---------- 1) 멈춘 지점의 용의 시스템: 리간드 + 구조 크기 ----------
    print('\n================ 1) 멈춘 지점의 용의 시스템 ================', flush=True)
    suspects = set()
    for shard in sorted(LAST_LOG):
        lo, hi = LAST_LOG[shard] + 1, LAST_LOG[shard] + 1000
        print(f'\n--- shard {shard} | hang 구간 pose {lo}~{hi} ---', flush=True)
        for sid in all_sys:
            if shard_of[sid] != shard:
                continue
            a, b = pose_window[sid]
            if b >= lo and a <= hi:
                suspects.add(sid)
                lm = ligand_metrics(smiles.get(sid, ''))
                print(f'  {sid}', flush=True)
                print(f'    poses {a}-{b} | CIF {info[sid]["size"]/MIB:8.2f} MiB '
                      f'| ligand heavy={lm["heavy"]} sym_ratio={lm["sym_ratio"]}', flush=True)

    # ---------- 2) 전체 구조 파일 크기 분포(= 수용체 규모) ----------
    print('\n================ 2) 전체 구조 크기 분포 ================', flush=True)
    sizes = sorted(v['size'] for v in info.values())
    print(f'CIF 크기(MiB)  중앙 {pct(sizes,0.5)/MIB:.2f} | 90% {pct(sizes,0.9)/MIB:.2f} '
          f'| 99% {pct(sizes,0.99)/MIB:.2f} | 최대 {sizes[-1]/MIB:.2f}', flush=True)
    med = pct(sizes, 0.5)
    for mult in (5, 10, 20):
        n = sum(1 for s in sizes if s >= med * mult)
        print(f'  중앙값의 {mult}배 이상: {n}개', flush=True)

    print(f'\n-- 구조 크기 상위 {args.top} (어느 샤드에 있는지 함께) --', flush=True)
    biggest = sorted(all_sys, key=lambda s: -info[s]['size'])[:args.top]
    hung = set(LAST_LOG)
    for sid in biggest:
        sh = shard_of[sid]
        a, b = pose_window[sid]
        flag = ''
        if sh in hung:
            lo, hi = LAST_LOG[sh] + 1, LAST_LOG[sh] + 1000
            flag = '  <== 멈춘 샤드' + ('  ★hang 구간 적중' if (b >= lo and a <= hi) else '')
        print(f'  {info[sid]["size"]/MIB:8.2f} MiB  shard {sh:<3} poses {a}-{b}  {sid}{flag}', flush=True)

    # ---------- 3) 멈춘 6개 샤드 vs 나머지: 크기 비교 ----------
    print('\n================ 3) 멈춘 샤드 vs 정상 샤드 ================', flush=True)
    hung_sizes = sorted(info[s]['size'] for s in all_sys if shard_of[s] in hung)
    ok_sizes = sorted(info[s]['size'] for s in all_sys if shard_of[s] not in hung)
    for name, xs in (('멈춘 6개 샤드', hung_sizes), ('정상 94개 샤드', ok_sizes)):
        print(f'  {name}: 시스템 {len(xs)}개 | 중앙 {pct(xs,0.5)/MIB:.2f} MiB '
              f'| 최대 {xs[-1]/MIB:.2f} MiB', flush=True)
    print(f'\n-- 멈춘 샤드별 가장 큰 시스템 --', flush=True)
    for shard in sorted(hung):
        lo, hi = LAST_LOG[shard] + 1, LAST_LOG[shard] + 1000
        mine = [s for s in all_sys if shard_of[s] == shard]
        top = sorted(mine, key=lambda s: -info[s]['size'])[:3]
        print(f'  shard {shard} (hang {lo}~{hi}):', flush=True)
        for sid in top:
            a, b = pose_window[sid]
            hit = ' ★hang 구간' if (b >= lo and a <= hi) else ''
            print(f'      {info[sid]["size"]/MIB:8.2f} MiB  poses {a}-{b}  {sid}{hit}', flush=True)

    print('\n(읽기 전용 조회였으며 아무 파일도 쓰지 않았습니다.)', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
