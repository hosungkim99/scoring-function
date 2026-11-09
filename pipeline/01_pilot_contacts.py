#!/usr/bin/env python3
"""Four-pose pilot only. No installs, no GT, no shared environment changes.

Run with the existing <CONDA_ENV> Python and -B. Outputs go below
ALLOWED_ROOT/Scoring_Function/out/pilot_contacts/<run_id>/.
Keep this script in Scoring_Function/pipeline/; SLURM stdout/stderr belong
in Scoring_Function/logs/ when a batch wrapper is added.
Outputs: features.parquet, feature_spec.json, run_report.json, RUN_COMPLETE.json.
This is category 1 of the eight-category project, not the full pipeline.
"""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import time
import uuid

ALLOWED_ROOT = Path('<DATA_ROOT>')
SOURCE_ROOT = Path('<CASP_ROOT>/targets_ligand/distill')
SYSTEM_ID = '1aon__1__1.D__1.BA_1.CA'
MODELS = ('af3', 'bt2', 'of3', 'pt2')
VERSION = 'pilot_contacts_0.1.2'
CUTOFFS = (3, 4, 5)

def feature_definitions():
    result = {'pl_min_distance_A': {'unit': 'angstrom', 'definition':
              'Minimum distance between selected protein and ligand heavy atoms.'}}
    for c in CUTOFFS:
        result.update({
            f'pl_pairs_lt_{c}A': {'unit': 'count', 'definition': f'Number of protein-ligand atom pairs with d < {c} angstrom.'},
            f'ligand_atoms_lt_{c}A': {'unit': 'count', 'definition': f'Unique ligand atoms with any protein atom at d < {c} angstrom.'},
            f'protein_atoms_lt_{c}A': {'unit': 'count', 'definition': f'Unique protein atoms with any ligand atom at d < {c} angstrom.'},
            f'protein_residues_lt_{c}A': {'unit': 'count', 'definition': f'Unique protein residues with any atom at d < {c} angstrom from a ligand atom.'},
        })
    return result

def specification():
    return {
        'schema_version': VERSION,
        'scope': 'Exactly four files: one known system, seed 1, sample 0 for each model.',
        'features': feature_definitions(),
        'input_selection': {
            'protein': 'Polymer entity, polypeptide type; if missing, every component must have peptide-linking chem_comp type.',
            'ligand': 'Exactly one organic non-polymer residue, C10 N5 O10 P2; identity is provisional ADP-compatible, not a verified chemical atom mapping.',
            'hydrogens': 'H and D excluded; coordinates unmodified.',
            'alternate_locations': 'Any explicit alternate location rejected in this pilot.',
            'occupancy': 'All coordinates must have occupancy 1 in this pilot; other inputs rejected.',
            'multiple_models_or_ligands': 'Rejected; no silent selection or merging.',
            'missing_residue_number': 'Retain null author and label number, use chain and component for the sole observed residue; duplicate atom names rejected.',
            'protein_OXT': 'Included if present; no atom synthesis or deletion.',
            'distance': 'Euclidean distance in the original common coordinate frame. Strict < cutoff.',
        },
        'candidate_groups': [
            {'id': 1, 'name': 'Atom pair contacts and distances', 'state': '13 pilot features implemented'},
            {'id': 2, 'name': 'Interactions and repulsion', 'state': 'pending', 'candidates': ['hydrogen bonds', 'hydrophobic contacts', 'salt bridges', 'pi interactions', 'clashes', 'fixed-pose Vina components']},
            {'id': 3, 'name': 'Surface and burial', 'state': 'pending', 'candidates': ['isolated ligand SASA', 'complex ligand SASA', 'ligand buried SASA', 'buried fraction', 'polar and nonpolar SASA']},
            {'id': 4, 'name': 'Ligand geometry and stability', 'state': 'pending', 'candidates': ['bond geometry', 'planarity', 'internal clashes', 'input energy', 'defined reference energy difference']},
            {'id': 5, 'name': 'Pocket fit', 'state': 'pending', 'candidates': ['predicted pocket score', 'pocket distance', 'residue contact overlap']},
            {'id': 6, 'name': 'Water-mediated interactions', 'state': 'pending', 'candidates': ['water bridge count', 'bridge geometry'], 'absent_input': 'null, not zero'},
            {'id': 7, 'name': 'Metal features', 'state': 'pending', 'candidates': ['metal presence and count', 'coordination distances', 'geometry'], 'absent_input': 'count zero, conditional geometry null'},
            {'id': 8, 'name': 'Ligand intrinsic descriptors', 'state': 'pending', 'candidates': ['MW', 'heavy atoms', 'rotatable bonds', 'HBD', 'HBA', 'TPSA', 'logP']},
        ],
    }

def checked_path(path):
    root = ALLOWED_ROOT
    if not root.is_absolute() or not root.is_dir() or root.resolve() != root:
        raise RuntimeError('Allowed root must exist and resolve exactly to the specified root.')
    p = Path(os.path.abspath(path))
    p.relative_to(root)  # Raises for a lexical escape.
    p.resolve(strict=False).relative_to(root)  # Raises for a symlink escape.
    cur = root
    for part in p.relative_to(root).parts:
        cur = cur / part
        if cur.is_symlink():
            raise RuntimeError(f'Refusing symlink in output path: {cur}')
    return p

def safe_mkdir(path):
    p = checked_path(path)
    cur = ALLOWED_ROOT
    for part in p.relative_to(ALLOWED_ROOT).parts:
        cur = checked_path(cur / part)
        if not cur.exists():
            cur.mkdir()
        if not cur.is_dir():
            raise RuntimeError(f'Not a directory: {cur}')
    return p

def write_new(path, payload):
    p = checked_path(path)
    with p.open('xb') as handle:
        handle.write(payload)

def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()

def category_rows(block, prefix):
    data = block.get_mmcif_category(prefix)
    if not data:
        return []
    sizes = {len(v) for v in data.values()}
    if len(sizes) != 1:
        raise ValueError(f'Inconsistent category lengths: {prefix}')
    return [{k: values[i] for k, values in data.items()} for i in range(next(iter(sizes)))]

def residue_key(atom):
    return tuple(atom.get(k) for k in ('label_asym_id', 'label_seq_id',
                                      'auth_seq_id', 'pdbx_PDB_ins_code', 'label_comp_id'))

def extract_one(path):
    import gemmi
    import numpy as np
    from scipy.spatial.distance import cdist
    raw = Path(path).read_bytes()
    block = gemmi.cif.read_string(raw.decode('utf-8')).sole_block()
    atoms = category_rows(block, '_atom_site.')
    if not atoms or len(atoms) > 200000:
        raise ValueError('Empty or unexpectedly large atom table for the pilot.')
    required = ('type_symbol', 'label_asym_id', 'label_entity_id', 'label_comp_id',
                'label_atom_id', 'Cartn_x', 'Cartn_y', 'Cartn_z', 'occupancy',
                'pdbx_PDB_model_num')
    if any(k not in atoms[0] for k in required):
        raise ValueError('Required atom_site field missing.')
    if len({a['pdbx_PDB_model_num'] for a in atoms}) != 1:
        raise ValueError('Multiple coordinate models are unsupported in this pilot.')
    if any(a.get('label_alt_id') not in (None, False, '.', '?', '') for a in atoms):
        raise ValueError('Alternate conformers require an explicit policy.')
    if any(abs(float(a['occupancy'])-1) > 1e-8 for a in atoms):
        raise ValueError('Non-unit occupancy requires an explicit policy.')
    entity_types = {r['id']: r['type'].lower() for r in category_rows(block, '_entity.')}
    poly_types = {r['entity_id']: r['type'].lower() for r in category_rows(block, '_entity_poly.')}
    comp_types = {r['id']: str(r.get('type', '')).lower() for r in category_rows(block, '_chem_comp.')}
    for a in atoms:
        if a['label_entity_id'] not in entity_types:
            raise ValueError('Atom entity not represented in entity metadata.')
    polymer_ids = {eid for eid, kind in entity_types.items() if kind == 'polymer'}
    protein_ids = set()
    for eid in polymer_ids:
        if eid in poly_types:
            if not poly_types[eid].startswith('polypeptide'):
                raise ValueError('Non-protein polymer is unsupported in this pilot.')
        elif not all('peptide linking' in comp_types.get(a['label_comp_id'], '')
                     for a in atoms if a['label_entity_id'] == eid):
            raise ValueError('Cannot establish protein entity type.')
        protein_ids.add(eid)
    heavy = [a for a in atoms if a['type_symbol'].upper() not in ('H', 'D')]
    protein = [a for a in heavy if a['label_entity_id'] in protein_ids]
    other = [a for a in heavy if a['label_entity_id'] not in protein_ids]
    if not protein or not other:
        raise ValueError('Protein or ligand atoms missing.')
    if any(entity_types[a['label_entity_id']] != 'non-polymer' for a in other):
        raise ValueError('Unsupported non-protein entity.')
    if len({residue_key(a) for a in other}) != 1:
        raise ValueError('More than one non-protein residue; do not merge automatically.')
    if Counter(a['type_symbol'].upper() for a in other) != Counter(C=10, N=5, O=10, P=2):
        raise ValueError('Expected known ADP-compatible heavy atom composition.')
    for group in (protein, other):
        ids = [(residue_key(a), a['label_atom_id']) for a in group]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate atom identifiers; residue grouping ambiguous.')
    xyz = lambda rows: np.asarray([[float(a[k]) for k in ('Cartn_x','Cartn_y','Cartn_z')] for a in rows], dtype=np.float64)
    pp, ll = xyz(protein), xyz(other)
    if not (np.isfinite(pp).all() and np.isfinite(ll).all()):
        raise ValueError('Non-finite coordinates.')
    # A known ligand has only 27 atoms, so this allocation is bounded by the pilot atom cap.
    distances = cdist(ll, pp)
    values = {'pl_min_distance_A': float(distances.min())}
    for cutoff in CUTOFFS:
        mask = distances < cutoff
        values[f'pl_pairs_lt_{cutoff}A'] = int(mask.sum())
        values[f'ligand_atoms_lt_{cutoff}A'] = int(mask.any(axis=1).sum())
        values[f'protein_atoms_lt_{cutoff}A'] = int(mask.any(axis=0).sum())
        values[f'protein_residues_lt_{cutoff}A'] = len({residue_key(a) for a, near in zip(protein,mask.any(axis=0)) if near})
    first = other[0]
    metadata = {
        'input_sha256': hashlib.sha256(raw).hexdigest(), 'input_size_bytes': len(raw),
        'ligand_instance_id': 'ligand_001',
        'ligand_chain': first['label_asym_id'], 'ligand_comp_id': first['label_comp_id'],
        'ligand_auth_seq_id': first.get('auth_seq_id') if first.get('auth_seq_id') not in (None,False) else None,
        'ligand_label_seq_id': first.get('label_seq_id') if first.get('label_seq_id') not in (None,False) else None,
        'protein_heavy_atoms': len(protein), 'ligand_heavy_atoms': len(other),
        'observed_ligand_count': 1,
    }
    return metadata, values

def run_pilot(source_root, explicit_files=None):
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    run_dir = safe_mkdir(ALLOWED_ROOT/'Scoring_Function'/'out'/'pilot_contacts'/run_id)
    for name in ('tmp','cache'):
        safe_mkdir(run_dir/name)
    for key in ('TMPDIR','TMP','TEMP'):
        os.environ[key] = str(run_dir/'tmp')
    os.environ['XDG_CACHE_HOME'] = str(run_dir/'cache')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key] = '1'
    import pandas as pd
    import pyarrow
    import gemmi
    spec = specification()
    spec_hash = hashlib.sha256(json_bytes(spec)).hexdigest()
    write_new(run_dir/'feature_spec.json', json_bytes(spec))
    manifest_path = Path(source_root)/SYSTEM_ID/'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if explicit_files is None else {}
    if explicit_files is None:
        parts = [str(manifest[k]) for k in ('sto','ligand_ccd')]
        if any(not p or '/' in p or '\\' in p or p in ('.','..') for p in parts):
            raise ValueError('Invalid manifest path component.')
        variant = '+'.join(parts)
        paths = [Path(source_root)/SYSTEM_ID/'results'/variant/'af3_msa'/m/'seed_1'/'sample_0'/'model.cif' for m in MODELS]
    else:
        paths = list(map(Path, explicit_files))
    rows=[]
    for model,path in zip(MODELS,paths):
        started=time.perf_counter()
        row={'system_id':SYSTEM_ID,'model':model,'seed':1,'sample':0,
             'pose_id':f'{SYSTEM_ID}|{model}|1|0','input_path':str(path.resolve()),
             'schema_version':VERSION,'config_sha256':spec_hash,
             'manifest_ligand_copies':manifest.get('ligand_copies'),
             'status':'success','error_type':None,'error_message':None}
        row.update(dict.fromkeys(feature_definitions()))
        try:
            meta,values=extract_one(path)
            row.update(meta); row.update(values)
            row['manifest_count_mismatch'] = (int(manifest['ligand_copies']) != 1) if manifest.get('ligand_copies') is not None else None
        except Exception as exc:
            row['status']='error'; row['error_type']=type(exc).__name__; row['error_message']=str(exc)
        row['elapsed_seconds']=time.perf_counter()-started
        rows.append(row)
    frame=pd.DataFrame(rows)
    if frame.duplicated(['system_id','model','seed','sample']).any():
        raise RuntimeError('Duplicate pose identifiers.')
    buf=io.BytesIO()
    frame.to_parquet(buf, engine='pyarrow', index=False, compression='snappy')
    write_new(run_dir/'features.parquet',buf.getvalue())
    restored=pd.read_parquet(run_dir/'features.parquet',engine='pyarrow')
    pd.testing.assert_frame_equal(restored, frame, check_dtype=False)
    report={'run_id':run_id,'scope':'four-pose category-1 pilot only',
            'created_utc':datetime.now(timezone.utc).isoformat(),
            'source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'python':sys.version,'python_executable':sys.executable,
            'versions':{k:version(k) for k in ('gemmi','numpy','scipy','pandas','pyarrow')},
            'config_sha256':spec_hash,'successes':sum(r['status']=='success' for r in rows),
            'failures':sum(r['status']!='success' for r in rows),'rows':rows,
            'gt_used':False,'coordinates_modified':False,'other_feature_groups':'pending'}
    write_new(run_dir/'run_report.json',json_bytes(report))
    write_new(run_dir/'RUN_COMPLETE.json',json_bytes({'outputs_written':True,'successes':report['successes'],'failures':report['failures']}))
    print('OUTPUT_DIR:',run_dir)
    print('STATUS:',report['successes'],'success,',report['failures'],'failed')
    print(frame[['model','status','pl_pairs_lt_3A','pl_pairs_lt_4A','pl_pairs_lt_5A','pl_min_distance_A']].to_string(index=False))
    if report['failures']:
        for r in rows:
            if r['status']!='success':print('ERROR:',r['model'],r['error_type'],r['error_message'])
    return 0 if report['failures']==0 else 2

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=SOURCE_ROOT)
    parser.add_argument('--files',nargs=4,metavar=('AF3','BT2','OF3','PT2'),help='Explicit four input files, preserving this order; does not change the allowed output root.')
    args=parser.parse_args()
    return run_pilot(args.source_root,args.files)

if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FATAL: {type(exc).__name__}: {exc}',file=sys.stderr)
        raise SystemExit(1)
