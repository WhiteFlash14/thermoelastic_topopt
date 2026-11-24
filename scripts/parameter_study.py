#!/usr/bin/env python3
"""Sweep the numerical settings that the paper does NOT report (RAMP penalties,
filter radius, MMA move limit, constraint scaling ...) and score every optimized
layout against a digitized paper figure.  The governing equations never change.

    python scripts/parameter_study.py examples/l_beam.yaml --mask paper_reference/fig4A_l_beam.npy \
        --grid ramp=3,5,8 filter_radius=1.5,2,2.5 move_limit=0.1,0.5,1.0 --jobs 6

Every ``--grid key=v1,v2,...`` is a key of the ``optimization:`` block of the
problem file (``ramp`` is a shortcut setting all three RAMP penalties).  The
full factorial combination is run in parallel.

Scores (design-domain elements, density thresholded at 0.5):
    agreement = fraction of elements with the same solid/void state as the paper
    IoU       = |solid_ours AND solid_paper| / |solid_ours OR solid_paper|
Outputs: study.csv (one row per run) and study.png (thumbnails, best first).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def score(mesh, rho, mask):
    ref = mask[mesh.elem_j, mesh.elem_i] > 0.5
    ours = rho > 0.5
    return float(np.mean(ref == ours)), float(np.sum(ref & ours) / max(np.sum(ref | ours), 1))


def _parse(v):
    try:
        return float(v)
    except ValueError:
        return v


def run_one(job):
    problem, mask_path, params = job
    from thermotopo import build, load_file, optimize
    from thermotopo.postprocess import summarize
    cfg = load_file(problem)
    o = cfg.setdefault('optimization', {})
    for k, v in params.items():
        if k == 'ramp':
            o.update(ramp_stiffness=v, ramp_conductivity=v, ramp_thermal_stress=v)
        elif k == 'symmetry':
            o['symmetry'] = [] if v in ('none', '') else [v]
        else:
            o[k] = v
    prob, os_, _ = build(cfg)
    os_.verbose = False
    st, hist = optimize(prob, os_)
    s, _ = summarize(prob, st, hist)
    agree, iou = score(prob.mesh, st['rho'], np.load(mask_path)) if mask_path else (np.nan, np.nan)
    row = dict(params, agreement=agree, iou=iou, C=s['compliance_Nm'], V=s['volume_fraction'],
               T=';'.join(f'{v:.2f}' for v in s['constrained_temperatures_K'].values()),
               max_vM_MPa=s['max_von_mises_MPa'], Fth_Fm=s['Fth_over_Fm'], gray=s['gray_fraction_0p1_0p9'],
               iterations=s['iterations'], status=s['status'], feasible=s['feasible'])
    m = prob.mesh
    return row, st['rho'], (m.elem_i, m.elem_j, m.nx, m.ny)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('problem')
    ap.add_argument('--mask', default=None, help='digitized reference (.npy, row 0 = bottom)')
    ap.add_argument('--grid', nargs='+', default=['ramp=3,5,8', 'filter_radius=1.5,2,2.5,3', 'move_limit=0.1,0.2'])
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    keys, values = [], []
    for g in a.grid:
        k, _, v = g.partition('=')
        keys.append(k)
        values.append([_parse(x) for x in v.split(',')])
    jobs = [(a.problem, a.mask, dict(zip(keys, combo))) for combo in itertools.product(*values)]
    out = Path(a.out or ROOT / 'results' / 'parameter_study' / Path(a.problem).stem)
    out.mkdir(parents=True, exist_ok=True)
    print(f'{len(jobs)} runs on {a.jobs} processes ...', flush=True)
    from multiprocessing import Pool
    with Pool(a.jobs) as pool:
        results = pool.map(run_one, jobs, chunksize=1)
    rows = [r for r, _, _ in results]
    with open(out / 'study.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order = np.argsort([-(r['agreement'] if r['agreement'] == r['agreement'] else -r['C']) for r in rows])
    n = len(results)
    nc = min(6, n)
    nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(3 * nc, 3.1 * nr), squeeze=False, layout='constrained')
    for ax in axs.ravel():
        ax.axis('off')
    for ax, k in zip(axs.ravel(), order):
        row, rho, (ei, ej, nx, ny) = results[k]
        img = np.full((ny, nx), np.nan)
        img[ej, ei] = rho
        ax.imshow(np.ma.masked_invalid(img), origin='lower', cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
        lab = ' '.join(f'{kk[:6]}={row[kk]:g}' if isinstance(row[kk], float) else f'{kk[:6]}={row[kk]}' for kk in keys)
        ax.set_title(f"{lab}\nagree={row['agreement']:.3f} C={row['C']:.4f}", fontsize=7)
    fig.savefig(out / 'study.png', dpi=110)
    for r in sorted(rows, key=lambda r: -(r['agreement'] if r['agreement'] == r['agreement'] else 0)):
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print('wrote', out)


if __name__ == '__main__':
    main()
