#!/usr/bin/env python3
"""Run thermo-elastic topology optimization for any 2D problem file.

Examples
--------
    python run.py examples/l_beam.yaml
    python run.py examples/biclamped_beam.yaml
    python run.py examples/my_part.yaml --check            # only draw the setup, no optimization
    python run.py examples/l_beam.yaml --set optimization.volume_fraction=0.45 --out results/l_beam_vf45
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from thermotopo import apply_overrides, build, load_file, optimize  # noqa: E402
from thermotopo.postprocess import plot_setup, save_all  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('problem', help='problem file (.yaml / .yml / .json)')
    ap.add_argument('--out', help='output folder (default: output.directory in the file, else results/<name>)')
    ap.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE',
                    help='override any entry, e.g. optimization.max_iterations=50')
    ap.add_argument('--check', action='store_true', help='build the model, draw setup.png and stop')
    ap.add_argument('--quiet', action='store_true', help='do not print every iteration')
    a = ap.parse_args(argv)

    cfg = apply_overrides(load_file(a.problem), a.set)
    prob, osets, info = build(cfg)
    osets.verbose = not a.quiet
    out = Path(a.out) if a.out else (ROOT / info['out'] if info['out'] else ROOT / 'results' / info['name'])
    out.mkdir(parents=True, exist_ok=True)
    mesh = prob.mesh
    print(f"[{info['name']}] {mesh.ne} elements ({mesh.nx} x {mesh.ny} grid, h = {mesh.h_file:g} {info['units']}), "
          f"{mesh.nn} nodes, {prob.n_design} design variables, {len(prob.tcons)} temperature constraint(s)")
    print(f"  total load = ({prob.Fm[0::2].sum():.4g}, {prob.Fm[1::2].sum():.4g}) N, "
          f"total heat = {prob.P.sum():.4g} W, volume limit = {prob.opt.volume_fraction}")
    plot_setup(prob, out / 'setup.png', info['units'], f"{info['name']} - problem setup")
    print(f'  wrote {out / "setup.png"}')
    if a.check:
        import numpy as np
        st =prob.analyze(prob.rho_from_x(prob.x_from_z(np.ones(prob.n_design))), sensitivities=False)
        print(f"  all-solid design: C = {st['C']:.5g} N m, constrained T = {np.round(st['T_measured'], 3)} K")
        return 0
    t0 = time.time()
    st, hist = optimize(prob, osets)
    s = save_all(prob, st, hist, out, cfg, info)
    print(f"  done in {time.time() - t0:.1f} s: {s['status']} after {s['iterations']} iterations")
    temps = s['constrained_temperatures_K']
    if len(temps) <= 4:
        tstr = ', '.join(f'{k} = {v:.2f} K' for k, v in temps.items())
    else:
        tstr = f'max of {len(temps)} constrained nodes = {max(temps.values()):.2f} K'
    print(f"  C = {s['compliance_Nm']:.6g} N m | V = {s['volume_fraction']:.4f} | max vM = {s['max_von_mises_MPa']:.1f} MPa"
          f" | {tstr} | ||Fth||/||Fm|| = {s['Fth_over_Fm']:.3f}")
    print(f'  results in {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
