#!/usr/bin/env python3
"""Self-checks of the implementation (takes ~10-20 s).  Run:  python verify.py

1. element matrices: rigid-body / constant-temperature null spaces, symmetry
2. thermal patch test: uniformly heated, statically determinate plate expands
   freely -> displacement = alpha*dT*x and zero stress
3. paper geometry: element counts, loads, heat input match Meng (2021)
4. finite-difference check of the adjoint sensitivities dC/dx and dT_i/dx
   (through the density filter and the symmetry map)
5. MMA on a small convex problem with a known optimum
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from thermotopo import build, load_file  # noqa: E402
from thermotopo.fem import element_matrices  # noqa: E402
from thermotopo.mma import MMA  # noqa: E402

OK = '\033[32mPASS\033[0m'


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f'{OK} {msg}')


def element_tests():
    em = element_matrices(1e-3, 1e-3, 210e9, 0.3, 43.0, 12.1e-6)
    ke, kt = em['ke0'], em['kt0']
    rx = np.tile([1.0, 0.0], 4)
    ry = np.tile([0.0, 1.0], 4)
    xy = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    rot = np.column_stack([-xy[:, 1], xy[:, 0]]).ravel()
    n = np.linalg.norm(ke)
    check(max(np.linalg.norm(ke @ v) for v in (rx, ry, rot)) / n < 1e-12, 'Q4 stiffness has 3 rigid-body modes')
    check(np.linalg.norm(kt @ np.ones(4)) / np.linalg.norm(kt) < 1e-12, 'conduction matrix: constant T is a null mode')
    check(np.allclose(ke, ke.T) and np.allclose(kt, kt.T), 'element matrices symmetric')
    check(abs(np.linalg.eigvalsh(ke)[3]) > 1e-6 * n, 'stiffness rank 5 (no spurious zero-energy modes)')


def patch_test():
    """10x10 mm plate, T = T_ref + 20 K everywhere, supports: node(0,0) xy, node(10,0) y."""
    cfg = dict(units='mm', geometry=dict(element_size=1.0, add=[dict(type='rectangle', x=[0, 10], y=[0, 10])]),
               material=dict(reference_temperature=300.0),
               mechanical=dict(supports=[dict(where=dict(point=[0, 0]), fix='xy'),
                                         dict(where=dict(point=[10, 0]), fix='y')], loads=[]),
               thermal=dict(fixed_temperature=[dict(where=dict(box=[[0, 0], [10, 10]]), value=320.0)], heat_sources=[]),
               optimization=dict(volume_fraction=1.0, filter_radius=1.0))
    prob, _, _ = build(cfg)
    st = prob.analyze(np.ones(prob.mesh.ne), sensitivities=False)
    U = st['U'].reshape(-1, 2)
    exact = 12.1e-6 * 20.0 * prob.mesh.nodes
    check(np.allclose(U, exact, rtol=1e-8, atol=1e-14), 'patch test: free thermal expansion u = alpha*dT*x')
    vm = prob.stress(st)
    check(vm.max() < 1e-3, f'patch test: zero thermal stress (max {vm.max():.2e} Pa)')


def paper_geometry():
    for f, ne, F, Q, nload in (('l_beam.yaml', 6400, -500.0, 0.12, 5), ('biclamped_beam.yaml', 7000, -3500.0, 0.67, 15)):
        prob, _, _ = build(load_file(ROOT / 'examples' / f))
        nl = int(np.count_nonzero(prob.Fm[1::2]))
        check(prob.mesh.ne == ne and np.isclose(prob.Fm[1::2].sum(), F) and np.isclose(prob.P.sum(), Q) and nl == nload,
              f'{f}: {ne} elements, F = {F} N on {nload} nodes, heat = {Q} W')
        st = prob.analyze(np.ones(prob.mesh.ne), sensitivities=False)
        check(np.isclose(st['C'], st['U'] @ (prob.mech.full_matrix(np.ones(prob.mesh.ne), prob.em['ke0']) @ st['U']), rtol=1e-8),
              f'{f}: C = F^T U = U^T K U (equilibrium)')


def gradient_test():
    for sym in ([], ['vertical']):
        cfg = load_file(ROOT / 'examples' / 'biclamped_beam.yaml')
        cfg['geometry']['element_size'] = 2.0
        cfg['mechanical']['loads'][0]['where'] = {'box': [[46, 34], [54, 36]]}
        cfg['optimization'].update(filter_radius=4.0, symmetry=sym, ramp_stiffness=5, ramp_conductivity=3,
                                   ramp_thermal_stress=4)
        prob, _, _ = build(cfg)
        rng = np.random.default_rng(1)
        z = rng.uniform(0.2, 0.9, prob.n_design)

        def evaluate(zz):
            return prob.analyze(prob.rho_from_x(prob.x_from_z(zz)))
        st = evaluate(z)
        gC = prob.grad_z(st['dC'])
        gT = prob.grad_z(st['dT'])
        picks = rng.choice(prob.n_design, 6, replace=False)
        errs = []
        for k in picks:
            e = 1e-6
            zp, zm = z.copy(), z.copy()
            zp[k] += e
            zm[k] -= e
            sp, sm = evaluate(zp), evaluate(zm)
            fdC = (sp['C'] - sm['C']) / (2 * e)
            fdT = (sp['T_measured'] - sm['T_measured']) / (2 * e)
            errs.append(abs(fdC - gC[k]) / max(abs(fdC), 1e-3 * np.abs(gC).max()))
            errs.append(np.max(np.abs(fdT - gT[:, k]) / np.maximum(np.abs(fdT), 1e-3 * np.abs(gT).max())))
        check(max(errs) < 1e-4, f'adjoint vs finite-difference gradients (symmetry={sym or "none"}), max rel. err {max(errs):.1e}')


def mma_test():
    # min (x1-2)^2 + (x2-1)^2  s.t.  x1 + x2 <= 2, 0<=x<=3  -> optimum (1.5, 0.5)
    mma = MMA(2, 1, 0.0, 3.0, move=0.5)
    x = np.array([3.0, 3.0])
    for _ in range(60):
        f0, df0 = (x[0] - 2) ** 2 + (x[1] - 1) ** 2, np.array([2 * (x[0] - 2), 2 * (x[1] - 1)])
        g, dg = np.array([x[0] + x[1] - 2.0]), np.array([[1.0, 1.0]])
        x = mma.update(x, f0, df0, g, dg)
    check(np.allclose(x, [1.5, 0.5], atol=1e-3), f'MMA converges on a test problem (x = {np.round(x, 4)})')


if __name__ == '__main__':
    element_tests()
    patch_test()
    paper_geometry()
    gradient_test()
    mma_test()
    print('\nAll checks passed.')
