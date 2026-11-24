"""MMA optimization loop with the stopping rule of Meng et al. (2021), Sec. 5:

  stop when the relative change of the objective is < tol (0.1 %) in
  ``window`` (10) consecutive iterations AND all constraints are satisfied,
  or when ``max_iterations`` (200) is reached.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .mma import MMA
from .problem import ThermoelasticCompliance


@dataclass
class OptimizerSettings:
    max_iterations: int = 200
    tolerance: float = 1e-3
    window: int = 10
    move_limit: float = 0.2
    initial_density: float = 1.0
    temperature_tolerance: float = 0.01     # K
    volume_tolerance: float = 1e-3          # relative
    objective_scale: float = 1.0            # MMA sees f0 = scale * C / C_initial
    temperature_scaling: str = 'rise'       # 'rise': (T-Tbar)/(Tbar-Tref) ; 'normalized': T/Tbar - 1
    verbose: bool = True


def optimize(prob: ThermoelasticCompliance, s: OptimizerSettings, callback=None):
    o = prob.opt
    nz, m = prob.n_design, prob.n_constraints
    z = np.full(nz, float(s.initial_density))
    z = np.clip(z, o.x_min, 1.0)
    mma = MMA(nz, m, o.x_min, 1.0, move=s.move_limit)
    Tlim = np.array([tc.limit for tc in prob.tcons])
    if s.temperature_scaling == 'normalized':
        Tscale = np.abs(Tlim)
    elif s.temperature_scaling == 'rise':
        Tscale = np.maximum(np.abs(Tlim - prob.mat.T_ref), 1.0)
    else:
        raise ValueError("temperature_scaling must be 'rise' or 'normalized'")
    history = []
    C0 = None
    t0 = time.time()
    status = 'max_iterations'
    st = None
    for it in range(1, s.max_iterations + 1):
        x = prob.x_from_z(z)
        rho = prob.rho_from_x(x)
        st = prob.analyze(rho)
        if C0 is None:
            C0 = abs(st['C']) if st['C'] != 0 else 1.0
        # objective & constraints (scaled to O(1) for MMA)
        f0 = s.objective_scale * st['C'] / C0
        df0 = s.objective_scale * prob.grad_z(st['dC']) / C0
        g = [st['volume'] / o.volume_fraction - 1.0]
        dg = [prob.grad_z(st['dV']) / o.volume_fraction]
        if len(Tlim):
            g.extend((st['T_measured'] - Tlim) / Tscale)
            dg.extend(prob.grad_z(st['dT']) / Tscale[:, None])
        g = np.asarray(g)
        dg = np.vstack(dg)
        rec = dict(iteration=it, compliance=st['C'], volume=st['volume'],
                   T_max_constrained=float(st['T_measured'].max()) if len(Tlim) else float('nan'),
                   T_max_domain=float(st['T'].max()),
                   grayness=float(np.mean(4 * rho * (1 - rho))),
                   change=float(np.max(np.abs(z - mma.xold1))) if mma.xold1 is not None else 1.0,
                   seconds=time.time() - t0)
        for i, tc in enumerate(prob.tcons):
            rec[f'T_{tc.name}'] = float(st['T_measured'][i])
        history.append(rec)
        if s.verbose:
            temps = ' '.join(f'{tc.name}={st["T_measured"][i]:.2f}K' for i, tc in enumerate(prob.tcons))
            print(f'it {it:3d}  C={st["C"]:.5e}  V={st["volume"]:.4f}  {temps}  '
                  f'gray={rec["grayness"]:.3f}  ch={rec["change"]:.3f}', flush=True)
        if callback:
            callback(it, st)
        feasible = (st['volume'] <= o.volume_fraction * (1 + s.volume_tolerance)
                    and (len(Tlim) == 0 or np.all(st['T_measured'] <= Tlim + s.temperature_tolerance)))
        if it > s.window and feasible:
            Cs = np.array([h['compliance'] for h in history[-(s.window + 1):]])
            rel = np.abs(np.diff(Cs)) / np.maximum(np.abs(Cs[1:]), 1e-30)
            if np.all(rel < s.tolerance):
                status = 'converged'
                break
        if it == s.max_iterations:
            break
        z = mma.update(z, f0, df0, g, dg)
    st['status'] = status
    st['iterations'] = len(history)
    st['x'] = prob.x_from_z(z)
    st['feasible'] = bool(feasible)
    return st, history
