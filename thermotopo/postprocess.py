"""Figures and files written for every run."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from .problem import ThermoelasticCompliance  # noqa: E402


def _extent(mesh):
    x0, y0, h = mesh.x0, mesh.y0, mesh.h_file
    return [x0, x0 + mesh.nx * h, y0, y0 + mesh.ny * h]


def _img(ax, mesh, values, cmap, vmin=None, vmax=None, title='', cbar=True, units='mm'):
    a = mesh.to_grid(values)
    im = ax.imshow(np.ma.masked_invalid(a), origin='lower', extent=_extent(mesh), cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(f'x ({units})')
    ax.set_ylabel(f'y ({units})')
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    return im


def plot_setup(prob: ThermoelasticCompliance, path, units='mm', title=''):
    """Draw domain + boundary conditions so the user can check the problem file."""
    mesh = prob.mesh
    fig, ax = plt.subplots(figsize=(7, 6), layout='constrained')
    val = np.full(mesh.ne, 0.25)
    val[mesh.passive_solid] = 0.9
    val[mesh.passive_void] = 0.0
    _img(ax, mesh, val, 'Greys', 0, 1, cbar=False, units=units)
    N = mesh.nodes_file
    fixed_nodes = np.unique(prob.mech.fixed // 2)
    ax.plot(*N[fixed_nodes].T, 's', ms=3, color='tab:blue', label='mechanical support')
    tfix = prob.therm.fixed
    ax.plot(*N[tfix].T, 'o', ms=2, mfc='none', color='tab:cyan', label='fixed temperature')
    span = max(mesh.nx, mesh.ny) * mesh.h_file
    for k, (ids, F) in enumerate(prob.load_groups):
        ax.plot(*N[ids].T, '.', ms=4, color='tab:red')
        nF = np.linalg.norm(F)
        if nF == 0:
            continue
        c = N[ids].mean(axis=0)
        d = F / nF * 0.12 * span
        ax.annotate('', xy=c, xytext=c - d, arrowprops=dict(arrowstyle='-|>', color='tab:red', lw=2.5))
        ax.annotate(f'{nF:g} N', c - d, xytext=(3, 3), textcoords='offset points', color='tab:red', fontsize=8)
        if k == 0:
            ax.plot([], [], '-', color='tab:red', lw=2.5, label='mechanical load (total)')
    hs = np.flatnonzero(prob.P > 0)
    ax.plot(*N[hs].T, '*', ms=12, color='orange', mec='k', label='heat source')
    for i, tc in enumerate(prob.tcons):
        p = (tc.weights[:, None] * N[tc.nodes]).sum(0)
        ax.annotate(f'{tc.name} <= {tc.limit:g} K', p, xytext=(4, 6), textcoords='offset points', fontsize=8)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)
    ax.set_title(title or 'Problem setup (gray = design domain, light = passive solid)')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_fields(prob, st, vm, path, units='mm', title='', T_range=None):
    """Paper-style panels: (A) density, (B) von Mises stress, (C) temperature."""
    mesh = prob.mesh
    fig, axs = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
    _img(axs[0], mesh, st['rho'], 'gray_r', 0, 1, '(A) density', units=units)
    _img(axs[1], mesh, vm * 1e-6, 'jet', 0, None, f'(B) von Mises stress, Eq. 15 (MPa), max {vm.max() * 1e-6:.1f}', units=units)
    Te = st['T'][mesh.cells].mean(axis=1)
    lo, hi = T_range if T_range else (None, None)
    _img(axs[2], mesh, Te, 'jet', lo, hi, f'(C) temperature (K), max {st["T"].max():.2f}', units=units)
    if title:
        fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_density(prob, st, path, units='mm'):
    mesh = prob.mesh
    w = mesh.nx / max(mesh.nx, mesh.ny)
    fig, ax = plt.subplots(figsize=(6 * w + 1.5, 6 * mesh.ny / max(mesh.nx, mesh.ny) + 0.8), layout='constrained')
    _img(ax, mesh, st['rho'], 'gray_r', 0, 1, 'optimized density', units=units)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_history(history, prob, path):
    it = [h['iteration'] for h in history]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
    axs[0].plot(it, [h['compliance'] for h in history])
    axs[0].set_ylabel('compliance C = F^T U (N m)')
    axs[1].plot(it, [h['volume'] for h in history])
    axs[1].axhline(prob.opt.volume_fraction, color='k', ls='--', lw=1, label='limit')
    axs[1].set_ylabel('volume fraction')
    axs[1].legend()
    for tc in prob.tcons:
        axs[2].plot(it, [h[f'T_{tc.name}'] for h in history], label=tc.name)
    lims = sorted({tc.limit for tc in prob.tcons})
    for L in lims:
        axs[2].axhline(L, color='k', ls='--', lw=1)
    axs[2].set_ylabel('constrained temperature (K)')
    if prob.tcons and len(prob.tcons) <= 8:
        axs[2].legend(fontsize=8)
    for a in axs:
        a.set_xlabel('MMA iteration')
        a.grid(alpha=0.3)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_vtk(prob, st, vm, path):
    """Legacy-ASCII VTK (open with ParaView): cell density/stress, nodal T and U."""
    mesh = prob.mesh
    N = mesh.nodes_file
    with open(path, 'w') as f:
        f.write('# vtk DataFile Version 3.0\nthermoelastic topology optimization\nASCII\nDATASET UNSTRUCTURED_GRID\n')
        f.write(f'POINTS {mesh.nn} double\n')
        np.savetxt(f, np.column_stack([N, np.zeros(mesh.nn)]), fmt='%.6g')
        f.write(f'CELLS {mesh.ne} {5 * mesh.ne}\n')
        np.savetxt(f, np.column_stack([np.full(mesh.ne, 4), mesh.cells]), fmt='%d')
        f.write(f'CELL_TYPES {mesh.ne}\n')
        np.savetxt(f, np.full(mesh.ne, 9), fmt='%d')
        f.write(f'CELL_DATA {mesh.ne}\nSCALARS density double 1\nLOOKUP_TABLE default\n')
        np.savetxt(f, st['rho'], fmt='%.6g')
        f.write('SCALARS von_mises_MPa double 1\nLOOKUP_TABLE default\n')
        np.savetxt(f, vm * 1e-6, fmt='%.6g')
        f.write(f'POINT_DATA {mesh.nn}\nSCALARS temperature_K double 1\nLOOKUP_TABLE default\n')
        np.savetxt(f, st['T'], fmt='%.6g')
        f.write('VECTORS displacement_m double\n')
        np.savetxt(f, np.column_stack([st['U'].reshape(-1, 2), np.zeros(mesh.nn)]), fmt='%.6g')


def summarize(prob, st, history):
    vm = prob.stress(st)
    r_all, r_free = prob.load_ratio(st)
    rho = st['rho']
    s = dict(
        compliance_Nm=st['C'],
        volume_fraction=st['volume'],
        volume_limit=prob.opt.volume_fraction,
        max_von_mises_MPa=float(vm.max() * 1e-6),
        max_von_mises_MPa_solid_only=float(vm[rho > 0.5].max() * 1e-6) if np.any(rho > 0.5) else float('nan'),
        T_max_domain_K=float(st['T'].max()),
        constrained_temperatures_K={tc.name: float(st['T_measured'][i]) for i, tc in enumerate(prob.tcons)},
        temperature_limits_K={tc.name: tc.limit for tc in prob.tcons},
        Fth_over_Fm=r_all,
        Fth_over_Fm_free_dofs=r_free,
        grayness_4rho1mrho=float(np.mean(4 * rho * (1 - rho))),
        gray_fraction_0p1_0p9=float(np.mean((rho > 0.1) & (rho < 0.9))),
        iterations=st.get('iterations'),
        status=st.get('status'),
        feasible=st.get('feasible'),
        elements=prob.mesh.ne,
        nodes=prob.mesh.nn,
        design_variables=prob.n_design,
        runtime_s=history[-1]['seconds'] if history else None,
    )
    return s, vm


def save_all(prob, st, history, outdir, cfg_used: dict, info: dict):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    units = info.get('units', 'mm')
    s, vm = summarize(prob, st, history)
    title = (f"{info.get('name', '')}: C = {s['compliance_Nm']:.4g} N m, V = {s['volume_fraction']:.3f}, "
             f"max sigma_vM = {s['max_von_mises_MPa']:.0f} MPa, T_max = {s['T_max_domain_K']:.2f} K, "
             f"{s['iterations']} it. ({s['status']})")
    T_range = info.get('plot', {}).get('temperature_range')
    plot_fields(prob, st, vm, out / 'fields.png', units, title, T_range)
    plot_density(prob, st, out / 'density.png', units)
    plot_history(history, prob, out / 'history.png')
    write_vtk(prob, st, vm, out / 'result.vtk')
    np.savez_compressed(out / 'result.npz', rho=st['rho'], x=st['x'], T=st['T'], U=st['U'], von_mises_Pa=vm,
                        nodes=prob.mesh.nodes_file, cells=prob.mesh.cells, elem_i=prob.mesh.elem_i,
                        elem_j=prob.mesh.elem_j, nx=prob.mesh.nx, ny=prob.mesh.ny, h=prob.mesh.h_file,
                        x0=prob.mesh.x0, y0=prob.mesh.y0)
    with open(out / 'history.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    (out / 'summary.json').write_text(json.dumps(s, indent=2))
    clean = {k: v for k, v in cfg_used.items() if not k.startswith('_')}
    try:
        import yaml
        (out / 'problem_used.yaml').write_text(yaml.safe_dump(clean, sort_keys=False))
    except ImportError:  # pragma: no cover
        (out / 'problem_used.json').write_text(json.dumps(clean, indent=2))
    return s
