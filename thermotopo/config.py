"""Read a problem file (YAML or JSON) and build mesh + problem + settings.

See ``examples/*.yaml`` and the README for the full format.  Minimal skeleton:

    name: my_part
    units: mm
    geometry:   {element_size: 1, add: [...shapes...], subtract: [...]}
    material:   {youngs_modulus: 210e9, poisson_ratio: 0.3, thermal_conductivity: 43,
                 thermal_expansion: 12.1e-6, thickness: 1, reference_temperature: 312}
    mechanical: {supports: [{where: ..., fix: xy}], loads: [{where: ..., force: [Fx, Fy]}]}
    thermal:    {fixed_temperature: [{where: ..., value: 312}],
                 heat_sources: [{name: H1, where: {point: [x, y]}, power: 0.06, max_temperature: 325}]}
    optimization: {volume_fraction: 0.5, filter_radius: 1.5, ...}
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from .geometry import Domain, make_shape
from .mesh import GridMesh
from .optimizer import OptimizerSettings
from .problem import Material, Settings, TemperatureConstraint, ThermoelasticCompliance

UNIT_SCALE = {'m': 1.0, 'mm': 1e-3, 'cm': 1e-2, 'um': 1e-6, 'in': 0.0254}

DEFAULT_MATERIAL = dict(youngs_modulus=210e9, poisson_ratio=0.3, thermal_conductivity=43.0,
                        thermal_expansion=12.1e-6, thickness=1.0, reference_temperature=312.0)
DEFAULT_OPT = dict(volume_fraction=0.5, filter_radius=None, ramp_stiffness=8.0, ramp_conductivity=8.0,
                   ramp_thermal_stress=8.0, conductivity_min_ratio=1e-3, x_min=1e-3,
                   stress_exponent=0.5, symmetry=[], max_iterations=200, tolerance=1e-3, window=10,
                   move_limit=0.2, initial_density=1.0, objective_scale=1.0, temperature_scaling='rise')


def load_file(path) -> dict:
    path = Path(path)
    text = path.read_text(encoding='utf-8')
    if path.suffix.lower() in ('.yaml', '.yml'):
        try:
            import yaml
        except ImportError as e:  # pragma: no cover
            raise ImportError('PyYAML is needed for .yaml files (pip install pyyaml) - or use a .json file') from e
        cfg = yaml.safe_load(text)
    else:
        cfg = json.loads(text)
    cfg['_base_dir'] = str(path.parent.resolve())
    cfg['_source'] = str(path)
    return cfg


def _num(v):
    """YAML 1.1 reads '1e-3' as a string; accept it as a number."""
    if isinstance(v, str):
        return float(v)
    return v


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """``key.sub=value`` strings from the command line (value parsed as YAML/JSON)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        key, _, val = item.partition('=')
        try:
            import yaml
            v = yaml.safe_load(val)
        except Exception:
            try:
                v = json.loads(val)
            except Exception:
                v = val
        d = cfg
        parts = key.split('.')
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    return cfg


def build(cfg: dict):
    """Return (problem, optimizer_settings, info dict)."""
    base = Path(cfg.get('_base_dir', '.'))
    units = cfg.get('units', 'mm')
    if units not in UNIT_SCALE:
        raise ValueError(f'units must be one of {list(UNIT_SCALE)}')
    sc = UNIT_SCALE[units]
    g = cfg['geometry']
    h = float(_num(g['element_size']))
    dom = Domain([make_shape(s, base) for s in g['add']], [make_shape(s, base) for s in g.get('subtract', [])])
    ps = g.get('passive_solid') or []
    pv = g.get('passive_void') or []
    mesh = GridMesh(dom, h, sc,
                    Domain([make_shape(s, base) for s in ps]) if ps else None,
                    Domain([make_shape(s, base) for s in pv]) if pv else None)

    m = {**DEFAULT_MATERIAL, **(cfg.get('material') or {})}
    mat = Material(E=float(_num(m['youngs_modulus'])), nu=float(_num(m['poisson_ratio'])),
                   k=float(_num(m['thermal_conductivity'])), alpha=float(_num(m['thermal_expansion'])),
                   thickness=float(_num(m['thickness'])) * sc, T_ref=float(_num(m['reference_temperature'])))

    o = {**DEFAULT_OPT, **(cfg.get('optimization') or {})}
    rmin = o['filter_radius'] if o['filter_radius'] is not None else 1.5 * h
    sym = o.get('symmetry') or []
    if isinstance(sym, str):
        sym = [] if sym.lower() in ('none', '') else [sym]
    settings = Settings(volume_fraction=float(_num(o['volume_fraction'])), filter_radius=float(_num(rmin)) * sc,
                        ramp_E=float(_num(o['ramp_stiffness'])), ramp_k=float(_num(o['ramp_conductivity'])),
                        ramp_beta=float(_num(o['ramp_thermal_stress'])),
                        k_min_ratio=float(_num(o['conductivity_min_ratio'])), x_min=float(_num(o['x_min'])),
                        stress_exponent=float(_num(o['stress_exponent'])), symmetry=list(sym))
    if not 0 < settings.volume_fraction <= 1:
        raise ValueError('volume_fraction must be in (0, 1]')
    os_ = OptimizerSettings(max_iterations=int(o['max_iterations']), tolerance=float(_num(o['tolerance'])),
                            window=int(o['window']), move_limit=float(_num(o['move_limit'])),
                            initial_density=float(_num(o['initial_density'])),
                            objective_scale=float(_num(o['objective_scale'])),
                            temperature_scaling=str(o['temperature_scaling']))

    mech = cfg.get('mechanical') or {}
    supports = []
    for s in mech.get('supports', []):
        ids, _ = mesh.select(s['where'])
        fix = str(s.get('fix', 'xy')).lower()
        if not set(fix) <= {'x', 'y'}:
            raise ValueError(f'support "fix" must be x, y or xy, got {fix!r}')
        supports.append((ids, fix))
    loads = []
    for ld in mech.get('loads', []):
        ids, w = mesh.select(ld['where'])
        F = np.asarray([_num(v) for v in ld['force']], float)
        if ld.get('per_node', False):
            w = np.ones(len(ids))
        loads.append((ids, w, F))

    th = cfg.get('thermal') or {}
    fixed_t = []
    for ft in th.get('fixed_temperature', []):
        ids, _ = mesh.select(ft['where'])
        fixed_t.append((ids, float(_num(ft.get('value', mat.T_ref)))))
    sources, tcons = [], []
    for i, hs in enumerate(th.get('heat_sources', [])):
        name = str(hs.get('name', f'H{i + 1}'))
        ids, w = mesh.select(hs['where'], interpolate_points=True)
        sources.append((ids, w, float(_num(hs['power']))))
        if hs.get('max_temperature') is not None:
            lim = float(_num(hs['max_temperature']))
            if 'point' in hs['where']:
                tcons.append(TemperatureConstraint(name, ids, w, lim))      # temperature at the point
            else:                                                            # every node of the region
                for k, nid in enumerate(ids):
                    tcons.append(TemperatureConstraint(f'{name}.{k}', np.array([nid]), np.array([1.0]), lim))
    for i, tc in enumerate(th.get('temperature_limits', [])):          # extra monitored points
        ids, w = mesh.select(tc['where'], interpolate_points=True)
        tcons.append(TemperatureConstraint(str(tc.get('name', f'P{i + 1}')), ids, w, float(_num(tc['max_temperature']))))

    prob = ThermoelasticCompliance(mesh, mat, settings, supports, loads, fixed_t, sources, tcons)
    info = dict(name=cfg.get('name', Path(cfg.get('_source', 'problem')).stem), units=units,
                description=cfg.get('description', ''), out=(cfg.get('output') or {}).get('directory'),
                plot=(cfg.get('output') or {}).get('plot', {}), reference=cfg.get('paper_reference'))
    return prob, os_, info
