"""Meng et al. (2021) Eq. (23): stiffest thermo-elastic design.

    min_rho  C = F^T U,              F = F_m + F_th(rho, T)
    s.t.     K_m(rho) U = F          (Eq. 4)
             K_T(rho) T = P          (Eq. 1)
             sum rho_e v_e <= V_f V_0
             T_i <= Tbar_i           (each heat-source node)
             0 < x_min <= x_e <= 1

RAMP interpolation (Eq. 18), density filter (Eq. 19-21), adjoint sensitivities
(Eq. 33-37 for C, Eq. 44-49 for T_i).  The relaxed centroid von Mises stress of
Eq. (15)-(16) is evaluated for reporting (it is *not* a constraint here).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from .fem import PHI, ReducedSystem, element_matrices, scatter_add
from .mesh import GridMesh


def ramp(rho, q):
    """RAMP: r = rho / (1 + q (1 - rho)) and dr/drho."""
    den = 1.0 + q * (1.0 - rho)
    return rho / den, (1.0 + q) / den ** 2


@dataclass
class Material:
    E: float = 210e9          # Pa
    nu: float = 0.3
    k: float = 43.0           # W/(m K)
    alpha: float = 12.1e-6    # 1/K
    thickness: float = 1e-3   # m
    T_ref: float = 312.0      # K, stress-free temperature (Eq. 11)


@dataclass
class Settings:
    volume_fraction: float = 0.5
    filter_radius: float = 1.5            # metres (converted by the config loader)
    ramp_E: float = 8.0
    ramp_k: float = 8.0
    ramp_beta: float = 8.0
    k_min_ratio: float = 1e-3             # lambda_min / lambda_0
    x_min: float = 1e-3
    stress_exponent: float = 0.5          # s in Eq. (15)
    symmetry: list = field(default_factory=list)


@dataclass
class TemperatureConstraint:
    name: str
    nodes: np.ndarray
    weights: np.ndarray
    limit: float               # K


class ThermoelasticCompliance:
    def __init__(self, mesh: GridMesh, mat: Material, opt: Settings,
                 supports, loads, fixed_temps, heat_sources, temp_constraints):
        """
        supports      : list of (node_ids, 'x'|'y'|'xy')
        loads         : list of (node_ids, weights, [Fx, Fy])      total force split by weights
        fixed_temps   : list of (node_ids, value_K)
        heat_sources  : list of (node_ids, weights, power_W)
        temp_constraints : list of TemperatureConstraint
        """
        self.mesh, self.mat, self.opt = mesh, mat, opt
        ne, nn = mesh.ne, mesh.nn
        self.em = element_matrices(mesh.h, mat.thickness, mat.E, mat.nu, mat.k, mat.alpha)
        # ---- mechanical system
        edof = np.empty((ne, 8), int)
        edof[:, 0::2] = 2 * mesh.cells
        edof[:, 1::2] = 2 * mesh.cells + 1
        self.edof = edof
        fixed = []
        for ids, comp in supports:
            if 'x' in comp:
                fixed.append(2 * ids)
            if 'y' in comp:
                fixed.append(2 * ids + 1)
        if not fixed:
            raise ValueError('no mechanical supports given - structure would be a free body')
        self.mech = ReducedSystem(edof, 2 * nn, np.unique(np.concatenate(fixed)))
        self.Fm = np.zeros(2 * nn)
        self.load_groups = [(ids, np.asarray(F, float) * np.sum(w)) for ids, w, F in loads]
        for ids, w, F in loads:
            np.add.at(self.Fm, 2 * ids, w * F[0])
            np.add.at(self.Fm, 2 * ids + 1, w * F[1])
        # ---- thermal system (theta = T - T_ref)
        self.theta_D = np.zeros(nn)
        fixed_t = []
        for ids, val in fixed_temps:
            self.theta_D[ids] = val - mat.T_ref
            fixed_t.append(ids)
        if not fixed_t:
            raise ValueError('no fixed-temperature boundary given - steady heat problem has no solution')
        self.therm = ReducedSystem(mesh.cells, nn, np.unique(np.concatenate(fixed_t)))
        self.P = np.zeros(nn)
        for ids, w, power in heat_sources:
            np.add.at(self.P, ids, w * power)
        self.tcons = list(temp_constraints)
        L = np.zeros((len(self.tcons), nn))
        for i, tc in enumerate(self.tcons):
            L[i, tc.nodes] = tc.weights
        self.L = L
        # ---- design variables: passive elements, density filter, symmetry
        self.passive_solid = mesh.passive_solid
        self.passive_void = mesh.passive_void
        self.passive = self.passive_solid | self.passive_void
        self._build_filter(opt.filter_radius)
        self._build_symmetry(opt.symmetry)
        self.n_constraints = 1 + len(self.tcons)

    # ------------------------------------------------------------------ setup
    def _build_filter(self, r):
        c = self.mesh.centers
        if r <= self.mesh.h * 1.0000001:
            self.H = sparse.identity(self.mesh.ne, format='csr')
        else:
            tree = cKDTree(c)
            pairs = tree.query_pairs(r, output_type='ndarray')
            i = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(self.mesh.ne)])
            j = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(self.mesh.ne)])
            w = np.maximum(0.0, r - np.linalg.norm(c[i] - c[j], axis=1))
            self.H = sparse.csr_matrix((w, (i, j)), shape=(self.mesh.ne, self.mesh.ne))
        self.Hs = np.asarray(self.H.sum(axis=1)).ravel()

    def _build_symmetry(self, axes):
        """Map from reduced design vector z to element design x: x = S z (active elements)."""
        ne = self.mesh.ne
        parent = np.arange(ne)

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for ax in axes or []:
            m = self.mesh.mirror_map(ax)
            for a, b in enumerate(m):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
        roots = np.array([find(a) for a in range(ne)])
        design = ~self.passive
        _, group = np.unique(roots[design], return_inverse=True)
        self.design_elems = np.flatnonzero(design)
        self.S = sparse.csr_matrix((np.ones(len(group)), (self.design_elems, group)),
                                   shape=(ne, group.max() + 1 if len(group) else 0))
        self.n_design = self.S.shape[1]

    # ------------------------------------------------------------ design maps
    def x_from_z(self, z):
        x = np.asarray(self.S @ z).ravel()
        x[self.passive_solid] = 1.0
        x[self.passive_void] = self.opt.x_min
        return x

    def rho_from_x(self, x):
        rho = np.asarray(self.H @ x).ravel() / self.Hs
        rho[self.passive_solid] = 1.0
        rho[self.passive_void] = self.opt.x_min
        return rho

    def grad_z(self, g_rho):
        """Chain rule through filter (Eq. 21), passive masking and symmetry."""
        g = np.array(g_rho, dtype=float, copy=True)
        g[..., self.passive] = 0.0
        g_x = np.asarray(self.H.T @ (g / self.Hs).T).T if g.ndim == 2 else np.asarray(self.H.T @ (g / self.Hs)).ravel()
        if g.ndim == 2:
            return np.asarray((self.S.T @ g_x.T).T)
        return np.asarray(self.S.T @ g_x).ravel()

    # --------------------------------------------------------------- analysis
    def interpolate(self, rho):
        o = self.opt
        rE, drE = ramp(rho, o.ramp_E)
        rk, drk = ramp(rho, o.ramp_k)
        rB, drB = ramp(rho, o.ramp_beta)
        kf = o.k_min_ratio + (1 - o.k_min_ratio) * rk
        dkf = (1 - o.k_min_ratio) * drk
        return rE, drE, kf, dkf, rB, drB

    def analyze(self, rho, sensitivities=True):
        em, mesh = self.em, self.mesh
        nn = mesh.nn
        rE, drE, kf, dkf, rB, drB = self.interpolate(rho)
        # ---- heat conduction  K_T theta = P   (Dirichlet part theta_D)
        th = self.therm
        KT = th.matrix(kf, em['kt0'])
        rhs = self.P.copy()
        if np.any(self.theta_D):
            KT_full = th.full_matrix(kf, em['kt0'])
            rhs = rhs - KT_full @ self.theta_D
        tfac = th.factor(KT)
        theta = self.theta_D.copy()
        theta[th.free] = tfac.solve(rhs[th.free])
        theta_e = theta[mesh.cells]
        # ---- thermo-elastic equilibrium  K U = Fm + Fth
        fth_e = rB[:, None] * (theta_e @ em['g0'].T)
        Fth = scatter_add(2 * nn, self.edof, fth_e)
        F = self.Fm + Fth
        me = self.mech
        K = me.matrix(rE, em['ke0'])
        mfac = me.factor(K)
        U = np.zeros(2 * nn)
        U[me.free] = mfac.solve(F[me.free])
        C = float(F @ U)
        T_meas = self.mat.T_ref + self.L @ theta
        vol = float(rho.mean())
        st = dict(rho=rho, theta=theta, T=theta + self.mat.T_ref, U=U, F=F, Fth=Fth,
                  C=C, volume=vol, T_measured=T_meas)
        if not sensitivities:
            return st
        Ue = U[self.edof]
        # ---- compliance sensitivity (adjoint, lambda_m = -U):
        # dC/drho_e = -dE u'k u + 2 dB u'g theta - 2 dk z'kt theta,   K_T z = sum_e rB g' u_e
        ze_rhs = scatter_add(nn, mesh.cells, rB[:, None] * (Ue @ em['g0']))
        z = np.zeros(nn)
        z[th.free] = tfac.solve(ze_rhs[th.free])
        ze = z[mesh.cells]
        dC = (-drE * np.einsum('ei,ij,ej->e', Ue, em['ke0'], Ue)
              + 2 * drB * np.einsum('ei,ij,ej->e', Ue, em['g0'], theta_e)
              - 2 * dkf * np.einsum('ei,ij,ej->e', ze, em['kt0'], theta_e))
        # ---- temperature constraints (Eq. 48-49): K_T psi_i = L_i
        if len(self.tcons):
            psi = np.zeros((nn, len(self.tcons)))
            psi[th.free] = tfac.solve(np.ascontiguousarray(self.L[:, th.free].T))
            dT = -dkf[None, :] * np.einsum('eia,ij,ej->ae', psi[mesh.cells], em['kt0'], theta_e)
        else:
            dT = np.zeros((0, mesh.ne))
        st['dC'] = dC
        st['dT'] = dT
        st['dV'] = np.full(mesh.ne, 1.0 / mesh.ne)
        return st

    # ---------------------------------------------------------- post-process
    def stress(self, st):
        """Relaxed centroid von Mises stress, Meng Eq. (15)-(16), in Pa."""
        rho = st['rho']
        rE, _, _, _, rB, _ = self.interpolate(rho)
        alpha_rho = self.mat.alpha * rB / rE                 # alpha(rho) = beta / E
        eps = st['U'][self.edof] @ self.em['Bc'].T
        th_c = st['theta'][self.mesh.cells].mean(axis=1)
        eps_m = eps - (alpha_rho * th_c)[:, None] * PHI
        sig = (rho ** self.opt.stress_exponent)[:, None] * (eps_m @ self.em['D0'].T)
        sx, sy, txy = sig.T
        return np.sqrt(np.maximum(sx * sx - sx * sy + sy * sy + 3 * txy * txy, 0.0))

    def load_ratio(self, st):
        """||F_th|| / ||F_m|| as reported in Meng Tables 1-3 (all DOFs and free DOFs)."""
        fr = self.mech.free
        return (float(np.linalg.norm(st['Fth']) / np.linalg.norm(self.Fm)),
                float(np.linalg.norm(st['Fth'][fr]) / max(np.linalg.norm(self.Fm[fr]), 1e-30)))
