"""HDG Stokes demo with mixed boundary conditions on [0,1] x [0,1].

Boundary conditions:
- top/bottom: Dirichlet velocity
- left/right: Neumann traction obtained from the analytical solution

This script uses a gradient-based HDG formulation for Stokes with mixed
Dirichlet/Neumann boundaries, following the structure of Formulation 2.24 in
Shannon & Bui-Thanh, "New HDG Methods for the Stokes and Oseen Equations".
"""

from __future__ import annotations

import math
import sys

import basix
import basix.ufl
import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.cpp.mesh import cell_num_entities
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block
from mpi4py import MPI
from petsc4py import PETSc
from ufl import div, dot, grad, inner

DIRICHLET_TAG = 1
NEUMANN_TAG = 2
NON_DIRICHLET_TAG = 3
ALL_FACETS_TAG = 4


def par_print(comm: MPI.Intracomm, msg: str) -> None:
    if comm.rank == 0:
        print(msg)
        sys.stdout.flush()


def norm_L2(comm: MPI.Intracomm, expr, measure=ufl.dx) -> np.floating:
    value = fem.assemble_scalar(fem.form(inner(expr, expr) * measure))
    return np.sqrt(comm.allreduce(value, op=MPI.SUM))


def compute_cell_boundary_facets(msh: mesh.Mesh) -> np.ndarray:
    tdim = msh.topology.dim
    fdim = tdim - 1
    n_f = cell_num_entities(msh.topology.cell_type, fdim)
    n_c = msh.topology.index_map(tdim).size_local
    return np.vstack((np.repeat(np.arange(n_c), n_f), np.tile(np.arange(n_f), n_c))).T.flatten()


def compute_boundary_facet_integration_entities(msh: mesh.Mesh, boundary_facets: np.ndarray) -> np.ndarray:
    """Return (cell, local_facet) pairs for the given exterior facets."""
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    msh.topology.create_connectivity(tdim, fdim)
    f_to_c = msh.topology.connectivity(fdim, tdim)
    c_to_f = msh.topology.connectivity(tdim, fdim)

    entities: list[int] = []
    for facet in boundary_facets:
        cells = f_to_c.links(facet)
        assert len(cells) == 1
        cell = cells[0]
        local_facets = c_to_f.links(cell)
        local_index = np.flatnonzero(local_facets == facet)
        assert len(local_index) == 1
        entities.extend([cell, int(local_index[0])])
    return np.asarray(entities, dtype=np.int32)


def exclude_integration_entities(all_entities: np.ndarray, excluded_entities: np.ndarray) -> np.ndarray:
    """Remove (cell, local_facet) pairs in ``excluded_entities`` from ``all_entities``."""
    all_pairs = all_entities.reshape(-1, 2)
    excluded = {tuple(pair) for pair in excluded_entities.reshape(-1, 2)}
    kept_pairs = [pair for pair in all_pairs if tuple(pair) not in excluded]
    if not kept_pairs:
        return np.empty(0, dtype=np.int32)
    return np.asarray(kept_pairs, dtype=np.int32).reshape(-1)



def velocity_exact(x):
    u1 = -x[0] ** 2 * (x[0] - 1.0) ** 2 * x[1] * (x[1] - 1.0) * (2.0 * x[1] - 1.0)
    u2 = x[0] * (x[0] - 1.0) * (2.0 * x[0] - 1.0) * x[1] ** 2 * (x[1] - 1.0) ** 2
    if isinstance(x, ufl.SpatialCoordinate):
        return ufl.as_vector((u1, u2))
    return np.vstack((u1, u2))



def pressure_exact(x):
    return x[0] ** 6 - x[1] ** 6



def top_bottom_boundary(x):
    return np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)



def left_right_boundary(x):
    return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)



def solve_with_petsc(A: PETSc.Mat, b: PETSc.Vec, comm: MPI.Intracomm, prefix: str) -> PETSc.Vec:
    attempts = [
        {
            "name": "mumps_lu",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "extra": {
                "mat_mumps_icntl_24": 1,
                "mat_mumps_icntl_25": 1,
            },
        },
        {
            "name": "superlu_dist_lu",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "superlu_dist",
            "extra": {},
        },
        {
            "name": "gmres_hypre",
            "ksp_type": "gmres",
            "pc_type": "hypre",
            "extra": {"ksp_rtol": 1.0e-10, "ksp_atol": 1.0e-12, "ksp_max_it": 5000},
        },
        {
            "name": "gmres_bjacobi",
            "ksp_type": "gmres",
            "pc_type": "bjacobi",
            "extra": {"ksp_rtol": 1.0e-10, "ksp_atol": 1.0e-12, "ksp_max_it": 5000},
        },
    ]

    last_error: str | None = None
    for i, attempt in enumerate(attempts):
        local_prefix = f"{prefix}_{i}_"
        opts = PETSc.Options()
        ksp = PETSc.KSP().create(comm)
        ksp.setOperators(A)
        ksp.setOptionsPrefix(local_prefix)
        opts[f"{local_prefix}ksp_type"] = attempt["ksp_type"]
        opts[f"{local_prefix}pc_type"] = attempt["pc_type"]
        if "pc_factor_mat_solver_type" in attempt:
            opts[f"{local_prefix}pc_factor_mat_solver_type"] = attempt["pc_factor_mat_solver_type"]
        for key, value in attempt["extra"].items():
            opts[f"{local_prefix}{key}"] = value
        ksp.setFromOptions()
        x = A.createVecRight()
        x.set(0.0)
        try:
            ksp.solve(b, x)
            if ksp.getConvergedReason() > 0:
                par_print(comm, f"Solver used: {attempt['name']}")
                return x
            last_error = f"{attempt['name']} failed with reason={ksp.getConvergedReason()}"
        except PETSc.Error as e:
            last_error = f"{attempt['name']} raised PETSc error {e.ierr}"

    raise RuntimeError(last_error or "PETSc solver failed without diagnostic information")



def solve_level(comm: MPI.Intracomm, n: int, k: int) -> tuple[float, float, float]:
    msh = mesh.create_unit_square(comm, n, n)
    tdim = msh.topology.dim
    fdim = tdim - 1
    gdim = msh.geometry.dim

    msh.topology.create_entities(fdim)
    facet_imap = msh.topology.index_map(fdim)
    num_facets = facet_imap.size_local + facet_imap.num_ghosts
    facets = np.arange(num_facets, dtype=np.int32)
    facet_mesh, facet_mesh_to_mesh = mesh.create_submesh(msh, fdim, facets)[:2]
    mesh_to_facet_mesh = np.full(num_facets, -1, dtype=np.int32)
    mesh_to_facet_mesh[facet_mesh_to_mesh] = np.arange(len(facet_mesh_to_mesh), dtype=np.int32)
    entity_maps = {facet_mesh: mesh_to_facet_mesh}

    G_el = basix.ufl.element("P", msh.basix_cell(), k, discontinuous=True, shape=(gdim, gdim))
    V_el = basix.ufl.element("P", msh.basix_cell(), k, discontinuous=True, shape=(gdim,))
    Q_el = basix.ufl.element("P", msh.basix_cell(), k, discontinuous=True)
    Vbar_el = basix.ufl.element("P", facet_mesh.basix_cell(), k, discontinuous=True, shape=(gdim,))

    G = fem.functionspace(msh, G_el)
    V = fem.functionspace(msh, V_el)
    Q = fem.functionspace(msh, Q_el)
    Vbar = fem.functionspace(facet_mesh, Vbar_el)
    W = ufl.MixedFunctionSpace(G, V, Q, Vbar)

    L_h, u_h, p_h, ubar_h = ufl.TrialFunctions(W)
    G_h, v_h, q_h, vbar_h = ufl.TestFunctions(W)

    all_cell_boundary_facets = compute_cell_boundary_facets(msh)
    dx_c = ufl.Measure("dx", domain=msh)

    dirichlet_facets = mesh.locate_entities_boundary(msh, fdim, top_bottom_boundary)
    neumann_facets = mesh.locate_entities_boundary(msh, fdim, left_right_boundary)
    dirichlet_cell_boundary_facets = compute_boundary_facet_integration_entities(msh, dirichlet_facets)
    neumann_cell_boundary_facets = compute_boundary_facet_integration_entities(msh, neumann_facets)
    non_dirichlet_cell_boundary_facets = exclude_integration_entities(
        all_cell_boundary_facets, dirichlet_cell_boundary_facets
    )
    ds_c = ufl.Measure(
        "ds",
        subdomain_data=[
            (ALL_FACETS_TAG, all_cell_boundary_facets),
            (DIRICHLET_TAG, dirichlet_cell_boundary_facets),
            (NEUMANN_TAG, neumann_cell_boundary_facets),
            (NON_DIRICHLET_TAG, non_dirichlet_cell_boundary_facets),
        ],
        domain=msh,
    )

    nu = fem.Constant(msh, dtype(1.0))
    inv_nu = fem.Constant(msh, dtype(1.0))
    x = ufl.SpatialCoordinate(msh)
    u_exact = velocity_exact(x)
    p_exact = pressure_exact(x)
    f = -nu * div(grad(u_exact)) + grad(p_exact)

    h = ufl.CellDiameter(msh)
    n_vec = ufl.FacetNormal(msh)
    tau = fem.Constant(msh, dtype(16.0 * k**2)) / h
    g_N = -nu * dot(grad(u_exact), n_vec) + p_exact * n_vec

    # Literature-based mixed-boundary HDG form with L = nu * grad(u):
    # (inv_nu L, G) + (u, div G) - <ubar, G n> = 0
    # (L, grad v) - (p, div v) + <tau (u - ubar), v> = (f, v)
    # -(u, grad q) + <ubar · n, q> = 0
    # < -L n + p n + tau (u - ubar), vbar >_{interior U Gamma_N} = <g_N, vbar>_{Gamma_N}
    A_form = (
        inv_nu * inner(L_h, G_h) * dx_c
        + inner(u_h, div(G_h)) * dx_c
        - inner(ubar_h, dot(G_h, n_vec)) * ds_c(ALL_FACETS_TAG)
        + inner(L_h, grad(v_h)) * dx_c
        - inner(p_h, div(v_h)) * dx_c
        + tau * inner(u_h - ubar_h, v_h) * ds_c(ALL_FACETS_TAG)
        - inner(u_h, grad(q_h)) * dx_c
        + inner(dot(ubar_h, n_vec), q_h) * ds_c(ALL_FACETS_TAG)
        + inner(-dot(L_h, n_vec) + p_h * n_vec + tau * (u_h - ubar_h), vbar_h) * ds_c(NON_DIRICHLET_TAG)
    )

    zero_tensor = fem.Constant(msh, np.zeros((gdim, gdim), dtype=PETSc.ScalarType))
    zero_scalar = fem.Constant(msh, dtype(0.0))
    L_blocked = [
        fem.form(inner(zero_tensor, G_h) * dx_c),
        fem.form(inner(f, v_h) * dx_c),
        fem.form(zero_scalar * q_h * dx_c),
        fem.form(inner(g_N, vbar_h) * ds_c(NEUMANN_TAG), entity_maps=entity_maps),
    ]
    A_blocked = fem.form(ufl.extract_blocks(A_form), entity_maps=entity_maps)

    facet_mesh_boundary_facets = mesh_to_facet_mesh[dirichlet_facets]
    facet_mesh_boundary_facets = facet_mesh_boundary_facets[facet_mesh_boundary_facets >= 0]
    facet_mesh.topology.create_connectivity(fdim, fdim)
    velocity_dofs = fem.locate_dofs_topological(Vbar, fdim, facet_mesh_boundary_facets)
    dirichlet_data = fem.Function(Vbar)
    dirichlet_data.interpolate(velocity_exact)
    velocity_bc = fem.dirichletbc(dirichlet_data, velocity_dofs)

    A = assemble_matrix_block(A_blocked, bcs=[velocity_bc])
    A.assemble()
    b = assemble_vector_block(L_blocked, A_blocked, bcs=[velocity_bc])
    x_vec = solve_with_petsc(A, b, msh.comm, prefix=f"stokes_numman_{n}")

    offset_L = G.dofmap.index_map.size_local * G.dofmap.index_map_bs
    offset_u = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    offset_p = Q.dofmap.index_map.size_local * Q.dofmap.index_map_bs
    offset_ubar = Vbar.dofmap.index_map.size_local * Vbar.dofmap.index_map_bs

    Lh = fem.Function(G)
    uh = fem.Function(V)
    ph = fem.Function(Q)
    ubarh = fem.Function(Vbar)
    sol = x_vec.array_r
    pos = 0
    Lh.x.array[:offset_L] = sol[pos : pos + offset_L]
    pos += offset_L
    uh.x.array[:offset_u] = sol[pos : pos + offset_u]
    pos += offset_u
    ph.x.array[:offset_p] = sol[pos : pos + offset_p]
    pos += offset_p
    ubarh.x.array[:offset_ubar] = sol[pos : pos + offset_ubar]
    Lh.x.scatter_forward()
    uh.x.scatter_forward()
    ph.x.scatter_forward()
    ubarh.x.scatter_forward()

    x = ufl.SpatialCoordinate(msh)
    u_exact = velocity_exact(x)
    p_exact = pressure_exact(x)
    e_u = norm_L2(msh.comm, uh - u_exact)
    e_p = norm_L2(msh.comm, ph - p_exact)
    e_div = norm_L2(msh.comm, div(uh))
    return float(e_u), float(e_p), float(e_div)



def convergence_rate(err_old: float, err_new: float) -> float:
    return math.log(err_old / err_new) / math.log(2.0)



def main() -> None:
    comm = MPI.COMM_WORLD
    levels = [8, 16, 32, 64]
    k = 1

    header = (
        f"{'n':>6} {'h':>10} {'||u-uh||':>14} {'rate_u':>10} "
        f"{'||p-ph||':>14} {'rate_p':>10} {'||div uh||':>14} {'rate_div':>10}"
    )
    par_print(comm, header)
    prev = None
    for n in levels:
        h = 1.0 / n
        e_u, e_p, e_div = solve_level(comm, n, k)
        if prev is None:
            rates = (float("nan"),) * 3
        else:
            rates = (
                convergence_rate(prev[0], e_u),
                convergence_rate(prev[1], e_p),
                convergence_rate(prev[2], e_div),
            )
        par_print(
            comm,
            f"{n:6d} {h:10.4e} {e_u:14.6e} {rates[0]:10.4f} "
            f"{e_p:14.6e} {rates[1]:10.4f} {e_div:14.6e} {rates[2]:10.4f}",
        )
        prev = (e_u, e_p, e_div)


if __name__ == "__main__":
    main()
