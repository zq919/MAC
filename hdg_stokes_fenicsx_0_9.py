"""FEniCSx 0.9.0 HDG Stokes demo in 2D (mixed boundaries).

This version follows the HDG polynomial-space definition requested by the user:

- V_h(K): [P_k(K)]^d  (discontinuous)
- Vbar_h(F): [P_k(F)]^d
- Q_h(K): P_{k-1}(K)
- Qbar_h(F): P_k(F)

Boundary conditions:
- top/bottom: Dirichlet velocity
- left/right: non-homogeneous Neumann traction (from exact solution)

The script runs a mesh-refinement study and reports:
- velocity L2 error and observed rate
- pressure L2 error and observed rate
- divergence L2 error and observed rate
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


def par_print(comm: MPI.Intracomm, msg: str) -> None:
    if comm.rank == 0:
        print(msg)
        sys.stdout.flush()


def norm_L2(comm: MPI.Intracomm, expr, measure=ufl.dx) -> np.floating:
    value = fem.assemble_scalar(fem.form(inner(expr, expr) * measure))
    return np.sqrt(comm.allreduce(value, op=MPI.SUM))


def compute_cell_boundary_facets(msh: mesh.Mesh) -> np.ndarray:
    """Return integration entities for all cell boundaries in ``msh``."""
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


def velocity_exact(x):
    """Exact velocity from the Stokes benchmark shown by the user."""
    u1 = -x[0] ** 2 * (x[0] - 1.0) ** 2 * x[1] * (x[1] - 1.0) * (2.0 * x[1] - 1.0)
    u2 = x[0] * (x[0] - 1.0) * (2.0 * x[0] - 1.0) * x[1] ** 2 * (x[1] - 1.0) ** 2
    if isinstance(x, ufl.SpatialCoordinate):
        return ufl.as_vector((u1, u2))
    return np.vstack((u1, u2))


def pressure_exact(x):
    """Exact pressure from the Stokes benchmark shown by the user."""
    return x[0] ** 6 - x[1] ** 6


def top_bottom_boundary(x):
    return np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)


def left_right_boundary(x):
    return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)


def solve_with_petsc(A: PETSc.Mat, b: PETSc.Vec, comm: MPI.Intracomm, prefix: str) -> PETSc.Vec:
    """Solve the linear system with a robust direct/fallback strategy."""
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
            "extra": {
                "ksp_rtol": 1.0e-10,
                "ksp_atol": 1.0e-12,
                "ksp_max_it": 5000,
            },
        },
        {
            "name": "gmres_bjacobi",
            "ksp_type": "gmres",
            "pc_type": "bjacobi",
            "extra": {
                "ksp_rtol": 1.0e-10,
                "ksp_atol": 1.0e-12,
                "ksp_max_it": 5000,
            },
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
    """Solve the HDG Stokes problem on an n x n mesh."""
    msh = mesh.create_unit_square(comm, n, n)
    tdim = msh.topology.dim
    fdim = tdim - 1
    gdim = msh.geometry.dim

    # Facet submesh used for trace spaces.
    msh.topology.create_entities(fdim)
    facet_imap = msh.topology.index_map(fdim)
    num_facets = facet_imap.size_local + facet_imap.num_ghosts
    facets = np.arange(num_facets, dtype=np.int32)
    facet_mesh, facet_mesh_to_mesh = mesh.create_submesh(msh, fdim, facets)[:2]
    mesh_to_facet_mesh = np.full(num_facets, -1, dtype=np.int32)
    mesh_to_facet_mesh[facet_mesh_to_mesh] = np.arange(len(facet_mesh_to_mesh), dtype=np.int32)
    entity_maps = {facet_mesh: mesh_to_facet_mesh}

    # HDG polynomial spaces requested by the user.
    V_el = basix.ufl.element("P", msh.basix_cell(), k, discontinuous=True, shape=(gdim,))
    Vbar_el = basix.ufl.element("P", facet_mesh.basix_cell(), k, discontinuous=True, shape=(gdim,))
    Q_el = basix.ufl.element("P", msh.basix_cell(), k - 1, discontinuous=True)
    Qbar_el = basix.ufl.element("P", facet_mesh.basix_cell(), k, discontinuous=True)

    V = fem.functionspace(msh, V_el)
    Vbar = fem.functionspace(facet_mesh, Vbar_el)
    Q = fem.functionspace(msh, Q_el)
    Qbar = fem.functionspace(facet_mesh, Qbar_el)
    W = ufl.MixedFunctionSpace(V, Vbar, Q, Qbar)

    u_h, ubar_h, p_h, pbar_h = ufl.TrialFunctions(W)
    v_h, vbar_h, q_h, qbar_h = ufl.TestFunctions(W)

    # Measures
    all_cell_boundary_facets = compute_cell_boundary_facets(msh)
    dx_c = ufl.Measure("dx", domain=msh)
    dx_f = ufl.Measure("dx", domain=facet_mesh)

    dirichlet_facets = mesh.locate_entities_boundary(msh, fdim, top_bottom_boundary)
    neumann_facets = mesh.locate_entities_boundary(msh, fdim, left_right_boundary)
    dirichlet_cell_boundary_facets = compute_boundary_facet_integration_entities(msh, dirichlet_facets)
    neumann_cell_boundary_facets = compute_boundary_facet_integration_entities(msh, neumann_facets)
    ds_c = ufl.Measure(
        "ds",
        subdomain_data=[
            (0, all_cell_boundary_facets),
            (DIRICHLET_TAG, dirichlet_cell_boundary_facets),
            (NEUMANN_TAG, neumann_cell_boundary_facets),
        ],
        domain=msh,
    )

    # Exact solution and forcing term
    nu = fem.Constant(msh, dtype(1.0))
    epsilon_p = fem.Constant(msh, dtype(1.0e-12))
    x = ufl.SpatialCoordinate(msh)
    u_exact = velocity_exact(x)
    p_exact = pressure_exact(x)
    f = -nu * div(grad(u_exact)) + grad(p_exact)

    # HDG coefficients
    h = ufl.CellDiameter(msh)
    n_vec = ufl.FacetNormal(msh)
    alpha = fem.Constant(msh, dtype(16.0 * k**2))
    g_N = -nu * dot(grad(u_exact), n_vec) + p_exact * n_vec

    # Non-homogeneous mixed-boundary HDG form:
    # - Dirichlet on Gamma_D imposed strongly on ubar_h.
    # - Neumann data enters via -<g_N, vbar_h>_{Gamma_N}.
    # - Gamma_N correction terms are included for pbar_h/vbar_h and
    #   ubar_h/qbar_h couplings following the requested variational form.
    a = (
        nu * inner(grad(u_h), grad(v_h)) * dx_c
        - nu * inner(u_h - ubar_h, dot(grad(v_h), n_vec)) * ds_c(0)
        - nu * inner(dot(grad(u_h), n_vec), v_h - vbar_h) * ds_c(0)
        + nu * (alpha / h) * inner(u_h - ubar_h, v_h - vbar_h) * ds_c(0)
    )
    b_vp = (
        -inner(p_h, div(v_h)) * dx_c
        + inner(pbar_h, dot(v_h, n_vec)) * ds_c(0)
        - inner(pbar_h, dot(vbar_h, n_vec)) * ds_c(NEUMANN_TAG)
    )
    b_uq = (
        -inner(q_h, div(u_h)) * dx_c
        + inner(dot(u_h, n_vec), qbar_h) * ds_c(0)
        - inner(dot(ubar_h, n_vec), qbar_h) * ds_c(NEUMANN_TAG)
    )
    A_form = a + b_vp + b_uq + epsilon_p * p_h * q_h * dx_c

    # Build RHS blocks manually. In FEniCSx 0.9.0, mixed-domain extraction from
    # a single linear form is less robust than explicit per-block forms.
    zero_scalar_c = fem.Constant(msh, dtype(0.0))
    zero_scalar_f = fem.Constant(facet_mesh, dtype(0.0))
    L_u = inner(f, v_h) * dx_c
    L_ubar = -inner(g_N, vbar_h) * ds_c(NEUMANN_TAG)
    L_p = zero_scalar_c * q_h * dx_c
    L_pbar = zero_scalar_f * qbar_h * dx_f

    # Compile forms
    A_blocked = fem.form(ufl.extract_blocks(A_form), entity_maps=entity_maps)
    L_blocked = [
        fem.form(L_u),
        fem.form(L_ubar, entity_maps=entity_maps),
        fem.form(L_p),
        fem.form(L_pbar, entity_maps=entity_maps),
    ]

    # Dirichlet condition on the trace velocity: top/bottom only.
    facet_mesh_boundary_facets = mesh_to_facet_mesh[dirichlet_facets]
    facet_mesh_boundary_facets = facet_mesh_boundary_facets[facet_mesh_boundary_facets >= 0]
    facet_mesh.topology.create_connectivity(fdim, fdim)
    velocity_dofs = fem.locate_dofs_topological(Vbar, fdim, facet_mesh_boundary_facets)
    ubar_bc = fem.Function(Vbar)
    ubar_bc.interpolate(velocity_exact)
    velocity_bc = fem.dirichletbc(ubar_bc, velocity_dofs)

    # Assemble matrix/vector.
    A = assemble_matrix_block(A_blocked, bcs=[velocity_bc])
    A.assemble()
    b = assemble_vector_block(L_blocked, A_blocked, bcs=[velocity_bc])

    # Solve with a robust direct/fallback PETSc strategy.
    x_vec = solve_with_petsc(A, b, msh.comm, prefix=f"hdg_stokes_{n}")

    # Scatter the monolithic solution into Functions.
    offset_u = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    offset_ubar = Vbar.dofmap.index_map.size_local * Vbar.dofmap.index_map_bs
    offset_p = Q.dofmap.index_map.size_local * Q.dofmap.index_map_bs
    offset_pbar = Qbar.dofmap.index_map.size_local * Qbar.dofmap.index_map_bs

    uh = fem.Function(V)
    ubarh = fem.Function(Vbar)
    ph = fem.Function(Q)
    pbarh = fem.Function(Qbar)
    sol = x_vec.array_r
    pos = 0
    uh.x.array[:offset_u] = sol[pos : pos + offset_u]
    pos += offset_u
    ubarh.x.array[:offset_ubar] = sol[pos : pos + offset_ubar]
    pos += offset_ubar
    ph.x.array[:offset_p] = sol[pos : pos + offset_p]
    pos += offset_p
    pbarh.x.array[:offset_pbar] = sol[pos : pos + offset_pbar]
    uh.x.scatter_forward()
    ubarh.x.scatter_forward()
    ph.x.scatter_forward()
    pbarh.x.scatter_forward()

    # Error norms
    x = ufl.SpatialCoordinate(msh)
    u_exact = velocity_exact(x)
    p_exact = pressure_exact(x)
    e_u = norm_L2(msh.comm, uh - u_exact)
    e_p = norm_L2(msh.comm, ph - p_exact)
    e_div = norm_L2(msh.comm, div(uh))
    return float(e_u), float(e_p), float(e_div)


def convergence_rate(err_old: float, err_new: float) -> float:
    return math.log(err_old / err_new) / math.log(2.0)


comm = MPI.COMM_WORLD
k = 2
levels = [8, 16, 32, 64]
results: list[tuple[int, float, float, float]] = []

for n in levels:
    results.append((n, *solve_level(comm, n, k)))

if comm.rank == 0:
    header = (
        " n |      ||u-u_h||_L2 | rate_u |      ||p-p_h||_L2 | rate_p | "
        "    ||div(u_h)||_L2 | rate_div"
    )
    print(header)
    print("-" * len(header))
    for i, (n, e_u, e_p, e_div) in enumerate(results):
        if i == 0:
            print(f"{n:2d} | {e_u:16.8e} |   ---  | {e_p:16.8e} |   ---  | {e_div:16.8e} |   ---")
        else:
            _, prev_u, prev_p, prev_div = results[i - 1]
            ru = convergence_rate(prev_u, e_u)
            rp = convergence_rate(prev_p, e_p)
            rd = convergence_rate(prev_div, e_div) if e_div > 0.0 and prev_div > 0.0 else float('nan')
            print(f"{n:2d} | {e_u:16.8e} | {ru:6.3f} | {e_p:16.8e} | {rp:6.3f} | {e_div:16.8e} | {rd:8.3f}")
