"""HDG Stokes demo with mixed boundary conditions on [0,1] x [0,1].

Boundary conditions:
- top/bottom: Dirichlet velocity
- left/right: Neumann traction obtained from the analytical solution

The script follows the FEniCSx 0.9.0 mixed-domain HDG assembly pattern and
implements the non-homogeneous mixed-boundary variational form in the style
used by the companion Stokes demo.
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

    nu = fem.Constant(msh, dtype(1.0))
    epsilon_p = fem.Constant(msh, dtype(1.0e-12))
    x = ufl.SpatialCoordinate(msh)
    u_exact = velocity_exact(x)
    p_exact = pressure_exact(x)
    f = -nu * div(grad(u_exact)) + grad(p_exact)

    h = ufl.CellDiameter(msh)
    n_vec = ufl.FacetNormal(msh)
    alpha = fem.Constant(msh, dtype(16.0 * k**2))

    g_N = -nu * dot(grad(u_exact), n_vec) + p_exact * n_vec
    # Non-homogeneous mixed-boundary HDG form in the primal (u, ubar, p, pbar)
    # style:
    #  - pressure-trace/velocity-trace consistency terms on Gamma_N are treated
    #    with the red correction terms from the provided variational equation;
    #  - non-homogeneous Dirichlet data is imposed strongly on ubar over Gamma_D;
    #  - non-homogeneous Neumann traction contributes through
    #    -<g_N, vbar>_{Gamma_N}.
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

    zero_scalar_c = fem.Constant(msh, dtype(0.0))
    zero_scalar_f = fem.Constant(facet_mesh, dtype(0.0))

    L_u = inner(f, v_h) * dx_c
    L_ubar = -inner(g_N, vbar_h) * ds_c(NEUMANN_TAG)
    L_p = zero_scalar_c * q_h * dx_c
    L_pbar = zero_scalar_f * qbar_h * dx_f

    A_blocked = fem.form(ufl.extract_blocks(A_form), entity_maps=entity_maps)
    L_blocked = [
        fem.form(L_u),
        fem.form(L_ubar, entity_maps=entity_maps),
        fem.form(L_p),
        fem.form(L_pbar, entity_maps=entity_maps),
    ]

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
