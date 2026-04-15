"""Biot-HDG (u, uhat, p, phat) demo following the provided variational form.

The program keeps the same high-level framework as the user's script:
- mixed HDG spaces on cell/facet meshes
- time stepping with backward Euler
- blocked assembly in FEniCSx 0.9.0
- PETSc direct/fallback solver strategy

Variational blocks implemented from the image:
- a_h(p_h, w_h)
- e_h(u_h, v_h)
- b_h / \tilde b_h coupling terms
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
from dolfinx.fem import Function, dirichletbc
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block
from mpi4py import MPI
from petsc4py import PETSc
from ufl import div, dot, grad, inner

DIRICHLET_TAG = 1


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


def u_ex_ufl(t, x, mu, lammda):
    return ufl.as_vector(
        (
            ufl.exp(-t)
            * (
                ufl.sin(2.0 * ufl.pi * x[1]) * (-1.0 + ufl.cos(2.0 * ufl.pi * x[0]))
                + 1.0 / (mu + lammda) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
            ),
            ufl.exp(-t)
            * (
                ufl.sin(2.0 * ufl.pi * x[0]) * (1.0 - ufl.cos(2.0 * ufl.pi * x[1]))
                + 1.0 / (mu + lammda) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
            ),
        )
    )


def u_ex_numpy(t, x, mu, lammda):
    return np.vstack(
        (
            np.exp(-t)
            * (
                np.sin(2.0 * np.pi * x[1]) * (-1.0 + np.cos(2.0 * np.pi * x[0]))
                + 1.0 / (mu + lammda) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
            ),
            np.exp(-t)
            * (
                np.sin(2.0 * np.pi * x[0]) * (1.0 - np.cos(2.0 * np.pi * x[1]))
                + 1.0 / (mu + lammda) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
            ),
        )
    )


def p_ex_ufl(t, x):
    return ufl.exp(-t) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


def p_ex_np(t, x):
    return np.exp(-t) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def epsilon(u):
    return ufl.sym(ufl.grad(u))


def sigma(u, mu, lammda):
    return 2.0 * mu * epsilon(u) + lammda * div(u) * ufl.Identity(ufl.shape(u)[0])


def source_u(t, x, mu, lammda, alph):
    u = u_ex_ufl(t, x, mu, lammda)
    p = p_ex_ufl(t, x)
    return -ufl.div(sigma(u, mu, lammda)) + alph * grad(p)


def source_p(t, x, mu, lammda, kappa, alph, c_0):
    u = u_ex_ufl(t, x, mu, lammda)
    p = p_ex_ufl(t, x)
    return -alph * div(ufl.diff(u, t)) - c_0 * ufl.diff(p, t) - kappa * div(grad(p))


def noslip_boundary(x):
    return np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0) | np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)


def solve_with_petsc(A: PETSc.Mat, b: PETSc.Vec, comm: MPI.Intracomm, prefix: str) -> PETSc.Vec:
    attempts = [
        {"name": "mumps_lu", "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
        {"name": "superlu_dist_lu", "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "superlu_dist"},
        {"name": "gmres_hypre", "ksp_type": "gmres", "pc_type": "hypre"},
    ]
    last_error: str | None = None
    for i, a in enumerate(attempts):
        local_prefix = f"{prefix}_{i}_"
        opts = PETSc.Options()
        ksp = PETSc.KSP().create(comm)
        ksp.setOperators(A)
        ksp.setOptionsPrefix(local_prefix)
        opts[f"{local_prefix}ksp_type"] = a["ksp_type"]
        opts[f"{local_prefix}pc_type"] = a["pc_type"]
        if "pc_factor_mat_solver_type" in a:
            opts[f"{local_prefix}pc_factor_mat_solver_type"] = a["pc_factor_mat_solver_type"]
            opts[f"{local_prefix}mat_mumps_icntl_24"] = 1
            opts[f"{local_prefix}mat_mumps_icntl_25"] = 1
        if a["ksp_type"] == "gmres":
            opts[f"{local_prefix}ksp_rtol"] = 1.0e-10
            opts[f"{local_prefix}ksp_atol"] = 1.0e-12
            opts[f"{local_prefix}ksp_max_it"] = 5000
        ksp.setFromOptions()
        x = A.createVecRight()
        x.set(0.0)
        try:
            ksp.solve(b, x)
            if ksp.getConvergedReason() > 0:
                par_print(comm, f"Solver used: {a['name']}")
                return x
            last_error = f"{a['name']} failed with reason={ksp.getConvergedReason()}"
        except PETSc.Error as e:
            last_error = f"{a['name']} raised PETSc error {e.ierr}"
    raise RuntimeError(last_error or "PETSc solver failed")


def convergence_rate(err_old: float, err_new: float) -> float:
    return math.log(err_old / err_new) / math.log(2.0)


def main() -> None:
    comm = MPI.COMM_WORLD
    k = 1
    Tfina = 1.0
    dt = 1.0 / 16.0
    nsteps = int(Tfina / dt)
    n = 32

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

    U_el = basix.ufl.element("P", msh.basix_cell(), k, discontinuous=True, shape=(gdim,))
    Ubar_el = basix.ufl.element("P", facet_mesh.basix_cell(), k, discontinuous=True, shape=(gdim,))
    P_el = basix.ufl.element("P", msh.basix_cell(), k, discontinuous=True)
    Pbar_el = basix.ufl.element("P", facet_mesh.basix_cell(), k, discontinuous=True)

    U = fem.functionspace(msh, U_el)
    Ubar = fem.functionspace(facet_mesh, Ubar_el)
    P = fem.functionspace(msh, P_el)
    Pbar = fem.functionspace(facet_mesh, Pbar_el)
    W = ufl.MixedFunctionSpace(U, Ubar, P, Pbar)

    u_h, uhat_h, p_h, phat_h = ufl.TrialFunctions(W)
    v_h, vhat_h, w_h, what_h = ufl.TestFunctions(W)

    u_prev = Function(U)
    p_prev = Function(P)

    all_cell_boundary_facets = compute_cell_boundary_facets(msh)
    dx_c = ufl.Measure("dx", domain=msh)
    dx_f = ufl.Measure("dx", domain=facet_mesh)
    dirichlet_facets = mesh.locate_entities_boundary(msh, fdim, noslip_boundary)
    dirichlet_cell_boundary_facets = compute_boundary_facet_integration_entities(msh, dirichlet_facets)
    ds_c = ufl.Measure("ds", subdomain_data=[(0, all_cell_boundary_facets), (DIRICHLET_TAG, dirichlet_cell_boundary_facets)], domain=msh)

    mu = fem.Constant(msh, PETSc.ScalarType(1.0))
    lammda = fem.Constant(msh, PETSc.ScalarType(1.0e4))
    c_0 = fem.Constant(msh, PETSc.ScalarType(1.0))
    alph = fem.Constant(msh, PETSc.ScalarType(1.0))
    kappa = fem.Constant(msh, PETSc.ScalarType(1.0))
    tau_u0 = fem.Constant(msh, PETSc.ScalarType(16.0 * k * k))
    tau_p0 = fem.Constant(msh, PETSc.ScalarType(16.0 * k * k))
    h = ufl.CellDiameter(msh)
    n_vec = ufl.FacetNormal(msh)
    tau_u = tau_u0 / h
    tau_p = tau_p0 / h

    x = ufl.SpatialCoordinate(msh)
    t0 = 0.0
    u_prev.interpolate(lambda xx: u_ex_numpy(t0, xx, 1.0, 1.0e4))
    p_prev.interpolate(lambda xx: p_ex_np(t0, xx))

    # e_h(u_h, v_h)
    e_h = (
        inner(2.0 * mu * epsilon(u_h), epsilon(v_h)) * dx_c
        + inner(lammda * div(u_h), div(v_h)) * dx_c
        - inner(dot(sigma(u_h, mu, lammda), n_vec), v_h - vhat_h) * ds_c(0)
        - inner(dot(sigma(v_h, mu, lammda), n_vec), u_h - uhat_h) * ds_c(0)
        + inner(tau_u * (u_h - uhat_h), v_h - vhat_h) * ds_c(0)
    )

    # a_h(p_h, w_h)
    a_h = (
        kappa * inner(grad(p_h), grad(w_h)) * dx_c
        - kappa * inner(dot(grad(p_h), n_vec), w_h - what_h) * ds_c(0)
        - kappa * inner(dot(grad(w_h), n_vec), p_h - phat_h) * ds_c(0)
        + kappa * inner(tau_p * (p_h - phat_h), w_h - what_h) * ds_c(0)
    )

    # \tilde b_h(v_h, p_h)
    b_vp_tilde = (
        -alph * inner(p_h, div(v_h)) * dx_c
        + alph * inner(phat_h, dot(v_h - vhat_h, n_vec)) * ds_c(0)
    )

    # continuity equation terms with backward Euler
    c_up = (
        (alph / dt) * inner(div(u_h), w_h) * dx_c
        - (alph / dt) * inner(dot(u_h, n_vec), what_h) * ds_c(0)
    )
    c_pp = (c_0 / dt) * inner(p_h, w_h) * dx_c

    A_form = e_h + b_vp_tilde + c_up + c_pp + a_h

    # RHS from previous step + source
    t = fem.Constant(msh, PETSc.ScalarType(0.0))
    f_u = source_u(t, x, mu, lammda, alph)
    f_p = source_p(t, x, mu, lammda, kappa, alph, c_0)

    L_u = inner(f_u, v_h) * dx_c
    L_uhat = fem.Constant(facet_mesh, np.zeros(gdim, dtype=PETSc.ScalarType))
    L_ubar = inner(L_uhat, vhat_h) * dx_f
    L_p = inner(f_p, w_h) * dx_c + (c_0 / dt) * inner(p_prev, w_h) * dx_c + (alph / dt) * inner(div(u_prev), w_h) * dx_c
    L_phat = -(alph / dt) * inner(dot(u_prev, n_vec), what_h) * ds_c(0)

    A_blocked = fem.form(ufl.extract_blocks(A_form), entity_maps=entity_maps)
    L_blocked = [
        fem.form(L_u),
        fem.form(L_ubar, entity_maps=entity_maps),
        fem.form(L_p),
        fem.form(L_phat, entity_maps=entity_maps),
    ]

    # Dirichlet on uhat
    facet_mesh_boundary_facets = mesh_to_facet_mesh[dirichlet_facets]
    facet_mesh_boundary_facets = facet_mesh_boundary_facets[facet_mesh_boundary_facets >= 0]
    facet_mesh.topology.create_connectivity(fdim, fdim)
    udofs = fem.locate_dofs_topological(Ubar, fdim, facet_mesh_boundary_facets)
    u_bc_fun = Function(Ubar)
    u_bc_fun.interpolate(lambda xx: u_ex_numpy(0.0, xx, 1.0, 1.0e4))
    bcs = [dirichletbc(u_bc_fun, udofs)]

    e_u = e_p = e_div = 0.0
    for step in range(nsteps):
        t.value = PETSc.ScalarType((step + 1) * dt)
        u_bc_fun.interpolate(lambda xx: u_ex_numpy(float(t.value), xx, 1.0, 1.0e4))

        A = assemble_matrix_block(A_blocked, bcs=bcs)
        A.assemble()
        b = assemble_vector_block(L_blocked, A_blocked, bcs=bcs)
        xvec = solve_with_petsc(A, b, msh.comm, prefix=f"biot_hdg_{step}")

        # scatter
        off_u = U.dofmap.index_map.size_local * U.dofmap.index_map_bs
        off_uh = Ubar.dofmap.index_map.size_local * Ubar.dofmap.index_map_bs
        off_p = P.dofmap.index_map.size_local * P.dofmap.index_map_bs
        off_ph = Pbar.dofmap.index_map.size_local * Pbar.dofmap.index_map_bs
        uh = Function(U)
        uhat = Function(Ubar)
        ph = Function(P)
        phat = Function(Pbar)
        sol = xvec.array_r
        pos = 0
        uh.x.array[:off_u] = sol[pos : pos + off_u]
        pos += off_u
        uhat.x.array[:off_uh] = sol[pos : pos + off_uh]
        pos += off_uh
        ph.x.array[:off_p] = sol[pos : pos + off_p]
        pos += off_p
        phat.x.array[:off_ph] = sol[pos : pos + off_ph]
        uh.x.scatter_forward()
        uhat.x.scatter_forward()
        ph.x.scatter_forward()
        phat.x.scatter_forward()

        u_prev.x.array[:] = uh.x.array
        p_prev.x.array[:] = ph.x.array
        u_prev.x.scatter_forward()
        p_prev.x.scatter_forward()

        u_ex = u_ex_ufl(t, x, mu, lammda)
        p_ex = p_ex_ufl(t, x)
        e_u = float(norm_L2(msh.comm, uh - u_ex))
        e_p = float(norm_L2(msh.comm, ph - p_ex))
        e_div = float(norm_L2(msh.comm, div(uh)))

    par_print(comm, f"Final t={float(t.value):.4f}: ||u-uh||={e_u:.6e}, ||p-ph||={e_p:.6e}, ||div u||={e_div:.6e}")


if __name__ == "__main__":
    main()
