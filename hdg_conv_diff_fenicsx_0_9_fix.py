"""FEniCSx 0.9.0-compatible HDG convection-diffusion demo.

This is a runnable adaptation of the mixed-domain HDG example for FEniCSx 0.9.0.
The main API differences from newer versions are:

1. ``mesh.create_submesh`` returns a NumPy array ``submesh_entity -> parent_entity``.
2. Mixed-domain forms must be compiled with ``fem.form(..., entity_maps=...)`` where
   ``entity_maps`` is a dict mapping the non-integration mesh to an array indexed by
   integration-domain entities.
3. The blocked problem is assembled with ``assemble_matrix_block`` /
   ``assemble_vector_block`` and solved manually with PETSc.
"""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import ufl
from mpi4py import MPI
from ufl import div, dot, grad, inner

if importlib.util.find_spec("petsc4py") is None:
    print("This script requires petsc4py.")
    raise SystemExit(0)

from petsc4py import PETSc

import dolfinx
from dolfinx import fem, mesh
from dolfinx.cpp.mesh import cell_num_entities
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block

if not dolfinx.has_petsc:
    print("This script requires DOLFINx to be compiled with PETSc enabled.")
    raise SystemExit(0)



def par_print(comm: MPI.Intracomm, msg: str) -> None:
    if comm.rank == 0:
        print(msg)
        sys.stdout.flush()



def norm_L2(comm: MPI.Intracomm, expr, measure=ufl.dx) -> np.floating:
    return np.sqrt(
        comm.allreduce(fem.assemble_scalar(fem.form(inner(expr, expr) * measure)), op=MPI.SUM)
    )



def compute_cell_boundary_facets(msh: mesh.Mesh) -> np.ndarray:
    """Return integration entities for all cell boundaries in ``msh``."""
    tdim = msh.topology.dim
    fdim = tdim - 1
    n_f = cell_num_entities(msh.topology.cell_type, fdim)
    n_c = msh.topology.index_map(tdim).size_local
    return np.vstack((np.repeat(np.arange(n_c), n_f), np.tile(np.arange(n_f), n_c))).T.flatten()



def u_e(x):
    """Exact solution."""
    module = ufl if isinstance(x, ufl.SpatialCoordinate) else np
    return module.sin(3.0 * module.pi * x[0]) * module.cos(2.0 * module.pi * x[1])


comm = MPI.COMM_WORLD
dtype = PETSc.ScalarType

# Create mesh
n = 16
msh = mesh.create_unit_square(comm, n, n)
tdim = msh.topology.dim
fdim = tdim - 1

# Create a facet submesh for the trace unknown.
msh.topology.create_entities(fdim)
facet_imap = msh.topology.index_map(fdim)
num_facets = facet_imap.size_local + facet_imap.num_ghosts
facets = np.arange(num_facets, dtype=np.int32)
facet_mesh, facet_mesh_to_mesh = mesh.create_submesh(msh, fdim, facets)[:2]

# Build the map required by mixed-domain form compilation:
# parent-mesh facet -> facet-submesh cell
mesh_to_facet_mesh = np.full(num_facets, -1, dtype=np.int32)
mesh_to_facet_mesh[facet_mesh_to_mesh] = np.arange(len(facet_mesh_to_mesh), dtype=np.int32)
entity_maps = {facet_mesh: mesh_to_facet_mesh}

# Function spaces
k = 3
V = fem.functionspace(msh, ("Discontinuous Lagrange", k))
Vbar = fem.functionspace(facet_mesh, ("Discontinuous Lagrange", k))
W = ufl.MixedFunctionSpace(V, Vbar)

u, ubar = ufl.TrialFunctions(W)
v, vbar = ufl.TestFunctions(W)

# Integration measures
dx_c = ufl.Measure("dx", domain=msh)
cell_boundary_facets = compute_cell_boundary_facets(msh)
cell_boundaries = 1
ds_c = ufl.Measure("ds", subdomain_data=[(cell_boundaries, cell_boundary_facets)], domain=msh)
dx_f = ufl.Measure("dx", domain=facet_mesh)

# PDE coefficients
h = ufl.CellDiameter(msh)
n = ufl.FacetNormal(msh)
kappa = fem.Constant(msh, dtype(1e-3))
gamma = 16.0 * k**2 / h
x = ufl.SpatialCoordinate(msh)
w = ufl.as_vector(
    (
        ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]),
    )
)
lmbda = ufl.conditional(ufl.gt(dot(w, n), 0), dtype(0.0), dtype(1.0))

# Bilinear form
# Diffusion
# Keep the same HDG structure as the original mixed-domain example.
a = (
    inner(kappa * grad(u), grad(v)) * dx_c
    - inner(kappa * (u - ubar), dot(grad(v), n)) * ds_c(cell_boundaries)
    - inner(dot(grad(u), n), kappa * (v - vbar)) * ds_c(cell_boundaries)
    + gamma * inner(kappa * (u - ubar), v - vbar) * ds_c(cell_boundaries)
)
# Advection
# Use upwind numerical flux through the trace variable.
a += -inner(w * u, grad(v)) * dx_c
a += inner(dot(w * (u - lmbda * (u - ubar)), n), v - vbar) * ds_c(cell_boundaries)

# Linear form
f = dot(w, grad(u_e(x))) - div(kappa * grad(u_e(x)))
L = inner(f, v) * dx_c + inner(fem.Constant(facet_mesh, dtype(0.0)), vbar) * dx_f

# Compile blocked forms for FEniCSx 0.9.0
a_blocked = fem.form(ufl.extract_blocks(a), entity_maps=entity_maps)
L_blocked = fem.form(ufl.extract_blocks(L))

# Dirichlet boundary conditions for the trace variable
msh_boundary_facets = mesh.exterior_facet_indices(msh.topology)
facet_mesh_boundary_facets = mesh_to_facet_mesh[msh_boundary_facets]
facet_mesh_boundary_facets = facet_mesh_boundary_facets[facet_mesh_boundary_facets >= 0]
facet_mesh.topology.create_connectivity(fdim, fdim)
dofs = fem.locate_dofs_topological(Vbar, fdim, facet_mesh_boundary_facets)
u_bc = fem.Function(Vbar)
u_bc.interpolate(u_e)
bc = fem.dirichletbc(u_bc, dofs)

# Assemble and solve the blocked linear system.
A = assemble_matrix_block(a_blocked, bcs=[bc])
A.assemble()
b = assemble_vector_block(L_blocked, a_blocked, bcs=[bc])

ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("superlu_dist")

x_vec = A.createVecRight()
try:
    ksp.solve(b, x_vec)
except PETSc.Error as e:  # type: ignore[attr-defined]
    if e.ierr == 92:
        par_print(comm, "The required PETSc LU solver/preconditioner is not available.")
        raise SystemExit(0)
    raise

# Scatter the monolithic block solution back into Functions.
u = fem.Function(V)
ubar = fem.Function(Vbar)
offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
u.x.array[:offset] = x_vec.array_r[:offset]
ubar.x.array[: len(x_vec.array_r) - offset] = x_vec.array_r[offset:]
u.x.scatter_forward()
ubar.x.scatter_forward()

# Optional output
try:
    from dolfinx.io import VTXWriter

    with VTXWriter(msh.comm, "u.bp", u, "bp4") as f_out:
        f_out.write(0.0)
    with VTXWriter(msh.comm, "ubar.bp", ubar, "bp4") as f_out:
        f_out.write(0.0)
except ImportError:
    par_print(comm, "ADIOS2 not available; skipping VTX output.")

# Errors
x = ufl.SpatialCoordinate(msh)
x_bar = ufl.SpatialCoordinate(facet_mesh)
e_u = norm_L2(msh.comm, u - u_e(x))
e_ubar = norm_L2(msh.comm, ubar - u_e(x_bar), measure=dx_f)
par_print(comm, f"e_u = {e_u}")
par_print(comm, f"e_ubar = {e_ubar}")
