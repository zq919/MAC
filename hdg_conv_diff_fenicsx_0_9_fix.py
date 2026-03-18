"""FEniCSx 0.9.0-compatible HDG convection-diffusion demo.

This is adapted from the mixed-domain demo and fixes the boundary-facet
submesh mapping for versions where ``mesh.create_submesh`` returns a NumPy
array instead of an ``EntityMap`` object.
"""

from dolfinx import fem, io, mesh
from dolfinx.cpp.mesh import cell_num_entities
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc
from ufl import div, dot, grad, inner
import numpy as np
import ufl

from utils import compute_cell_boundary_int_entities, norm_L2


def u_e(x):
    "Function to represent the exact solution."
    module = ufl if isinstance(x, ufl.SpatialCoordinate) else np
    return module.sin(3.0 * module.pi * x[0]) * module.cos(2.0 * module.pi * x[1])


def boundary(x):
    "Mark the domain boundary."
    return (
        np.isclose(x[0], 0.0)
        | np.isclose(x[0], 1.0)
        | np.isclose(x[1], 0.0)
        | np.isclose(x[1], 1.0)
    )


def parent_to_sub_entities(entity_map, parent_entities, num_parent_entities):
    """Map parent-mesh entities to submesh entities.

    FEniCSx 0.9.0 returns ``entity_map`` as a NumPy array that maps
    ``submesh_entity -> parent_entity``. Newer releases may return an
    ``EntityMap`` object with ``sub_topology_to_topology(..., inverse=True)``.
    This helper supports both APIs.
    """
    if hasattr(entity_map, "sub_topology_to_topology"):
        return entity_map.sub_topology_to_topology(parent_entities, inverse=True)

    inverse = np.full(num_parent_entities, -1, dtype=np.int32)
    inverse[np.asarray(entity_map, dtype=np.int32)] = np.arange(len(entity_map), dtype=np.int32)
    return inverse[parent_entities]


comm = MPI.COMM_WORLD
n = 16
msh = mesh.create_unit_square(comm, n, n)

tdim = msh.topology.dim
fdim = tdim - 1
num_cell_facets = cell_num_entities(msh.topology.cell_type, fdim)
msh.topology.create_entities(fdim)
facet_imap = msh.topology.index_map(fdim)
num_facets = facet_imap.size_local + facet_imap.num_ghosts
facets = np.arange(num_facets, dtype=np.int32)
facet_mesh, facet_mesh_emap = mesh.create_submesh(msh, fdim, facets)[0:2]

k = 3
V = fem.functionspace(msh, ("Discontinuous Lagrange", k))
Vbar = fem.functionspace(facet_mesh, ("Discontinuous Lagrange", k))
W = ufl.MixedFunctionSpace(V, Vbar)

u, ubar = ufl.TrialFunctions(W)
v, vbar = ufl.TestFunctions(W)

cell_boundary_facets = compute_cell_boundary_int_entities(msh)
dx_c = ufl.Measure("dx", domain=msh)
cell_boundaries = 0
ds_c = ufl.Measure("ds", subdomain_data=[(cell_boundaries, cell_boundary_facets)], domain=msh)
dx_f = ufl.Measure("dx", domain=facet_mesh)

entity_maps = [facet_mesh_emap]

h = ufl.CellDiameter(msh)
n = ufl.FacetNormal(msh)
kappa = fem.Constant(msh, PETSc.ScalarType(1e-3))
gamma = 16.0 * k**2 / h

a = (
    inner(kappa * grad(u), grad(v)) * dx_c
    - inner(kappa * (u - ubar), dot(grad(v), n)) * ds_c(cell_boundaries)
    - inner(dot(grad(u), n), kappa * (v - vbar)) * ds_c(cell_boundaries)
    + gamma * inner(kappa * (u - ubar), v - vbar) * ds_c(cell_boundaries)
)

x = ufl.SpatialCoordinate(msh)
w = ufl.as_vector(
    (
        ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]),
    )
)
lmbda = ufl.conditional(ufl.gt(dot(w, n), 0), 0, 1)
a += -inner(w * u, grad(v)) * dx_c + inner(
    dot(w * (u - lmbda * (u - ubar)), n), v - vbar
) * ds_c(cell_boundaries)

f = dot(w, grad(u_e(x))) - div(kappa * grad(u_e(x)))
L = inner(f, v) * dx_c + inner(fem.Constant(facet_mesh, 0.0), vbar) * dx_f

msh_boundary_facets = mesh.locate_entities_boundary(msh, fdim, boundary)
facet_mesh_boundary_facets = parent_to_sub_entities(
    facet_mesh_emap, msh_boundary_facets, num_facets
)
facet_mesh_boundary_facets = facet_mesh_boundary_facets[facet_mesh_boundary_facets >= 0]

facet_mesh.topology.create_connectivity(fdim, fdim)
dofs = fem.locate_dofs_topological(Vbar, fdim, facet_mesh_boundary_facets)
u_bc = fem.Function(Vbar)
u_bc.interpolate(u_e)
bc = fem.dirichletbc(u_bc, dofs)
bcs = [bc]

u, ubar = fem.Function(V), fem.Function(Vbar)
petsc_opts = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "superlu_dist",
}
problem = LinearProblem(
    ufl.extract_blocks(a),
    ufl.extract_blocks(L),
    u=[u, ubar],
    bcs=bcs,
    kind="mpi",
    petsc_options_prefix="hdg_conv_diff_",
    petsc_options=petsc_opts,
    entity_maps=entity_maps,
)
problem.solve()

with io.VTXWriter(msh.comm, "u.bp", u) as f:
    f.write(0.0)
with io.VTXWriter(msh.comm, "ubar.bp", ubar) as f:
    f.write(0.0)

x = ufl.SpatialCoordinate(msh)
e_L2 = norm_L2(msh.comm, u - u_e(x))

if comm.rank == 0:
    print(f"e_L2 = {e_L2}")
