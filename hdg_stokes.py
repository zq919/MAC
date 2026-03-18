from __future__ import annotations

import ctypes
import json
import sys
from pathlib import Path

import cffi
import numpy as np
import ufl
from basix.ufl import element
from dolfinx import fem, io, jit, la, mesh
from dolfinx.common import Timer, TimingType, list_timings
from dolfinx.fem import (
    Constant,
    Form,
    Function,
    IntegralType,
    assemble_scalar,
    dirichletbc,
    form,
    form_cpp_class,
    functionspace,
    locate_dofs_geometrical,
    locate_dofs_topological,
)
from mpi4py import MPI
from petsc4py import PETSc

from assemble import assemble_matrix_block, assemble_vector_block, pack_coefficients


SCALAR_DTYPE = np.dtype(PETSc.ScalarType)
REAL_DTYPE = np.dtype(PETSc.RealType)
if SCALAR_DTYPE != np.dtype(np.float64) or REAL_DTYPE != np.dtype(np.float64):
    raise RuntimeError("This script currently targets real-valued float64 DOLFINx 0.9.0 builds.")

SCALAR_CTYPE = ctypes.c_double
REAL_CTYPE = ctypes.c_double
INT_CTYPE = ctypes.c_int32
PERM_CTYPE = ctypes.c_uint8
KERNEL_SIGNATURE = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(SCALAR_CTYPE),
    ctypes.POINTER(SCALAR_CTYPE),
    ctypes.POINTER(SCALAR_CTYPE),
    ctypes.POINTER(REAL_CTYPE),
    ctypes.POINTER(INT_CTYPE),
    ctypes.POINTER(PERM_CTYPE),
)

ffi = cffi.FFI()
_CALLBACKS: list[ctypes._CFuncPtr] = []



def par_print(message: str, comm: MPI.Intracomm) -> None:
    if comm.rank == 0:
        print(message)
        sys.stdout.flush()


class LoggedTimer:
    def __init__(self, name: str, comm: MPI.Intracomm):
        self._timer = Timer(name)
        par_print(name, comm)

    def stop(self) -> float:
        return self._timer.stop()



def reorder_mesh(_msh) -> None:
    """Compatibility placeholder for the old example."""



def _measure(domain):
    return ufl.dx(domain=domain)



def norm_L2(comm: MPI.Intracomm, expr) -> float:
    domain = expr.ufl_domain()
    local_value = assemble_scalar(form(ufl.inner(expr, expr) * _measure(domain), dtype=PETSc.ScalarType))
    return float(np.sqrt(comm.allreduce(local_value, op=MPI.SUM)))



def domain_average(domain, expr) -> float:
    numerator = assemble_scalar(form(expr * _measure(domain), dtype=PETSc.ScalarType))
    denominator = assemble_scalar(form(1 * _measure(domain), dtype=PETSc.ScalarType))
    num = domain.comm.allreduce(numerator, op=MPI.SUM)
    den = domain.comm.allreduce(denominator, op=MPI.SUM)
    return float(num / den)



def normal_jump_error(msh, uh) -> float:
    facet_normal = ufl.FacetNormal(msh)
    jump_form = form(
        ufl.inner(ufl.jump(uh, facet_normal), ufl.jump(uh, facet_normal)) * ufl.dS(domain=msh),
        dtype=PETSc.ScalarType,
    )
    local_value = assemble_scalar(jump_form)
    return float(np.sqrt(msh.comm.allreduce(local_value, op=MPI.SUM)))



def build_kernel_getter(dtype_name: str):
    def _kernel(ufcx_form, integral_type: IntegralType, local_index: int = 0):
        offsets = ufcx_form.form_integral_offsets
        integral_index = getattr(integral_type, "value", integral_type)
        start = offsets[integral_index]
        return getattr(ufcx_form.form_integrals[start + local_index], f"tabulate_tensor_{dtype_name}")

    return _kernel



def build_custom_form(spaces, integrals, coeffs, constants, integration_mesh, entity_map=None):
    formtype = form_cpp_class(PETSc.ScalarType)  # type: ignore[arg-type]
    entity_maps = {} if entity_map is None else {entity_map[0]: entity_map[1]}
    return Form(
        formtype(
            [space._cpp_object for space in spaces],
            integrals,
            [coefficient._cpp_object for coefficient in coeffs],
            [constant._cpp_object for constant in constants],
            False,
            entity_maps,
            integration_mesh,
        )
    )



def _as_array(ptr, shape, *, ctype):
    size = int(np.prod(shape))
    return np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctype)), shape=(size,)).reshape(shape)



def _register_callback(pyfun):
    callback = KERNEL_SIGNATURE(pyfun)
    _CALLBACKS.append(callback)
    return callback



def boundary(x, tdim: int):
    lr = np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)
    tb = np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)
    if tdim == 2:
        return lr | tb
    fb = np.isclose(x[2], 0.0) | np.isclose(x[2], 1.0)
    return lr | tb | fb



def main() -> None:
    total_timer = Timer("TOTAL")
    comm = MPI.COMM_WORLD
    timings: dict[str, float] = {}

    mesh_resolution = 8
    viscosity = 1.0
    polynomial_degree = 2
    num_time_steps = 1
    use_direct_solver = False

    timer = LoggedTimer(f"Create mesh (n = {mesh_resolution})", comm)
    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, ghost_mode=mesh.GhostMode.none)
    timings["create_mesh"] = timer.stop()

    tdim = msh.topology.dim
    fdim = tdim - 1

    timer = LoggedTimer("Reorder mesh", comm)
    reorder_mesh(msh)
    timings["reorder_mesh"] = timer.stop()

    timer = LoggedTimer("Create facet mesh", comm)
    msh.topology.create_entities(fdim)
    facet_imap = msh.topology.index_map(fdim)
    num_facets = facet_imap.size_local + facet_imap.num_ghosts
    facets = np.arange(num_facets, dtype=np.int32)
    facet_mesh, entity_map, _, _ = mesh.create_submesh(msh, fdim, facets)
    timings["create_facet_mesh"] = timer.stop()

    timer = LoggedTimer("Create function spaces", comm)
    V = functionspace(
        msh,
        element("Discontinuous Lagrange", msh.basix_cell(), polynomial_degree, shape=(tdim,), dtype=PETSc.RealType),
    )
    Q = functionspace(
        msh,
        element("Discontinuous Lagrange", msh.basix_cell(), polynomial_degree - 1, dtype=PETSc.RealType),
    )
    Vbar = functionspace(
        facet_mesh,
        element(
            "Discontinuous Lagrange",
            facet_mesh.basix_cell(),
            polynomial_degree,
            shape=(tdim,),
            dtype=PETSc.RealType,
        ),
    )
    Qbar = functionspace(
        facet_mesh,
        element("Discontinuous Lagrange", facet_mesh.basix_cell(), polynomial_degree, dtype=PETSc.RealType),
    )
    timings["create_function_spaces"] = timer.stop()

    timer = LoggedTimer("Define problem", comm)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    q = ufl.TestFunction(Q)
    ubar = ufl.TrialFunction(Vbar)
    vbar = ufl.TestFunction(Vbar)
    pbar = ufl.TrialFunction(Qbar)
    qbar = ufl.TestFunction(Qbar)

    V_ele_space_dim = V.element.space_dimension
    Vbar_ele_space_dim = Vbar.element.space_dimension
    Q_ele_space_dim = Q.element.space_dimension
    Qbar_ele_space_dim = Qbar.element.space_dimension
    num_cell_facets = msh.ufl_cell().num_facets()
    num_dofs_g = msh.geometry.dofmap.shape[1]

    h = ufl.CellDiameter(msh)
    facet_normal = ufl.FacetNormal(msh)
    gamma = polynomial_degree**2 / h
    gamma *= 6.0 if tdim == 2 else 10.0

    def u_e(x, module=np):
        if tdim == 2:
            ux = module.sin(module.pi * x[0]) * module.cos(module.pi * x[1])
            uy = -module.sin(module.pi * x[1]) * module.cos(module.pi * x[0])
            return np.stack((ux, uy)) if module is np else ufl.as_vector((ux, uy))
        ux = module.sin(module.pi * x[0]) * module.cos(module.pi * x[1]) - module.sin(module.pi * x[0]) * module.cos(module.pi * x[2])
        uy = module.sin(module.pi * x[1]) * module.cos(module.pi * x[2]) - module.sin(module.pi * x[1]) * module.cos(module.pi * x[0])
        uz = module.sin(module.pi * x[2]) * module.cos(module.pi * x[0]) - module.sin(module.pi * x[2]) * module.cos(module.pi * x[1])
        return np.stack((ux, uy, uz)) if module is np else ufl.as_vector((ux, uy, uz))

    def p_e(x, module=np):
        if tdim == 2:
            return module.sin(module.pi * x[0]) * module.sin(module.pi * x[1])
        return module.sin(module.pi * x[0]) * module.sin(module.pi * x[1]) * module.sin(module.pi * x[2])

    dx_c = ufl.Measure("dx", domain=msh)
    ds_c = ufl.Measure("ds", domain=msh)
    x_coord = ufl.SpatialCoordinate(msh)
    forcing = -viscosity * ufl.div(ufl.grad(u_e(x_coord, ufl))) + ufl.grad(p_e(x_coord, ufl))

    u_n = Function(V)
    delta_t = Constant(msh, PETSc.ScalarType(1e16))
    nu = Constant(msh, PETSc.ScalarType(viscosity))

    a_00 = (
        ufl.inner(u / delta_t, v) * dx_c
        + nu
        * (
            ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_c
            + gamma * ufl.inner(u, v) * ds_c
            - (ufl.inner(u, ufl.dot(ufl.grad(v), facet_normal)) + ufl.inner(v, ufl.dot(ufl.grad(u), facet_normal))) * ds_c
        )
    )
    a_10 = -ufl.inner(q, ufl.div(u)) * dx_c
    a_20 = nu * (ufl.inner(vbar, ufl.dot(ufl.grad(u), facet_normal)) * ds_c - gamma * ufl.inner(vbar, u) * ds_c)
    a_30 = ufl.inner(ufl.dot(u, facet_normal), qbar) * ds_c
    a_22 = nu * gamma * ufl.inner(ubar, vbar) * ds_c
    p_11 = h / nu * ufl.inner(pbar, qbar) * ds_c
    L_0 = ufl.inner(forcing + u_n / delta_t, v) * dx_c
    timings["define_problem"] = timer.stop()

    timer = LoggedTimer("Create inverse entity map", comm)
    inv_entity_map = np.full(num_facets, -1, dtype=np.int32)
    for local_index, parent_facet in enumerate(entity_map):
        inv_entity_map[parent_facet] = local_index
    timings["create_inv_ent_map"] = timer.stop()

    timer = LoggedTimer("JIT kernels", comm)
    scalar_name = SCALAR_DTYPE.name
    get_kernel = build_kernel_getter(scalar_name)

    ufcx_form_00, _, _ = jit.ffcx_jit(msh.comm, a_00, form_compiler_options={"scalar_type": PETSc.ScalarType})
    kernel_00_cell = get_kernel(ufcx_form_00, IntegralType.cell)
    kernel_00_facet = get_kernel(ufcx_form_00, IntegralType.exterior_facet)

    ufcx_form_10, _, _ = jit.ffcx_jit(msh.comm, a_10, form_compiler_options={"scalar_type": PETSc.ScalarType})
    kernel_10 = get_kernel(ufcx_form_10, IntegralType.cell)

    ufcx_form_20, _, _ = jit.ffcx_jit(msh.comm, a_20, form_compiler_options={"scalar_type": PETSc.ScalarType})
    kernel_20 = get_kernel(ufcx_form_20, IntegralType.exterior_facet)

    ufcx_form_30, _, _ = jit.ffcx_jit(msh.comm, a_30, form_compiler_options={"scalar_type": PETSc.ScalarType})
    kernel_30 = get_kernel(ufcx_form_30, IntegralType.exterior_facet)

    ufcx_form_22, _, _ = jit.ffcx_jit(msh.comm, a_22, form_compiler_options={"scalar_type": PETSc.ScalarType})
    kernel_22 = get_kernel(ufcx_form_22, IntegralType.exterior_facet)

    ufcx_form_p11, _, _ = jit.ffcx_jit(msh.comm, p_11, form_compiler_options={"scalar_type": PETSc.ScalarType})
    kernel_p11 = get_kernel(ufcx_form_p11, IntegralType.exterior_facet)

    ufcx_form_0, _, _ = jit.ffcx_jit(msh.comm, L_0, form_compiler_options={"scalar_type": PETSc.ScalarType})
    kernel_0 = get_kernel(ufcx_form_0, IntegralType.cell)

    null64 = np.zeros(0, dtype=REAL_DTYPE)
    null32 = np.zeros(0, dtype=np.int32)
    null8 = np.zeros(0, dtype=np.uint8)
    constants_size = 2

    def compute_mats(coords, constants):
        visc = np.array([constants[1]], dtype=SCALAR_DTYPE)
        A_00 = np.zeros((V_ele_space_dim, V_ele_space_dim), dtype=SCALAR_DTYPE)
        A_10 = np.zeros((Q_ele_space_dim, V_ele_space_dim), dtype=SCALAR_DTYPE)
        A_20 = np.zeros((num_cell_facets * Vbar_ele_space_dim, V_ele_space_dim), dtype=SCALAR_DTYPE)
        A_20_f = np.zeros((Vbar_ele_space_dim, V_ele_space_dim), dtype=SCALAR_DTYPE)
        A_30 = np.zeros((num_cell_facets * Qbar_ele_space_dim, V_ele_space_dim), dtype=SCALAR_DTYPE)
        A_30_f = np.zeros((Qbar_ele_space_dim, V_ele_space_dim), dtype=SCALAR_DTYPE)
        A_22 = np.zeros((num_cell_facets * Vbar_ele_space_dim, num_cell_facets * Vbar_ele_space_dim), dtype=SCALAR_DTYPE)
        A_22_f = np.zeros((Vbar_ele_space_dim, Vbar_ele_space_dim), dtype=SCALAR_DTYPE)

        kernel_00_cell(ffi.from_buffer(A_00), ffi.from_buffer(null64), ffi.from_buffer(constants), ffi.from_buffer(coords), ffi.from_buffer(null32), ffi.from_buffer(null8))
        kernel_10(ffi.from_buffer(A_10), ffi.from_buffer(null64), ffi.from_buffer(null64), ffi.from_buffer(coords), ffi.from_buffer(null32), ffi.from_buffer(null8))

        entity_local_index = np.zeros(1, dtype=np.int32)
        for local_facet in range(num_cell_facets):
            entity_local_index[0] = local_facet
            A_20_f.fill(0.0)
            A_30_f.fill(0.0)
            A_22_f.fill(0.0)

            kernel_00_facet(ffi.from_buffer(A_00), ffi.from_buffer(null64), ffi.from_buffer(constants), ffi.from_buffer(coords), ffi.from_buffer(entity_local_index), ffi.from_buffer(null8))
            kernel_20(ffi.from_buffer(A_20_f), ffi.from_buffer(null64), ffi.from_buffer(visc), ffi.from_buffer(coords), ffi.from_buffer(entity_local_index), ffi.from_buffer(null8))
            kernel_30(ffi.from_buffer(A_30_f), ffi.from_buffer(null64), ffi.from_buffer(null64), ffi.from_buffer(coords), ffi.from_buffer(entity_local_index), ffi.from_buffer(null8))
            kernel_22(ffi.from_buffer(A_22_f), ffi.from_buffer(null64), ffi.from_buffer(visc), ffi.from_buffer(coords), ffi.from_buffer(entity_local_index), ffi.from_buffer(null8))

            row0 = local_facet * Vbar_ele_space_dim
            row1 = row0 + Vbar_ele_space_dim
            A_20[row0:row1, :] = A_20_f
            prow0 = local_facet * Qbar_ele_space_dim
            prow1 = prow0 + Qbar_ele_space_dim
            A_30[prow0:prow1, :] = A_30_f
            A_22[row0:row1, row0:row1] = A_22_f
        return A_00, A_10, A_20, A_30, A_22

    def compute_tilde_mats(A_00, A_10, A_20, A_30):
        A_tilde = np.zeros((V_ele_space_dim + Q_ele_space_dim, V_ele_space_dim + Q_ele_space_dim), dtype=SCALAR_DTYPE)
        A_tilde[:V_ele_space_dim, :V_ele_space_dim] = A_00
        A_tilde[V_ele_space_dim:, :V_ele_space_dim] = A_10
        A_tilde[:V_ele_space_dim, V_ele_space_dim:] = A_10.T

        B_tilde = np.zeros((num_cell_facets * Vbar_ele_space_dim, V_ele_space_dim + Q_ele_space_dim), dtype=SCALAR_DTYPE)
        B_tilde[:, :V_ele_space_dim] = A_20

        C_tilde = np.zeros((num_cell_facets * Qbar_ele_space_dim, V_ele_space_dim + Q_ele_space_dim), dtype=SCALAR_DTYPE)
        C_tilde[:, :V_ele_space_dim] = A_30
        return A_tilde, B_tilde, C_tilde

    def compute_L_tilde(coords, constants, coeffs):
        b_0 = np.zeros(V_ele_space_dim, dtype=SCALAR_DTYPE)
        kernel_0(ffi.from_buffer(b_0), ffi.from_buffer(coeffs), ffi.from_buffer(constants), ffi.from_buffer(coords), ffi.from_buffer(null32), ffi.from_buffer(null8))
        L_tilde = np.zeros(V_ele_space_dim + Q_ele_space_dim, dtype=SCALAR_DTYPE)
        L_tilde[:V_ele_space_dim] = b_0
        return L_tilde

    def tabulate_tensor_a00(A_ptr, _w_ptr, c_ptr, coords_ptr, _entity_ptr, _perm_ptr):
        A_local = _as_array(A_ptr, (num_cell_facets * Vbar_ele_space_dim, num_cell_facets * Vbar_ele_space_dim), ctype=SCALAR_CTYPE)
        coords = _as_array(coords_ptr, (num_dofs_g, 3), ctype=REAL_CTYPE)
        constants = _as_array(c_ptr, (constants_size,), ctype=SCALAR_CTYPE)
        A_00, A_10, A_20, A_30, A_22 = compute_mats(coords, constants)
        A_tilde, B_tilde, _ = compute_tilde_mats(A_00, A_10, A_20, A_30)
        A_local[:, :] = A_22 - B_tilde @ np.linalg.solve(A_tilde, B_tilde.T)

    def tabulate_tensor_a01(A_ptr, _w_ptr, c_ptr, coords_ptr, _entity_ptr, _perm_ptr):
        A_local = _as_array(A_ptr, (num_cell_facets * Vbar_ele_space_dim, num_cell_facets * Qbar_ele_space_dim), ctype=SCALAR_CTYPE)
        coords = _as_array(coords_ptr, (num_dofs_g, 3), ctype=REAL_CTYPE)
        constants = _as_array(c_ptr, (constants_size,), ctype=SCALAR_CTYPE)
        A_00, A_10, A_20, A_30, _ = compute_mats(coords, constants)
        A_tilde, B_tilde, C_tilde = compute_tilde_mats(A_00, A_10, A_20, A_30)
        A_local[:, :] = -B_tilde @ np.linalg.solve(A_tilde, C_tilde.T)

    def tabulate_tensor_a10(A_ptr, _w_ptr, c_ptr, coords_ptr, _entity_ptr, _perm_ptr):
        A_local = _as_array(A_ptr, (num_cell_facets * Qbar_ele_space_dim, num_cell_facets * Vbar_ele_space_dim), ctype=SCALAR_CTYPE)
        coords = _as_array(coords_ptr, (num_dofs_g, 3), ctype=REAL_CTYPE)
        constants = _as_array(c_ptr, (constants_size,), ctype=SCALAR_CTYPE)
        A_00, A_10, A_20, A_30, _ = compute_mats(coords, constants)
        A_tilde, B_tilde, C_tilde = compute_tilde_mats(A_00, A_10, A_20, A_30)
        A_local[:, :] = -C_tilde @ np.linalg.solve(A_tilde, B_tilde.T)

    def tabulate_tensor_a11(A_ptr, _w_ptr, c_ptr, coords_ptr, _entity_ptr, _perm_ptr):
        A_local = _as_array(A_ptr, (num_cell_facets * Qbar_ele_space_dim, num_cell_facets * Qbar_ele_space_dim), ctype=SCALAR_CTYPE)
        coords = _as_array(coords_ptr, (num_dofs_g, 3), ctype=REAL_CTYPE)
        constants = _as_array(c_ptr, (constants_size,), ctype=SCALAR_CTYPE)
        A_00, A_10, A_20, A_30, _ = compute_mats(coords, constants)
        A_tilde, _, C_tilde = compute_tilde_mats(A_00, A_10, A_20, A_30)
        A_local[:, :] = -C_tilde @ np.linalg.solve(A_tilde, C_tilde.T)

    def tabulate_tensor_p00(P_ptr, _w_ptr, c_ptr, coords_ptr, _entity_ptr, _perm_ptr):
        P_local = _as_array(P_ptr, (num_cell_facets * Vbar_ele_space_dim, num_cell_facets * Vbar_ele_space_dim), ctype=SCALAR_CTYPE)
        coords = _as_array(coords_ptr, (num_dofs_g, 3), ctype=REAL_CTYPE)
        constants = _as_array(c_ptr, (constants_size,), ctype=SCALAR_CTYPE)
        A_00, _, A_20, _, A_22 = compute_mats(coords, constants)
        P_local[:, :] = A_22 - A_20 @ np.linalg.solve(A_00, A_20.T)

    def tabulate_tensor_p11(P_ptr, _w_ptr, c_ptr, coords_ptr, _entity_ptr, _perm_ptr):
        P_local = _as_array(P_ptr, (num_cell_facets * Qbar_ele_space_dim, num_cell_facets * Qbar_ele_space_dim), ctype=SCALAR_CTYPE)
        coords = _as_array(coords_ptr, (num_dofs_g, 3), ctype=REAL_CTYPE)
        constants = _as_array(c_ptr, (1,), ctype=SCALAR_CTYPE)
        P_local.fill(0.0)
        P_11_f = np.zeros((Qbar_ele_space_dim, Qbar_ele_space_dim), dtype=SCALAR_DTYPE)
        entity_local_index = np.zeros(1, dtype=np.int32)
        for local_facet in range(num_cell_facets):
            entity_local_index[0] = local_facet
            P_11_f.fill(0.0)
            kernel_p11(ffi.from_buffer(P_11_f), ffi.from_buffer(null64), ffi.from_buffer(constants), ffi.from_buffer(coords), ffi.from_buffer(entity_local_index), ffi.from_buffer(null8))
            row0 = local_facet * Qbar_ele_space_dim
            row1 = row0 + Qbar_ele_space_dim
            P_local[row0:row1, row0:row1] = P_11_f

    def tabulate_tensor_L0(b_ptr, w_ptr, c_ptr, coords_ptr, _entity_ptr, _perm_ptr):
        b_local = _as_array(b_ptr, (num_cell_facets * Vbar_ele_space_dim,), ctype=SCALAR_CTYPE)
        coords = _as_array(coords_ptr, (num_dofs_g, 3), ctype=REAL_CTYPE)
        constants = _as_array(c_ptr, (constants_size,), ctype=SCALAR_CTYPE)
        coeffs = _as_array(w_ptr, (V_ele_space_dim,), ctype=SCALAR_CTYPE)
        A_00, A_10, A_20, A_30, _ = compute_mats(coords, constants)
        A_tilde, B_tilde, _ = compute_tilde_mats(A_00, A_10, A_20, A_30)
        L_tilde = compute_L_tilde(coords, constants, coeffs)
        b_local[:] = -B_tilde @ np.linalg.solve(A_tilde, L_tilde)

    def tabulate_tensor_L1(b_ptr, w_ptr, c_ptr, coords_ptr, _entity_ptr, _perm_ptr):
        b_local = _as_array(b_ptr, (num_cell_facets * Qbar_ele_space_dim,), ctype=SCALAR_CTYPE)
        coords = _as_array(coords_ptr, (num_dofs_g, 3), ctype=REAL_CTYPE)
        constants = _as_array(c_ptr, (constants_size,), ctype=SCALAR_CTYPE)
        coeffs = _as_array(w_ptr, (V_ele_space_dim,), ctype=SCALAR_CTYPE)
        A_00, A_10, A_20, A_30, _ = compute_mats(coords, constants)
        A_tilde, _, C_tilde = compute_tilde_mats(A_00, A_10, A_20, A_30)
        L_tilde = compute_L_tilde(coords, constants, coeffs)
        b_local[:] = -C_tilde @ np.linalg.solve(A_tilde, L_tilde)

    def backsub_u(x_ptr, w_ptr, c_ptr, coords_ptr, _entity_ptr, _perm_ptr):
        x_local = _as_array(x_ptr, (V_ele_space_dim,), ctype=SCALAR_CTYPE)
        coords = _as_array(coords_ptr, (num_dofs_g, 3), ctype=REAL_CTYPE)
        w = _as_array(w_ptr, (num_cell_facets * (Vbar_ele_space_dim + Qbar_ele_space_dim) + V_ele_space_dim,), ctype=SCALAR_CTYPE)
        offset_ubar = num_cell_facets * Vbar_ele_space_dim
        offset_pbar = offset_ubar + num_cell_facets * Qbar_ele_space_dim
        u_bar = w[:offset_ubar]
        p_bar = w[offset_ubar:offset_pbar]
        u_prev = w[offset_pbar:]
        constants = _as_array(c_ptr, (constants_size,), ctype=SCALAR_CTYPE)
        A_00, A_10, A_20, A_30, _ = compute_mats(coords, constants)
        A_tilde, B_tilde, C_tilde = compute_tilde_mats(A_00, A_10, A_20, A_30)
        L_tilde = compute_L_tilde(coords, constants, u_prev)
        x_local[:] = np.linalg.solve(A_tilde, L_tilde - B_tilde.T @ u_bar - C_tilde.T @ p_bar)[:V_ele_space_dim]

    def backsub_p(x_ptr, w_ptr, c_ptr, coords_ptr, _entity_ptr, _perm_ptr):
        x_local = _as_array(x_ptr, (Q_ele_space_dim,), ctype=SCALAR_CTYPE)
        coords = _as_array(coords_ptr, (num_dofs_g, 3), ctype=REAL_CTYPE)
        w = _as_array(w_ptr, (num_cell_facets * (Vbar_ele_space_dim + Qbar_ele_space_dim) + V_ele_space_dim,), ctype=SCALAR_CTYPE)
        offset_ubar = num_cell_facets * Vbar_ele_space_dim
        offset_pbar = offset_ubar + num_cell_facets * Qbar_ele_space_dim
        u_bar = w[:offset_ubar]
        p_bar = w[offset_ubar:offset_pbar]
        u_prev = w[offset_pbar:]
        constants = _as_array(c_ptr, (constants_size,), ctype=SCALAR_CTYPE)
        A_00, A_10, A_20, A_30, _ = compute_mats(coords, constants)
        A_tilde, B_tilde, C_tilde = compute_tilde_mats(A_00, A_10, A_20, A_30)
        L_tilde = compute_L_tilde(coords, constants, u_prev)
        x_local[:] = np.linalg.solve(A_tilde, L_tilde - B_tilde.T @ u_bar - C_tilde.T @ p_bar)[V_ele_space_dim:]

    cb_a00 = _register_callback(tabulate_tensor_a00)
    cb_a01 = _register_callback(tabulate_tensor_a01)
    cb_a10 = _register_callback(tabulate_tensor_a10)
    cb_a11 = _register_callback(tabulate_tensor_a11)
    cb_p00 = _register_callback(tabulate_tensor_p00)
    cb_p11 = _register_callback(tabulate_tensor_p11)
    cb_L0 = _register_callback(tabulate_tensor_L0)
    cb_L1 = _register_callback(tabulate_tensor_L1)
    cb_backsub_u = _register_callback(backsub_u)
    cb_backsub_p = _register_callback(backsub_p)
    timings["jit_kernels"] = timer.stop()

    timer = LoggedTimer("Create forms", comm)
    cells = np.arange(msh.topology.index_map(tdim).size_local, dtype=np.int32)
    perms = np.array([], dtype=np.int8)
    integration_mesh = msh._cpp_object if hasattr(msh, "_cpp_object") else msh
    facet_entity_map = (facet_mesh._cpp_object if hasattr(facet_mesh, "_cpp_object") else facet_mesh, inv_entity_map)

    def callback_addr(callback):
        return ctypes.cast(callback, ctypes.c_void_p).value

    a00 = build_custom_form([Vbar, Vbar], {IntegralType.cell: [(-1, callback_addr(cb_a00), cells, perms)]}, [], [delta_t, nu], integration_mesh, facet_entity_map)
    a01 = build_custom_form([Vbar, Qbar], {IntegralType.cell: [(-1, callback_addr(cb_a01), cells, perms)]}, [], [delta_t, nu], integration_mesh, facet_entity_map)
    a10 = build_custom_form([Qbar, Vbar], {IntegralType.cell: [(-1, callback_addr(cb_a10), cells, perms)]}, [], [delta_t, nu], integration_mesh, facet_entity_map)
    a11 = build_custom_form([Qbar, Qbar], {IntegralType.cell: [(-1, callback_addr(cb_a11), cells, perms)]}, [], [delta_t, nu], integration_mesh, facet_entity_map)
    a = [[a00, a01], [a10, a11]]

    p00 = build_custom_form([Vbar, Vbar], {IntegralType.cell: [(-1, callback_addr(cb_p00), cells, perms)]}, [], [delta_t, nu], integration_mesh, facet_entity_map)
    p11_form = build_custom_form([Qbar, Qbar], {IntegralType.cell: [(-1, callback_addr(cb_p11), cells, perms)]}, [], [nu], integration_mesh, facet_entity_map)
    p = [[p00, None], [None, p11_form]]

    L0 = build_custom_form([Vbar], {IntegralType.cell: [(-1, callback_addr(cb_L0), cells, perms)]}, [u_n], [delta_t, nu], integration_mesh, facet_entity_map)
    L1 = build_custom_form([Qbar], {IntegralType.cell: [(-1, callback_addr(cb_L1), cells, perms)]}, [u_n], [delta_t, nu], integration_mesh, facet_entity_map)
    L = [L0, L1]
    timings["create_forms"] = timer.stop()

    timer = LoggedTimer("Boundary conditions", comm)
    msh_boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: boundary(x, tdim))
    facet_mesh_boundary_facets = inv_entity_map[msh_boundary_facets]
    facet_mesh_boundary_facets = facet_mesh_boundary_facets[facet_mesh_boundary_facets >= 0]

    dofs = locate_dofs_topological(Vbar, fdim, facet_mesh_boundary_facets)
    u_bc = Function(Vbar)
    u_bc.interpolate(u_e)
    bc_ubar = dirichletbc(u_bc, dofs)

    pressure_dofs = locate_dofs_geometrical(
        Qbar,
        lambda x: np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0)),
    )
    pressure_dofs = np.array([pressure_dofs[0]], dtype=np.int32) if len(pressure_dofs) > 0 else np.array([], dtype=np.int32)
    bc_pbar = dirichletbc(PETSc.ScalarType(0.0), pressure_dofs, Qbar)

    bcs = [bc_ubar]
    if use_direct_solver:
        bcs.append(bc_pbar)
    timings["bcs"] = timer.stop()

    timer = LoggedTimer("Assemble matrix", comm)
    A = assemble_matrix_block(a, bcs=bcs)
    A.assemble()
    timings["assemble_mat"] = timer.stop()

    b = fem.petsc.create_vector_block(L)

    if use_direct_solver:
        ksp = PETSc.KSP().create(msh.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("superlu_dist")
    else:
        timer = LoggedTimer("Assemble preconditioner", comm)
        P = assemble_matrix_block(p, bcs=bcs)
        P.assemble()
        timings["assemble_pre"] = timer.stop()

        timer = LoggedTimer("Setup solver", comm)
        offset_ubar = Vbar.dofmap.index_map.local_range[0] * Vbar.dofmap.index_map_bs + Qbar.dofmap.index_map.local_range[0]
        offset_pbar = offset_ubar + Vbar.dofmap.index_map.size_local * Vbar.dofmap.index_map_bs
        is_ubar = PETSc.IS().createStride(Vbar.dofmap.index_map.size_local * Vbar.dofmap.index_map_bs, offset_ubar, 1, comm=PETSc.COMM_SELF)
        is_pbar = PETSc.IS().createStride(Qbar.dofmap.index_map.size_local, offset_pbar, 1, comm=PETSc.COMM_SELF)

        null_vec = A.createVecLeft()
        offset = Vbar.dofmap.index_map.size_local * Vbar.dofmap.index_map_bs
        null_vec.array[offset:] = 1.0
        null_vec.normalize()
        A.setNullSpace(PETSc.NullSpace().create(vectors=[null_vec]))

        ksp = PETSc.KSP().create(msh.comm)
        ksp.setOperators(A, P)
        ksp.setTolerances(rtol=1e-12)
        ksp.setType("minres")
        ksp.getPC().setType("fieldsplit")
        ksp.getPC().setFieldSplitIS(("u", is_ubar), ("p", is_pbar))

        ksp_u, ksp_p = ksp.getPC().getFieldSplitSubKSP()
        ksp_u.setType("preonly")
        ksp_u.getPC().setType("hypre")
        ksp_p.setType("preonly")
        ksp_p.getPC().setType("sor")

        opts = PETSc.Options()
        opts["ksp_monitor"] = None
        opts["ksp_view"] = None
        opts["fieldsplit_u_pc_hypre_type"] = "boomeramg"
        opts["fieldsplit_u_pc_hypre_boomeramg_cycle_type"] = "V"
        opts["fieldsplit_u_pc_hypre_boomeramg_agg_nl"] = 1
        opts["fieldsplit_u_pc_hypre_boomeramg_agg_num_paths"] = 1
        opts["fieldsplit_u_pc_hypre_boomeramg_strong_threshold"] = 0.5 if tdim == 2 else 0.75
        opts["options_left"] = None
        ksp.setFromOptions()
        timings["setup_solver"] = timer.stop()

    x = A.createVecRight()

    u_h = Function(V)
    u_h.name = "u"
    p_h = Function(Q)
    p_h.name = "p"
    ubar_h = Function(Vbar)
    ubar_h.name = "ubar"
    pbar_h = Function(Qbar)
    pbar_h.name = "pbar"

    timer = LoggedTimer("Write initial condition to file", comm)
    output_dir = Path.cwd()
    u_file = io.VTXWriter(msh.comm, output_dir / "u.bp", [u_h._cpp_object])
    p_file = io.VTXWriter(msh.comm, output_dir / "p.bp", [p_h._cpp_object])
    ubar_file = io.VTXWriter(msh.comm, output_dir / "ubar.bp", [ubar_h._cpp_object])
    pbar_file = io.VTXWriter(msh.comm, output_dir / "pbar.bp", [pbar_h._cpp_object])
    u_file.write(0.0)
    p_file.write(0.0)
    ubar_file.write(0.0)
    pbar_file.write(0.0)
    timings["write_init"] = timer.stop()

    u_form = build_custom_form([V], {IntegralType.cell: [(-1, callback_addr(cb_backsub_u), cells, perms)]}, [ubar_h, pbar_h, u_n], [delta_t, nu], integration_mesh, facet_entity_map)
    p_form = build_custom_form([Q], {IntegralType.cell: [(-1, callback_addr(cb_backsub_p), cells, perms)]}, [ubar_h, pbar_h, u_n], [delta_t, nu], integration_mesh, facet_entity_map)

    current_time = 0.0
    timings["assemble_vec"] = 0.0
    timings["solve"] = 0.0
    timings["recov_facet_sol"] = 0.0
    timings["backsub"] = 0.0
    timings["write"] = 0.0

    for _ in range(num_time_steps):
        current_time += float(delta_t.value)
        par_print(f"\nt = {current_time}", comm)

        timer = LoggedTimer("Assemble vector", comm)
        with b.localForm() as b_loc:
            b_loc.set(0)
        assemble_vector_block(b, L, a, bcs=bcs)
        timings["assemble_vec"] += timer.stop()

        timer = LoggedTimer("Solve", comm)
        ksp.solve(b, x)
        timings["solve"] += timer.stop()

        timer = LoggedTimer("Recover facet solution", comm)
        offset = Vbar.dofmap.index_map.size_local * Vbar.dofmap.index_map_bs
        ubar_h.x.array[:offset] = x.array_r[:offset]
        pbar_h.x.array[: len(x.array_r) - offset] = x.array_r[offset:]
        ubar_h.x.scatter_forward()
        pbar_h.x.scatter_forward()
        timings["recov_facet_sol"] += timer.stop()

        timer = LoggedTimer("Backsubstitution", comm)
        coeffs_u = pack_coefficients(u_form)
        u_h.x.array[:] = 0.0
        fem.assemble_vector(u_h.x.array, u_form, coeffs=coeffs_u)
        u_h.x.scatter_reverse(la.InsertMode.add)
        u_h.x.scatter_forward()

        coeffs_p = pack_coefficients(p_form)
        p_h.x.array[:] = 0.0
        fem.assemble_vector(p_h.x.array, p_form, coeffs=coeffs_p)
        p_h.x.scatter_reverse(la.InsertMode.add)
        p_h.x.scatter_forward()
        timings["backsub"] += timer.stop()

        timer = LoggedTimer("Write to file", comm)
        u_file.write(current_time)
        p_file.write(current_time)
        ubar_file.write(current_time)
        pbar_file.write(current_time)
        timings["write"] += timer.stop()

        u_n.x.array[:] = u_h.x.array
        u_n.x.scatter_forward()

    timer = LoggedTimer("Compute error in facet solution", comm)
    xbar = ufl.SpatialCoordinate(facet_mesh)
    e_ubar = norm_L2(comm, ubar_h - u_e(xbar, ufl))
    pbar_h_avg = domain_average(facet_mesh, pbar_h)
    pbar_e_avg = domain_average(facet_mesh, p_e(xbar, ufl))
    e_pbar = norm_L2(comm, (pbar_h - pbar_h_avg) - (p_e(xbar, ufl) - pbar_e_avg))
    timings["compute_error_facet"] = timer.stop()

    timer = LoggedTimer("Compute errors", comm)
    x_cell = ufl.SpatialCoordinate(msh)
    e_u = norm_L2(comm, u_h - u_e(x_cell, ufl))
    e_div_u = norm_L2(comm, ufl.div(u_h))
    e_jump_u = normal_jump_error(msh, u_h)
    p_h_avg = domain_average(msh, p_h)
    p_e_avg = domain_average(msh, p_e(x_cell, ufl))
    e_p = norm_L2(comm, (p_h - p_h_avg) - (p_e(x_cell, ufl) - p_e_avg))
    timings["compute_errors_cell"] = timer.stop()

    num_cells = msh.topology.index_map(tdim).size_global
    num_dofs_V = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    num_dofs_Q = Q.dofmap.index_map.size_global
    num_dofs_Vbar = Vbar.dofmap.index_map.size_global * Vbar.dofmap.index_map_bs
    num_dofs_Qbar = Qbar.dofmap.index_map.size_global
    dofs_sc = num_dofs_Vbar + num_dofs_Qbar
    total_dofs = num_dofs_V + num_dofs_Q + dofs_sc

    timings["total"] = total_timer.stop()
    results = {
        "data": {
            "num_proc": comm.size,
            "num_cells": num_cells,
            "num_dofs_V": num_dofs_V,
            "num_dofs_Q": num_dofs_Q,
            "num_dofs_Vbar": num_dofs_Vbar,
            "num_dofs_Qbar": num_dofs_Qbar,
            "dofs_sc": dofs_sc,
            "total_dofs": total_dofs,
            "e_u": e_u,
            "e_div_u": e_div_u,
            "e_jump_u": e_jump_u,
            "e_p": e_p,
            "e_ubar": e_ubar,
            "e_pbar": e_pbar,
            "its": ksp.its,
        },
        "timings": {name: comm.allreduce(value, op=MPI.MAX) for name, value in timings.items()},
    }

    for name, value in results["data"].items():
        par_print(f"{name} = {value}", comm)

    if comm.rank == 0:
        with open(output_dir / f"results_{comm.size}.json", "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)

    list_timings(MPI.COMM_WORLD, [TimingType.wall, TimingType.user])


if __name__ == "__main__":
    main()
