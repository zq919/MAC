"""Compatibility assembly helpers for the HDG examples on DOLFINx 0.9.0."""

from __future__ import annotations

import collections.abc
from collections.abc import Sequence
from typing import Any

from petsc4py import PETSc

from dolfinx.fem.assemble import pack_coefficients as _pack_coefficients
from dolfinx.fem.petsc import (
    apply_lifting as _apply_lifting,
    assemble_matrix as _assemble_matrix,
    assemble_matrix_block as _assemble_matrix_block,
    assemble_vector as _assemble_vector,
    assemble_vector_block as _assemble_vector_block,
    create_matrix,
    create_matrix_block,
)


__all__ = [
    "apply_lifting",
    "assemble_matrix",
    "assemble_matrix_block",
    "assemble_vector",
    "assemble_vector_block",
    "create_matrix",
    "create_matrix_block",
    "pack_coefficients",
]



def pack_coefficients(form):
    """Pack coefficients for Python `Form` wrappers and nested form containers."""

    def _pack(obj):
        if obj is None:
            return {}
        if isinstance(obj, collections.abc.Iterable) and not hasattr(obj, "_cpp_object"):
            return [_pack(sub_obj) for sub_obj in obj]
        return _pack_coefficients(getattr(obj, "_cpp_object", obj))

    return _pack(form)



def apply_lifting(
    b: PETSc.Vec,
    a: Sequence[Any],
    bcs: Sequence[Sequence[Any]],
    x0: Sequence[PETSc.Vec] | None = None,
    scale: float = 1.0,
    constants=None,
    coeffs=None,
) -> None:
    """Compatibility wrapper using the legacy ``scale`` keyword."""
    _apply_lifting(b, a, bcs, [] if x0 is None else list(x0), scale, constants, coeffs)



def assemble_vector(L, constants=None, coeffs=None):
    """Assemble a linear form into a PETSc vector."""
    return _assemble_vector(L, constants=constants, coeffs=coeffs)



def assemble_matrix(a, bcs=None, diagonal: float = 1.0, constants=None, coeffs=None):
    """Assemble a bilinear form into a PETSc matrix."""
    return _assemble_matrix(
        a,
        bcs=[] if bcs is None else list(bcs),
        diagonal=diagonal,
        constants=constants,
        coeffs=coeffs,
    )



def assemble_matrix_block(a, bcs=None, diagonal: float = 1.0, constants=None, coeffs=None):
    """Assemble blocked bilinear forms into a PETSc matrix."""
    return _assemble_matrix_block(
        a,
        bcs=[] if bcs is None else list(bcs),
        diagonal=diagonal,
        constants=constants,
        coeffs=coeffs,
    )



def assemble_vector_block(
    first,
    second,
    third=None,
    bcs=None,
    x0: PETSc.Vec | None = None,
    scale: float = 1.0,
    constants_L=None,
    coeffs_L=None,
    constants_a=None,
    coeffs_a=None,
):
    """Compatibility wrapper supporting both DOLFINx 0.9.0 and legacy call styles."""
    kwargs = dict(
        bcs=[] if bcs is None else list(bcs),
        x0=x0,
        alpha=scale,
        constants_L=constants_L,
        coeffs_L=coeffs_L,
        constants_a=constants_a,
        coeffs_a=coeffs_a,
    )
    if isinstance(first, PETSc.Vec):
        return _assemble_vector_block(first, second, third, **kwargs)
    return _assemble_vector_block(first, second, **kwargs)
