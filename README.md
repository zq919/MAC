# MAC

## FEniCSx 0.9.0 runnable HDG convection-diffusion example

This repository now includes `hdg_conv_diff_fenicsx_0_9_fix.py`, a standalone
HDG convection-diffusion script written against the FEniCSx 0.9.0 API.

For the mathematical statement of the HDG method implemented in the script, see `VARIATIONAL_FORM.md`.

## Why the original code still failed on 0.9.0

Besides the `create_submesh(...)` entity-map difference, FEniCSx 0.9.0 also
expects mixed-domain forms to be compiled with `fem.form(..., entity_maps=...)`
using a dictionary keyed by the non-integration mesh, and the blocked linear
system should be assembled with `assemble_matrix_block` /
`assemble_vector_block` before solving it with PETSc.

## What this script does

- builds the inverse map `mesh_to_facet_mesh` from parent-mesh facets to the
  facet submesh;
- compiles the mixed-domain blocked forms with
  `entity_maps = {facet_mesh: mesh_to_facet_mesh}`;
- assembles and solves the HDG linear system with PETSc block assembly tools;
- applies Dirichlet data on the trace space and reports `e_u` and `e_ubar`.
