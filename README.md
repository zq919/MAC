# MAC

## FEniCSx 0.9.0 compatibility note

This repository now includes `hdg_conv_diff_fenicsx_0_9_fix.py`, a patched HDG
convection-diffusion example for FEniCSx 0.9.0.

### What changed

The original code assumed that `mesh.create_submesh(...)` returned an object
with a `sub_topology_to_topology(..., inverse=True)` method. In FEniCSx 0.9.0,
the returned entity map is a NumPy array, so the parent-mesh boundary facets
must be mapped back to submesh facets by explicitly building the inverse map.

### Key fix

Use `parent_to_sub_entities(...)` to support both APIs:

- Newer FEniCSx releases: call `sub_topology_to_topology(..., inverse=True)`.
- FEniCSx 0.9.0: invert the `submesh_entity -> parent_entity` NumPy array and
  index it with the parent boundary facets.
