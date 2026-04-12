# MAC

## FEniCSx 0.9.0 HDG demos

This repository contains FEniCSx 0.9.0-oriented HDG examples and notes.

### Available scripts

- `hdg_conv_diff_fenicsx_0_9_fix.py`: HDG convection-diffusion example on the unit square.
- `hdg_stokes_fenicsx_0_9.py`: 2D HDG Stokes example using the HDG polynomial spaces
- `stokes-numman.py`: HDG Stokes example on `[0,1] x [0,1]` with Dirichlet velocity on the top/bottom boundaries and exact-solution Neumann traction on the left/right boundaries, using a gradient-based mixed-boundary HDG formulation with

  $$
  G_h(K)=[P_k(K)]^{d\times d},\qquad
  V_h(K)=[P_k(K)]^d,\qquad
  Q_h(K)=P_k(K),\qquad
  \bar V_h(F)=[P_k(F)]^d,
  $$

  following the mixed-boundary HDG structure of Shannon & Bui-Thanh, *New HDG Methods for the Stokes and Oseen Equations*.

  together with the analytical solution

  $$
  u_1 = -x^2(x-1)^2 y(y-1)(2y-1),\qquad
  u_2 = x(x-1)(2x-1)y^2(y-1)^2,\qquad
  p = x^6 - y^6,
  $$

  and reporting refinement-table errors/rates for velocity, pressure, and divergence.

### Notes

- `VARIATIONAL_FORM.md` explains the strong/variational formulations used by the HDG
  convection-diffusion and Stokes scripts.
- The mixed-domain FEniCSx 0.9.0 examples in this repository use facet submeshes,
  `entity_maps`, and blocked PETSc assembly.
