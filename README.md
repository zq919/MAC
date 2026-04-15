# MAC

## FEniCSx 0.9.0 HDG demos

This repository contains FEniCSx 0.9.0-oriented HDG examples and notes.

### Available scripts

- `hdg_conv_diff_fenicsx_0_9_fix.py`: HDG convection-diffusion example on the unit square.
- `hdg_stokes_fenicsx_0_9.py`: 2D HDG Stokes example using the HDG polynomial spaces
- `Biot_HDG_cou.py`: unsteady Biot HDG demo using mixed block spaces `(u, ubar, p, pbar)` and the variational blocks `e_h`, `a_h`, and `\tilde b_h` with backward-Euler time coupling.
- `stokes-numman.py`: HDG Stokes example on `[0,1] x [0,1]` with Dirichlet velocity on the top/bottom boundaries and exact-solution Neumann traction on the left/right boundaries, using a mixed-boundary primal HDG formulation with

  $$
  V_h(K)=[P_k(K)]^d,\qquad
  \bar V_h(F)=[P_k(F)]^d,\qquad
  Q_h(K)=P_{k-1}(K),\qquad
  \bar Q_h(F)=P_k(F),
  $$

  following the non-homogeneous mixed-boundary variational structure used in `stokes-numman.py`.

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
