"""
  ViscousFormulationBassiRebay1()

The classical BR1 flux from

- F. Bassi, S. Rebay (1997)
  A High-Order Accurate Discontinuous Finite Element Method for
  the Numerical Solution of the Compressible Navier-Stokes Equations
  [DOI: 10.1006/jcph.1996.5572](https://doi.org/10.1006/jcph.1996.5572)
"""
struct ViscousFormulationBassiRebay1 end

"""
    ViscousFormulationLocalDG(penalty_parameter)

The local DG (LDG) flux from "The Local Discontinuous Galerkin Method for Time-Dependent
Convection-Diffusion Systems" by Cockburn and Shu (1998).

The parabolic "upwinding" vector is currently implemented for `TreeMesh`; for all other mesh types,
the LDG solver is equivalent to [`ViscousFormulationBassiRebay1`](@ref) with an LDG-type penalization.

- Cockburn and Shu (1998).
  The Local Discontinuous Galerkin Method for Time-Dependent
  Convection-Diffusion Systems
  [DOI: 10.1137/S0036142997316712](https://doi.org/10.1137/S0036142997316712)
- Cockburn and Dong (2007)  
  An Analysis of the Minimal Dissipation Local Discontinuous 
  Galerkin Method for Convection–Diffusion Problems.
  [DOI: 10.1007/s10915-007-9130-3](https://doi.org/10.1007/s10915-007-9130-3)
"""
struct ViscousFormulationLocalDG{P}
    penalty_parameter::P
end

ViscousFormulationLocalDG() = ViscousFormulationLocalDG(nothing)

default_parabolic_solver() = ViscousFormulationBassiRebay1()
