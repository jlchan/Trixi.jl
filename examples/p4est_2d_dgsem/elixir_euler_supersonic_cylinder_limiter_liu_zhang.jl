# Channel flow around a cylinder at Mach 3
#
# Boundary conditions are supersonic Mach 3 inflow at the left portion of the domain
# and supersonic outflow at the right portion of the domain. The top and bottom of the
# channel as well as the cylinder are treated as Euler slip wall boundaries.
# This flow results in strong shock reflections / interactions as well as Kelvin-Helmholtz
# instabilities at later times as two Mach stems form above and below the cylinder.
#
# For complete details on the problem setup see Section 5.7 of the paper:
# - Jean-Luc Guermond, Murtazo Nazarov, Bojan Popov, and Ignacio Tomas (2018)
#   Second-Order Invariant Domain Preserving Approximation of the Euler Equations using Convex Limiting.
#   [DOI: 10.1137/17M1149961](https://doi.org/10.1137/17M1149961)
#
# Keywords: supersonic flow, shock capturing, unstructured curved mesh, positivity preservation, compressible Euler, 2D

using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

@inline function initial_condition_high_mach_flow(x, t, equations::CompressibleEulerEquations2D)
    # set the freestream flow parameters
    rho_freestream = 1.4
    v1 = 3.0
    v2 = 0.0
    p_freestream = 1.0

    prim = SVector(rho_freestream, v1, v2, p_freestream)
    return prim2cons(prim, equations)
end

initial_condition = initial_condition_high_mach_flow

# Supersonic inflow boundary condition.
# Calculate the boundary flux entirely from the external solution state, i.e., set
# external solution state values for everything entering the domain.
@inline function boundary_condition_supersonic_inflow(u_inner,
                                                      normal_direction::AbstractVector,
                                                      x, t, surface_flux_function,
                                                      equations::CompressibleEulerEquations2D)
    u_boundary = initial_condition_high_mach_flow(x, t, equations)
    return flux(u_boundary, normal_direction, equations)
end

# Supersonic outflow boundary condition.
# Calculate the boundary flux entirely from the internal solution state. Analogous to supersonic inflow
# except all the solution state values are set from the internal solution as everything leaves the domain
@inline function boundary_condition_outflow(u_inner, normal_direction::AbstractVector, x, t,
                                            surface_flux_function,
                                            equations::CompressibleEulerEquations2D)
    return flux(u_inner, normal_direction, equations)
end

boundary_conditions = (; Bottom = boundary_condition_slip_wall,
                         Circle = boundary_condition_slip_wall,
                         Top = boundary_condition_slip_wall,
                         Right = boundary_condition_outflow,
                         Left = boundary_condition_supersonic_inflow)

surface_flux = flux_lax_friedrichs

polydeg = 3
basis = LobattoLegendreBasis(polydeg)
indicator_ec = IndicatorEntropyCorrection(equations, basis; 
                                          scaling = 2)
# indicator_sc = IndicatorHennemannGassner(equations, basis,
#                                          alpha_max = 0.05,
#                                          alpha_min = 0.00,
#                                          alpha_smooth = true,
#                                          variable = density_pressure)
# indicator = IndicatorEntropyCorrectionShockCapturingCombined(indicator_ec, indicator_sc)

volume_integral_default = VolumeIntegralWeakForm()
volume_integral_entropy_stable = VolumeIntegralPureLGLFiniteVolume(surface_flux)
volume_integral = VolumeIntegralAdaptive(indicator_ec,
                                         volume_integral_default,
                                         volume_integral_entropy_stable)
solver = DGSEM(basis, surface_flux, volume_integral)

# Get the unstructured quad mesh from a file (downloads the file if not available locally)
mesh_file = Trixi.download("https://gist.githubusercontent.com/andrewwinters5000/a08f78f6b185b63c3baeff911a63f628/raw/addac716ea0541f588b9d2bd3f92f643eb27b88f/abaqus_cylinder_in_channel.inp",
                           joinpath(@__DIR__, "abaqus_cylinder_in_channel.inp"))

mesh = P4estMesh{2}(mesh_file; initial_refinement_level=3)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers

tspan = (0.0, 20.0)
ode = semidiscretize(semi, tspan)

# using Plots
# pd = PlotData2D(ode.u0, semi)
# plot(getmesh(pd))

# Callbacks

summary_callback = SummaryCallback()

alive_callback = AliveCallback(analysis_interval = analysis_interval)

# The SaveRestartCallback allows to save a file from which a Trixi.jl simulation can be restarted
save_restart = SaveRestartCallback(interval = 1000,
                                   save_final_restart = true)

save_solution = SaveSolutionCallback(interval = 5000,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

callbacks = CallbackSet(summary_callback, 
                        alive_callback,
                        save_solution, save_restart)

local_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (1.0e-8, 1.0e-8),
                                                     variables = (Trixi.density,
                                                                  energy_internal))
global_limiter! = PositivityPreservingLimiterLiuZhang(local_limiter!, semi;
                                                      record_davis_yin_iterations = true)

ode_solver = RDPK3SpFSAL49(; stage_limiter! = global_limiter!,
                             step_limiter! = global_limiter!)

###############################################################################
# run the simulation
sol = solve(ode, ode_solver;
            adaptive = true, dt = 1e-7, abstol = 1e-5, reltol = 1e-4,
            saveat=LinRange(tspan..., 400), callback = callbacks);

# using Plots
# @gif for i in eachindex(sol.u)
#     pd = PlotData2D(sol.u[i], semi)
#     plot(pd["rho"], clims=(0.02, 7.5), title="Time: $(sol.t[i])", dpi=400)
# end fps=10