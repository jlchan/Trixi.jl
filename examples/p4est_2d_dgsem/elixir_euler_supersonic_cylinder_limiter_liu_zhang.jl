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

@inline function boundary_condition_outflow_general(u_inner,
                                                    normal_direction::AbstractVector, x, t,
                                                    surface_flux_function,
                                                    equations::CompressibleEulerEquations2D)

    # This would be for the general case where we need to check the magnitude of the local Mach number
    norm_ = norm(normal_direction)
    # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
    normal = normal_direction / norm_

    # Rotate the internal solution state
    u_local = Trixi.rotate_to_x(u_inner, normal, equations)

    # Compute the primitive variables
    rho_local, v_normal, v_tangent, p_local = cons2prim(u_local, equations)

    # Compute local Mach number
    a_local = sqrt(equations.gamma * p_local / rho_local)
    Mach_local = abs(v_normal / a_local)
    if Mach_local <= 1.0 # The `if` is not needed in this elixir but kept for generality
        # In general, `p_local` need not be available from the initial condition
        p_local = pressure(initial_condition_subsonic(x, t, equations), equations)
    end

    # Create the `u_surface` solution state where the local pressure is possibly set from an external value
    prim = SVector(rho_local, v_normal, v_tangent, p_local)
    u_boundary = prim2cons(prim, equations)
    u_surface = Trixi.rotate_from_x(u_boundary, normal, equations)

    # Compute the flux using the appropriate mixture of internal / external solution states
    return flux(u_surface, normal_direction, equations)
end

initial_condition = initial_condition_high_mach_flow

boundary_conditions = (; Bottom = boundary_condition_slip_wall,
                         Circle = boundary_condition_slip_wall,
                         Top = boundary_condition_slip_wall,
                         Right = boundary_condition_outflow_general,
                         Left = BoundaryConditionDirichlet(initial_condition_high_mach_flow))

surface_flux = flux_lax_friedrichs

polydeg = 3
basis = LobattoLegendreBasis(polydeg)
indicator_ec = IndicatorEntropyCorrection(equations, basis; scaling = 2)
# indicator_sc = IndicatorHennemannGassner(equations, basis,
#                                          alpha_max = 0.1,
#                                          alpha_min = 0.00,
#                                          alpha_smooth = true,
#                                          variable = density_pressure)
# indicator_combined = 
#     IndicatorEntropyCorrectionShockCapturingCombined(indicator_ec, indicator_sc)

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

alive_callback = AliveCallback(alive_interval = 2000)

# The SaveRestartCallback allows to save a file from which a Trixi.jl simulation can be restarted
save_restart = SaveRestartCallback(interval = 1000,
                                   save_final_restart = true)

save_solution = SaveSolutionCallback(interval = 1000,
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

ode_solver = RDPK3SpFSAL35(; stage_limiter! = global_limiter!,
                             step_limiter! = global_limiter!)

###############################################################################
# run the simulation
sol = solve(ode, ode_solver;
            adaptive = true, dt = 1e-7, abstol = 1e-5, reltol = 1e-3,
            # adaptive = false, dt = 1,
            saveat=LinRange(tspan..., 800), callback = callbacks);

# using Plots
# @gif for i in eachindex(sol.u)
#     pd = PlotData2D(sol.u[i], semi)
#     plot(pd["rho"], clims=(0.0, 7.0), title="Time: $(sol.t[i])", dpi=400)
# end 

# using Plots
# pd = PlotData2D(sol.u[end], semi)
# plot(pd["rho"], clims=(0.0, 7.0), title="Time: $(sol.t[end])", dpi=400)
