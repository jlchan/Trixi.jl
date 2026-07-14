using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

"""
    initial_condition_medium_sedov_blast_wave(x, t, equations::CompressibleEulerEquations3D)

The Sedov blast wave setup based on example 35.1.4 from Flash
- https://flash.rochester.edu/site/flashcode/user_support/flash4_ug_4p8.pdf
with smaller strength of the initial discontinuity.
"""
function initial_condition_medium_sedov_blast_wave(x, t,
                                                   equations::CompressibleEulerEquations3D)
    # Set up polar coordinates
    inicenter = SVector(0.0, 0.0, 0.0)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    z_norm = x[3] - inicenter[3]
    r = sqrt(x_norm^2 + y_norm^2 + z_norm^2)

    # Setup based on example 35.1.4 in https://flash.rochester.edu/site/flashcode/user_support/flash4_ug_4p8.pdf
    r0 = 0.21875 # = 3.5 * smallest dx (for domain length=4 and max-ref=6)
    E = 1.0
    p0_inner = 3 * (equations.gamma - 1) * E / (4 * pi * r0^2)
    p0_outer = 1.0e-6

    # Calculate primitive variables
    rho = 1.0
    v1 = 0.0
    v2 = 0.0
    v3 = 0.0
    p = r > r0 ? p0_outer : p0_inner
    if r ≈ r0
        p = 0.5f0 * (p0_inner + p0_outer)
    end

    return prim2cons(SVector(rho, v1, v2, v3, p), equations)
end

initial_condition = initial_condition_medium_sedov_blast_wave

surface_flux = flux_lax_friedrichs
basis = LobattoLegendreBasis(3)
indicator_ec = IndicatorEntropyCorrection(equations, basis)
volume_integral_default = VolumeIntegralWeakForm()
volume_integral_entropy_stable = VolumeIntegralPureLGLFiniteVolume(surface_flux)
volume_integral = VolumeIntegralAdaptive(indicator_ec,
                                         volume_integral_default,
                                         volume_integral_entropy_stable)

solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-1.0, -1.0, -1.0)
coordinates_max = (1.0, 1.0, 1.0)

trees_per_dimension = (16, 16, 16)
mesh = P4estMesh(trees_per_dimension,
                 polydeg = 3,
                 coordinates_min = coordinates_min,
                 coordinates_max = coordinates_max,
                 periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1e-1)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 500
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 1.3)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

local_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (1e-6, 1e-6),
                                                     variables = (Trixi.density,
                                                                  pressure))
global_limiter! = PositivityPreservingLimiterLiuZhang(local_limiter!, semi;
                                                      record_davis_yin_iterations = true)

ode_solver = CarpenterKennedy2N54(; stage_limiter! = global_limiter!,
                                  step_limiter! = global_limiter!,
                                  williamson_condition = false)

sol = solve(ode, ode_solver;
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
