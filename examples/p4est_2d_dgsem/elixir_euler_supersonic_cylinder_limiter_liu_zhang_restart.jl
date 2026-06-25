using OrdinaryDiffEqLowStorageRK
using Trixi
using Accessors: @reset

###############################################################################
# Restart supersonic cylinder simulation with Liu-Zhang limiter
#
# Continues the simulation from a restart file written by
# elixir_euler_supersonic_cylinder_limiter_liu_zhang.jl, switching to the
# combined entropy-correction and shock-capturing volume integral.

base_elixir = "elixir_euler_supersonic_cylinder_limiter_liu_zhang.jl"
trixi_include(@__MODULE__, joinpath(@__DIR__, base_elixir),
              tspan = (0.0, 1.0e-10))

###############################################################################
# adapt the parameters that have changed compared to the base elixir

# Note: If you get a restart file from somewhere else, you need to provide
# appropriate setups in the elixir loading a restart file

output_directory = normpath(joinpath(@__DIR__, "..", "..", "out"))
restart_file = "restart_000015000.h5" 
restart_filename = joinpath(output_directory, restart_file)

mesh = load_mesh(restart_filename)

indicator_ec = IndicatorEntropyCorrection(equations, basis; 
                                          scaling = 2)

# indicator_sc = IndicatorHennemannGassner(equations, basis,
#                                          alpha_max = 0.05,
#                                          alpha_min = 0.00,
#                                          alpha_smooth = true,
#                                          variable = density_pressure)
# indicator = IndicatorEntropyCorrectionShockCapturingCombined(indicator_ec, indicator_sc)
# surface_flux = flux_hllc
surface_flux = flux_lax_friedrichs
volume_integral = VolumeIntegralAdaptive(indicator_ec,
                                         volume_integral_default,
                                         volume_integral_entropy_stable)
solver = DGSEM(basis, surface_flux, volume_integral)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_conditions)

local_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (1.0e-8, 1.0e-8),
                                                     variables = (Trixi.density,
                                                                  energy_internal))
global_limiter! = PositivityPreservingLimiterLiuZhang(local_limiter!, semi;
                                                      record_davis_yin_iterations = true)
ode_solver = RDPK3SpFSAL49(; stage_limiter! = global_limiter!,
                             step_limiter! = global_limiter!)

analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
callbacks = CallbackSet(summary_callback,
                        analysis_callback, 
                        alive_callback)

t_end = 20.0
tspan = (load_time(restart_filename), t_end)
dt_restart = load_dt(restart_filename)
ode = semidiscretize(semi, tspan, restart_filename)

# Do not overwrite snapshots from the original simulation.
@reset save_solution.condition.save_initial_solution = false
@reset save_solution.condition.output_directory = output_directory
@reset save_restart.condition.output_directory = output_directory
@reset analysis_callback.affect!.output_directory = output_directory

integrator = init(ode, ode_solver;
                  adaptive = true, dt = dt_restart,
                  abstol = 1e-5, reltol = 1e-4,
                  saveat=LinRange(tspan..., 200), callback = callbacks);

# Continue restart file numbering from the original simulation.
load_timestep!(integrator, restart_filename)

###############################################################################
# run the simulation

sol = solve!(integrator)

using Plots
@gif for i in eachindex(sol.u)
    pd = PlotData2D(sol.u[i], semi)
    plot(pd["rho"], clims=(0.02, 7.5), title="Time: $(sol.t[i])", dpi=300)
end

# pd = PlotData2D(sol.u[end], semi)
# plot(pd["rho"], clims=(0.02, 7.5), title="Time: $(sol.t[end])", dpi=300)
# plot(getmesh(pd))

# (; node_coordinates) = semi.cache.elements
# scatter!(vec(node_coordinates[1,:,:,:]), vec(node_coordinates[2,:,:,:]), color=:black, markersize=1)
# x = reshape(node_coordinates[1,:,:,:], :, nelements(solver, semi.cache))
# y = reshape(node_coordinates[2,:,:,:], :, nelements(solver, semi.cache))
# xc = vec(sum(x, dims=1) / size(x, 1))
# yc = vec(sum(y, dims=1) / size(y, 1))

# av = indicator_ec.cache.alpha
# av = repeat(av', size(x, 1), 1)

# @inline function local_mach_number(u, equations::CompressibleEulerEquations2D)
#     rho = Trixi.density(u, equations)
#     v = Trixi.velocity(u, equations)          # SVector(v1, v2)
#     p = Trixi.pressure(u, equations)
#     c = sqrt(equations.gamma * p / rho) # sound speed
#     return norm(v) / c                  # magnitude Mach number
# end

# u = Trixi.wrap_array_native(ode.u0, semi)
# u = reshape(reinterpret(SVector{nvariables(equations), Float64}, u), :, nelements(solver, semi.cache))
# z = local_mach_number.(u, equations)

# scatter(vec(x), vec(y), zcolor=vec(z),
#         # clims=(0.0, 0.1),
#         clims=(0.0, 10),
#         msw = 0, markersize=1, ratio=1, 
#         leg=false, colorbar=true)

# outflow = findall(@. abs(x - 4.0) < 100 * eps())        # right boundary
# z[outflow]