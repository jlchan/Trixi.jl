
# using Pkg;  Pkg.activate(temp=true)
using OrdinaryDiffEq, DiffEqCallbacks
using Trixi
using LinearAlgebra

#################################################################################
# hack viscous terms into the rhs

using StaticArrays
using LinearAlgebra: mul!
using Trixi: DGMultiFluxDiffPeriodicFDSBP, BoundaryConditionPeriodic
using Trixi: @trixi_timeit, timer, @threaded
using Trixi: create_cache, rhs!

# set mach number
@inline get_mach() = .1

@inline function evaluate_viscous_coefficients(q, equations)
  rho, v1, v2, v3, T = q
  Re = 1600
  Pr = .72

  # use non-dimensional Sutherland's law:
  # mu_ref * (T/T_ref)^(3/2) * (Tref/Tref + S/Tref) / (T/Tref + S/Tref)
  # Here, S/Tref is computed from the table in https://www.cfd-online.com/Wiki/Sutherland%27s_law
  S_Tref = 110.4 / 273.15
  mu_T = T^(1.5) * (1 + S_Tref) / (T + S_Tref)

  mu = mu_T / Re
  lambda = -2/3 * mu
  kappa = equations.gamma / Pr
  return mu, lambda, kappa
end

@inline function temperature(u, equations::CompressibleEulerEquations3D)
  rho, rho_v1, rho_v2, rho_v3, rho_e = u

  @unpack gamma = equations

  Ma = get_mach()
  c_v = inv(gamma * (gamma - 1) * Ma^2)

  inv_rho = inv(rho)
  internal_energy = (rho_e - 0.5 * (rho_v1^2 + rho_v2^2 + rho_v3^2) * inv_rho) * inv_rho
  return internal_energy / c_v
end

# applies a differentiation operator to a 3D grid along the direction specified by `orientation`.
# keyword arguments `beta` specifies whether or not to accumulate
# keyword argument `alpha` is just there for consistency with SummationByPartsOperators.
function LinearAlgebra.mul!(out::AbstractArray{T, 3},
                            D::SummationByPartsOperators.AbstractDerivativeOperator,
                            orientation::Int, u::AbstractArray{T, 3};
                            alpha=true, beta=false) where {T}
  N = length(SummationByPartsOperators.grid(D)) # number of grid points
  if orientation==1 # compute du/dr
    @threaded for i in 1:N, j in 1:N
      mul!(view(out, :, i, j), D, view(u, :, i, j), alpha, beta)
    end
  elseif orientation==2 # compute du/ds
    @threaded for i in 1:N, j in 1:N
      mul!(view(out, i, :, j), D, view(u, i, :, j), alpha, beta)
    end
  else # if orientation==3 # compute du/dz
    @threaded for i in 1:N, j in 1:N
      mul!(view(out, i, j, :), D, view(u, i, j, :), alpha, beta)
    end
  end
end

function Trixi.create_cache(mesh::DGMultiMesh, equations::CompressibleEulerEquations3D,
                            dg::DGMultiFluxDiffPeriodicFDSBP, RealT, uEltype)

  rd = dg.basis
  md = mesh.md

  # for use with flux differencing schemes
  Qrst_skew = Trixi.compute_flux_differencing_SBP_matrices(dg)

  # storage for volume quadrature values, face quadrature values, flux values
  nvars = Trixi.nvariables(equations)
  u_values = Trixi.allocate_nested_array(uEltype, nvars, size(md.xq), dg)

  # dummy storage to allow for compatibility with DGMultiFluxDiffSBP if Threads.nthreads() > 2
  if Threads.nthreads() <= 2
    fluxdiff_local_threaded = [zeros(SVector{nvars, uEltype}, size(md.xq, 1))]
  else
    fluxdiff_local_threaded = nothing
  end

  # cache for viscous terms
  N = length(SummationByPartsOperators.grid(dg.basis.approximationType))
  v1, v2, v3, T = ntuple(_ -> reshape(similar(md.xq), N, N, N), 4)
  dv1dx, dv2dy, dv3dz, div_velocity = ntuple(_ -> reshape(similar(md.xq), N, N, N), 4)
  dv1dy_plus_dv2dx, dv1dz_plus_dv3dx, dv2dz_plus_dv3dy = ntuple(_ -> reshape(similar(md.xq), N, N, N), 3)
  tau_11, tau_12, tau_13, tau_22, tau_23, tau_33 = ntuple(_ -> reshape(similar(md.xq), N, N, N), 6)
  dTdx, dTdy, dTdz = ntuple(_ -> reshape(similar(md.xq), N, N, N), 3)
  kappa_tilde_x, kappa_tilde_y, kappa_tilde_z = ntuple(_ -> reshape(similar(md.xq), N, N, N), 3)
  rhs2, rhs3, rhs4, rhs5, rhs_heat = ntuple(_ -> reshape(similar(md.xq), N, N, N), 5)

  # for computing enstrophy
  velocity = SVector{3}(ntuple(_ -> reshape(similar(md.xq), N, N, N), 3))
  grad_velocity = SMatrix{3, 3}(ntuple(_ -> reshape(similar(md.xq), N, N, N), 9))

  return (; md, Qrst_skew, u_values, fluxdiff_local_threaded,
            invJ = inv.(md.J), inv_wq = inv.(rd.wq),
            v1, v2, v3, T,
            dv1dx, dv2dy, dv3dz, div_velocity,
            dv1dy_plus_dv2dx, dv1dz_plus_dv3dx, dv2dz_plus_dv3dy,
            tau_11, tau_12, tau_13, tau_22, tau_23, tau_33,
            dTdx, dTdy, dTdz,
            kappa_tilde_x, kappa_tilde_y, kappa_tilde_z,
            rhs2, rhs3, rhs4, rhs5, rhs_heat,
            velocity, grad_velocity)
end

calc_viscous_terms_naive!(du, u, semi, t) =
  calc_viscous_terms_naive!(du, u, Trixi.mesh_equations_solver_cache(semi)...)

function calc_viscous_terms_naive!(du, u, mesh, equations, dg, cache)
  @unpack v1, v2, v3, T = cache
  @unpack dv1dx, dv2dy, dv3dz, div_velocity = cache
  @unpack dv1dy_plus_dv2dx, dv1dz_plus_dv3dx, dv2dz_plus_dv3dy = cache
  @unpack tau_11, tau_12, tau_13, tau_22, tau_23, tau_33 = cache
  @unpack dTdx, dTdy, dTdz = cache
  @unpack kappa_tilde_x, kappa_tilde_y, kappa_tilde_z = cache

  # rename variables
  energy_flux_x, energy_flux_y, energy_flux_z = kappa_tilde_x, kappa_tilde_y, kappa_tilde_z

  @unpack rhs2, rhs3, rhs4, rhs5, rhs_heat = cache

  @unpack rxJ, syJ, tzJ = mesh.md
  @unpack invJ = cache

  D = dg.basis.approximationType # FDSBP operator

  @trixi_timeit timer() "compute primitive vars" begin
  @threaded for i in eachindex(v1)
    rho, rho_v1, rho_v2, rho_v3, rho_e = u[i]
    inv_rho = inv(rho)
    v1[i] = rho_v1 * inv_rho
    v2[i] = rho_v2 * inv_rho
    v3[i] = rho_v3 * inv_rho
    T[i]  = temperature(u[i], equations)
  end
  end

  @trixi_timeit timer() "reset rhs vectors" begin
  map(x -> fill!(x, zero(eltype(x))), (dv1dx, dv2dy, dv3dz, dv1dy_plus_dv2dx, dv1dz_plus_dv3dx, dv2dz_plus_dv3dy))
  map(x -> fill!(x, zero(eltype(x))), (dTdx, dTdy, dTdz))
  end

  @trixi_timeit timer() "compute velocity derivs" begin
  # compute velocity derivatives
  mul!(dv1dx, D, 1, v1)
  mul!(dv2dy, D, 2, v2)
  mul!(dv3dz, D, 3, v3)
  mul!(dv1dy_plus_dv2dx, D, 2, v1)
  mul!(dv1dy_plus_dv2dx, D, 1, v2, beta=true) # accumulates into the output
  mul!(dv1dz_plus_dv3dx, D, 3, v1)
  mul!(dv1dz_plus_dv3dx, D, 1, v3, beta=true) # accumulates into the output
  mul!(dv2dz_plus_dv3dy, D, 3, v2)
  mul!(dv2dz_plus_dv3dy, D, 2, v3, beta=true) # accumulates into the output

  # temperature derivatives
  mul!(dTdx, D, 1, T)
  mul!(dTdy, D, 2, T)
  mul!(dTdz, D, 3, T)
  end

  @trixi_timeit timer() "compute tau and kappatilde" begin
  @threaded for i in eachindex(tau_11)
    q = SVector{5}(u[i][1], v1[i], v2[i], v3[i], T[i])
    mu, lambda, kappa = evaluate_viscous_coefficients(q, equations)

    div_velocity = dv1dx[i] + dv2dy[i] + dv3dz[i]
    tau_11[i] = 2 * mu * dv1dx[i] + lambda * div_velocity
    tau_22[i] = 2 * mu * dv2dy[i] + lambda * div_velocity
    tau_33[i] = 2 * mu * dv3dz[i] + lambda * div_velocity
    tau_12[i] = mu * dv1dy_plus_dv2dx[i]
    tau_13[i] = mu * dv1dz_plus_dv3dx[i]
    tau_23[i] = mu * dv2dz_plus_dv3dy[i]

    energy_flux_x[i] = tau_11[i] * v1[i] + tau_12[i] * v2[i] + tau_13[i] * v3[i] + kappa * dTdx[i]
    energy_flux_y[i] = tau_12[i] * v1[i] + tau_22[i] * v2[i] + tau_23[i] * v3[i] + kappa * dTdy[i]
    energy_flux_z[i] = tau_13[i] * v1[i] + tau_23[i] * v2[i] + tau_33[i] * v3[i] + kappa * dTdz[i]
  end
  end

  @trixi_timeit timer() "compute second derivatives" begin
  # viscous momentum terms - ∑_j D_j * τ_ij
  fill!(rhs2, zero(eltype(rhs2)))
  fill!(rhs3, zero(eltype(rhs3)))
  fill!(rhs4, zero(eltype(rhs4)))
  fill!(rhs5, zero(eltype(rhs5)))
  mul!(rhs2, D, 1, tau_11)
  mul!(rhs2, D, 2, tau_12, beta=true)
  mul!(rhs2, D, 3, tau_13, beta=true)
  mul!(rhs3, D, 1, tau_12)
  mul!(rhs3, D, 2, tau_22, beta=true)
  mul!(rhs3, D, 3, tau_23, beta=true)
  mul!(rhs4, D, 1, tau_13)
  mul!(rhs4, D, 2, tau_23, beta=true)
  mul!(rhs4, D, 3, tau_33, beta=true)
  # compute energy rhs
  mul!(rhs5, D, 1, energy_flux_x)
  mul!(rhs5, D, 2, energy_flux_y, beta=true)
  mul!(rhs5, D, 3, energy_flux_z, beta=true)
  end

  @threaded for i in eachindex(du)
    du[i] = du[i] + SVector{5}(0.0, rhs2[i], rhs3[i], rhs4[i], rhs5[i])
  end

  return nothing
end

calc_viscous_terms_edoh!(du, u, semi, t) =
  calc_viscous_terms_edoh!(du, u, Trixi.mesh_equations_solver_cache(semi)...)

function calc_viscous_terms_edoh!(du, u, mesh, equations, dg, cache)
  @unpack v1, v2, v3, T = cache
  @unpack dv1dx, dv2dy, dv3dz, div_velocity = cache
  @unpack dv1dy_plus_dv2dx, dv1dz_plus_dv3dx, dv2dz_plus_dv3dy = cache
  @unpack tau_11, tau_12, tau_13, tau_22, tau_23, tau_33 = cache
  @unpack dTdx, dTdy, dTdz = cache
  @unpack kappa_tilde_x, kappa_tilde_y, kappa_tilde_z = cache
  @unpack rhs2, rhs3, rhs4, rhs5, rhs_heat = cache

  @unpack rxJ, syJ, tzJ = mesh.md
  @unpack invJ = cache

  D = dg.basis.approximationType # FDSBP operator

  @trixi_timeit timer() "compute primitive vars" begin
  @threaded for i in eachindex(v1)
    rho, rho_v1, rho_v2, rho_v3, rho_e = u[i]
    inv_rho = inv(rho)
    v1[i] = rho_v1 * inv_rho
    v2[i] = rho_v2 * inv_rho
    v3[i] = rho_v3 * inv_rho
    T[i]  = temperature(u[i], equations)
  end
  end

  @trixi_timeit timer() "reset rhs vectors" begin
    map(x -> fill!(x, zero(eltype(x))), (dv1dx, dv2dy, dv3dz, dv1dy_plus_dv2dx, dv1dz_plus_dv3dx, dv2dz_plus_dv3dy))
    map(x -> fill!(x, zero(eltype(x))), (dTdx, dTdy, dTdz))
  end

  @trixi_timeit timer() "compute velocity derivs" begin
  # compute velocity derivatives
  mul!(dv1dx, D, 1, v1)
  mul!(dv2dy, D, 2, v2)
  mul!(dv3dz, D, 3, v3)
  mul!(dv1dy_plus_dv2dx, D, 2, v1)
  mul!(dv1dy_plus_dv2dx, D, 1, v2, beta=true) # accumulates into the output
  mul!(dv1dz_plus_dv3dx, D, 3, v1)
  mul!(dv1dz_plus_dv3dx, D, 1, v3, beta=true) # accumulates into the output
  mul!(dv2dz_plus_dv3dy, D, 3, v2)
  mul!(dv2dz_plus_dv3dy, D, 2, v3, beta=true) # accumulates into the output

  # temperature derivatives
  mul!(dTdx, D, 1, T)
  mul!(dTdy, D, 2, T)
  mul!(dTdz, D, 3, T)
  end

  # @infiltrate

  @trixi_timeit timer() "compute tau and kappatilde" begin
  @threaded for i in eachindex(tau_11)
    q = SVector{5}(u[i][1], v1[i], v2[i], v3[i], T[i])
    mu, lambda, kappa = evaluate_viscous_coefficients(q, equations)

    div_velocity = dv1dx[i] + dv2dy[i] + dv3dz[i]
    tau_11[i] = 2 * mu * dv1dx[i] + lambda * div_velocity
    tau_22[i] = 2 * mu * dv2dy[i] + lambda * div_velocity
    tau_33[i] = 2 * mu * dv3dz[i] + lambda * div_velocity
    tau_12[i] = mu * dv1dy_plus_dv2dx[i]
    tau_13[i] = mu * dv1dz_plus_dv3dx[i]
    tau_23[i] = mu * dv2dz_plus_dv3dy[i]

    kappa_invT = kappa / T[i]
    kappa_tilde_x[i] = kappa_invT * dTdx[i]
    kappa_tilde_y[i] = kappa_invT * dTdy[i]
    kappa_tilde_z[i] = kappa_invT * dTdz[i]
  end
  end

  @trixi_timeit timer() "compute second derivatives" begin
  # compute T * D_j * kappa_tilde_j + kappa_tilde_j * D_j * T
  fill!(rhs_heat, zero(eltype(rhs_heat)))
  mul!(rhs_heat, D, 1, kappa_tilde_x)
  mul!(rhs_heat, D, 2, kappa_tilde_y, beta=true)
  mul!(rhs_heat, D, 3, kappa_tilde_z, beta=true)
  @threaded for i in eachindex(rhs_heat)
    rhs_heat[i] *= T[i] # at this point, rhs_heat[i] = ∑_j D_j * kappa_tilde_j
    rhs_heat[i] = rhs_heat[i] + kappa_tilde_x[i] * dTdx[i] +
                                kappa_tilde_y[i] * dTdy[i] +
                                kappa_tilde_z[i] * dTdz[i]
  end

  # viscous momentum terms - ∑_j D_j * τ_ij
  fill!(rhs2, zero(eltype(rhs2)))
  fill!(rhs3, zero(eltype(rhs3)))
  fill!(rhs4, zero(eltype(rhs4)))
  fill!(rhs5, zero(eltype(rhs5)))
  mul!(rhs2, D, 1, tau_11)
  mul!(rhs2, D, 2, tau_12, beta=true)
  mul!(rhs2, D, 3, tau_13, beta=true)
  mul!(rhs3, D, 1, tau_12)
  mul!(rhs3, D, 2, tau_22, beta=true)
  mul!(rhs3, D, 3, tau_23, beta=true)
  mul!(rhs4, D, 1, tau_13)
  mul!(rhs4, D, 2, tau_23, beta=true)
  mul!(rhs4, D, 3, tau_33, beta=true)
  end
  # split form rhs for energy:
  # rhs2 * v1 + tau_11 * dv1dx + tau_12 * dv1dy + tau_13 * dv1dz
  #   + rhs3 * v2 + tau_21 * dv2dx + tau_22 * dv2dy + tau_23 * dv2dz
  #   + rhs4 * v3 + tau_31 * dv3dx + tau_32 * dv3dy + tau_33 * dv3dz
  # = rhs2 * v1 + rhs3 * v2 + rhs4 * v3 + ...
  #   ... + tau_11 * dv1dx + tau_22 * dv2dy + tau_33 * dv3dz +
  #   ... + tau_12 * (dv1dy + dv2dx) + tau_13 * (dv1dz + dv3dx) + tau_23 * (dv2dz + dv3dx)

  @trixi_timeit timer() "accumulate rhs5 and du" begin
  @threaded for i in eachindex(rhs5)
    # ∑_j u_i * D_j * tau_ij + ∑_j tau_ij * D_j * u_i
    rhs5[i] = rhs2[i] * v1[i] + rhs3[i] * v2[i] + rhs4[i] * v3[i] +
              tau_11[i] * dv1dx[i] + tau_22[i] * dv2dy[i] + tau_33[i] * dv3dz[i] +
              tau_12[i] * dv1dy_plus_dv2dx[i] +
              tau_13[i] * dv1dz_plus_dv3dx[i] +
              tau_23[i] * dv2dz_plus_dv3dy[i] +
              rhs_heat[i]
  end

  @threaded for i in eachindex(du)
    du[i] = du[i] + SVector{5}(0.0, rhs2[i], rhs3[i], rhs4[i], rhs5[i])
  end
  end

  return nothing
end

rhs_naive!(du, u, semi, t) = rhs_naive!(du, u, t, Trixi.mesh_equations_solver_cache(semi)...)
rhs_split!(du, u, semi, t) = rhs_split!(du, u, t, Trixi.mesh_equations_solver_cache(semi)...)

function rhs_naive!(du, u, t, mesh, equations::CompressibleEulerEquations3D, dg::DGMulti, cache)

  @trixi_timeit timer() "reset ∂u/∂t" Trixi.reset_du!(du, dg, cache)

  @trixi_timeit timer() "volume integral" Trixi.calc_volume_integral!(
    du, u, mesh, Trixi.have_nonconservative_terms(equations), equations,
    dg.volume_integral, dg, cache)

  @trixi_timeit timer() "Jacobian" Trixi.invert_jacobian!(
    du, mesh, equations, dg, cache)

  @trixi_timeit timer() "viscous terms" calc_viscous_terms_naive!(du, u, mesh, equations, dg, cache)

  return nothing
end

function rhs_split!(du, u, t, mesh, equations::CompressibleEulerEquations3D, dg::DGMulti, cache)

  @trixi_timeit timer() "reset ∂u/∂t" Trixi.reset_du!(du, dg, cache)

  @trixi_timeit timer() "volume integral" Trixi.calc_volume_integral!(
    du, u, mesh, Trixi.have_nonconservative_terms(equations), equations,
    dg.volume_integral, dg, cache)

  @trixi_timeit timer() "Jacobian" Trixi.invert_jacobian!(
    du, mesh, equations, dg, cache)

  @trixi_timeit timer() "viscous terms" calc_viscous_terms_edoh!(du, u, mesh, equations, dg, cache)

  return nothing
end


###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

function initial_condition_taylor_green_vortex(x, t, equations::CompressibleEulerEquations3D)
  A  = 1.0 # magnitude of speed
  Ma = get_mach() # maximum Mach number

  rho = 1.0
  v1  =  A * sin(x[1]) * cos(x[2]) * cos(x[3])
  v2  = -A * cos(x[1]) * sin(x[2]) * cos(x[3])
  v3  = 0.0
  p   = (A / Ma)^2 * rho / equations.gamma # scaling to get Ms
  p   = p + 1.0/16.0 * A^2 * rho * (cos(2*x[1])*cos(2*x[3]) + 2*cos(2*x[2]) + 2*cos(2*x[1]) + cos(2*x[2])*cos(2*x[3]))

  return prim2cons(SVector(rho, v1, v2, v3, p), equations)
end

# 3D version of KHI
function initial_condition_kelvin_helmholtz_instability(x, t, equations::CompressibleEulerEquations3D)
  slope = 15
  B = tanh(slope * x[2] + 7.5) - tanh(slope * x[2] - 7.5)
  rho = 0.5 + 0.75 * B
  v1 = 0.5 * (B - 1)
  v2 = 0.1 * sin(2 * x[1]) * sin(2 * x[3])
  v3 = 0.1 * sin(2 * x[1]) * sin(2 * x[3])
  p  = 1.0
  return prim2cons(SVector(rho, v1, v2, v3, p), equations)
end

# initial_condition = initial_condition_kelvin_helmholtz_instability
initial_condition = initial_condition_taylor_green_vortex

volume_flux  = flux_ranocha
dg = DGMulti(element_type = Hex(),
             approximation_type = periodic_derivative_operator(
               derivative_order=1, accuracy_order=4, xmin=-pi, xmax=pi, N=32),
             surface_flux = nothing,
             volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

mesh = DGMultiMesh(dg, coordinates_min=(-pi, -pi, -pi),
                       coordinates_max=( pi,  pi,  pi))

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 20.0)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, uEltype=real(dg))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

function compute_tgv_quantities(u, t, integrator)
  @unpack mesh, cache = integrator.p
  @unpack velocity, grad_velocity = cache

  dg = integrator.p.solver
  D = dg.basis.approximationType # FDSBP operator

  # compute |D_i * u_j|
  @threaded for i in eachindex(u)
    rho, rho_v1, rho_v2, rho_v3, rho_e = u[i]
    velocity[1][i] = rho_v1 / rho
    velocity[2][i] = rho_v2 / rho
    velocity[3][i] = rho_v3 / rho
  end
  fill!.(grad_velocity, zero(eltype(u[1])))
  for i in 1:3, j in 1:3
    mul!(grad_velocity[i, j], D, i, velocity[j])
  end

  # integrate quantities
  ke, enstrophy, S = ntuple(_->0.0, 3)
  rho_squared, T_squared, rho_avg, T_avg = ntuple(_->0.0, 4)
  for i in eachindex(u)
    rho, rho_v1, rho_v2, rho_v3, rho_e = u[i]
    ke += mesh.md.wJq[i] * (rho_v1^2 + rho_v2^2 + rho_v3^2) / rho
    S += mesh.md.wJq[i] * Trixi.entropy(u[i], equations)

    T = temperature(u[i], equations)
    rho_squared += mesh.md.wJq[i] * rho^2
    rho_avg     += mesh.md.wJq[i] * rho
    T_squared   += mesh.md.wJq[i] * T^2
    T_avg       += mesh.md.wJq[i] * T
    for j in 1:3, k in 1:3
      enstrophy += mesh.md.wJq[i] * grad_velocity[j, k][i]^2
    end
  end
  rho_RMS = rho_squared - rho_avg^2
  T_RMS = T_squared - T_avg^2
  return ke, enstrophy, S, rho_RMS, T_RMS
end

tsave = LinRange(tspan..., 50)
saved_values_naive, saved_values_split, saved_values_entropy = ntuple(_ -> SavedValues(Float64, NTuple{5, Float64}), 3)

function create_callback_set(saved_values)
  return CallbackSet(summary_callback, analysis_callback, alive_callback,
                     SavingCallback(compute_tgv_quantities, saved_values, saveat=tsave))
end

###############################################################################
# run the simulation

# ode = semidiscretize(semi, tspan)
sol_naive = solve(ODEProblem(rhs_naive!, compute_coefficients(first(tspan), semi), tspan, semi),
                  RDPK3SpFSAL49(), abstol = 1.0e-7, reltol = 1.0e-7,
                  dt = 1e-4, save_everystep = false,
                  callback = create_callback_set(saved_values_split))

sol_split = solve(ODEProblem(rhs_split!, compute_coefficients(first(tspan), semi), tspan, semi),
                  RDPK3SpFSAL49(), abstol = 1.0e-7, reltol = 1.0e-7,
                  dt = 1e-4, save_everystep = false,
                  callback = create_callback_set(saved_values_split))

summary_callback() # print the timer summary
