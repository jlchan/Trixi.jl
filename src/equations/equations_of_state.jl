# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    AbstractEquationOfState

The interface for an `AbstractEquationOfState` requires specifying
the following four functions: 
- `pressure(V, T, eos)`
- `energy_internal_specific(V, T, eos)`, the specific internal energy
- `entropy_specific(V, T, eos)`, the specific entropy
- `speed_of_sound(V, T, eos)`

where `eos = equations.equation_of_state`.
`entropy_specific` is required to calculate the mathematical entropy and entropy variables,
and `speed_of_sound` is required to calculate wavespeed estimates for e.g., [`FluxLaxFriedrichs`](@ref).
    
Additional functions can also be specialized to particular equations of state to improve 
efficiency. 
"""
abstract type AbstractEquationOfState end

include("equation_of_state_ideal_gas.jl")
include("equation_of_state_vdw.jl")
include("equation_of_state_peng_robinson.jl")

#######################################################
#
# Some general fallback routines are provided below
# 
#######################################################

function gibbs_free_energy(V, T, eos)
    s = entropy_specific(V, T, eos)
    p = pressure(V, T, eos)
    e_internal = energy_internal_specific(V, T, eos)
    h = e_internal + p * V
    return h - T * s
end

# compute c_v = de/dT
@inline function heat_capacity_constant_volume(V, T, eos::AbstractEquationOfState)
    return ForwardDiff.derivative(T -> energy_internal_specific(V, T, eos), T)
end

# this is used in [`flux_terashima_etal`](@ref) and [`flux_terashima_etal_central`](@ref)
function calc_pressure_derivatives(V, T, eos)
    dpdV_T = ForwardDiff.derivative(V -> pressure(V, T, eos), V)
    dpdT_V = ForwardDiff.derivative(T -> pressure(V, T, eos), T)
    return dpdT_V, dpdV_T
end

# relative tolerance, initial guess, and maximum number of iterations 
# for the Newton solver for temperature. 
eos_newton_tol(eos::AbstractEquationOfState) = 10 * eps()
eos_initial_temperature(V, e_internal, eos::AbstractEquationOfState) = 1
eos_initial_specific_volume(gibbs, T, eos::AbstractEquationOfState) = 1
eos_min_specific_volume(eos) = 0.0
eos_newton_maxiter(eos) = 20

"""
    temperature(V, e_internal, eos::AbstractEquationOfState; initial_T = 1.0, 
                tol = eos_newton_tol(eos), maxiter = 100)

Calculates the temperature as a function of specific volume `V` and internal energy `e`
by using Newton's method to determine `T` such that `energy_internal_specific(V, T, eos) = e`.
Note that the tolerance may need to be adjusted based on the specific equation of state. 
"""
function temperature(V, e_internal, eos::AbstractEquationOfState;
                     initial_T = eos_initial_temperature(V, e_internal, eos),
                     tol = eos_newton_tol(eos), maxiter = eos_newton_maxiter(eos))
    T = initial_T
    de = energy_internal_specific(V, T, eos) - e_internal
    iter = 1
    while abs(de) > tol * abs(e_internal) && iter < maxiter
        de = energy_internal_specific(V, T, eos) - e_internal

        # for thermodynamically admissible states, c_v = de_dT_V > 0, which should 
        # guarantee convergence of this iteration.
        de_dT_V = heat_capacity_constant_volume(V, T, eos)

        # guard against negative temperatures
        T = max(tol, T - de / de_dT_V)
        iter += 1
    end
    if iter == maxiter
        @warn "Solver in `temperature(V, T, eos)` did not converge within $maxiter iterations. " *
              "Final states: iter = $iter, V, e_internal = $V, $(e_internal) with de = $de"
    end
    return T
end

# determines specific volume from Gibbs free energy and temperature
function specific_volume(gibbs, T, eos::AbstractEquationOfState;
                         initial_V = eos_initial_specific_volume(gibbs, T, eos),
                         tol = eos_newton_tol(eos), maxiter = eos_newton_maxiter(eos))
    V = initial_V
    dgibbs = gibbs_free_energy(V, T, eos) - gibbs
    iter = 1
    while abs(dgibbs) > tol * abs(gibbs) && iter < maxiter

        # for thermodynamically admissible states, dgdV_T = V * dpdV at constant T
        _, dpdV_T = calc_pressure_derivatives(model, V, T)
        dgdV_T = V * dpdV_T

        # guard against negative temperatures
        V = max(eos_min_specific_volume(eos) + tol, V - dgibbs / dgdV_T)

        dgibbs = gibbs_free_energy(V, T, eos) - gibbs

        iter += 1
    end
    if iter == maxiter
        @warn "Solver in `specific_volume(gibbs, T, eos)` did not converge within $maxiter iterations. " *
              "Final states: iter = $iter, V, T = $V, $T with dgibbs = $dgibbs"
    end
    return T
end

# helper function used in [`flux_terashima_etal`](@ref) and [`flux_terashima_etal_central`](@ref)
@inline function drho_e_internal_drho_at_const_p(V, T, eos::AbstractEquationOfState)
    rho = inv(V)
    e_internal = energy_internal_specific(V, T, eos)

    dpdT_V, dpdV_T = calc_pressure_derivatives(V, T, eos)
    dpdrho_T = dpdV_T * (-V / rho) # V = inv(rho), so dVdrho = -1/rho^2 = -V^2. 
    de_dV_T = T * dpdT_V - pressure(V, T, eos)
    drho_e_internal_drho_T = e_internal + rho * de_dV_T * (-V / rho) # d(rho_e)/drho_|T = e + rho * de_dV|T * dVdrho

    c_v = heat_capacity_constant_volume(V, T, eos)
    return ((-rho * c_v) / (dpdT_V) * dpdrho_T + drho_e_internal_drho_T)
end
end # @muladd
