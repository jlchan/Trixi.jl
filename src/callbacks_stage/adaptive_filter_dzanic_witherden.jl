# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    PositivityPreservingFilterDzanicWitherden(; thresholds, variables,
                                                tolerance = 1.0e-8,
                                                max_iterations_rootfinding = 20)

Positivity-preserving adaptive modal filter of
- Dzanic, Witherden (2022)
  Positivity-preserving entropy-based adaptive filtering for discontinuous
  spectral element methods
  [doi: 10.1016/j.jcp.2022.111501](https://doi.org/10.1016/j.jcp.2022.111501)

The filter is applied to all scalar `variables` in their given order using the
associated `thresholds` to determine the minimal acceptable values. A single
element-wise filter strength is computed such that all constraints are satisfied
at all nodes. If the element mean violates a threshold, an error is thrown.

The parameters `tolerance` and `max_iterations_rootfinding` control the Illinois 
modified regula falsi root solver used to determine the filter strength.
"""
struct PositivityPreservingFilterDzanicWitherden{N,
                                                 Thresholds <:
                                                 NTuple{N, <:Real},
                                                 Variables <: NTuple{N, Any},
                                                 RealT <: Real}
    thresholds::Thresholds
    variables::Variables
    tolerance::RealT
    max_iterations_rootfinding::Int
end

function PositivityPreservingFilterDzanicWitherden(; thresholds, variables,
                                                     tolerance = 1.0e-8,
                                                     max_iterations_rootfinding = 20)
    return PositivityPreservingFilterDzanicWitherden(thresholds, variables,
                                                     tolerance,
                                                     max_iterations_rootfinding)
end

function (limiter!::PositivityPreservingFilterDzanicWitherden)(u_ode,
                                                               integrator,
                                                               semi::AbstractSemidiscretization,
                                                               t)
    u = wrap_array(u_ode, semi)
    @trixi_timeit timer() "positivity-preserving adaptive filter" begin
        adaptive_filter_dzanic_witherden!(u, limiter!.thresholds, limiter!.variables,
                                          limiter!.tolerance, limiter!.max_iterations_rootfinding,
                                          mesh_equations_solver_cache(semi)...)
    end

    return nothing
end

include("adaptive_filter_dzanic_witherden_dg1d.jl")
end # @muladd
