# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@inline function constraint_residual(u_node, threshold::Real, variable, equations)
    return variable(u_node, equations) - threshold
end

@inline function constraint_residual(u_node, thresholds::NTuple{N, <:Real},
                                     variables::NTuple{N, Any},
                                     equations) where {N}
    threshold = first(thresholds)
    remaining_thresholds = Base.tail(thresholds)
    variable = first(variables)
    remaining_variables = Base.tail(variables)

    residual = constraint_residual(u_node, threshold, variable, equations)
    remaining_residual = constraint_residual(u_node, remaining_thresholds,
                                             remaining_variables, equations)
    return min(residual, remaining_residual)
end

@inline function constraint_residual(u_node, thresholds::Tuple{}, variables::Tuple{},
                                     equations)
    return typemax(eltype(u_node))
end

@inline function filter_power(f, degree_sq)
    return f^degree_sq
end

@inline function filtered_node_vars_at_index(contrib, f, node_index, n_nodes,
                                             ::Val{N_VARS}) where {
                                                                   N_VARS}
    RealT = eltype(contrib)
    return SVector(ntuple(Val(N_VARS)) do variable_index
                       res = zero(RealT)
                       for k in 1:n_nodes
                           degree_sq = (k - 1)^2
                           res += filter_power(f, degree_sq) *
                                  contrib[variable_index, k, node_index]
                       end
                       res
                   end)
end

@inline function solve_filter_strength_illinois(g, f_invalid, tolerance, max_iterations)
    RealT = typeof(f_invalid)
    f_valid = zero(RealT)
    g_valid = g(f_valid)
    g_invalid = g(f_invalid)

    if g_valid < zero(RealT)
        error("element mean violates positivity constraints; " *
              "adaptive filter cannot recover a constraint-satisfying state")
    end

    side = 0

    for _ in 1:max_iterations
        if abs(f_invalid - f_valid) <= tolerance
            break
        end

        f_candidate = f_invalid -
                      g_invalid * (f_invalid - f_valid) / (g_invalid - g_valid)
        g_candidate = g(f_candidate)

        if abs(g_candidate) <= tolerance
            if g_candidate >= zero(RealT)
                f_valid = f_candidate
            end
            break
        end

        if g_candidate >= zero(RealT)
            if side == 1
                g_invalid *= 0.5f0
            end
            f_valid = f_candidate
            g_valid = g_candidate
            side = 1
        else
            if side == 2
                g_valid *= 0.5f0
            end
            f_invalid = f_candidate
            g_invalid = g_candidate
            side = 2
        end
    end

    return f_valid
end

function adaptive_filter_dzanic_witherden!(u, thresholds::NTuple{N, <:Real},
                                           variables::NTuple{N, Any},
                                           tolerance::Real, max_iterations::Int,
                                           mesh::AbstractMesh{1}, equations,
                                           dg::DGSEM, cache) where {N}
    @unpack inverse_vandermonde_legendre = dg.basis
    vandermonde = inv(inverse_vandermonde_legendre)

    n_nodes = nnodes(dg)
    n_vars = Val(nvariables(equations))

    @threaded for element in eachelement(dg, cache)
        needs_filtering = false
        for i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, element)
            if constraint_residual(u_node, thresholds, variables, equations) <
               zero(eltype(u))
                needs_filtering = true
                break
            end
        end
        needs_filtering || continue

        u_mean = compute_u_mean(u, element, mesh, equations, dg, cache)
        if constraint_residual(u_mean, thresholds, variables, equations) <
           zero(eltype(u))
            error("element mean violates positivity constraints; " *
                  "adaptive filter cannot recover a constraint-satisfying state")
        end

        contrib = zeros(eltype(u), nvariables(equations), n_nodes, n_nodes)
        modal = zeros(eltype(u), n_nodes)
        u_nodal = zeros(eltype(u), n_nodes)

        for v in eachvariable(equations)
            for i in eachnode(dg)
                u_nodal[i] = u[v, i, element]
            end
            multiply_scalar_dimensionwise!(modal, inverse_vandermonde_legendre, u_nodal)
            for k in 1:n_nodes
                for i in eachnode(dg)
                    contrib[v, k, i] = vandermonde[i, k] * modal[k]
                end
            end
        end

        f_upper = one(eltype(u))

        for i in eachnode(dg)
            u_filtered = filtered_node_vars_at_index(contrib, f_upper, i, n_nodes,
                                                     n_vars)
            if constraint_residual(u_filtered, thresholds, variables,
                                   equations) >= zero(eltype(u))
                continue
            end

            function g(f)
                u_node = filtered_node_vars_at_index(contrib, f, i, n_nodes, n_vars)
                return constraint_residual(u_node, thresholds, variables, equations)
            end

            f_upper = solve_filter_strength_illinois(g, f_upper, tolerance,
                                                     max_iterations)
        end

        for i in eachnode(dg)
            u_filtered = filtered_node_vars_at_index(contrib, f_upper, i, n_nodes,
                                                     n_vars)
            set_node_vars!(u, u_filtered, equations, dg, i, element)
        end
    end

    return nothing
end
end # @muladd
