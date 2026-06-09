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
    variable = first(variables)
    remaining_thresholds = Base.tail(thresholds)
    remaining_variables = Base.tail(variables)

    residual = constraint_residual(u_node, threshold, variable, equations)
    remaining_residual = constraint_residual(u_node, remaining_thresholds,
                                             remaining_variables, equations)
    
    # if any constraint_residual is negative, then taking the minimum will detect the violation.
    return min(residual, remaining_residual)
end

# terminate recursion
@inline function constraint_residual(u_node, thresholds::Tuple{}, variables::Tuple{},
                                     equations)
    return typemax(eltype(u_node))
end

@inline function satisfies_constraints(u_node, thresholds, variables, equations;
                                       tolerance = zero(eltype(u_node)))
    return constraint_residual(u_node, thresholds, variables, equations) >= -tolerance
end

@inline function filter_constraint_residual(modal_contributions, f, node_index, n_nodes,
                                            n_vars, thresholds, variables, equations)
    u_node = filtered_solution_at_node(modal_contributions, f, node_index, equations)
    return constraint_residual(u_node, thresholds, variables, equations)
end

@inline function filtered_solution_at_node(modal_contributions, f, node_index, 
                                           equations::AbstractEquations{NDIMS, NVARS}) where {NDIMS, NVARS}
    RealT = eltype(modal_contributions)
    return SVector(ntuple(Val(NVARS)) do variable_index
                       res = zero(RealT)
                       for k in axes(modal_contributions, 2)
                           degree_sq = (k - 1)^2
                           res += f^degree_sq *
                                  modal_contributions[variable_index, k, node_index]

                        #    # this is equivalent to Zhang-Shu limiting
                        #    if k == 1
                        #       # first mode is not filtered
                        #       res += modal_contributions[variable_index, k, node_index]
                        #    else
                        #       res += f * modal_contributions[variable_index, k, node_index]
                        #    end
                       end
                       res
                   end)
end

@inline function solve_for_filter_strength_illinois(modal_contributions, node_index,
                                                    n_nodes, n_vars, thresholds,
                                                    variables, equations, 
                                                    f_inadmissible,
                                                    tolerance, 
                                                    max_iterations_rootfinding)

    return 0.0
                                                        
    RealT = typeof(f_inadmissible)
    f_admissible = zero(RealT)
    g_valid = filter_constraint_residual(modal_contributions, f_admissible, node_index,
                                         n_nodes, n_vars, thresholds, variables,
                                         equations)
    g_invalid = filter_constraint_residual(modal_contributions, f_inadmissible,
                                           node_index,
                                           n_nodes, n_vars, thresholds, variables,
                                           equations)

    # if g_valid < zero(RealT)
    #     error("element mean violates positivity constraints; " *
    #           "adaptive filter cannot recover a constraint-satisfying state")
    # end

    side = 0

    for _ in 1:max_iterations_rootfinding
        if abs(f_inadmissible - f_admissible) <= tolerance
            break
        end

        f_candidate = f_inadmissible -
                      g_invalid * (f_inadmissible - f_admissible) /
                      (g_invalid - g_valid)
        g_candidate = filter_constraint_residual(modal_contributions, f_candidate,
                                                 node_index, n_nodes, n_vars,
                                                 thresholds, variables, equations)

        if abs(g_candidate) <= tolerance
            # if the residual is non-negative, then the filtered 
            # solution satisfies the constraints.
            if g_candidate >= zero(RealT)
                f_admissible = f_candidate
            end
            break
        end

        if g_candidate >= zero(RealT)
            if side == 1
                g_invalid *= 0.5f0
            end
            f_admissible = f_candidate
            g_valid = g_candidate
            side = 1
        else
            if side == 2
                g_valid *= 0.5f0
            end
            f_inadmissible = f_candidate
            g_invalid = g_candidate
            side = 2
        end
    end

    return f_admissible
end

function adaptive_filter_dzanic_witherden!(u, thresholds, variables,
                                           tolerance, max_iterations_rootfinding,
                                           mesh::AbstractMesh{1}, equations,
                                           dg::DGSEM, cache) 
    (; inverse_vandermonde_legendre) = dg.basis
    vandermonde = inv(inverse_vandermonde_legendre)

    n_nodes = nnodes(dg)
    n_vars = Val(nvariables(equations))

    @threaded for element in eachelement(dg, cache)
        violates_positivity = false
        for i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, element)
            if !satisfies_constraints(u_node, thresholds, variables, equations)
                violates_positivity = true
                break
            end
        end
        violates_positivity || continue

        u_mean = compute_u_mean(u, element, mesh, equations, dg, cache)
        if !satisfies_constraints(u_mean, thresholds, variables, equations)
            @warn "cell average = $(u_mean) violates positivity constraints; " *
                  "adaptive filter cannot recover a constraint-satisfying state"
        end

        # precompute modal_contributions[:,i] = vandermonde[:,i] * (vandermonde \ u)
        # --> modal_contributions * [f^(2k) for k in 0:n_nodes-1] returns the filtered
        # solution at the nodes
        modal_contributions = zeros(eltype(u), nvariables(equations), n_nodes, n_nodes)
        modal = zeros(eltype(u), n_nodes)
        u_nodal = zeros(eltype(u), n_nodes)

        for v in eachvariable(equations)
            for i in eachnode(dg)
                u_nodal[i] = u[v, i, element]
            end
            multiply_scalar_dimensionwise!(modal, inverse_vandermonde_legendre, u_nodal)
            for ii in 1:n_nodes
                for i in eachnode(dg)
                    modal_contributions[v, ii, i] = vandermonde[i, ii] * modal[ii]
                end
            end
        end

        # the filter is applied via ∑ f^(2k) û_k, where û_k are the modal coefficients
        # we initialize f = 1 and solve for a value of f that satisfies the constraints.
        f_upper = one(eltype(u))
        for i in eachnode(dg)
            
            u_filtered = filtered_solution_at_node(modal_contributions, f_upper, i,
                                                     equations)
            satisfies_constraints(u_filtered, thresholds, variables, equations) &&
                continue

            # solve for filter strength f that satisfies the constraints
            # note that f_upper is passed in as a new upper bound for the 
            # bracketing root finding algorithm after each node.              
            f_upper = solve_for_filter_strength_illinois(modal_contributions, i, n_nodes,
                                                         n_vars, thresholds, variables,
                                                         equations, f_upper, tolerance,
                                                         max_iterations_rootfinding)
        end

        # apply the filter to the solution
        for i in eachnode(dg)
            u_filtered = filtered_solution_at_node(modal_contributions, f_upper, i,
                                                     equations)
            set_node_vars!(u, u_filtered, equations, dg, i, element)
        end
    end

    return nothing
end
end # @muladd
