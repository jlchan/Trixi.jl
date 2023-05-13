# by default, return an empty tuple for volume integral caches
function create_cache(mesh::DGMultiMesh{NDIMS}, equations,
                      volume_integral::VolumeIntegralShockCapturingHG,
                      dg::DGMultiFluxDiff{<:GaussSBP}, RealT, uEltype) where {NDIMS}
  element_ids_dg   = Int[]
  element_ids_dgfv = Int[]

  # build element to element (EToE) connectivity for smoothing of
  # shock capturing parameters.
  FToF = mesh.md.FToF # num_faces x num_elements matrix
  EToE = similar(FToF)
  for e in axes(FToF, 2)
    for f in axes(FToF, 1)
      neighbor_face_index = FToF[f, e]

      # reverse-engineer element index from face. Assumes all elements
      # have the same number of faces.
      neighbor_element_index = ((neighbor_face_index - 1) ÷ dg.basis.num_faces) + 1
      EToE[f, e] = neighbor_element_index
    end
  end

  # create sparse hybridized operators for low order scheme
  Qrst, E = StartUpDG.sparse_low_order_SBP_operators(dg.basis)
  Brst = map(n -> Diagonal(n .* dg.basis.wf), dg.basis.nrstJ)
  sparse_hybridized_SBP_operators = map((Q, B) -> [Q-Q' E'*B; -B*E zeros(size(B))], Qrst, Brst)

  # find joint sparsity pattern of the entire matrix. store as a transpose
  # for faster iteration through the rows.
  sparsity_pattern = transpose(sum(map(A -> abs.(A), sparse_hybridized_SBP_operators)) .> 100 * eps())

  return (; element_ids_dg, element_ids_dgfv, sparse_hybridized_SBP_operators, sparsity_pattern, EToE)
end


# this method is used when the indicator is constructed as for shock-capturing volume integrals
function create_cache(::Type{IndicatorHennemannGassner}, equations::AbstractEquations,
                      basis::RefElemData{NDIMS}) where NDIMS

  alpha = Vector{real(basis)}()
  alpha_tmp = similar(alpha)

  A = Vector{real(basis)}
  indicator_threaded  = [A(undef, nnodes(basis)) for _ in 1:Threads.nthreads()]
  modal_threaded      = [A(undef, nnodes(basis)) for _ in 1:Threads.nthreads()]

  # initialize inverse Vandermonde matrices at Gauss-Legendre nodes
  (; N) = basis
  gauss_node_coordinates_1D, _ = StartUpDG.gauss_quad(0, 0, N)
  VDM_1D = StartUpDG.vandermonde(Line(), N, gauss_node_coordinates_1D)
  inverse_vandermonde = SimpleKronecker(NDIMS, inv(VDM_1D))

  return (; alpha, alpha_tmp, indicator_threaded, modal_threaded, inverse_vandermonde)
end


function (indicator_hg::IndicatorHennemannGassner)(u, mesh::DGMultiMesh,
                                                   equations, dg::DGMulti{NDIMS}, cache;
                                                   kwargs...) where {NDIMS}
  (; alpha_max, alpha_min, alpha_smooth, variable) = indicator_hg
  (; alpha, alpha_tmp, indicator_threaded, modal_threaded, inverse_vandermonde) = indicator_hg.cache

  resize!(alpha, nelements(mesh, dg))
  if alpha_smooth
    resize!(alpha_tmp, nelements(mesh, dg))
  end

  # magic parameters
  threshold = 0.5 * 10^(-1.8 * (dg.basis.N + 1)^0.25)
  parameter_s = log((1 - 0.0001) / 0.0001)

  @threaded for element in eachelement(mesh, dg)
    indicator = indicator_threaded[Threads.threadid()]
    modal_ = modal_threaded[Threads.threadid()]

    # Calculate indicator variables at *Gauss* nodes.
    for i in eachnode(dg)
      indicator[i] = indicator_hg.variable(u[i, element], equations)
    end

    # multiply by invVDM::SimpleKronecker
    LinearAlgebra.mul!(modal_, inverse_vandermonde, indicator)

    # reshape into a matrix over each element
    modal = reshape(modal_, ntuple(_ -> dg.basis.N + 1, NDIMS))

    # Calculate total energies for all modes, without highest, without two highest
    total_energy = sum(x -> x^2, modal)

    # TODO: check if this allocates
    clip_1_ranges = ntuple(_ -> Base.OneTo(dg.basis.N), NDIMS)
    clip_2_ranges = ntuple(_ -> Base.OneTo(dg.basis.N - 1), NDIMS)
    total_energy_clip1 = sum(x -> x^2, view(modal, clip_1_ranges...))
    total_energy_clip2 = sum(x -> x^2, view(modal, clip_2_ranges...))

    # Calculate energy in higher modes
    if !(iszero(total_energy))
      energy_frac_1 = (total_energy - total_energy_clip1) / total_energy
    else
      energy_frac_1 = zero(total_energy)
    end
    if !(iszero(total_energy_clip1))
      energy_frac_2 = (total_energy_clip1 - total_energy_clip2) / total_energy_clip1
    else
      energy_frac_2 = zero(total_energy_clip1)
    end
    energy = max(energy_frac_1, energy_frac_2)

    alpha_element = 1 / (1 + exp(-parameter_s / threshold * (energy - threshold)))

    # Take care of the case close to pure DG
    if alpha_element < alpha_min
      alpha_element = zero(alpha_element)
    end

    # Take care of the case close to pure FV
    if alpha_element > 1 - alpha_min
      alpha_element = one(alpha_element)
    end

    # Clip the maximum amount of FV allowed
    alpha[element] = min(alpha_max, alpha_element)
  end

  # smooth element indices after they're all computed
  if alpha_smooth
    apply_smoothing!(mesh, alpha, alpha_tmp, dg, cache)
  end

  return alpha
end

# Diffuse alpha values by setting each alpha to at least 50% of neighboring elements' alpha
function apply_smoothing!(mesh::DGMultiMesh, alpha, alpha_tmp, dg::DGMulti, cache)

  # Copy alpha values such that smoothing is indpedenent of the element access order
  alpha_tmp .= alpha

  # smooth alpha with its neighboring value
  for element in eachelement(mesh, dg)
    for face in Base.OneTo(StartUpDG.num_faces(dg.basis.element_type))
      neighboring_element = cache.EToE[face, element]
      alpha_neighbor = alpha_tmp[neighboring_element]
      alpha[element]  = max(alpha[element], 0.5 * alpha_neighbor)
    end
  end

end

#     pure_and_blended_element_ids!(element_ids_dg, element_ids_dgfv, alpha, dg, cache)
#
# Given blending factors `alpha` and the solver `dg`, fill
# `element_ids_dg` with the IDs of elements using a pure DG scheme and
# `element_ids_dgfv` with the IDs of elements using a blended DG-FV scheme.
function pure_and_blended_element_ids!(element_ids_dg, element_ids_dgfv, alpha, mesh::DGMultiMesh, dg::DGMulti)
  empty!(element_ids_dg)
  empty!(element_ids_dgfv)

  for element in eachelement(mesh, dg)
    # Clip blending factor for values close to zero (-> pure DG)
    dg_only = isapprox(alpha[element], 0, atol=1e-12)
    if dg_only
      push!(element_ids_dg, element)
    else
      push!(element_ids_dgfv, element)
    end
  end

  return nothing
end

function calc_volume_integral!(du, u,
                               mesh::DGMultiMesh,
                               have_nonconservative_terms, equations,
                               volume_integral::VolumeIntegralShockCapturingHG,
                               dg::DGMultiFluxDiff, cache)

  @unpack element_ids_dg, element_ids_dgfv = cache
  @unpack volume_flux_dg, volume_flux_fv, indicator = volume_integral

  # Calculate blending factors α: u = u_DG * (1 - α) + u_FV * α
  alpha = @trixi_timeit timer() "blending factors" indicator(u, mesh, equations, dg, cache)

  # Determine element ids for DG-only and blended DG-FV volume integral
  pure_and_blended_element_ids!(element_ids_dg, element_ids_dgfv, alpha, mesh, dg)

  # Loop over pure DG elements
  @trixi_timeit timer() "pure DG" @threaded for idx_element in eachindex(element_ids_dg)
    element = element_ids_dg[idx_element]
    flux_differencing_kernel!(du, u, element, mesh, have_nonconservative_terms,
                              equations, volume_flux_dg, dg, cache)
  end

  # Loop over blended DG-FV elements
  @trixi_timeit timer() "blended DG-FV" @threaded for idx_element in eachindex(element_ids_dgfv)
    element = element_ids_dgfv[idx_element]
    alpha_element = alpha[element]

    # Calculate DG volume integral contribution
    flux_differencing_kernel!(du, u, element, mesh,
                              have_nonconservative_terms, equations,
                              volume_flux_dg, dg, cache, 1 - alpha_element)

    # # Calculate "FV" low order volume integral contribution
    low_order_flux_differencing_kernel!(du, u, element, mesh,
                                        have_nonconservative_terms, equations,
                                        volume_flux_fv, dg, cache, alpha_element)

    # blend them together via r_high * (1 - alpha) + r_low * (alpha)
  end

  return nothing
end

# computes an algebraic low order method with internal dissipation.
# TODO: implement for curved meshes
function low_order_flux_differencing_kernel!(du, u, element, mesh::DGMultiMesh,
                                             have_nonconservative_terms::False, equations,
                                             volume_flux_fv, dg::DGMultiFluxDiff,
                                             cache, alpha=true)

  (; sparsity_pattern, sparse_hybridized_SBP_operators) = cache

  # accumulates output from flux differencing
  rhs_local = cache.rhs_local_threaded[Threads.threadid()]
  fill!(rhs_local, zero(eltype(rhs_local)))

  # TODO: add flux differencing loop using non-symmetric `volume_flux_fv`
  u_local = view(cache.entropy_projected_u_values, :, element)

  A_base = parent(sparsity_pattern) # the adjoint of a SparseMatrixCSC is basically a SparseMatrixCSR
  row_ids, rows = axes(A, 2), rowvals(A_base)
  for i in row_ids
    u_i = u_local[i]
    du_i = rhs_local[i]
    for id in nzrange(A_base, i)
      j = rows[id]
      u_j = u_local[j]
      # TODO: scale n_ij by geometric terms as well!
      n_ij = SVector(getindex.(sparse_hybridized_SBP_operators, i, j))
      n_ij_norm = norm(n_ij)
      f_ij = volume_flux_fv(u_i, u_j, n_ij / n_ij_norm, equations)
      du_i = du_i + f_ij * n_ij_norm
    end
    rhs_local[i] = du_i
  end

  # Here, we exploit that under a Gauss nodal basis the structure of the projection
  # matrix `Ph = [diagm(1 ./ wq), projection_matrix_gauss_to_face]` such that
  # `Ph * [u; uf] = (u ./ wq) + projection_matrix_gauss_to_face * uf`.
  volume_indices = Base.OneTo(dg.basis.Nq)
  face_indices = (dg.basis.Nq + 1):(dg.basis.Nq + dg.basis.Nfq)
  local_volume_flux = view(rhs_local, volume_indices)
  local_face_flux = view(rhs_local, face_indices)

  # initialize rhs_volume_local = projection_matrix_gauss_to_face * local_face_flux
  rhs_volume_local = cache.rhs_volume_local_threaded[Threads.threadid()]
  apply_to_each_field(mul_by!(cache.projection_matrix_gauss_to_face), rhs_volume_local, local_face_flux)

  # accumulate volume contributions at Gauss nodes
  for i in eachindex(rhs_volume_local)
    du[i, element] = alpha * (rhs_volume_local[i] + local_volume_flux[i] * cache.inv_gauss_weights[i])
  end

end
