
"""
    DGMulti(approximation_type::AbstractDerivativeOperator;
            element_type::AbstractElemShape,
            surface_flux=flux_central,
            surface_integral=SurfaceIntegralWeakForm(surface_flux),
            volume_integral=VolumeIntegralWeakForm(),
            kwargs...)

Create a summation by parts (SBP) discretization on the given `element_type`
using a tensor product structure.

For more info, see the documentations of
[StartUpDG.jl](https://jlchan.github.io/StartUpDG.jl/dev/)
and
[SummationByPartsOperators.jl](https://ranocha.de/SummationByPartsOperators.jl/stable/).
"""
function DGMulti(approximation_type::AbstractDerivativeOperator;
                 element_type::AbstractElemShape,
                 surface_flux=flux_central,
                 surface_integral=SurfaceIntegralWeakForm(surface_flux),
                 volume_integral=VolumeIntegralWeakForm(),
                 kwargs...)

  rd = RefElemData(element_type, approximation_type; kwargs...)
  return DG(rd, nothing #= mortar =#, surface_integral, volume_integral)
end

function DGMulti(element_type::AbstractElemShape,
                 approximation_type::AbstractDerivativeOperator,
                 volume_integral,
                 surface_integral;
                 kwargs...)

  DGMulti(approximation_type, element_type=element_type,
          surface_integral=surface_integral, volume_integral=volume_integral)
end



function construct_1d_operators(D::AbstractDerivativeOperator, tol)
  # StartUpDG assumes nodes from -1 to +1
  nodes_1d = collect(grid(D))
  M = SummationByPartsOperators.mass_matrix(D)
  if M isa UniformScaling
    weights_1d = M * ones(Bool, length(nodes_1d))
  else
    weights_1d = diag(M)
  end
  xmin, xmax = extrema(nodes_1d)
  factor = 2 / (xmax - xmin)
  @. nodes_1d = factor * (nodes_1d - xmin) - 1
  @. weights_1d = factor * weights_1d

  D_1d = droptol!(inv(factor) * sparse(D), tol)
  I_1d = Diagonal(ones(Bool, length(nodes_1d)))

  return nodes_1d, weights_1d, D_1d, I_1d
end


function StartUpDG.RefElemData(element_type::Line,
                               D::AbstractDerivativeOperator;
                               tol = 100*eps())

  approximation_type = D
  N = SummationByPartsOperators.accuracy_order(D) # kind of polynomial degree

  # 1D operators
  nodes_1d, weights_1d, D_1d = construct_1d_operators(D, tol)

  # volume
  rq = r = nodes_1d
  wq = weights_1d
  Dr = D_1d
  M = Diagonal(wq)
  Pq = LinearAlgebra.I
  Vq = LinearAlgebra.I

  VDM = nothing # unused generalized Vandermonde matrix

  rst = (r,)
  rstq = (rq,)
  Drst = (Dr,)

  # face
  face_vertices = StartUpDG.face_vertices(element_type)
  face_mask = [1, length(nodes_1d)]

  rf = [-1.0; 1.0]
  nrJ = [-1.0; 1.0]
  wf = [1.0; 1.0]
  Vf = sparse([1, 2], [1, length(nodes_1d)], [1.0, 1.0])
  LIFT = Diagonal(wq) \ (Vf' * Diagonal(wf))

  rstf = (rf,)
  nrstJ = (nrJ,)

  # low order interpolation nodes
  r1 = StartUpDG.nodes(element_type, 1)
  V1 = StartUpDG.vandermonde(element_type, 1, r) / StartUpDG.vandermonde(element_type, 1, r1)

  return RefElemData(
    element_type, approximation_type, N,
    face_vertices, V1,
    rst, VDM, face_mask,
    N, rst, LinearAlgebra.I, # plotting
    rstq, wq, Vq, # quadrature
    rstf, wf, Vf, nrstJ, # faces
    M, Pq, Drst, LIFT)
end


function StartUpDG.RefElemData(element_type::Quad,
                               D::AbstractDerivativeOperator;
                               tol = 100*eps())

  approximation_type = D
  N = SummationByPartsOperators.accuracy_order(D) # kind of polynomial degree

  # 1D operators
  nodes_1d, weights_1d, D_1d, I_1d = construct_1d_operators(D, tol)

  # volume
  s, r = vec.(StartUpDG.NodesAndModes.meshgrid(nodes_1d)) # this is to match
                                                          # ordering of nrstJ
  rq = r; sq = s
  wr, ws = vec.(StartUpDG.NodesAndModes.meshgrid(weights_1d))
  wq = wr .* ws
  Dr = kron(I_1d, D_1d)
  Ds = kron(D_1d, I_1d)
  M = Diagonal(wq)
  Pq = LinearAlgebra.I
  Vq = LinearAlgebra.I

  VDM = nothing # unused generalized Vandermonde matrix

  rst = (r, s)
  rstq = (rq, sq)
  Drst = (Dr, Ds)

  # face
  face_vertices = StartUpDG.face_vertices(element_type)
  face_mask = vcat(StartUpDG.find_face_nodes(element_type, r, s)...)

  rf, sf, wf, nrJ, nsJ = StartUpDG.init_face_data(element_type,
    quad_rule_face=(nodes_1d, weights_1d))
  Vf = sparse(eachindex(face_mask), face_mask, ones(Bool, length(face_mask)))
  LIFT = Diagonal(wq) \ (Vf' * Diagonal(wf))

  rstf = (rf, sf)
  nrstJ = (nrJ, nsJ)

  # low order interpolation nodes
  r1, s1 = StartUpDG.nodes(element_type, 1)
  V1 = StartUpDG.vandermonde(element_type, 1, r, s) / StartUpDG.vandermonde(element_type, 1, r1, s1)

  return RefElemData(
    element_type, approximation_type, N,
    face_vertices, V1,
    rst, VDM, face_mask,
    N, rst, LinearAlgebra.I, # plotting
    rstq, wq, Vq, # quadrature
    rstf, wf, Vf, nrstJ, # faces
    M, Pq, Drst, LIFT)
end


function StartUpDG.RefElemData(element_type::Hex,
                               D::AbstractDerivativeOperator;
                               tol = 100*eps())

  approximation_type = D
  N = SummationByPartsOperators.accuracy_order(D) # kind of polynomial degree

  # 1D operators
  nodes_1d, weights_1d, D_1d, I_1d = construct_1d_operators(D, tol)

  # volume
  # to match ordering of nrstJ
  s, r, t = vec.(StartUpDG.NodesAndModes.meshgrid(nodes_1d, nodes_1d, nodes_1d))
  rq = r; sq = s; tq = t
  wr, ws, wt = vec.(StartUpDG.NodesAndModes.meshgrid(weights_1d, weights_1d, weights_1d))
  wq = wr .* ws .* wt
  Dr = kron(I_1d, I_1d, D_1d)
  Ds = kron(I_1d, D_1d, I_1d)
  Dt = kron(D_1d, I_1d, I_1d)
  M = Diagonal(wq)
  Pq = LinearAlgebra.I
  Vq = LinearAlgebra.I

  VDM = nothing # unused generalized Vandermonde matrix

  rst = (r, s, t)
  rstq = (rq, sq, tq)
  Drst = (Dr, Ds, Dt)

  # face
  face_vertices = StartUpDG.face_vertices(element_type)
  face_mask = vcat(StartUpDG.find_face_nodes(element_type, r, s, t)...)

  rf, sf, tf, wf, nrJ, nsJ, ntJ = let
    rf, sf = vec.(StartUpDG.NodesAndModes.meshgrid(nodes_1d, nodes_1d))
    wr, ws = vec.(StartUpDG.NodesAndModes.meshgrid(weights_1d, weights_1d))
    wf = wr .* ws
    StartUpDG.init_face_data(element_type, quad_rule_face=(rf, sf, wf))
  end
  Vf = sparse(eachindex(face_mask), face_mask, ones(Bool, length(face_mask)))
  LIFT = Diagonal(wq) \ (Vf' * Diagonal(wf))

  rstf = (rf, sf, tf)
  nrstJ = (nrJ, nsJ, ntJ)

  # low order interpolation nodes
  r1, s1, t1 = StartUpDG.nodes(element_type, 1)
  V1 = StartUpDG.vandermonde(element_type, 1, r, s, t) / StartUpDG.vandermonde(element_type, 1, r1, s1, t1)

  return RefElemData(
    element_type, approximation_type, N,
    face_vertices, V1,
    rst, VDM, face_mask,
    N, rst, LinearAlgebra.I, # plotting
    rstq, wq, Vq, # quadrature
    rstf, wf, Vf, nrstJ, # faces
    M, Pq, Drst, LIFT)
end


function Base.show(io::IO, mime::MIME"text/plain", rd::RefElemData{NDIMS, ElementType, ApproximationType}) where {NDIMS, ElementType<:StartUpDG.AbstractElemShape, ApproximationType<:AbstractDerivativeOperator}
  @nospecialize rd
  print(io, "RefElemData for an approximation using an ")
  show(IOContext(io, :compact => true), rd.approximationType)
  print(io, " on $(rd.elementType) element")
end

function Base.show(io::IO, rd::RefElemData{NDIMS, ElementType, ApproximationType}) where {NDIMS, ElementType<:StartUpDG.AbstractElemShape, ApproximationType<:AbstractDerivativeOperator}
  @nospecialize rd
  print(io, "RefElemData{", summary(rd.approximationType), ", ", rd.elementType, "}")
end

const DGMultiPeriodicFDSBP{ApproxType, ElemType} =
  DGMulti{NDIMS, ElemType, ApproxType, SurfaceIntegral, VolumeIntegral} where {NDIMS, ElemType, ApproxType<:SummationByPartsOperators.PeriodicDerivativeOperator, SurfaceIntegral, VolumeIntegral}

"""
  CartesianMesh(dg::DGMultiPeriodicFDSBP{NDIMS})

Constructs a single-element `mesh::AbstractMeshData` for a single periodic element given
a DGMulti with approximation_type <: SummationByPartsOperators.AbstractDerivativeOperator.
"""
function CartesianMesh(dg::DGMultiPeriodicFDSBP{NDIMS}) where {NDIMS}

  rd = dg.basis

  e = ones(size(rd.r))
  z = zero.(e)

  VXYZ = ntuple(_ -> nothing, NDIMS)
  EToV = NaN # StartUpDG.jl uses size(EToV, 1) for the number of elements, this lets us reuse that.
  FToF = nothing

  xyz = xyzq = rd.rst # TODO: extend to mapped domains
  xyzf = ntuple(_ -> nothing, NDIMS)
  wJq = diag(rd.M)

  # arrays of connectivity indices between face nodes
  mapM = mapP = mapB = nothing

  # volume geofacs Gij = dx_i/dxhat_j
  rstxyzJ = @SMatrix [e z; z e] # TODO: extend to mapped domains
  J = e

  # surface geofacs
  nxyzJ = ntuple(_ -> nothing, NDIMS)
  Jf = nothing

  is_periodic = ntuple(_->true, NDIMS)

  md = MeshData(VXYZ, EToV, FToF, xyz, xyzf, xyzq, wJq,
                mapM, mapP, mapB, rstxyzJ, J, nxyzJ, Jf,
                is_periodic)

  boundary_faces = []
  return VertexMappedMesh{NDIMS, rd.elementType, typeof(md), 0, typeof(boundary_faces)}(md, boundary_faces)
end

# `estimate_h` uses that `Jf / J = O(h^{NDIMS-1}) / O(h^{NDIMS}) = O(h)`. However,
# since we do not initialize `Jf` here, we specialize `estimate_h` based on the grid
# provided by SummationByPartsOperators.jl.
function StartUpDG.estimate_h(e, rd::RefElemData{NDIMS, ElementType, ApproximationType}, md)  where {NDIMS, ElementType, ApproximationType<:SummationByPartsOperators.PeriodicDerivativeOperator}
  D = rd.approximationType
  grid = SummationByPartsOperators.grid(D)
  return grid[2] - grid[1]
end

# do nothing for interface terms if using a periodic operator
prolong2interfaces!(cache, u, mesh, equations, surface_integral, dg::DGMultiPeriodicFDSBP) = nothing
calc_interface_flux!(cache, surface_integral, mesh, have_nonconservative_terms::Val{false},
                     equations, dg::DGMultiPeriodicFDSBP) = nothing
calc_surface_integral!(du, u, surface_integral, mesh, equations,
                       dg::DGMultiPeriodicFDSBP, cache) = nothing
