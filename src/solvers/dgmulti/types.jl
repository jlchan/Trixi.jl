# We include type aliases outside of @muladd blocks to avoid Revise conflicts.
# See https://github.com/trixi-framework/Trixi.jl/issues/801.

# `DGMulti` refers to both multiple DG types (polynomial/SBP, simplices/quads/hexes) as well as
# the use of multi-dimensional operators in the solver.
const DGMulti{NDIMS, ElemType, ApproxType, SurfaceIntegral, VolumeIntegral} =
  DG{<:RefElemData{NDIMS, ElemType, ApproxType}, Mortar, SurfaceIntegral, VolumeIntegral} where {Mortar}

# Type aliases. The first parameter is `ApproxType` since it is more commonly used for dispatch.
const DGMultiWeakForm{ApproxType, ElemType} =
DGMulti{NDIMS, ElemType, ApproxType, <:SurfaceIntegralWeakForm, <:VolumeIntegralWeakForm} where {NDIMS}

const DGMultiFluxDiff{ApproxType, ElemType} =
DGMulti{NDIMS, ElemType, ApproxType, <:AbstractSurfaceIntegral, <:VolumeIntegralFluxDifferencing} where {NDIMS}

  # By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

# these are necessary for pretty printing
polydeg(dg::DGMulti) = dg.basis.N
Base.summary(io::IO, dg::DG) where {DG <: DGMulti} = print(io, "DGMulti(polydeg=$(polydeg(dg)))")
Base.real(rd::RefElemData{NDIMS, Elem, ApproxType, Nfaces, RealT}) where {NDIMS, Elem, ApproxType, Nfaces, RealT} = RealT

"""
    DGMulti(; polydeg::Integer,
              element_type::AbstractElemShape,
              approximation_type=Polynomial(),
              surface_flux=flux_central,
              surface_integral=SurfaceIntegralWeakForm(surface_flux),
              volume_integral=VolumeIntegralWeakForm(),
              RefElemData_kwargs...)

Create a discontinuous Galerkin method which uses
- approximations of polynomial degree `polydeg`
- element type `element_type` (`Tri()`, `Quad()`, `Tet()`, and `Hex()` currently supported)

Optional:
- `approximation_type` (default is `Polynomial()`; `SBP()` also supported for `Tri()`, `Quad()`,
  and `Hex()` element types).
- `RefElemData_kwargs` are additional keyword arguments for `RefElemData`, such as `quad_rule_vol`.
  For more info, see the [StartUpDG.jl docs](https://jlchan.github.io/StartUpDG.jl/dev/).
"""
function DGMulti(; polydeg::Integer,
                   element_type::AbstractElemShape,
                   approximation_type=Polynomial(),
                   surface_flux=flux_central,
                   surface_integral=SurfaceIntegralWeakForm(surface_flux),
                   volume_integral=VolumeIntegralWeakForm(),
                   kwargs...)

  # call dispatchable constructor
  DGMulti(element_type, approximation_type, volume_integral, surface_integral;
          polydeg=polydeg, surface_flux=surface_flux, kwargs...)
end

# dispatchable constructor for DGMulti to allow for specialization
function DGMulti(element_type::AbstractElemShape,
                 approximation_type,
                 volume_integral,
                 surface_integral;
                 polydeg::Integer,
                 surface_flux,
                 kwargs...)

  rd = RefElemData(element_type, approximation_type, polydeg; kwargs...)
  return DG(rd, nothing #= mortar =#, surface_integral, volume_integral)
end

# now that DGMulti is defined, we can define constructors for VertexMappedMesh which use dg::DGMulti
"""
    VertexMappedMesh(vertex_coordinates, EToV, dg::DGMulti;
                     is_on_boundary = nothing,
                     is_periodic::NTuple{NDIMS, Bool} = ntuple(_->false, NDIMS)) where {NDIMS, Tv}

Constructor which uses `dg::DGMulti` instead of `rd::RefElemData`.
"""
VertexMappedMesh(vertex_coordinates, EToV, dg::DGMulti; kwargs...) =
  VertexMappedMesh(vertex_coordinates, EToV, dg.basis; kwargs...)

"""
    VertexMappedMesh(triangulateIO, dg::DGMulti, boundary_dict::Dict{Symbol, Int})

Constructor which uses `dg::DGMulti` instead of `rd::RefElemData`.
"""
VertexMappedMesh(triangulateIO, dg::DGMulti, boundary_dict::Dict{Symbol, Int}) =
  VertexMappedMesh(triangulateIO, dg.basis, boundary_dict)

# Todo: DGMulti. Add traits for dispatch on affine/curved meshes here.

# Matrix type for lazy construction of physical differentiation matrices
# Constructs a lazy linear combination of B = ∑_i coeffs[i] * A[i]
struct LazyMatrixLinearCombo{Tcoeffs, N, Tv, TA <: AbstractMatrix{Tv}} <: AbstractMatrix{Tv}
  matrices::NTuple{N, TA}
  coeffs::NTuple{N, Tcoeffs}
  function LazyMatrixLinearCombo(matrices, coeffs)
    @assert all(matrix -> size(matrix) == size(first(matrices)), matrices)
    new{typeof(first(coeffs)), length(matrices), eltype(first(matrices)), typeof(first(matrices))}(matrices, coeffs)
  end
end
Base.eltype(A::LazyMatrixLinearCombo) = eltype(first(A.matrices))
Base.IndexStyle(A::LazyMatrixLinearCombo) = IndexCartesian()
Base.size(A::LazyMatrixLinearCombo) = size(first(A.matrices))

@inline function Base.getindex(A::LazyMatrixLinearCombo{<:Real, N}, i, j) where {N}
  val = zero(eltype(A))
  for k in Base.OneTo(N)
    val = val + A.coeffs[k] * getindex(A.matrices[k], i, j)
  end
  return val
end

"""
    SurfaceIntegralHybrid(surface_flux=flux_central, surface_dissipation=nothing)

This evaluates a hybrid surface flux, which evaluates the usual surface flux at an interface,
as well as a surface dissipation term at a set of "hybrid" interface nodes.
"""
struct SurfaceIntegralHybrid{SurfaceIntegral, SurfaceDissipation, OperatorType, FaceCache} <: AbstractSurfaceIntegral
  surface_integral::SurfaceIntegral
  surface_dissipation::SurfaceDissipation
  interp_to_hybrid_nodes::OperatorType    # interpolation from face nodes to hybrid face nodes
  project_from_hybrid_nodes::OperatorType # projection from hybrid face nodes to original face nodes
  face_cache::FaceCache # cache to store temporary face variables
end

# We pass in dg::DGMulti since we need the polynomial degree and information about the approximation type
# to construct interpolation and projection operators used for this type of "discrete" surface integral.
function SurfaceIntegralHybrid(surface_dissipation,
                               dg::DGMulti{2, Quad, <:SBP}, equations;
                               hybrid_quadrature = StartUpDG.gauss_quad(0, 0, dg.basis.N),
                               uEltype = real(dg))
  nvars = nvariables(equations)

  polydeg = dg.basis.N
  face_nodes, face_weights = StartUpDG.gauss_lobatto_quad(0, 0, polydeg)
  hybrid_face_nodes, hybrid_face_weights = hybrid_quadrature

  # interpolation and projection matrices
  inv_face_mass_matrix = diagm(inv.(face_weights))
  face_vandermonde_matrix = StartUpDG.vandermonde(Line(), polydeg, face_nodes)
  interp_to_hybrid_nodes = StartUpDG.vandermonde(Line(), polydeg, hybrid_face_nodes) / face_vandermonde_matrix
  project_from_hybrid_nodes = inv_face_mass_matrix * (interp_to_hybrid_nodes' * diagm(hybrid_face_weights))

  # cache for temporary storage of variables.
  # TODO: DGMulti. Assumes that the number of hybrid nodes is the same as the number of face nodes
  # This can be made true for 2D elements, but not for 3D simplices.
  entropy_vars_mine = StructArray{SVector{nvars, uEltype}}(ntuple(_->zeros(uEltype, length(face_nodes)), nvars))
  interpolated_entropy_vars_mine = StructArray{SVector{nvars, uEltype}}(ntuple(_->zeros(uEltype, length(hybrid_face_nodes)), nvars))
  face_cache = (; entropy_vars_mine,
                  entropy_vars_other = similar(entropy_vars_mine),
                  interpolated_entropy_vars_mine,
                  interpolated_entropy_vars_other = similar(interpolated_entropy_vars_mine),
                  flux_tmp = similar(interpolated_entropy_vars_mine),
                  projected_flux_tmp = similar(entropy_vars_mine))

  return SurfaceIntegralHybrid(dg.surface_integral, surface_dissipation,
                               interp_to_hybrid_nodes, project_from_hybrid_nodes,
                               face_cache)
end

function Base.show(io::IO, ::MIME"text/plain", integral::SurfaceIntegralHybrid)
  @nospecialize integral # reduce precompilation time

  if get(io, :compact, false)
    show(io, integral)
  else
    setup = [
            "surface integral" => integral.surface_integral,
            "surface dissipation" => integral.surface_dissipation,
            "operator type" => typeof(integral.interp_to_hybrid_nodes),
            "cache type" => typeof(integral.face_cache)
            ]
    summary_box(io, "SurfaceIntegralHybrid", setup)
  end
end


end # @muladd
