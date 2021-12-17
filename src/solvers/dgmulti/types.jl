# Note: we define type aliases outside of the @muladd block to avoid Revise breaking when code
# inside the @muladd block is edited. See https://github.com/trixi-framework/Trixi.jl/issues/801
# for more details.

# `DGMulti` refers to both multiple DG types (polynomial/SBP, simplices/quads/hexes) as well as
# the use of multi-dimensional operators in the solver.
const DGMulti{NDIMS, MeshType, ApproxType, SurfaceIntegral, VolumeIntegral} =
  DG{<:RefElemData{NDIMS, MeshType, ApproxType}, Mortar, SurfaceIntegral, VolumeIntegral} where {Mortar}

# Type aliases. The first parameter is `ApproxType` since it is more commonly used for dispatch.
const DGMultiWeakForm{ApproxType, MeshType} =
  DGMulti{NDIMS, MeshType, ApproxType, <:SurfaceIntegralWeakForm, <:VolumeIntegralWeakForm} where {NDIMS}

const DGMultiFluxDiff{ApproxType, MeshType} =
  DGMulti{NDIMS, MeshType, ApproxType, <:SurfaceIntegralWeakForm, <:VolumeIntegralFluxDifferencing} where {NDIMS}

const DGMultiFluxDiffSBP{ApproxType, MeshType} =
  DGMulti{NDIMS, MeshType, ApproxType, <:SurfaceIntegralWeakForm, <:VolumeIntegralFluxDifferencing} where {NDIMS, ApproxType<:Union{SBP, AbstractDerivativeOperator}}

const DGMultiSBP{ApproxType, MeshType} =
  DGMulti{NDIMS, MeshType, ApproxType, SurfaceIntegral, VolumeIntegral} where {NDIMS, MeshType, ApproxType<:Union{SBP, AbstractDerivativeOperator}, SurfaceIntegral, VolumeIntegral}


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
function DGMulti(; polydeg=nothing,
                   element_type::AbstractElemShape,
                   approximation_type=Polynomial(),
                   surface_flux=flux_central,
                   surface_integral=SurfaceIntegralWeakForm(surface_flux),
                   volume_integral=VolumeIntegralWeakForm(),
                   kwargs...)

  # call dispatchable constructor
  DGMulti(element_type, approximation_type, volume_integral, surface_integral;
          polydeg=polydeg, kwargs...)
end

# dispatchable constructor for DGMulti to allow for specialization
function DGMulti(element_type::AbstractElemShape,
                 approximation_type,
                 volume_integral,
                 surface_integral;
                 polydeg::Integer,
                 kwargs...)

  rd = RefElemData(element_type, approximation_type, polydeg; kwargs...)
  return DG(rd, nothing #= mortar =#, surface_integral, volume_integral)
end

# now that `DGMulti` is defined, we can define constructors for `DGMultiMesh` which use `dg::DGMulti`
"""
    DGMultiMesh(vertex_coordinates, EToV, dg::DGMulti;
                is_on_boundary = nothing,
                periodicity::NTuple{NDIMS, Bool} = ntuple(_->false, NDIMS)) where {NDIMS, Tv}

Constructor which uses `dg::DGMulti` instead of `rd::RefElemData`.
"""
DGMultiMesh(vertex_coordinates, EToV, dg::DGMulti; kwargs...) =
  DGMultiMesh(vertex_coordinates, EToV, dg.basis; kwargs...)

"""
    DGMultiMesh(triangulateIO, dg::DGMulti, boundary_dict::Dict{Symbol, Int})

Constructor which uses `dg::DGMulti` instead of `rd::RefElemData`.
"""
DGMultiMesh(triangulateIO, dg::DGMulti, boundary_dict::Dict{Symbol, Int}) =
  DGMultiMesh(triangulateIO, dg.basis, boundary_dict)

"""
    DGMultiMesh(dg::DGMulti; cells_per_dimension,
                coordinates_min=(-1.0, -1.0), coordinates_max=(1.0, 1.0),
                is_on_boundary=nothing,
                periodicity=ntuple(_ -> false, NDIMS))

Constructs a Cartesian [`DGMultiMesh`](@ref) with element type `dg.basis.elementType`. The domain is
the tensor product of the intervals `[coordinates_min[i], coordinates_max[i]]`.
- `is_on_boundary` specifies boundary using a `Dict{Symbol, <:Function}`
- `periodicity` is a tuple of `Bool`s specifying periodicity = `true`/`false` in the (x,y,z) direction.
"""
function DGMultiMesh(dg::DGMulti{NDIMS}; cells_per_dimension,
                     coordinates_min=ntuple(_ -> -one(real(dg)), NDIMS),
                     coordinates_max=ntuple(_ -> one(real(dg)), NDIMS),
                     is_on_boundary=nothing,
                     periodicity=ntuple(_ -> false, NDIMS), kwargs...) where {NDIMS}

  if haskey(kwargs, :is_periodic)
    # TODO: DGMulti. Deprecate `is_periodic` in version 0.5
    Base.depwarn("keyword argument `is_periodic` is now `periodicity`.", :DGMultiMesh)
    periodicity=kwargs[:is_periodic]
  end

  vertex_coordinates, EToV = StartUpDG.uniform_mesh(dg.basis.elementType, cells_per_dimension...)
  domain_lengths = coordinates_max .- coordinates_min
  for i in 1:NDIMS
    @. vertex_coordinates[i] = 0.5 * (vertex_coordinates[i] + 1) * domain_lengths[i] + coordinates_min[i]
  end

  return DGMultiMesh(vertex_coordinates, EToV, dg, is_on_boundary=is_on_boundary, periodicity=periodicity)
end

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

# `SimpleKronecker` lazily stores a Kronecker product `kron(ntuple(A, NDIMS)...)`.
# This object also allocates some temporary storage to enable the fast computation
# of matrix-vector products.
struct SimpleKronecker{NDIMS, TA, Ttmp}
  A::TA
  tmp_storage::Ttmp # temporary array used for Kronecker multiplication
end

# constructor for SimpleKronecker which requires specifying only `NDIMS` and
# the 1D matrix `A`.
function SimpleKronecker(NDIMS, A, eltype_A=eltype(A))
  @assert size(A, 1) == size(A, 2) # check if square
  tmp_storage=[zeros(eltype_A, ntuple(_ -> size(A, 2), NDIMS)...) for _ in 1:Threads.nthreads()]
  return SimpleKronecker{NDIMS, typeof(A), typeof(tmp_storage)}(A, tmp_storage)
end

# Computes `b = kron(A, A) * x` in an optimized fashion
function LinearAlgebra.mul!(b_in, A_kronecker::SimpleKronecker{2}, x_in)

  @unpack A = A_kronecker
  tmp_storage = A_kronecker.tmp_storage[Threads.threadid()]
  n = size(A, 2)

  # copy `x_in` to `tmp_storage` to avoid mutating the input
  @assert length(tmp_storage) == length(x_in)
  for i in eachindex(tmp_storage)
    tmp_storage[i] = x_in[i]
  end
  x = reshape(tmp_storage, n, n)
  b = reshape(b_in, n, n)

  @turbo for j in 1:n, i in 1:n
    tmp = zero(eltype(x))
    for ii in 1:n
      tmp = tmp + A[i, ii] * x[ii, j]
    end
    b[i, j] = tmp
  end

  @turbo for j in 1:n, i in 1:n
    tmp = zero(eltype(x))
    for jj in 1:n
      tmp = tmp + A[j, jj] * b[i, jj]
    end
    x[i, j] = tmp
  end

  @turbo for i in eachindex(b_in)
    b_in[i] = x[i]
  end

  return nothing
end

# Computes `b = kron(A, A, A) * x` in an optimized fashion
function LinearAlgebra.mul!(b_in, A_kronecker::SimpleKronecker{3}, x_in)

  @unpack A = A_kronecker
  tmp_storage = A_kronecker.tmp_storage[Threads.threadid()]
  n = size(A, 2)

  # copy `x_in` to `tmp_storage` to avoid mutating the input
  for i in eachindex(tmp_storage)
    tmp_storage[i] = x_in[i]
  end
  x = reshape(tmp_storage, n, n, n)
  b = reshape(b_in, n, n, n)

  @turbo for k in 1:n, j in 1:n, i in 1:n
    tmp = zero(eltype(x))
    for ii in 1:n
      tmp = tmp + A[i, ii] * x[ii, j, k]
    end
    b[i, j, k] = tmp
  end

  @turbo for k in 1:n, j in 1:n, i in 1:n
    tmp = zero(eltype(x))
    for jj in 1:n
      tmp = tmp + A[j, jj] * b[i, jj, k]
    end
    x[i, j, k] = tmp
  end

  @turbo for k in 1:n, j in 1:n, i in 1:n
    tmp = zero(eltype(x))
    for kk in 1:n
      tmp = tmp + A[k, kk] * x[i, j, kk]
    end
    b[i, j, k] = tmp
  end

  return nothing
end



end # @muladd
