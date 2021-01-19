# TODO: Intercepting "externally owned" method calls by dispatching on type
# parameters (rather than outermost wrapper type) is generally bad practice, but
# refactoring this code to use a different mechanism would be a lot of work. In
# the future, this interception code would be more easily/robustly replaced by
# using a tool like https://github.com/jrevels/Cassette.jl.

abstract type AbstractMutable end

function Base.sum(a::AbstractArray{<:AbstractMutable})
    return operate(sum, a)
end

function LinearAlgebra.dot(
    lhs::AbstractArray{<:AbstractMutable},
    rhs::AbstractArray,
)
    return operate(LinearAlgebra.dot, lhs, rhs)
end

function LinearAlgebra.dot(
    lhs::AbstractArray,
    rhs::AbstractArray{<:AbstractMutable},
)
    return operate(LinearAlgebra.dot, lhs, rhs)
end

function LinearAlgebra.dot(
    lhs::AbstractArray{<:AbstractMutable},
    rhs::AbstractArray{<:AbstractMutable},
)
    return operate(LinearAlgebra.dot, lhs, rhs)
end

# Special-case because the the base version wants to do
# fill!(::Array{AbstractVariableRef}, zero(GenericAffExpr{Float64,eltype(x)}))
_one_indexed(A) = all(x -> isa(x, Base.OneTo), axes(A))
if VERSION <= v"1.2"
    function LinearAlgebra.diagm_container(
        kv::Pair{<:Integer,<:AbstractVector{<:AbstractMutable}}...,
    )
        T = promote_type(map(x -> eltype(x.second), kv)...)
        U = promote_type(T, promote_operation(zero, T))
        n = mapreduce(x -> length(x.second) + abs(x.first), max, kv)
        return zeros(U, n, n)
    end
else
    function LinearAlgebra.diagm_container(
        size,
        kv::Pair{<:Integer,<:AbstractVector{<:AbstractMutable}}...,
    )
        T = promote_type(map(x -> promote_type(eltype(x.second)), kv)...)
        U = promote_type(T, promote_operation(zero, T))
        return zeros(U, LinearAlgebra.diagm_size(size, kv...)...)
    end
end
function LinearAlgebra.diagm(x::AbstractVector{<:AbstractMutable})
    # `LinearAlgebra.diagm` doesn't work for non-one-indexed arrays in general.
    @assert _one_indexed(x)
    ZeroType = promote_operation(zero, eltype(x))
    return LinearAlgebra.diagm(0 => copyto!(similar(x, ZeroType), x))
end

###############################################################################
# Interception of Base's matrix/vector arithmetic machinery

# Redirect calls with `eltype(ret) <: AbstractMutable` to `_mul!` to
# replace it with an implementation more efficient than `generic_matmatmul!` and
# `generic_matvecmul!` since it takes into account the mutability of the
# arithmetic. We need `args...` because SparseArrays` also gives `α` and `β`
# arguments.

function _mul!(output, A, B, α, β)
    # See SparseArrays/src/linalg.jl
    if !isone(β)
        if iszero(β)
            mutable_operate!(zero, output)
        else
            rmul!(output, scaling(β))
        end
    end
    return mutable_operate!(add_mul, output, A, B, scaling(α))
end

function _mul!(output, A, B, α)
    mutable_operate!(zero, output)
    return mutable_operate!(add_mul, output, A, B, scaling(α))
end

function _mul!(output, A, B)
    return mutable_operate_to!(output, *, A, B)
end

function LinearAlgebra.mul!(
    ret::AbstractMatrix{<:AbstractMutable},
    A::AbstractVecOrMat,
    B::AbstractVecOrMat,
    args::Vararg{Any,N},
) where {N}
    _mul!(ret, A, B, args...)
end

function LinearAlgebra.mul!(
    ret::AbstractVector{<:AbstractMutable},
    A::AbstractVecOrMat,
    B::AbstractVector,
    args::Vararg{Any,N},
) where {N}
    _mul!(ret, A, B, args...)
end

# SparseArrays promotes the element types of `A` and `B` to the same type
# which always produce quadratic expressions for JuMP even if only one of them
# was affine and the other one constant. Moreover, it does not always go through
# `LinearAlgebra.mul!` which prevents us from using mutability of the arithmetic.
# For this reason we intercept the calls and redirect them to `mul`.

# A few are overwritten below but many more need to be redirected to `mul` in
# `linalg.jl`.

const MatrixLike = [
    T -> SparseMat{<:T},
    T -> StridedVector{<:T},
    T -> StridedMatrix{<:T},
    T -> LinearAlgebra.Adjoint{<:T, <:SparseMat},
    T -> LinearAlgebra.Symmetric{<:T, <:SparseMat},
    T -> LinearAlgebra.Hermitian{<:T, <:SparseMat},
    T -> LinearAlgebra.Diagonal{<:T, <:SparseMat},
    T -> LinearAlgebra.UpperTriangular{<:T, <:SparseMat},
    T -> LinearAlgebra.LowerTriangular{<:T, <:SparseMat},
    T -> LinearAlgebra.UnitLowerTriangular{<:T, <:SparseMat},
    T -> LinearAlgebra.UnitUpperTriangular{<:T, <:SparseMat},
]

for f_A in MatrixLike
    A, mut_A = f_A(Any), f_A(AbstractMutable)
    for f_B in MatrixLike
        B, mut_B = f_B(Any), f_B(AbstractMutable)
        @eval begin
            Base.:*(a::$(mut_A), b::$(B)) = mul(a, b)
            Base.:*(a::$(A), b::$(mut_B)) = mul(a, b)
            Base.:*(a::$(mut_A), b::$(mut_B)) = mul(a, b)
        end
    end
end

# Base doesn't define efficient fallbacks for sparse array arithmetic involving
# non-`<:Number` scalar elements, so we define some of these for
# `<:AbstractMutable` scalar elements here.

for (S, T) in [
    (Number, AbstractMutable),
    (Scaling, AbstractMutable),
    (AbstractMutable, Any),
]
    @eval begin
        function Base.:*(A::$S, B::SparseMat{<:$T})
            return SparseMat(
                B.m,
                B.n,
                copy(B.colptr),
                copy(SparseArrays.rowvals(B)),
                A .* SparseArrays.nonzeros(B),
            )
        end
        function Base.:*(A::SparseMat{<:$T}, B::$S)
            return SparseMat(
                A.m,
                A.n,
                copy(A.colptr),
                copy(SparseArrays.rowvals(A)),
                SparseArrays.nonzeros(A) .* B,
            )
        end
    end
end

function Base.:/(A::SparseMat{<:AbstractMutable}, B::Scaling)
    return SparseMat(
        A.m,
        A.n,
        copy(A.colptr),
        copy(SparseArrays.rowvals(A)),
        SparseArrays.nonzeros(A) ./ B,
    )
end

# Base assumes that the element type is unaffected by `-`
function Base.:-(A::SparseMat{<:AbstractMutable})
    return SparseMat(
        A.m,
        A.n,
        copy(A.colptr),
        copy(SparseArrays.rowvals(A)),
        -SparseArrays.nonzeros(A),
    )
end

# Matrix(::SparseMatrixCSC) assumes that `zero` does not affect the element type of `S`.
function Base.Matrix(S::SparseMat{T}) where {T<:AbstractMutable}
    U = promote_operation(+, promote_operation(zero, T), T)
    A = Matrix{U}(undef, size(S)...)
    mutable_operate!(zero, A)
    return mutable_operate!(+, A, S)
end

# +(::SparseMatrixCSC) is not defined for generic types in Base.
Base.:+(A::AbstractArray{<:AbstractMutable}) = A

# Fix https://github.com/JuliaLang/julia/issues/32374 as done in
# https://github.com/JuliaLang/julia/pull/32375. This hack should
# be removed once we drop Julia v1.0.
function Base.:-(A::LinearAlgebra.Symmetric{<:AbstractMutable})
    return LinearAlgebra.Symmetric(-parent(A), LinearAlgebra.sym_uplo(A.uplo))
end

function Base.:-(A::LinearAlgebra.Hermitian{<:AbstractMutable})
    return LinearAlgebra.Hermitian(-parent(A), LinearAlgebra.sym_uplo(A.uplo))
end

function Base.:*(α::AbstractMutable, A::AbstractArray)
    return α .* A
end
function Base.:*(A::AbstractArray, α::AbstractMutable)
    return A .* α
end

# Needed for Julia v1.0, otherwise, `broadcast(*, α, A)` gives a `Array` and
# not a `Symmetric`.
function Base.:*(α::Number, A::LinearAlgebra.Symmetric{<:AbstractMutable})
    return LinearAlgebra.Symmetric(α * parent(A), LinearAlgebra.sym_uplo(A.uplo))
end

function Base.:*(α::Number, A::LinearAlgebra.Hermitian{<:AbstractMutable})
    return LinearAlgebra.Hermitian(α * parent(A), LinearAlgebra.sym_uplo(A.uplo))
end

# These three have specific methods that just redirect to `Matrix{T}` which
# does not work, e.g. if `zero(T)` has a different type than `T`.
function Base.Matrix(
    x::Union{
        LinearAlgebra.Tridiagonal{T},
        LinearAlgebra.UpperTriangular{T},
        LinearAlgebra.LowerTriangular{T},
    }
) where {T<:AbstractMutable}
    return Matrix{promote_type(promote_operation(zero, T), T)}(x)
end

# Needed for Julia v1.1 only. If `parent(A)` is for instance `Diagonal`, the
# `eltype` of `B` might be different form the `eltype` of `A`.
function Matrix(A::LinearAlgebra.Symmetric{<:AbstractMutable})
    B = LinearAlgebra.copytri!(convert(Matrix, copy(A.data)), A.uplo)
    for i = 1:size(A, 1)
        # `B[i, i]` is used instead of `A[i, i]` on Julia v1.1 hence the need
        # to overwrite it for `AbstractMutable`.
        B[i, i] = LinearAlgebra.symmetric(
            A[i, i],
            LinearAlgebra.sym_uplo(A.uplo),
        )::LinearAlgebra.symmetric_type(eltype(A.data))
    end
    return B
end

function Matrix(A::LinearAlgebra.Hermitian{<:AbstractMutable})
    B = LinearAlgebra.copytri!(convert(Matrix, copy(A.data)), A.uplo, true)
    for i = 1:size(A, 1)
        # `B[i, i]` is used instead of `A[i, i]` on Julia v1.1 hence the need
        # to overwrite it for `AbstractMutable`.
        B[i, i] = LinearAlgebra.hermitian(
            A[i, i],
            LinearAlgebra.sym_uplo(A.uplo),
        )::LinearAlgebra.hermitian_type(eltype(A.data))
    end
    return B
end

# Called in `getindex` of `LinearAlgebra.LowerTriangular` and
# `LinearAlgebra.UpperTriangular` as the elements may be `Array` for which
# `zero` is only defined for instances but not for the type. For
# `AbstractMutable` we assume that `zero` for the instance is the same than for
# the type by default.
Base.zero(x::AbstractMutable) = zero(typeof(x))

# This was fixed in https://github.com/JuliaLang/julia/pull/36194 but then
# reverted. Fixed again in https://github.com/JuliaLang/julia/pull/38789/ but
# not merged yet.
#
# To determine whether the funtion is zero preserving, `LinearAlgebra` calls
# `zero` on the `eltype` of the broadcasted object and then check `_iszero`.
# `_iszero(x)` redirects to `iszero(x)` for numbers and to `x == 0` otherwise.
# `x == 0` returns false for types that implement `iszero` but not `==` such as
# `DummyBigInt` and MOI functions.
LinearAlgebra._iszero(x::AbstractMutable) = iszero(x)
