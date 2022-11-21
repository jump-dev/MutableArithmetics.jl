# Copyright (c) 2019 MutableArithmetics.jl contributors
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v.2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.

# TODO: this file contains a large number of method specializations to intercept
# "externally owned" method calls by dispatching on type parameters (rather than
# outermost wrapper type). This is generally bad practice, but refactoring this
# code to use a different mechanism would be a lot of work. In the future, this
# interception code would be more easily/robustly replaced by using a tool like
# https://github.com/jrevels/Cassette.jl.

abstract type AbstractMutable end

function Base.sum(a::AbstractArray{<:AbstractMutable})
    return operate(sum, a)
end

# When doing `x'y` where the elements of `x` and/or `y` are arrays, redirecting
# to `dot(x, y)` is not equivalent to `x'y` as it will call dot recursively on
# the elements of `x` and `y`. See
# https://github.com/JuliaLang/julia/issues/35174
# For this reason, a `_dot_nonrecursive` function was added that does not
# recursively call `dot`:
# https://github.com/JuliaLang/julia/commit/eae3216416453b53631afa6c803591cf2c5ae5b3
#
# However, it does not exploit mutability and returns
# `zero(eltype(lhs)) * zero(eltype(rhs))` in case the arrays are empty which
# creates type instability for some types for which this type is not invariant
# under addition.

# TODO: LinearAlgebra should have a documented function so that we don't have to
# overload an internal function
function LinearAlgebra._dot_nonrecursive(
    lhs::AbstractArray{<:AbstractMutable},
    rhs::AbstractArray,
)
    return fused_map_reduce(add_mul, lhs, rhs)
end
function LinearAlgebra._dot_nonrecursive(
    lhs::AbstractArray,
    rhs::AbstractArray{<:AbstractMutable},
)
    return fused_map_reduce(add_mul, lhs, rhs)
end
function LinearAlgebra._dot_nonrecursive(
    lhs::AbstractArray{<:AbstractMutable},
    rhs::AbstractArray{<:AbstractMutable},
)
    return fused_map_reduce(add_mul, lhs, rhs)
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

function LinearAlgebra.diagm_container(
    size,
    kv::Pair{<:Integer,<:AbstractVector{<:AbstractMutable}}...,
)
    T = promote_type(map(x -> promote_type(eltype(x.second)), kv)...)
    U = promote_type(T, promote_operation(zero, T))
    return zeros(U, LinearAlgebra.diagm_size(size, kv...)...)
end

function LinearAlgebra.diagm(x::AbstractVector{<:AbstractMutable})
    # `LinearAlgebra.diagm` doesn't work for non-one-indexed arrays in general.
    @assert _one_indexed(x)
    ZeroType = promote_operation(zero, eltype(x))
    return LinearAlgebra.diagm(0 => copyto!(similar(x, ZeroType), x))
end

# SparseArrays promotes the element types of `A` and `B` to the same type which,
# always produce quadratic expressions for JuMP even if only one of them was
# affine and the other one constant. Moreover, it does not always go through
# `LinearAlgebra.mul!` which prevents us from using mutability of the
# arithmetic. For this reason we intercept the calls and redirect them to `mul`.

# A few are overwritten below but many more need to be redirected to `mul` in
# `linalg.jl`.

Base.:*(A::_SparseMat{<:AbstractMutable}, x::StridedVector) = mul(A, x)

Base.:*(A::_SparseMat, x::StridedVector{<:AbstractMutable}) = mul(A, x)

function Base.:*(
    A::_SparseMat{<:AbstractMutable},
    x::StridedVector{<:AbstractMutable},
)
    return mul(A, x)
end

# These six methods are needed on Julia v1.2 and earlier
function Base.:*(
    A::LinearAlgebra.Adjoint{<:AbstractMutable,<:_SparseMat},
    x::StridedVector,
)
    return mul(A, x)
end

function Base.:*(
    A::LinearAlgebra.Adjoint{<:Any,<:_SparseMat},
    x::StridedVector{<:AbstractMutable},
)
    return mul(A, x)
end

function Base.:*(
    A::LinearAlgebra.Adjoint{<:AbstractMutable,<:_SparseMat},
    x::StridedVector{<:AbstractMutable},
)
    return mul(A, x)
end

function Base.:*(
    A::LinearAlgebra.Transpose{<:AbstractMutable,<:_SparseMat},
    x::StridedVector,
)
    return mul(A, x)
end

function Base.:*(
    A::LinearAlgebra.Transpose{<:Any,<:_SparseMat},
    x::StridedVector{<:AbstractMutable},
)
    return mul(A, x)
end

function Base.:*(
    A::LinearAlgebra.Transpose{<:AbstractMutable,<:_SparseMat},
    x::StridedVector{<:AbstractMutable},
)
    return mul(A, x)
end

function Base.:*(
    A::_SparseMat{<:AbstractMutable},
    B::_SparseMat{<:AbstractMutable},
)
    return mul(A, B)
end

Base.:*(A::_SparseMat{<:Any}, B::_SparseMat{<:AbstractMutable}) = mul(A, B)

Base.:*(A::_SparseMat{<:AbstractMutable}, B::_SparseMat{<:Any}) = mul(A, B)

function Base.:*(
    A::_SparseMat{<:AbstractMutable},
    B::LinearAlgebra.Adjoint{<:AbstractMutable,<:_SparseMat},
)
    return mul(A, B)
end

function Base.:*(
    A::_SparseMat{<:Any},
    B::LinearAlgebra.Adjoint{<:AbstractMutable,<:_SparseMat},
)
    return mul(A, B)
end

function Base.:*(
    A::_SparseMat{<:AbstractMutable},
    B::LinearAlgebra.Adjoint{<:Any,<:_SparseMat},
)
    return mul(A, B)
end

function Base.:*(
    A::LinearAlgebra.Adjoint{<:AbstractMutable,<:_SparseMat},
    B::_SparseMat{<:AbstractMutable},
)
    return mul(A, B)
end

function Base.:*(
    A::LinearAlgebra.Adjoint{<:Any,<:_SparseMat},
    B::_SparseMat{<:AbstractMutable},
)
    return mul(A, B)
end

function Base.:*(
    A::LinearAlgebra.Adjoint{<:AbstractMutable,<:_SparseMat},
    B::_SparseMat{<:Any},
)
    return mul(A, B)
end

function Base.:*(
    A::LinearAlgebra.Transpose{<:AbstractMutable,<:_SparseMat},
    B::_SparseMat{<:AbstractMutable},
)
    return mul(A, B)
end

function Base.:*(
    A::LinearAlgebra.Transpose{<:Any,<:_SparseMat},
    B::_SparseMat{<:AbstractMutable},
)
    return mul(A, B)
end

function Base.:*(
    A::LinearAlgebra.Transpose{<:AbstractMutable,<:_SparseMat},
    B::_SparseMat{<:Any},
)
    return mul(A, B)
end

function Base.:*(
    A::StridedMatrix{<:AbstractMutable},
    B::_SparseMat{<:AbstractMutable},
)
    return mul(A, B)
end

Base.:*(A::StridedMatrix{<:Any}, B::_SparseMat{<:AbstractMutable}) = mul(A, B)

Base.:*(A::StridedMatrix{<:AbstractMutable}, B::_SparseMat{<:Any}) = mul(A, B)

function Base.:*(
    A::_SparseMat{<:AbstractMutable},
    B::StridedMatrix{<:AbstractMutable},
)
    return mul(A, B)
end

Base.:*(A::_SparseMat{<:Any}, B::StridedMatrix{<:AbstractMutable}) = mul(A, B)

Base.:*(A::_SparseMat{<:AbstractMutable}, B::StridedMatrix{<:Any}) = mul(A, B)

function Base.:*(
    A::LinearAlgebra.Adjoint{<:AbstractMutable,<:_SparseMat},
    B::StridedMatrix{<:AbstractMutable},
)
    return mul(A, B)
end

function Base.:*(
    A::LinearAlgebra.Adjoint{<:Any,<:_SparseMat},
    B::StridedMatrix{<:AbstractMutable},
)
    return mul(A, B)
end

function Base.:*(
    A::LinearAlgebra.Adjoint{<:AbstractMutable,<:_SparseMat},
    B::StridedMatrix{<:Any},
)
    return mul(A, B)
end

function Base.:*(
    A::LinearAlgebra.Transpose{<:AbstractMutable,<:_SparseMat},
    B::StridedMatrix{<:AbstractMutable},
)
    return mul(A, B)
end

function Base.:*(
    A::LinearAlgebra.Transpose{<:Any,<:_SparseMat},
    B::StridedMatrix{<:AbstractMutable},
)
    return mul(A, B)
end

function Base.:*(
    A::LinearAlgebra.Transpose{<:AbstractMutable,<:_SparseMat},
    B::StridedMatrix{<:Any},
)
    return mul(A, B)
end

const StridedMaybeAdjOrTransMat{T} = Union{
    StridedMatrix{T},
    LinearAlgebra.Adjoint{T,<:StridedMatrix},
    LinearAlgebra.Transpose{T,<:StridedMatrix},
}

# See https://github.com/JuliaLang/julia/pull/37898
# The default fallback only used `promote_type` so it may get its wrong, e.g.,
# for JuMP and MultivariatePolynomials.
if VERSION >= v"1.7.0-DEV.1284"
    _mat_mat_scalar(A, B, γ) = operate!!(*, operate(*, A, B), γ)
    function LinearAlgebra.mat_mat_scalar(
        A::StridedMaybeAdjOrTransMat{<:AbstractMutable},
        B::StridedMaybeAdjOrTransMat,
        γ,
    )
        return _mat_mat_scalar(A, B, γ)
    end
    function LinearAlgebra.mat_mat_scalar(
        A::StridedMaybeAdjOrTransMat,
        B::StridedMaybeAdjOrTransMat{<:AbstractMutable},
        γ,
    )
        return _mat_mat_scalar(A, B, γ)
    end
    function LinearAlgebra.mat_mat_scalar(
        A::StridedMaybeAdjOrTransMat{<:AbstractMutable},
        B::StridedMaybeAdjOrTransMat{<:AbstractMutable},
        γ,
    )
        return _mat_mat_scalar(A, B, γ)
    end
end

# Base doesn't define efficient fallbacks for sparse array arithmetic involving
# non-`<:Number` scalar elements, so we define some of these for
# `<:AbstractMutable` scalar elements here.

for (S, T) in [
    (LinearAlgebra.UniformScaling, AbstractMutable),
    (Number, AbstractMutable),
    (AbstractMutable, Any),
]
    @eval begin
        function Base.:*(A::$S, B::_SparseMat{<:$T})
            return _SparseMat(
                B.m,
                B.n,
                copy(B.colptr),
                copy(SparseArrays.rowvals(B)),
                A .* SparseArrays.nonzeros(B),
            )
        end
        function Base.:*(B::_SparseMat{<:$T}, A::$S)
            return _SparseMat(
                B.m,
                B.n,
                copy(B.colptr),
                copy(SparseArrays.rowvals(B)),
                SparseArrays.nonzeros(B) .* A,
            )
        end
    end
end

function Base.:/(
    A::_SparseMat{<:AbstractMutable},
    B::LinearAlgebra.UniformScaling,
)
    return _SparseMat(
        A.m,
        A.n,
        copy(A.colptr),
        copy(SparseArrays.rowvals(A)),
        SparseArrays.nonzeros(A) ./ B,
    )
end

function Base.:/(A::_SparseMat{<:AbstractMutable}, B::Number)
    return _SparseMat(
        A.m,
        A.n,
        copy(A.colptr),
        copy(SparseArrays.rowvals(A)),
        SparseArrays.nonzeros(A) ./ B,
    )
end

# Base assumes that the element type is unaffected by `-`
function Base.:-(A::_SparseMat{<:AbstractMutable})
    return _SparseMat(
        A.m,
        A.n,
        copy(A.colptr),
        copy(SparseArrays.rowvals(A)),
        -SparseArrays.nonzeros(A),
    )
end

# Matrix(::SparseMatrixCSC) assumes that `zero` does not affect the element
# type of `S`.
function Base.Matrix(S::_SparseMat{T}) where {T<:AbstractMutable}
    U = promote_operation(+, promote_operation(zero, T), T)
    A = Matrix{U}(undef, size(S)...)
    operate!(zero, A)
    return operate!(+, A, S)
end

# +(::SparseMatrixCSC) is not defined for generic types in Base.
Base.:+(A::AbstractArray{<:AbstractMutable}) = A

Base.:*(α::AbstractMutable, A::AbstractArray) = α .* A

Base.:*(A::AbstractArray, α::AbstractMutable) = A .* α

# Needed for Julia v1.0, otherwise, `broadcast(*, α, A)` gives a `Array` and
# not a `Symmetric`.
function Base.:*(α::Number, A::LinearAlgebra.Symmetric{<:AbstractMutable})
    return LinearAlgebra.Symmetric(
        α * parent(A),
        LinearAlgebra.sym_uplo(A.uplo),
    )
end

function Base.:*(α::Number, A::LinearAlgebra.Hermitian{<:AbstractMutable})
    return LinearAlgebra.Hermitian(
        α * parent(A),
        LinearAlgebra.sym_uplo(A.uplo),
    )
end

# Fix ambiguity identified by Aqua.jl.
function Base.:*(α::Real, A::LinearAlgebra.Hermitian{<:AbstractMutable})
    return LinearAlgebra.Hermitian(
        α * parent(A),
        LinearAlgebra.sym_uplo(A.uplo),
    )
end

# These three have specific methods that just redirect to `Matrix{T}` which
# does not work, e.g. if `zero(T)` has a different type than `T`.

function Base.Matrix(x::LinearAlgebra.Tridiagonal{T}) where {T<:AbstractMutable}
    return Matrix{promote_type(promote_operation(zero, T), T)}(x)
end

function Base.Matrix(
    x::LinearAlgebra.UpperTriangular{T},
) where {T<:AbstractMutable}
    return Matrix{promote_type(promote_operation(zero, T), T)}(x)
end

function Base.Matrix(
    x::LinearAlgebra.LowerTriangular{T},
) where {T<:AbstractMutable}
    return Matrix{promote_type(promote_operation(zero, T), T)}(x)
end

# Needed for Julia v1.1 only. If `parent(A)` is for instance `Diagonal`, the
# `eltype` of `B` might be different form the `eltype` of `A`.
function Matrix(A::LinearAlgebra.Symmetric{<:AbstractMutable})
    B = LinearAlgebra.copytri!(convert(Matrix, copy(A.data)), A.uplo)
    for i in 1:size(A, 1)
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
    for i in 1:size(A, 1)
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
# reverted. Fixed again in https://github.com/JuliaLang/julia/pull/38789/.
if VERSION >= v"1.7.0-DEV.872"
    # `AbstractMutable` objects are more likely to implement `iszero` than `==`
    # with `Int`.
    LinearAlgebra.iszerodefined(::Type{<:AbstractMutable}) = true
else
    # To determine whether the funtion is zero preserving, `LinearAlgebra` calls
    # `zero` on the `eltype` of the broadcasted object and then check `_iszero`.
    # `_iszero(x)` redirects to `iszero(x)` for numbers and to `x == 0`
    # otherwise.
    # `x == 0` returns false for types that implement `iszero` but not `==` such
    # as `DummyBigInt` and MOI functions.
    LinearAlgebra._iszero(x::AbstractMutable) = iszero(x)
end
