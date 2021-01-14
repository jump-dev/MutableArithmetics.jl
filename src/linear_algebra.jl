mutability(::Type{<:Array}) = IsMutable()
mutable_copy(A::Array) = copy_if_mutable.(A)

# Sum

function promote_operation(
    op::Union{typeof(+),typeof(-)},
    ::Type{Array{S,N}},
    ::Type{Array{T,N}},
) where {S,T,N}
    return Array{promote_operation(op, S, T),N}
end

function promote_operation(
    op::Union{typeof(+),typeof(-)},
    ::Type{LinearAlgebra.UniformScaling{S}},
    ::Type{Matrix{T}},
) where {S,T}
    return Matrix{promote_operation(op, S, T)}
end

function promote_operation(
    op::Union{typeof(+),typeof(-)},
    ::Type{Matrix{T}},
    ::Type{LinearAlgebra.UniformScaling{S}},
) where {S,T}
    return Matrix{promote_operation(op, S, T)}
end

function promote_operation(
    op::Union{typeof(+),typeof(-)},
    ::Type{Matrix{T}},
    ::Type{<:LinearAlgebra.Symmetric{S}},
) where {S,T}
    return Matrix{promote_operation(op, S, T)}
end

# Only `Scaling`
function mutable_operate!(
    op::Union{typeof(+),typeof(-)},
    A::Matrix,
    B::LinearAlgebra.UniformScaling,
)
    n = LinearAlgebra.checksquare(A)
    for i = 1:n
        A[i, i] = operate!(op, A[i, i], B)
    end
    return A
end

function mutable_operate!(
    op::AddSubMul,
    A::Matrix,
    B::Scaling,
    C::Scaling,
    D::Vararg{Scaling,N},
) where {N}
    return mutable_operate!(add_sub_op(op), A, *(B, C, D...))
end

mul_rhs(::typeof(+)) = add_mul
mul_rhs(::typeof(-)) = sub_mul

# `Scaling` and `Array`
function _mutable_operate!(
    op::Union{typeof(+),typeof(-)},
    A::Array{S,N},
    B::Union{Array{T,N},LinearAlgebra.Symmetric{T}},
    left_factors::Tuple,
    right_factors::Tuple,
) where {S,T,N}
    for i in eachindex(A)
        A[i] = operate!(mul_rhs(op), A[i], left_factors..., B[i], right_factors...)
    end
    return A
end

function _check_dims(A, B)
    if size(A) != size(B)
        throw(
            DimensionMismatch(
                "Cannot sum matrices of size `$(size(A))` and size `$(size(B))`, the size of the two matrices must be equal.",
            ),
        )
    end
end

function mutable_operate!(
    op::Union{typeof(+),typeof(-)},
    A::Array{S,N},
    B::AbstractArray{T,N},
) where {S,T,N}
    _check_dims(A, B)
    return _mutable_operate!(op, A, B, tuple(), tuple())
end

function mutable_operate!(
    op::AddSubMul,
    A::Array{S,N},
    B::AbstractArray{T,N},
    α::Vararg{Scaling,M},
) where {S,T,N,M}
    _check_dims(A, B)
    return _mutable_operate!(add_sub_op(op), A, B, tuple(), α)
end

function mutable_operate!(
    op::AddSubMul,
    A::Array{S,N},
    α::Scaling,
    B::AbstractArray{T,N},
    β::Vararg{Scaling,M},
) where {S,T,N,M}
    _check_dims(A, B)
    return _mutable_operate!(add_sub_op(op), A, B, (α,), β)
end

function mutable_operate!(
    op::AddSubMul,
    A::Array{S,N},
    α1::Scaling,
    α2::Scaling,
    B::AbstractArray{T,N},
    β::Vararg{Scaling,M},
) where {S,T,N,M}
    _check_dims(A, B)
    return _mutable_operate!(add_sub_op(op), A, B, (α1, α2), β)
end

# Fallback, we may be able to be more efficient in more cases by adding more specialized methods
function mutable_operate!(op::AddSubMul, A::Array, x, y, args::Vararg{Any,N}) where {N}
    return mutable_operate!(op, A, x, *(y, args...))
end

# Product

function similar_array_type(::Type{LinearAlgebra.Symmetric{T,MT}}, ::Type{S}) where {S,T,MT}
    return LinearAlgebra.Symmetric{S,similar_array_type(MT, S)}
end

similar_array_type(::Type{Array{T,N}}, ::Type{S}) where {S,T,N} = Array{S,N}

function promote_operation(
    op::typeof(*),
    A::Type{<:AbstractArray{T}},
    ::Type{S},
) where {S,T}
    return similar_array_type(A, promote_operation(op, T, S))
end

function promote_operation(
    op::typeof(*),
    ::Type{S},
    A::Type{<:AbstractArray{T}},
) where {S,T}
    return similar_array_type(A, promote_operation(op, S, T))
end

# `{S}` and `{T}` are used to avoid ambiguity with above methods.
function promote_operation(
    op::typeof(*),
    A::Type{<:AbstractArray{S}},
    B::Type{<:AbstractArray{T}},
) where {S,T}
    return promote_array_mul(A, B)
end

function promote_sum_mul(T::Type, S::Type)
    U = promote_operation(*, T, S)
    return promote_operation(+, U, U)
end

function promote_array_mul(::Type{Matrix{S}}, ::Type{Vector{T}}) where {S,T}
    return Vector{promote_sum_mul(S, T)}
end

function promote_array_mul(
    ::Type{<:AbstractMatrix{S}},
    ::Type{<:AbstractMatrix{T}},
) where {S,T}
    return Matrix{promote_sum_mul(S, T)}
end

function promote_array_mul(
    ::Type{<:AbstractMatrix{S}},
    ::Type{<:AbstractVector{T}},
) where {S,T}
    return Vector{promote_sum_mul(S, T)}
end

################################################################################
# We roll our own matmul here (instead of using Julia's generic fallbacks)
# because doing so allows us to accumulate the expressions for the inner loops
# in-place.
# Additionally, Julia's generic fallbacks can be finnicky when your array
# elements aren't `<:Number`.

# This method of `mul!` is adapted from upstream Julia. Note that we
# confuse transpose with adjoint.
#=
> Copyright (c) 2009-2018: Jeff Bezanson, Stefan Karpinski, Viral B. Shah,
> and other contributors:
>
> https://github.com/JuliaLang/julia/contributors
>
> Permission is hereby granted, free of charge, to any person obtaining
> a copy of this software and associated documentation files (the
> "Software"), to deal in the Software without restriction, including
> without limitation the rights to use, copy, modify, merge, publish,
> distribute, sublicense, and/or sell copies of the Software, and to
> permit persons to whom the Software is furnished to do so, subject to
> the following conditions:
>
> The above copyright notice and this permission notice shall be
> included in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
> EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
> MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
> NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
> LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
> OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
> WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
=#

function _dim_check(C::AbstractVector, A::AbstractMatrix, B::AbstractVector)
    mB = length(B)
    mA, nA = size(A)
    if mB != nA
        throw(
            DimensionMismatch("matrix A has dimensions ($mA,$nA), vector B has length $mB"),
        )
    end
    if mA != length(C)
        throw(DimensionMismatch("result C has length $(length(C)), needs length $mA"))
    end
end

function _dim_check(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
    mB, nB = size(B)
    mA, nA = size(A)
    if mB != nA
        throw(
            DimensionMismatch(
                "matrix A has dimensions ($mA,$nA), matrix B has dimensions ($mB,$nB)",
            ),
        )
    end
    if size(C, 1) != mA || size(C, 2) != nB
        throw(DimensionMismatch("result C has dimensions $(size(C)), needs ($mA,$nB)"))
    end
end

function _add_mul_array(C::Vector, A::AbstractMatrix, B::AbstractVector)
    Astride = size(A, 1)
    # We need a buffer to hold the intermediate multiplication.
    mul_buffer = buffer_for(add_mul, eltype(C), eltype(A), eltype(B))
    @inbounds begin
        for k in eachindex(B)
            aoffs = (k - 1) * Astride
            b = B[k]
            for i in Base.OneTo(size(A, 1))
                C[i] = buffered_operate!(mul_buffer, add_mul, C[i], A[aoffs+i], b)
            end
        end
    end # @inbounds
    return C
end

# This is incorrect if `C` is `LinearAlgebra.Symmetric` as we modify twice the
# same diagonal element.
function _add_mul_array(C::Matrix, A::AbstractMatrix, B::AbstractMatrix)
    mul_buffer = buffer_for(add_mul, eltype(C), eltype(A), eltype(B))
    @inbounds begin
        for i = 1:size(A, 1), j = 1:size(B, 2)
            Ctmp = C[i, j]
            for k = 1:size(A, 2)
                Ctmp = buffered_operate!(mul_buffer, add_mul, Ctmp, A[i, k], B[k, j])
            end
            C[i, j] = Ctmp
        end
    end # @inbounds
    return C
end

function mutable_operate!(
    ::typeof(add_mul),
    C::VecOrMat,
    A::AbstractMatrix,
    B::AbstractVecOrMat,
)
    _dim_check(C, A, B)
    _add_mul_array(C, A, B)
end

function mutable_operate!(::typeof(zero), C::Union{Vector,Matrix})
    # C may contain undefined values so we cannot call `zero!`
    for i in eachindex(C)
        @inbounds C[i] = zero(eltype(C))
    end
end

function mutable_operate_to!(
    C::AbstractArray,
    ::typeof(*),
    A::AbstractArray,
    B::AbstractArray,
)
    mutable_operate!(zero, C)
    return mutable_operate!(add_mul, C, A, B)
end

function undef_array(::Type{Array{T,N}}, axes::Vararg{Base.OneTo,N}) where {T,N}
    return Array{T,N}(undef, length.(axes))
end

# Does what `LinearAlgebra/src/matmul.jl` does for abstract
# matrices and vector, estimate the resulting element type,
# allocate the resulting array but it redirects to `mul_to!` instead of
# `LinearAlgebra.mul!`.
function operate(::typeof(*), A::AbstractMatrix{S}, B::AbstractVector{T}) where {T,S}
    C = undef_array(promote_array_mul(typeof(A), typeof(B)), axes(A, 1))
    return mutable_operate_to!(C, *, A, B)
end

function operate(::typeof(*), A::AbstractMatrix{S}, B::AbstractMatrix{T}) where {T,S}
    C = undef_array(promote_array_mul(typeof(A), typeof(B)), axes(A, 1), axes(B, 2))
    return mutable_operate_to!(C, *, A, B)
end

#mutable_copy(A::LinearAlgebra.Symmetric) = LinearAlgebra.Symmetric(mutable_copy(parent(A)), LinearAlgebra.sym_uplo(A.uplo))
# Broadcast applies the transpose
#mutable_copy(A::LinearAlgebra.Transpose) = LinearAlgebra.Transpose(mutable_copy(parent(A)))
#mutable_copy(A::LinearAlgebra.Adjoint) = LinearAlgebra.Adjoint(mutable_copy(parent(A)))

const TransposeOrAdjoint{T,MT} =
    Union{LinearAlgebra.Transpose{T,MT},LinearAlgebra.Adjoint{T,MT}}

_mirror_transpose_or_adjoint(x, ::LinearAlgebra.Transpose) = LinearAlgebra.transpose(x)

_mirror_transpose_or_adjoint(x, ::LinearAlgebra.Adjoint) = LinearAlgebra.adjoint(x)

_mirror_transpose_or_adjoint(
    A::Type{<:AbstractArray{T}},
    ::Type{<:LinearAlgebra.Transpose},
) where {T} = LinearAlgebra.Transpose{T,A}

_mirror_transpose_or_adjoint(
    A::Type{<:AbstractArray{T}},
    ::Type{<:LinearAlgebra.Adjoint},
) where {T} = LinearAlgebra.Adjoint{T,A}

similar_array_type(TA::Type{<:TransposeOrAdjoint{T,A}}, ::Type{S}) where {S,T,A} =
    _mirror_transpose_or_adjoint(similar_array_type(A, S), TA)

# dot product
function promote_array_mul(
    ::Type{<:TransposeOrAdjoint{S,<:AbstractVector}},
    ::Type{<:AbstractVector{T}},
) where {S,T}
    return promote_sum_mul(S, T)
end

function promote_array_mul(
    A::Type{<:TransposeOrAdjoint{S,V}},
    M::Type{<:AbstractMatrix{T}},
) where {S,T,V<:AbstractVector}
    B = promote_array_mul(_mirror_transpose_or_adjoint(M, A), V)
    return _mirror_transpose_or_adjoint(B, A)
end

function operate(
    ::typeof(*),
    x::LinearAlgebra.Adjoint{<:Any,<:AbstractVector},
    y::AbstractVector,
)
    return operate(LinearAlgebra.dot, parent(x), y)
end

function operate(
    ::typeof(*),
    x::TransposeOrAdjoint{<:Any,<:AbstractVector},
    y::AbstractMatrix,
)
    return _mirror_transpose_or_adjoint(
        operate(*, _mirror_transpose_or_adjoint(y, x), parent(x)),
        x,
    )
end

function operate(
    ::typeof(*),
    x::LinearAlgebra.Transpose{<:Any,<:AbstractVector},
    y::AbstractVector,
)
    lx = length(x)
    if lx != length(y)
        throw(
            DimensionMismatch(
                "first array has length $(lx) which does not match the length of the second, $(length(y)).",
            ),
        )
    end
    SumType = promote_sum_mul(eltype(x), eltype(y))
    if iszero(lx)
        return zero(SumType)
    end
    # We need a buffer to hold the intermediate multiplication.
    mul_buffer = buffer_for(add_mul, SumType, eltype(x), eltype(y))
    s = zero(SumType)
    for (Ix, Iy) in zip(eachindex(x), eachindex(y))
        s = @inbounds buffered_operate!(mul_buffer, add_mul, s, x[Ix], y[Iy])
    end
    return s
end

function operate(::typeof(LinearAlgebra.dot), x::AbstractArray, y::AbstractArray)
    lx = length(x)
    if lx != length(y)
        throw(
            DimensionMismatch(
                "first array has length $(lx) which does not match the length of the second, $(length(y)).",
            ),
        )
    end
    if iszero(lx)
        return LinearAlgebra.dot(zero(eltype(x)), zero(eltype(y)))
    end
    # We need a buffer to hold the intermediate multiplication.
    SumType = promote_sum_mul(eltype(x), eltype(y))
    mul_buffer = buffer_for(add_mul, SumType, eltype(x), eltype(y))
    s = zero(SumType)
    for (Ix, Iy) in zip(eachindex(x), eachindex(y))
        s = @inbounds buffered_operate!(
            mul_buffer,
            add_mul,
            s,
            LinearAlgebra.adjoint(x[Ix]),
            y[Iy],
        )
    end
    return s
end
