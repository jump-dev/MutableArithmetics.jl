import LinearAlgebra

mutability(::Type{<:Vector}) = IsMutable()
function promote_operation(::typeof(*), ::Type{<:AbstractMatrix{S}}, ::Type{<:AbstractVector{T}}) where {S, T}
    return Vector{Base.promote_op(LinearAlgebra.matprod, S, T)}
end
function mutable_operate_to!(C::Vector, ::typeof(*), A::AbstractMatrix, B::AbstractVector)
    if mutability(eltype(C), add_mul, eltype(C), eltype(A), eltype(B)) isa NotMutable
        return LinearAlgebra.mul!(C, A, B)
    end
    # If `mutability(S, muladd!, T, U)` is `NotMutable`, we might as well redirect to `LinearAlgebra.mul!(C, A, B)`
    # in which case we can do `muladd_buf_impl!(mul_buffer, A[aoffs + i], b, C[i])` here instead of
    # `A[aoffs + i] = muladd_buf!(mul_buffer, A[aoffs + i], b, C[i])`
    mB = length(B)
    mA, nA = (size(A, 1), size(A, 2)) # lapack_size is not exposed.
    if mB != nA
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA), vector B has length $mB"))
    end
    if mA != length(C)
        throw(DimensionMismatch("result C has length $(length(C)), needs length $mA"))
    end

    Astride = size(A, 1)
    @inbounds begin
    for i = 1:mA
        C[i] = zero!(C[i])
    end

    # We need a buffer to hold the intermediate multiplication.
    mul_buffer = zero(promote_operation(*, eltype(A), eltype(B)))
    @inbounds for k = Base.OneTo(mB)
        aoffs = (k-1) * Astride
        b = B[k]
        for i = Base.OneTo(mA)
            # `C[i] = muladd_buf!(mul_buffer, C[i], A[aoffs + i], b)`
            mutable_buffered_operate!(mul_buffer, add_mul, C[i], A[aoffs + i], b)
        end
    end
    end
    return C
end

function mul(A::AbstractMatrix{S}, B::AbstractVector{T}) where {T, S}
    U = Base.promote_op(LinearAlgebra.matprod, S, T)
    C = similar(B, U, axes(A, 1))
    # C now contains only undefined values, we need to fill this with actual zeros
    for i in eachindex(C)
        @inbounds C[i] = zero(U)
    end
    return mul_to!(C, A, B)
end
