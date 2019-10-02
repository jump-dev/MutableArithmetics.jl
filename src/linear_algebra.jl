import LinearAlgebra

function mutability(::Type{<:Vector}, ::typeof(mul_to!),
                    ::Type{<:AbstractVecOrMat}, ::Type{<:AbstractVector})
    return IsMutable() # Assume the element type of the first vector is of correct type which is the case if it is called from `mul`
end
function mul_to_impl!(C::Vector, A::AbstractVecOrMat, B::AbstractVector)
    if mutability(eltype(C), muladd!, eltype(A), eltype(B)) isa NotMutable
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
    mul_buffer = zero(zero(eltype(A)) * zero(eltype(B)))
    for k = 1:mB
        aoffs = (k-1)*Astride
        b = B[k]
        for i = 1:mA
            # `C[i] = muladd_buf!(mul_buffer, C[i], A[aoffs + i], b)`
            muladd_buf_impl!(mul_buffer, C[i], A[aoffs + i], b)
        end
    end
    end # @inbounds
    return C
end

function mul(A::AbstractVecOrMat{T}, B::AbstractVector{S}) where {T, S}
    TS = Base.promote_op(LinearAlgebra.matprod, T, S)
    C = similar(B, TS, axes(A,1))
    # C now contains only undefined values, we need to fill this with actual zeros
    for i in eachindex(C)
        @inbounds C[i] = zero(TS)
    end
    return mul_to!(C, A, B)
end
