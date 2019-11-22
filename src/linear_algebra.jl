mutability(::Type{<:Array}) = IsMutable()

# Sum

function promote_operation(op::Union{typeof(+), typeof(-)}, ::Type{Array{S, N}}, ::Type{Array{T, N}}) where {S, T, N}
    return Array{promote_operation(op, S, T), N}
end
function mutable_operate!(op::Union{typeof(+), typeof(-)}, A::Array{S, N}, B::Array{T, N}) where {S, T, N}
    for i in eachindex(A)
        A[i] = operate!(op, A[i], B[i])
    end
    return A
end

# UniformScaling
function promote_operation(op::typeof(+), ::Type{Array{T, 2}}, ::Type{LinearAlgebra.UniformScaling{S}}) where {S, T}
    return Array{promote_operation(op, T, S), 2}
end
function promote_operation(op::typeof(+), ::Type{LinearAlgebra.UniformScaling{S}}, ::Type{Array{T, 2}}) where {S, T}
    return Array{promote_operation(op, S, T), 2}
end
function mutable_operate!(::typeof(+), A::Matrix, B::LinearAlgebra.UniformScaling)
    n = LinearAlgebra.checksquare(A)
    for i in 1:n
        A[i, i] = operate!(+, A[i, i], B)
    end
    return A
end
function mutable_operate!(::typeof(add_mul), A::Matrix, B::Scaling, C::Scaling, D::Vararg{Scaling, N}) where N
    return mutable_operate!(+, A, *(B, C, D...))
end
function mutable_operate!(::typeof(add_mul), A::Array{S, N}, B::Array{T, N}, α::Vararg{Scaling, M}) where {S, T, N, M}
    for i in eachindex(A)
        A[i] = operate!(add_mul, A[i], B[i], α...)
    end
    return A
end
function mutable_operate!(::typeof(add_mul), A::Array{S, N}, α::Scaling, B::Array{T, N}, β::Vararg{Scaling, M}) where {S, T, N, M}
    for i in eachindex(A)
        A[i] = operate!(add_mul, A[i], α, B[i], β...)
    end
    return A
end
function mutable_operate!(::typeof(add_mul), A::Array{S, N}, α1::Scaling, α2::Scaling, B::Array{T, N}, β::Vararg{Scaling, M}) where {S, T, N, M}
    return mutable_operate!(add_mul, A, α1 * α2, B, β...)
end

# Product

function promote_operation(op::typeof(*), ::Type{Array{T, N}}, ::Type{S}) where {S, T, N}
    return Array{promote_operation(op, T, S), N}
end
function promote_operation(op::typeof(*), ::Type{S}, ::Type{Array{T, N}}) where {S, T, N}
    return Array{promote_operation(op, S, T), N}
end

function promote_operation(::typeof(*), ::Type{Matrix{S}}, ::Type{Vector{T}}) where {S, T}
    return Vector{Base.promote_op(LinearAlgebra.matprod, S, T)}
end
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
        for k = Base.OneTo(mB)
            aoffs = (k-1) * Astride
            b = B[k]
            for i = Base.OneTo(mA)
                mutable_buffered_operate!(mul_buffer, add_mul, C[i], A[aoffs + i], b)
            end
        end
    end # @inbounds
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
