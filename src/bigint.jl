mutability(::Type{BigInt}) = IsMutable()
mutable_copy(x::BigInt) = deepcopy(x)

# zero
promote_operation(::typeof(zero), ::Type{BigInt}) = BigInt
mutable_operate!(::typeof(zero), x::BigInt) = Base.GMP.MPZ.set_si!(x, 0)

# one
promote_operation(::typeof(one), ::Type{BigInt}) = BigInt
mutable_operate!(::typeof(one), x::BigInt) = Base.GMP.MPZ.set_si!(x, 1)

# +
promote_operation(::typeof(+), ::Vararg{Type{BigInt},N}) where {N} = BigInt
function mutable_operate_to!(output::BigInt, ::typeof(+), a::BigInt, b::BigInt)
    return Base.GMP.MPZ.add!(output, a, b)
end
#function mutable_operate_to!(output::BigInt, op::typeof(+), a::BigInt, b::LinearAlgebra.UniformScaling)
#    return mutable_operate_to!(output, op, a, b.λ)
#end

# -
promote_operation(::typeof(-), ::Vararg{Type{BigInt},N}) where {N} = BigInt
function mutable_operate_to!(output::BigInt, ::typeof(-), a::BigInt, b::BigInt)
    return Base.GMP.MPZ.sub!(output, a, b)
end

# *
promote_operation(::typeof(*), ::Vararg{Type{BigInt},N}) where {N} = BigInt
function mutable_operate_to!(output::BigInt, ::typeof(*), a::BigInt, b::BigInt)
    return Base.GMP.MPZ.mul!(output, a, b)
end

function mutable_operate_to!(
    output::BigInt,
    op::Union{typeof(+),typeof(-),typeof(*)},
    a::BigInt,
    b::BigInt,
    c::Vararg{BigInt,N},
) where {N}
    mutable_operate_to!(output, op, a, b)
    return mutable_operate!(op, output, c...)
end
function mutable_operate!(op::Function, x::BigInt, args::Vararg{Any,N}) where {N}
    mutable_operate_to!(x, op, x, args...)
end

# add_mul and sub_mul
# Buffer to hold the product
buffer_for(::AddSubMul, args::Vararg{Type{BigInt},N}) where {N} = BigInt()
function mutable_operate_to!(
    output::BigInt,
    op::AddSubMul,
    x::BigInt,
    y::BigInt,
    z::BigInt,
    args::Vararg{BigInt,N},
) where {N}
    return mutable_buffered_operate_to!(BigInt(), output, op, x, y, z, args...)
end

function mutable_buffered_operate_to!(
    buffer::BigInt,
    output::BigInt,
    op::AddSubMul,
    a::BigInt,
    x::BigInt,
    y::BigInt,
    args::Vararg{BigInt,N},
) where {N}
    mutable_operate_to!(buffer, *, x, y, args...)
    return mutable_operate_to!(output, add_sub_op(op), a, buffer)
end
function mutable_buffered_operate!(
    buffer::BigInt,
    op::AddSubMul,
    x::BigInt,
    args::Vararg{Any,N},
) where {N}
    return mutable_buffered_operate_to!(buffer, x, op, x, args...)
end

scaling_to_bigint(x::BigInt) = x
scaling_to_bigint(x::Number) = convert(BigInt, x)
scaling_to_bigint(J::LinearAlgebra.UniformScaling) = scaling_to_bigint(J.λ)
function mutable_operate_to!(
    output::BigInt,
    op::Union{typeof(+),typeof(-),typeof(*)},
    args::Vararg{Scaling,N},
) where {N}
    return mutable_operate_to!(output, op, scaling_to_bigint.(args)...)
end
function mutable_operate_to!(
    output::BigInt,
    op::AddSubMul,
    x::Scaling,
    y::Scaling,
    z::Scaling,
    args::Vararg{Scaling,N},
) where {N}
    return mutable_operate_to!(
        output,
        op,
        scaling_to_bigint(x),
        scaling_to_bigint(y),
        scaling_to_bigint(z),
        scaling_to_bigint.(args)...,
    )
end
# Called for instance if `args` is `(v', v)` for a vector `v`.
function mutable_operate_to!(
    output::BigInt,
    op::AddSubMul,
    x,
    y,
    z,
    args::Vararg{Any,N},
) where {N}
    return mutable_operate_to!(output, add_sub_op(op), x, *(y, z, args...))
end
