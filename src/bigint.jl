mutability(::Type{BigInt}) = IsMutable()
mutable_copy(x::BigInt) = deepcopy(x)

# zero
promote_operation(::typeof(zero), ::Type{BigInt}) = BigInt
mutable_operate!(::typeof(zero), x::BigInt) = Base.GMP.MPZ.set_si!(x, 0)

# one
promote_operation(::typeof(one), ::Type{BigInt}) = BigInt
mutable_operate!(::typeof(one), x::BigInt) = Base.GMP.MPZ.set_si!(x, 1)

# +
promote_operation(::typeof(+), ::Vararg{Type{BigInt}, N}) where {N} = BigInt
function mutable_operate_to!(output::BigInt, ::typeof(+), a::BigInt, b::BigInt)
    return Base.GMP.MPZ.add!(output, a, b)
end
#function mutable_operate_to!(output::BigInt, op::typeof(+), a::BigInt, b::LinearAlgebra.UniformScaling)
#    return mutable_operate_to!(output, op, a, b.λ)
#end

# *
promote_operation(::typeof(*), ::Vararg{Type{BigInt}, N}) where {N} = BigInt
function mutable_operate_to!(output::BigInt, ::typeof(*), a::BigInt, b::BigInt)
    return Base.GMP.MPZ.mul!(output, a, b)
end

function mutable_operate_to!(output::BigInt, op::Union{typeof(*), typeof(+)},
                             a::BigInt, b::BigInt, c::Vararg{BigInt, N}) where N
    mutable_operate_to!(output, op, a, b)
    return mutable_operate!(op, output, c...)
end
function mutable_operate!(op::Function, x::BigInt, args::Vararg{Any, N}) where N
    mutable_operate_to!(x, op, x, args...)
end

# add_mul
# Buffer to hold the product
buffer_for(::typeof(add_mul), args::Vararg{Type{BigInt}, N}) where {N} = BigInt()
function mutable_operate_to!(output::BigInt, ::typeof(add_mul), x::BigInt, y::BigInt, z::BigInt, args::Vararg{BigInt, N}) where N
    return mutable_buffered_operate_to!(BigInt(), output, add_mul, x, y, z, args...)
end

function mutable_buffered_operate_to!(buffer::BigInt, output::BigInt, ::typeof(add_mul),
                                      a::BigInt, x::BigInt, y::BigInt, args::Vararg{BigInt, N}) where N
    mutable_operate_to!(buffer, *, x, y, args...)
    return mutable_operate_to!(output, +, a, buffer)
end
function mutable_buffered_operate!(buffer::BigInt, op::typeof(add_mul), x::BigInt, args::Vararg{Any, N}) where N
    return mutable_buffered_operate_to!(buffer, x, op, x, args...)
end

scaling_to_bigint(x::BigInt) = x
scaling_to_bigint(x::Number) = convert(BigInt, x)
scaling_to_bigint(J::LinearAlgebra.UniformScaling) = scaling_to_bigint(J.λ)
function mutable_operate_to!(output::BigInt, op::Union{typeof(+), typeof(*)}, args::Vararg{Scaling, N}) where N
    return mutable_operate_to!(output, op, scaling_to_bigint.(args)...)
end
function mutable_operate_to!(output::BigInt, op::typeof(add_mul), x, y, z, args::Vararg{Scaling, N}) where N
    return mutable_operate_to!(
        output, op, scaling_to_bigint(x), scaling_to_bigint(y),
        scaling_to_bigint(z), scaling_to_bigint.(args)...)
end
