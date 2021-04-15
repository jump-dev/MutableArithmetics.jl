mutability(::Type{BigFloat}) = IsMutable()
mutable_copy(x::BigFloat) = deepcopy(x)

@static if VERSION >= v"1.1.0-DEV.683"
    const MPFRRoundingMode = Base.MPFR.MPFRRoundingMode
else
    const MPFRRoundingMode = Int32
end

# zero
promote_operation(::typeof(zero), ::Type{BigFloat}) = BigFloat
function _set_si!(x::BigFloat, value)
    ccall(
        (:mpfr_set_si, :libmpfr),
        Int32,
        (Ref{BigFloat}, Clong, MPFRRoundingMode),
        x,
        value,
        Base.MPFR.ROUNDING_MODE[],
    )
    return x
end
mutable_operate!(::typeof(zero), x::BigFloat) = _set_si!(x, 0)

# one
promote_operation(::typeof(one), ::Type{BigFloat}) = BigFloat
mutable_operate!(::typeof(one), x::BigFloat) = _set_si!(x, 1)

# +
promote_operation(::typeof(+), ::Vararg{Type{BigFloat},N}) where {N} = BigFloat
function mutable_operate_to!(output::BigFloat, ::typeof(+), a::BigFloat, b::BigFloat)
    ccall(
        (:mpfr_add, :libmpfr),
        Int32,
        (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, MPFRRoundingMode),
        output,
        a,
        b,
        Base.MPFR.ROUNDING_MODE[],
    )
    return output
end
#function mutable_operate_to!(output::BigFloat, op::typeof(+), a::BigFloat, b::LinearAlgebra.UniformScaling)
#    return mutable_operate_to!(output, op, a, b.λ)
#end

# -
promote_operation(::typeof(-), ::Vararg{Type{BigFloat},N}) where {N} = BigFloat
function mutable_operate_to!(output::BigFloat, ::typeof(-), a::BigFloat, b::BigFloat)
    ccall(
        (:mpfr_sub, :libmpfr),
        Int32,
        (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, MPFRRoundingMode),
        output,
        a,
        b,
        Base.MPFR.ROUNDING_MODE[],
    )
    return output
end

# *
promote_operation(::typeof(*), ::Vararg{Type{BigFloat},N}) where {N} = BigFloat
function mutable_operate_to!(output::BigFloat, ::typeof(*), a::BigFloat, b::BigFloat)
    ccall(
        (:mpfr_mul, :libmpfr),
        Int32,
        (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, MPFRRoundingMode),
        output,
        a,
        b,
        Base.MPFR.ROUNDING_MODE[],
    )
    return output
end

function mutable_operate_to!(
    output::BigFloat,
    op::Union{typeof(+),typeof(-),typeof(*)},
    a::BigFloat,
    b::BigFloat,
    c::Vararg{BigFloat,N},
) where {N}
    mutable_operate_to!(output, op, a, b)
    return mutable_operate!(op, output, c...)
end
function mutable_operate!(op::Function, x::BigFloat, args::Vararg{Any,N}) where {N}
    mutable_operate_to!(x, op, x, args...)
end

# add_mul and sub_mul
# Buffer to hold the product
buffer_for(::AddSubMul, args::Vararg{Type{BigFloat},N}) where {N} = BigFloat()
function mutable_operate_to!(
    output::BigFloat,
    op::AddSubMul,
    x::BigFloat,
    y::BigFloat,
    z::BigFloat,
    args::Vararg{BigFloat,N},
) where {N}
    return mutable_buffered_operate_to!(BigFloat(), output, op, x, y, z, args...)
end

function mutable_buffered_operate_to!(
    buffer::BigFloat,
    output::BigFloat,
    op::AddSubMul,
    a::BigFloat,
    x::BigFloat,
    y::BigFloat,
    args::Vararg{BigFloat,N},
) where {N}
    mutable_operate_to!(buffer, *, x, y, args...)
    return mutable_operate_to!(output, add_sub_op(op), a, buffer)
end
function mutable_buffered_operate!(
    buffer::BigFloat,
    op::AddSubMul,
    x::BigFloat,
    args::Vararg{Any,N},
) where {N}
    return mutable_buffered_operate_to!(buffer, x, op, x, args...)
end

function _scaling_to_bigfloat(x)
    return convert(BigFloat, scaling_to_number(x))
end
function mutable_operate_to!(
    output::BigFloat,
    op::Union{typeof(+),typeof(-),typeof(*)},
    args::Vararg{Scaling,N},
) where {N}
    return mutable_operate_to!(output, op, _scaling_to_bigfloat.(args)...)
end
function mutable_operate_to!(
    output::BigFloat,
    op::AddSubMul,
    x::Scaling,
    y::Scaling,
    z::Scaling,
    args::Vararg{Scaling,N},
) where {N}
    return mutable_operate_to!(
        output,
        op,
        _scaling_to_bigfloat(x),
        _scaling_to_bigfloat(y),
        _scaling_to_bigfloat(z),
        _scaling_to_bigfloat.(args)...,
    )
end
# Called for instance if `args` is `(v', v)` for a vector `v`.
function mutable_operate_to!(
    output::BigFloat,
    op::AddSubMul,
    x,
    y,
    z,
    args::Vararg{Any,N},
) where {N}
    return mutable_operate_to!(output, add_sub_op(op), x, *(y, z, args...))
end
