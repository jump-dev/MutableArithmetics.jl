# Copyright (c) 2019 MutableArithmetics.jl contributors
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v.2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.

# This file contains methods to implement the MutableArithmetics API for
# Base.BigFloat.

mutability(::Type{BigFloat}) = IsMutable()

# Copied from `deepcopy_internal` implementation in Julia:
# https://github.com/JuliaLang/julia/blob/7d41d1eb610cad490cbaece8887f9bbd2a775021/base/mpfr.jl#L1041-L1050
function mutable_copy(x::BigFloat)
    d = x._d
    d′ = GC.@preserve d unsafe_string(pointer(d), sizeof(d)) # creates a definitely-new String
    return Base.MPFR._BigFloat(x.prec, x.sign, x.exp, d′)
end

const _MPFRRoundingMode = Base.MPFR.MPFRRoundingMode

# zero

promote_operation(::typeof(zero), ::Type{BigFloat}) = BigFloat

function _set_si!(x::BigFloat, value)
    ccall(
        (:mpfr_set_si, :libmpfr),
        Int32,
        (Ref{BigFloat}, Clong, _MPFRRoundingMode),
        x,
        value,
        Base.MPFR.ROUNDING_MODE[],
    )
    return x
end
operate!(::typeof(zero), x::BigFloat) = _set_si!(x, 0)

# one

promote_operation(::typeof(one), ::Type{BigFloat}) = BigFloat

operate!(::typeof(one), x::BigFloat) = _set_si!(x, 1)

# +

promote_operation(::typeof(+), ::Vararg{Type{BigFloat},N}) where {N} = BigFloat

function operate_to!(output::BigFloat, ::typeof(+), a::BigFloat, b::BigFloat)
    ccall(
        (:mpfr_add, :libmpfr),
        Int32,
        (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, _MPFRRoundingMode),
        output,
        a,
        b,
        Base.MPFR.ROUNDING_MODE[],
    )
    return output
end

# -

promote_operation(::typeof(-), ::Vararg{Type{BigFloat},N}) where {N} = BigFloat

function operate_to!(output::BigFloat, ::typeof(-), a::BigFloat, b::BigFloat)
    ccall(
        (:mpfr_sub, :libmpfr),
        Int32,
        (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, _MPFRRoundingMode),
        output,
        a,
        b,
        Base.MPFR.ROUNDING_MODE[],
    )
    return output
end

# *

promote_operation(::typeof(*), ::Vararg{Type{BigFloat},N}) where {N} = BigFloat

function operate_to!(output::BigFloat, ::typeof(*), a::BigFloat, b::BigFloat)
    ccall(
        (:mpfr_mul, :libmpfr),
        Int32,
        (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, _MPFRRoundingMode),
        output,
        a,
        b,
        Base.MPFR.ROUNDING_MODE[],
    )
    return output
end

function operate_to!(
    output::BigFloat,
    op::Union{typeof(+),typeof(-),typeof(*)},
    a::BigFloat,
    b::BigFloat,
    c::Vararg{BigFloat,N},
) where {N}
    operate_to!(output, op, a, b)
    return operate!(op, output, c...)
end

function operate!(op::Function, x::BigFloat, args::Vararg{Any,N}) where {N}
    return operate_to!(x, op, x, args...)
end

# add_mul and sub_mul

# Buffer to hold the product
buffer_for(::AddSubMul, args::Vararg{Type{BigFloat},N}) where {N} = BigFloat()

function operate_to!(
    output::BigFloat,
    op::AddSubMul,
    x::BigFloat,
    y::BigFloat,
    z::BigFloat,
    args::Vararg{BigFloat,N},
) where {N}
    return buffered_operate_to!(BigFloat(), output, op, x, y, z, args...)
end

function buffered_operate_to!(
    buffer::BigFloat,
    output::BigFloat,
    op::AddSubMul,
    a::BigFloat,
    x::BigFloat,
    y::BigFloat,
    args::Vararg{BigFloat,N},
) where {N}
    operate_to!(buffer, *, x, y, args...)
    return operate_to!(output, add_sub_op(op), a, buffer)
end

function buffered_operate!(
    buffer::BigFloat,
    op::AddSubMul,
    x::BigFloat,
    args::Vararg{Any,N},
) where {N}
    return buffered_operate_to!(buffer, x, op, x, args...)
end

_scaling_to_bigfloat(x) = _scaling_to(BigFloat, x)

function operate_to!(
    output::BigFloat,
    op::Union{typeof(+),typeof(-),typeof(*)},
    args::Vararg{Scaling,N},
) where {N}
    return operate_to!(output, op, _scaling_to_bigfloat.(args)...)
end

function operate_to!(
    output::BigFloat,
    op::AddSubMul,
    x::Scaling,
    y::Scaling,
    z::Scaling,
    args::Vararg{Scaling,N},
) where {N}
    return operate_to!(
        output,
        op,
        _scaling_to_bigfloat(x),
        _scaling_to_bigfloat(y),
        _scaling_to_bigfloat(z),
        _scaling_to_bigfloat.(args)...,
    )
end

# Called for instance if `args` is `(v', v)` for a vector `v`.
function operate_to!(
    output::BigFloat,
    op::AddSubMul,
    x,
    y,
    z,
    args::Vararg{Any,N},
) where {N}
    return operate_to!(output, add_sub_op(op), x, *(y, z, args...))
end
