# This file contains methods to implement the MutableArithmetics API for
# Base.Rational{T}.

# `Rational` is a `struct`, not a `mutable struct`, so if `T` is not
# mutable, we cannot mutate the rational.
mutability(::Type{Rational{T}}) where {T} = mutability(T)

mutable_copy(x::Rational) = Rational(mutable_copy(x.num), mutable_copy(x.den))

# zero

promote_operation(::typeof(zero), ::Type{Rational{T}}) where {T} = Rational{T}

function operate!(::typeof(zero), x::Rational)
    operate!(zero, x.num)
    operate!(one, x.den)
    return x
end

# one

promote_operation(::typeof(one), ::Type{Rational{T}}) where {T} = Rational{T}

function operate!(::typeof(one), x::Rational)
    operate!(one, x.num)
    operate!(one, x.den)
    return x
end

# +
function promote_operation(
    ::Union{typeof(+),typeof(-)},
    ::Type{Rational{S}},
    ::Type{Rational{T}}
) where {S,T}
    return Rational{promote_sum_mul(S, T)}
end

function promote_operation(
    op::Union{typeof(+),typeof(-)},
    ::Type{Rational{S}},
    ::Type{I}
) where {S,I<:Integer}
    return promote_operation(op, Rational{S}, Rational{I})
end

function promote_operation(
    op::Union{typeof(+),typeof(-)},
    ::Type{I},
    ::Type{Rational{S}}
) where {S,I<:Integer}
    return promote_operation(op, Rational{S}, Rational{I})
end

function operate_to!(output::Rational, ::typeof(+), x::Rational, y::Rational)
    xd, yd = Base.divgcd(promote(x.den, y.den)...)
    # TODO: Use `checked_mul` and `checked_add` like in Base
    operate_to!(output.num, *, x.num, yd)
    operate!(add_mul, output.num, y.num, xd)
    operate_to!(output.den, *, x.den, yd)
    return output
end

function operate_to!(output::Rational, ::typeof(+), x::Rational, y::Integer)
    # TODO Use `checked_mul` and `checked_add` like in Base
    operate_to!(output.num, *, x.den, y)
    operate!(+, output.num, x.num)
    operate_to!(output.den, *, x.den, oftype(x.den, 1))
    return output
end

function operate_to!(output::Rational, ::typeof(+), y::Integer, x::Rational)
    return operate_to!(output, +, x, y)
end

# unary -
function operate_to!(output::Rational, ::typeof(-), x::Rational)
    operate_to!(output.num, -, x.num)
    operate_to!(output.den, copy, x.den)
    return output
end

# binary -
function operate_to!(output::Rational, ::typeof(-), x::Rational, y::Rational)
    xd, yd = Base.divgcd(promote(x.den, y.den)...)
    # TODO: Use `checked_mul` and `checked_sub` like in Base
    operate_to!(output.num, *, x.num, yd)
    operate!(sub_mul, output.num, y.num, xd)
    operate_to!(output.den, *, x.den, yd)
    return output
end

function operate_to!(output::Rational, ::typeof(-), x::Rational, y::Integer)
    # TODO Use `checked_mul` and `checked_sub` like in Base
    operate_to!(output.num, *, x.den, y)
    operate!(-, output.num)
    operate!(+, output.num, x.num)
    operate_to!(output.den, copy, x.den)
    return output
end

function operate_to!(output::Rational, ::typeof(-), y::Integer, x::Rational)
    # TODO Use `checked_mul` and `checked_sub` like in Base
    operate_to!(output, -, x, y)
    operate_to!(output, -, output)
    return output
end

# *

function promote_operation(
    ::typeof(*),
    ::Type{Rational{S}},
    ::Type{Rational{T}},
) where {S,T}
    return Rational{promote_operation(*, S, T)}
end

function promote_operation(
    ::typeof(*),
    ::Type{Rational{S}},
    ::Type{I}
) where {S,I<:Integer}
    return promote_operation(*, Rational{S}, Rational{I})
end

function promote_operation(
    ::typeof(*),
    ::Type{I},
    ::Type{Rational{S}}
) where {S,I<:Integer}
    return promote_operation(*, Rational{S}, Rational{I})
end

function operate_to!(output::Rational, ::typeof(*), x::Rational, y::Rational)
    xn, yd = Base.divgcd(promote(x.num, y.den)...)
    xd, yn = Base.divgcd(promote(x.den, y.num)...)
    operate_to!(output.num, *, xn, yn)
    operate_to!(output.den, *, xd, yd)
    return output
end

function operate_to!(output::Rational, ::typeof(*), x::Rational, y::Integer)
    xn = x.num
    xd, yn = Base.divgcd(promote(x.den, y)...)
    operate_to!(output.num, *, xn, yn)
    operate_to!(output.den, copy, x.den)
    return output
end

function operate_to!(output::Rational, ::typeof(*), y::Integer, x::Rational)
   return operate_to!(output, *, x, y)
end

# //
function operate_to!(output::Rational, op::Union{typeof(/), typeof(//)}, x::Rational, y::Rational)
    xn, yn = Base.divgcd(promote(x.num, y.num)...)
    xd, yd = Base.divgcd(promote(x.den, y.den)...)
    operate_to!(output.num, *, xn, yd)
    operate_to!(output.den, *, xd, yn)
    return output
end

function operate_to!(output::Rational, op::Union{typeof(/), typeof(//)}, x::Rational, y::Integer)
    xn, yn = Base.divgcd(promote(x.num, y)...)
    operate_to!(output.num, copy, xn)
    operate_to!(output.den, *, x.den, yn)
    return output
end

function operate_to!(output::Rational, op::Union{typeof(/), typeof(//)}, x::Integer, y::Rational)
    xn, yd = Base.divgcd(promote(x, y.den)...)
    operate_to!(output.num, *, xn, yd)
    operate_to!(output.den, copy, y.num)
    return output
end

function operate_to!(output::Rational, op::Union{typeof(/), typeof(//)}, x::Integer, y::Integer)
    n, d = Base.divgcd(promote(x, y)...)
    operate_to!(output.num, copy, n)
    operate_to!(output.den, copy, d)
    return output
end

# gcd

function promote_operation(
    ::Union{typeof(gcd),typeof(lcm)},
    ::Type{Rational{S}},
    ::Type{Rational{T}},
) where {S,T}
    return Rational{promote_operation(gcd, S, T)}
end

function operate_to!(output::Rational, ::typeof(gcd), a::Rational, b::Rational)
    operate_to!(output.num, gcd, a.num, b.num)
    operate_to!(output.den, lcm, a.den, b.den)
    return output
end

function operate_to!(output::Rational, ::typeof(lcm), a::Rational, b::Rational)
    operate_to!(output.num, lcm, a.num, b.num)
    operate_to!(output.den, gcd, a.den, b.den)
    return output
end

function operate_to!(
    output::Rational,
    op::Union{typeof(+),typeof(-),typeof(*)},
    a::Rational,
    b::Rational,
    c::Vararg{Rational,N},
) where {N}
    operate_to!(output, op, a, b)
    return operate!(op, output, c...)
end

function operate!(op::Function, x::Rational, args::Vararg{Any,N}) where {N}
    return operate_to!(x, op, x, args...)
end

# add_mul and sub_mul

# Buffer to hold the product
function buffer_for(::AddSubMul, args::Vararg{Type{<:Rational},N}) where {N}
    return zero(promote_operation(*, args...))
end

function operate_to!(
    output::Rational,
    op::AddSubMul,
    x::Rational,
    y::Rational,
    z::Rational,
    args::Vararg{Rational,N},
) where {N}
    buffer = buffer_for(op, typeof(x), typeof(y), typeof(z), typeof.(args)...)
    return buffered_operate_to!(buffer, output, op, x, y, z, args...)
end

function buffered_operate_to!(
    buffer::Rational,
    output::Rational,
    op::AddSubMul,
    a::Rational,
    x::Rational,
    y::Rational,
    args::Vararg{Rational,N},
) where {N}
    operate_to!(buffer, *, x, y, args...)
    return operate_to!(output, add_sub_op(op), a, buffer)
end

function buffered_operate!(
    buffer::Rational,
    op::AddSubMul,
    x::Rational,
    args::Vararg{Any,N},
) where {N}
    return buffered_operate_to!(buffer, x, op, x, args...)
end

function operate_to!(
    output::Rational,
    op::Union{typeof(+),typeof(-),typeof(*)},
    args::Vararg{Scaling,N},
) where {N}
    return operate_to!(output, op, _scaling_to.(typeof(output), args)...)
end

function operate_to!(
    output::Rational,
    op::AddSubMul,
    x::Scaling,
    y::Scaling,
    z::Scaling,
    args::Vararg{Scaling,N},
) where {N}
    return operate_to!(
        output,
        op,
        _scaling_to(typeof(output), x),
        _scaling_to(typeof(output), y),
        _scaling_to(typeof(output), z),
        _scaling_to.(typeof(output), args)...,
    )
end

# Called for instance if `args` is `(v', v)` for a vector `v`.
function operate_to!(
    output::Rational,
    op::AddSubMul,
    x,
    y,
    z,
    args::Vararg{Any,N},
) where {N}
    return operate_to!(output, add_sub_op(op), x, *(y, z, args...))
end
