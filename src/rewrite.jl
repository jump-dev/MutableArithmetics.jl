# Copyright (c) 2019 MutableArithmetics.jl contributors
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v.2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.

"""
    @rewrite(expr)

Return the value of `expr` exploiting the mutability of the temporary
expressions created for the computation of the result.

## Examples

The expression
```julia
MA.@rewrite(x + y * z + u * v * w)
```
is rewritten into
```julia
MA.add_mul!!(
    MA.add_mul!!(
        MA.copy_if_mutable(x),
        y, z),
    u, v, w)
```
"""
macro rewrite(expr)
    return rewrite_and_return(expr)
end

struct Zero end

## We need to copy `x` as it will be used as might be given by the user and be
## given as first argument of `operate!!`.
#Base.:(+)(zero::Zero, x) = copy_if_mutable(x)
## `add_mul(zero, ...)` redirects to `muladd(..., zero)` which calls `... + zero`.
#Base.:(+)(x, zero::Zero) = copy_if_mutable(x)
function operate(::typeof(add_mul), ::Zero, args::Vararg{Any,N}) where {N}
    return operate(*, args...)
end

function operate(::typeof(sub_mul), ::Zero, x)
    # `operate(*, x)` would redirect to `copy_if_mutable(x)` which would be a
    # useless copy.
    return operate(-, x)
end

function operate(::typeof(sub_mul), ::Zero, x, y, args::Vararg{Any,N}) where {N}
    return operate(-, operate(*, x, y, args...))
end

broadcast!!(::Union{typeof(add_mul),typeof(+)}, ::Zero, x) = copy_if_mutable(x)

broadcast!!(::typeof(add_mul), ::Zero, x, y) = x * y

# Needed in `@rewrite(1 .+ sum(1 for i in 1:0) * 1^2)`
Base.:*(z::Zero, ::Any) = z
Base.:*(::Any, z::Zero) = z
Base.:*(z::Zero, ::Zero) = z
Base.:+(::Zero, x::Any) = x
Base.:+(x::Any, ::Zero) = x
Base.:+(z::Zero, ::Zero) = z
Base.:-(::Zero, x::Any) = -x
Base.:-(x::Any, ::Zero) = x
Base.:-(z::Zero, ::Zero) = z
Base.:-(z::Zero) = z
Base.:+(z::Zero) = z
Base.:*(z::Zero) = z

function Base.:/(z::Zero, x::Any)
    if iszero(x)
        throw(DivideError())
    else
        return z
    end
end

# Needed by `@rewrite(BigInt(1) .+ sum(1 for i in 1:0) * 1^2)`
# since we don't require mutable type to support Zero in
# `mutable_operate!`.
_any_zero() = false
_any_zero(::Any, args::Vararg{Any,N}) where {N} = _any_zero(args...)
_any_zero(::Zero, ::Vararg{Any,N}) where {N} = true

function operate!!(
    op::Union{typeof(add_mul),typeof(sub_mul)},
    x,
    args::Vararg{Any,N},
) where {N}
    if _any_zero(args...)
        return x
    else
        return operate_fallback!!(mutability(x, op, x, args...), op, x, args...)
    end
end

# Needed for `@rewrite(BigInt(1) .+ sum(1 for i in 1:0) * 1^2)`
Base.broadcastable(z::Zero) = Ref(z)
Base.ndims(::Type{Zero}) = 0
Base.length(::Zero) = 1
Base.iterate(z::Zero) = (z, nothing)
Base.iterate(::Zero, ::Nothing) = nothing

"""
    rewrite(x)

Rewrite the expression `x` as specified in [`@rewrite`](@ref).
Returns a variable name as `Symbol` and the rewritten expression assigning the
value of the expression `x` to the variable.
"""
function rewrite(x)
    return MutableArithmetics2.rewrite(x)
end

"""
    rewrite_and_return(x)

Rewrite the expression `x` as specified in [`@rewrite`](@ref).
Return the rewritten expression returning the result.
"""
function rewrite_and_return(x)
    return MutableArithmetics2.rewrite_and_return(x)
end
