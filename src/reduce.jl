# Copyright (c) 2019 MutableArithmetics.jl contributors
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v.2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.

_check_same_length(::Any) = nothing

function _check_same_length(a, b, c::Vararg{Any,N}) where {N}
    if length(a) != length(b)
        throw(
            DimensionMismatch(
                "one array has length $(length(a)) which does not match the " *
                "length of the next one, $(length(b)).",
            ),
        )
    end
    return _check_same_length(b, c...)
end

reduce_op(op::AddSubMul) = add_sub_op(op)

reduce_op(::typeof(add_dot)) = +

map_op(::AddSubMul) = *

map_op(::typeof(add_dot)) = LinearAlgebra.dot

# We need a generated function here to help with type stability.
@generated function _accumulator(f::F, op::O, args::Vararg{Any,N}) where {F,O,N}
    expr = Expr(:call, :(map_op(op)))
    for i in 1:N
        push!(expr.args, :(zero(f(args[$i]))))
    end
    return expr
end

function fused_map_reduce(op::F, args::Vararg{Any,N}) where {F<:Function,N}
    _check_same_length(args...)
    if length(args[1]) == 0
        # If there are no arguments, we need to use the eltype of each vector to
        # infer the result type.
        return _accumulator(eltype, op, args...)
    end
    # If there are arguments, we use the first element. We don't use the eltype
    # because non-concrete vectors might have an eltype that isn't amenable to
    # zero(T). Ideally, this would be an error. But LinearAlgebra.dot supports
    # it so we do too.
    accumulator = _accumulator(first, op, args...)
    T = typeof(accumulator)
    buffer = buffer_for(op, T, eltype.(args)...)
    for I in zip(eachindex.(args)...)
        accumulator =
            buffered_operate!!(buffer, op, accumulator, getindex.(args, I)...)
    end
    return accumulator
end

function operate(::typeof(sum), a::AbstractArray)
    return mapreduce(
        identity,
        add!!,
        a;
        init = zero(promote_operation(+, eltype(a), eltype(a))),
    )
end
