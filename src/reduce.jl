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

neutral_element(::typeof(+), ::Type{T}) where {T} = zero(T)
neutral_element(::typeof(+), ::Type{T}) where {T<:AbstractArray} = Zero()

"""
    zero_for(::Type{T}, inputs...)

Return a fresh additive identity for accumulating values from `inputs` with
promoted result type `T`. The inputs are iterables of scalar operands and must
not be mutated. The returned value can be mutated independently of the inputs
and of zeros returned by other calls.

The default returns `zero(T)` without inspecting the inputs.
For array-valued results, it retains the symbolic [`Zero`](@ref) identity.
Types whose zeros need value information can specialize this function to scan
the inputs before constructing the zero.
For example, algebra elements can prepare a common parent for all operands.

If empty inputs provide no context, a specialization can use `zero(T)` when
available. Otherwise it must report that no zero can be constructed.
"""
function zero_for(::Type{T}, inputs::Vararg{Any,N}) where {T,N}
    return neutral_element(+, T)
end

map_op(::AddSubMul) = *

map_op(::typeof(add_dot)) = LinearAlgebra.dot

function promote_map_reduce(op::Function, args::Vararg{Any,N}) where {N}
    return promote_operation(
        op,
        promote_operation(map_op(op), args...),
        args...,
    )
end

_concrete_eltype(x) = isempty(x) ? eltype(x) : typeof(first(x))

# Implicit sparse zeros add no context and need not be constructed or indexed.
_zero_entries(input) = input
_zero_inputs(op, args::Vararg{Any,N}) where {N} = map(_zero_entries, args)
# Prepare from the factors used by add_dot: adjoint can change their context.
function _zero_inputs(::typeof(add_dot), a, b)
    return (Iterators.map(adjoint, _zero_entries(a)), _zero_entries(b))
end

"""
    fused_map_reduce(op, args...)

Reduce corresponding entries of `args` using the fused operation `op`.
The accumulator is prepared by [`zero_for`](@ref) from the promoted result type
and the inputs (with the left operands adjointed for `add_dot`).
"""
function fused_map_reduce(op::F, args::Vararg{Any,N}) where {F<:Function,N}
    _check_same_length(args...)
    T = promote_map_reduce(op, _concrete_eltype.(args)...)
    accumulator = zero_for(T, _zero_inputs(op, args...)...)
    buffer = buffer_for(op, T, eltype.(args)...)
    for I in zip(eachindex.(args)...)
        accumulator =
            buffered_operate!!(buffer, op, accumulator, getindex.(args, I)...)
    end
    return accumulator
end

function operate(
    ::typeof(sum),
    a::AbstractArray;
    init = zero_for(
        promote_operation(+, eltype(a), eltype(a)),
        _zero_entries(a),
    ),
)
    return mapreduce(identity, add!!, a; init)
end
