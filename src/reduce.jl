function _same_length(a) end
function _same_length(a, b, c::Vararg{Any,N}) where {N}
    if length(a) != length(b)
        throw(
            DimensionMismatch(
                "one array has length $(length(a)) which does not match the length of the next one, $(length(b)).",
            ),
        )
    end
    _same_length(b, c...)
end
reduce_op(op::AddSubMul) = add_sub_op(op)
reduce_op(op::typeof(add_dot)) = +
neutral_element(::typeof(+), T::Type) = zero(T)
map_op(::AddSubMul) = *
map_op(::typeof(add_dot)) = LinearAlgebra.dot
function promote_map_reduce(op::Function, args::Vararg{Any,N}) where {N}
    T = promote_operation(
        op,
        promote_operation(map_op(op), args...),
        args...,
    )
end

function fused_map_reduce(
    op::F,
    args::Vararg{Any,N},
) where {F<:Function,N}
    _same_length(args...)
    T = promote_map_reduce(op, eltype.(args)...)
    accumulator = neutral_element(reduce_op(op), T)
    buffer = buffer_for(op, T, eltype.(args)...)
    for I in zip(eachindex.(args)...)
        accumulator = buffered_operate!(
            buffer,
            op,
            accumulator,
            getindex.(args, I)...,
        )
    end
    return accumulator
end
function operate(::typeof(sum), a::AbstractArray)
    return mapreduce(
        identity,
        add!,
        a,
        init = zero(promote_operation(+, eltype(a), eltype(a))),
    )
end
