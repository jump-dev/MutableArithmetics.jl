function operate(::typeof(sum), a::AbstractArray)
    return mapreduce(
        identity,
        add!,
        a,
        init = zero(promote_operation(+, eltype(a), eltype(a))),
    )
end
