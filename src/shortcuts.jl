"""
    add_to!(a, b, c)

Return the sum of `b` and `c`, possibly modifying `a`.
"""
add_to!(output, args...) = operate_to!(output, +, args...)

"""
    add!(a, b, ...)

Return the sum of `a`, `b`, ..., possibly modifying `a`.
"""
add!(args...) = operate!(+, args...)

"""
    mul_to!(a, b, c)

Return the product of `b` and `c`, possibly modifying `a`.
"""
mul_to!(output, args::Vararg{Any, N}) where {N} = operate_to!(output, *, args...)

"""
    mul!(a, b, ...)

Return the product of `a`, `b`, ..., possibly modifying `a`.
"""
mul!(args...) = operate!(*, args...)

"""
    add_mul(a, args...)

Return `a + *(args...)`. Note that `add_mul(a, b, c) = muladd(b, c, a)`.
"""
function add_mul end
add_mul(a, b, c) = muladd(b, c, a)

function promote_operation(::typeof(add_mul), T::Type, args::Vararg{Type, N}) where N
    return promote_operation(+, T, promote_operation(*, args...))
end

"""
    add_mul_to!(output, args...)

Return `add_mul(args...)`, possibly modifying `output`.
"""
add_mul_to!(output, args...) = operate_to!(output, add_mul, args...)

"""
    add_mul!(args...)

Return `add_mul(args...)`, possibly modifying `args[1]`.
"""
add_mul!(args...) = operate!(add_mul, args...)

"""
    add_mul_buf_to!(buffer, output, args...)

Return `add_mul(args...)`, possibly modifying `output` and `buffer`.
"""
function add_mul_buf_to!(buffer, output, args...)
    buffered_operate_to!(buffer, output, add_mul, args...)
end

"""
    add_mul_buf!(buffer, args...)

Return `add_mul(args...)`, possibly modifying `args[1]` and `buffer`.
"""
function add_mul_buf!(buffer, args...)
    buffered_operate!(buffer, add_mul, args...)
end

"""
    zero!(a)

Return `zero(a)`, possibly modifying `a`.
"""
zero!(a) = operate!(zero, a)

"""
    one!(a)

Return `one(a)`, possibly modifying `a`.
"""
one!(a) = operate!(one, a)
