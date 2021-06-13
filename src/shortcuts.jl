"""
    add_to!(a, b, c)

Return the sum of `b` and `c`, possibly modifying `a`.
"""
add_to!(output, args::Vararg{Any,N}) where {N} = operate_to!(output, +, args...)

"""
    add!(a, b, ...)

Return the sum of `a`, `b`, ..., possibly modifying `a`.
"""
add!(args::Vararg{Any,N}) where {N} = operate!(+, args...)

"""
    sub_to!(output, a, b)

Return the `a - b`, possibly modifying `output`.
"""
sub_to!(output, a, b) = operate_to!(output, -, a, b)

"""
    sub!(a, b)

Return `a - b`, possibly modifying `a`.
"""
sub!(a, b) = operate!(-, a, b)

"""
    mul_to!(a, b, c, ...)

Return the product of `b`, `c`, ..., possibly modifying `a`.
"""
mul_to!(output, args::Vararg{Any,N}) where {N} = operate_to!(output, *, args...)

"""
    mul!(a, b, ...)

Return the product of `a`, `b`, ..., possibly modifying `a`.
"""
mul!(args::Vararg{Any,N}) where {N} = operate!(*, args...)

"""
    mul(a, b, ...)

Shortcut for `operate(*, a, b, ...)`, see [`operate`](@ref).
"""
mul(args::Vararg{Any,N}) where {N} = operate(*, args...)

"""
    gcd_to!(a, b, c, ...)

Return the `gcd` of `b`, `c`, ..., possibly modifying `a`.
"""
gcd_to!(output, args::Vararg{Any,N}) where {N} = operate_to!(output, gcd, args...)

"""
    gcd!(a, b, ...)

Return the `gcd` of `a`, `b`, ..., possibly modifying `a`.
"""
gcd!(args::Vararg{Any,N}) where {N} = operate!(gcd, args...)

"""
    add_mul_to!(output, args...)

Return `add_mul(args...)`, possibly modifying `output`.
"""
add_mul_to!(output, args::Vararg{Any,N}) where {N} = operate_to!(output, add_mul, args...)

"""
    add_mul!(args...)

Return `add_mul(args...)`, possibly modifying `args[1]`.
"""
add_mul!(args::Vararg{Any,N}) where {N} = operate!(add_mul, args...)

"""
    add_mul_buf_to!(buffer, output, args...)

Return `add_mul(args...)`, possibly modifying `output` and `buffer`.
"""
function add_mul_buf_to!(buffer, output, args::Vararg{Any,N}) where {N}
    buffered_operate_to!(buffer, output, add_mul, args...)
end

"""
    add_mul_buf!(buffer, args...)

Return `add_mul(args...)`, possibly modifying `args[1]` and `buffer`.
"""
function add_mul_buf!(buffer, args::Vararg{Any,N}) where {N}
    buffered_operate!(buffer, add_mul, args...)
end

"""
    sub_mul_to!(output, args...)

Return `sub_mul(args...)`, possibly modifying `output`.
"""
sub_mul_to!(output, args::Vararg{Any,N}) where {N} = operate_to!(output, sub_mul, args...)

"""
    sub_mul!(args...)

Return `sub_mul(args...)`, possibly modifying `args[1]`.
"""
sub_mul!(args::Vararg{Any,N}) where {N} = operate!(sub_mul, args...)

"""
    sub_mul_buf_to!(buffer, output, args...)

Return `sub_mul(args...)`, possibly modifying `output` and `buffer`.
"""
function sub_mul_buf_to!(buffer, output, args::Vararg{Any,N}) where {N}
    buffered_operate_to!(buffer, output, sub_mul, args...)
end

"""
    sub_mul_buf!(buffer, args...)

Return `sub_mul(args...)`, possibly modifying `args[1]` and `buffer`.
"""
function sub_mul_buf!(buffer, args::Vararg{Any,N}) where {N}
    buffered_operate!(buffer, sub_mul, args...)
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
