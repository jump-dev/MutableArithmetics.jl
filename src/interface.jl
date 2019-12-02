# Example of mutable types that can implement this API: BigInt, Array, JuMP.AffExpr, MultivariatePolynomials.AbstractPolynomial
# `mutable_operate!(add_mul, ...)` is similar to `JuMP.add_to_expression(...)`
# `operate!(add_mul, ...)` is similar to `JuMP.destructive_add(...)`
# `operate!` is similar to `MOI.Utilities.operate!`

"""
    promote_operation(op::Function, ArgsTypes::Type...)

Returns the type returned to the call `operate(op, args...)` where the types of
the arguments `args` are `ArgsTypes`.
"""
function promote_operation end
# Julia v1.0.x has trouble with inference with the `Vararg` method, see
# https://travis-ci.org/JuliaOpt/JuMP.jl/jobs/617606373
function promote_operation(op::Function, x::Type, y::Type)
    return typeof(op(zero(x), zero(y)))
end
function promote_operation(op::Function, args::Vararg{Type, N}) where N
    return typeof(op(zero.(args)...))
end
promote_operation(::typeof(*), ::Type{T}) where {T} = T
function promote_operation(::typeof(*), ::Type{S}, ::Type{T}, ::Type{U}, args::Vararg{Type, N}) where {S, T, U, N}
    return promote_operation(*, promote_operation(*, S, T), U, args...)
end

# Helpful error for common mistake
function promote_operation(op::Union{typeof(+), typeof(-), typeof(add_mul)}, A::Type{<:Array}, α::Type{<:Number})
    error("Operation `$op` between `$A` and `$α` is not allowed. You should use broadcast.")
end
function promote_operation(op::Union{typeof(+), typeof(-), typeof(add_mul)}, α::Type{<:Number}, A::Type{<:Array})
    error("Operation `$op` between `$α` and `$A` is not allowed. You should use broadcast.")
end

# Define Traits
abstract type MutableTrait end
struct IsMutable <: MutableTrait end
struct NotMutable <: MutableTrait end

"""
    mutability(T::Type, ::typeof(op), args::Type...)::MutableTrait

Return `IsMutable` to indicate an object of type `T` can be modified to be
equal to `op(args...)`.
"""
function mutability(T::Type, op, args::Vararg{Type, N}) where N
    if mutability(T) isa IsMutable && promote_operation(op, args...) == T
        return IsMutable()
    else
        return NotMutable()
    end
end
mutability(x, op, args::Vararg{Any, N}) where {N} = mutability(typeof(x), op, typeof.(args)...)
mutability(::Type) = NotMutable()

# `copy(::BigInt)` and `copy(::Array)` does not copy its elements so we need `deepcopy`.
function mutable_copy end
mutable_copy(x) = deepcopy(x)
mutable_copy(A::AbstractArray) = mutable_copy.(A)
copy_if_mutable_fallback(::NotMutable, x) = x
copy_if_mutable_fallback(::IsMutable, x) = mutable_copy(x)
copy_if_mutable(x) = copy_if_mutable_fallback(mutability(typeof(x)), x)

function mutable_operate_to_fallback(::NotMutable, output, op::Function, args...)
    throw(ArgumentError("Cannot call `mutable_operate_to!($output, $op, $(args...))` as `$output` cannot be modifed to equal the result of the operation. Use `operate!` or `operate_to!` instead which returns the value of the result (possibly modifying the first argument) to write generic code that also works when the type cannot be modified."))
end

function mutable_operate_to_fallback(::IsMutable, output, op::Function, args...)
    error("`mutable_operate_to!($(typeof(output)), $op, ", join(typeof.(args), ", "),
          ")` is not implemented yet.")
end

"""
    mutable_operate_to!(output, op::Function, args...)

Modify the value of `output` to be equal to the value of `op(args...)`. Can
only be called if `mutability(output, op, args...)` returns `true`.
"""
function mutable_operate_to!(output, op::Function, args::Vararg{Any, N}) where N
    mutable_operate_to_fallback(mutability(output, op, args...), output, op, args...)
end

"""
    mutable_operate!(op::Function, args...)

Modify the value of `args[1]` to be equal to the value of `op(args...)`. Can
only be called if `mutability(args[1], op, args...)` returns `true`.
"""
function mutable_operate!(op::Function, args::Vararg{Any, N}) where N
    mutable_operate_to!(args[1], op, args...)
end

buffer_for(::Function, args::Vararg{Type, N}) where {N} = nothing

"""
    mutable_buffered_operate_to!(buffer, output, op::Function, args...)

Modify the value of `output` to be equal to the value of `op(args...)`,
possibly modifying `buffer`. Can only be called if
`mutability(output, op, args...)` returns `true`.
"""
function mutable_buffered_operate_to!(::Nothing, output, op::Function, args::Vararg{Any, N}) where N
    return mutable_operate_to!(output, op, args...)
end

"""
    mutable_buffered_operate!(buffer, op::Function, args...)

Modify the value of `args[1]` to be equal to the value of `op(args...)`,
possibly modifying `buffer`. Can only be called if
`mutability(args[1], op, args...)` returns `true`.
"""
function mutable_buffered_operate!(buffer, op::Function, args::Vararg{Any, N}) where N
    mutable_buffered_operate_to!(buffer, args[1], op, args...)
end
function mutable_buffered_operate!(::Nothing, op::Function, args::Vararg{Any, N}) where N
    return mutable_operate!(op, args...)
end

"""
    operate_to!(output, op::Function, args...)

Returns the value of `op(args...)`, possibly modifying `output`.
"""
function operate_to!(output, op::Function, args::Vararg{Any, N}) where N
    return operate_to_fallback!(mutability(output, op, args...), output, op, args...)
end

function operate_to_fallback!(::NotMutable, output, op::Function, args::Vararg{Any, N}) where N
    return op(args...)
end
function operate_to_fallback!(::IsMutable, output, op::Function, args::Vararg{Any, N}) where N
    return mutable_operate_to!(output, op, args...)
end

"""
    operate!(op::Function, args...)

Returns the value of `op(args...)`, possibly modifying `args[1]`.
"""
function operate!(op::Function, args::Vararg{Any, N}) where N
    return operate_fallback!(mutability(args[1], op, args...), op, args...)
end

function operate_fallback!(::NotMutable, op::Function, args::Vararg{Any, N}) where N
    return op(args...)
end
function operate_fallback!(::IsMutable, op::Function, args::Vararg{Any, N}) where N
    return mutable_operate!(op, args...)
end

"""
    buffered_operate_to!(buffer, output, op::Function, args...)

Returns the value of `op(args...)`, possibly modifying `buffer` and `output`.
"""
function buffered_operate_to!(buffer, output, op::Function, args::Vararg{Any, N}) where N
    return buffered_operate_to_fallback!(mutability(output, op, args...),
                                         buffer, output, op, args...)
end

function buffered_operate_to_fallback!(::NotMutable, buffer, output, op::Function, args::Vararg{Any, N}) where N
    return op(args...)
end
function buffered_operate_to_fallback!(::IsMutable, buffer, output, op::Function, args::Vararg{Any, N}) where N
    return mutable_buffered_operate_to!(buffer, output, op, args...)
end

"""
    buffered_operate!(buffer, op::Function, args...)

Returns the value of `op(args...)`, possibly modifying `buffer`.
"""
function buffered_operate!(buffer, op::Function, args::Vararg{Any, N}) where N
    return buffered_operate_fallback!(mutability(args[1], op, args...),
                                      buffer, op, args...)
end

function buffered_operate_fallback!(::NotMutable, buffer, op::Function, args::Vararg{Any, N}) where N
    return op(args...)
end
function buffered_operate_fallback!(::IsMutable, buffer, op::Function, args::Vararg{Any, N}) where N
    return mutable_buffered_operate!(buffer, op, args...)
end
