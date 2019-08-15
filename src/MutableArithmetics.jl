#  Copyright 2019, Gilles Peiffer, Benoît Legat, Sascha Timme, and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################

module MutableArithmetics

import LinearAlgebra

# These functions suffixed by `_impl!` fail if the first argument cannot be modified to be the result

# Example of mutable types that can implement this API: BigInt, Array, JuMP.AffExpr, MultivariatePolynomials.AbstractPolynomial
# `..._impl!` functions are similar to `JuMP.add_to_expression`
# `...!` functions are similar to `JuMP.destructive_add` and `MOI.Utilities.operate!`

"""
    add_to_impl!(a, b, c)

Write the result of the sum of `b` and `c` to `a`.
"""
function add_to_impl! end

"""
    add_impl!(a, b)

Write the result of the sum of `a` and `b` to `a`.
"""
function add_impl! end
# Fallback
add_impl!(a, b) = add_to_impl!(a, a, b)

"""
    mul_to_impl!(a, b, c)

Write the result of the product of `b` and `c` to `a`.
"""
function mul_to_impl! end

"""
    mul_impl!(a, b)

Write the result of the product of `a` and `b` to `a`.
"""
function mul_impl! end
# Fallback
mul_impl!(a, b) = mul_to_impl!(a, a, b)

"""
    muladd_to_impl!(a, b, c, d)

Write the result of `muladd(c, d, b)` to `a`.
"""
function muladd_to_impl! end

"""
    muladd_buf_to_impl!(buf, a, b, c, d)

Write the result of `muladd(c, d, b)` to `a`, possibly modifying `buf`.
"""
function muladd_buf_to_impl! end
# Fallback
function muladd_buf_to_impl!(buf, a, b, c, d)
    mul_to_impl!(buf, c, d)
    return add_to_impl!(a, b, buf)
end

"""
    muladd_impl!(a, b, c)

Write the result of `muladd(b, c, a)` to `a`.
"""
function muladd_impl! end
# Fallback
muladd_impl!(a, b, c) = muladd_to_impl!(a, a, b, c)

"""
    muladd_buf_impl!(buf, a, b, c)

Write the result of `muladd(b, c, a)` to `a`, possibly modifying `buf`.
"""
function muladd_buf_impl! end
# No fallback as it depends on the type of arguments whether we should
# redirect to `muladd_buf_to_impl!` or `muladd_impl!`


"""
    zero_impl!(a)

Write the result of `zero(a)` to `a`.
"""
function zero_impl! end

"""
    one_impl!(a)

Write the result of `one(a)` to `a`.
"""
function one_impl! end


# Define Traits
abstract type MutableTrait end
struct IsMutable <: MutableTrait end
struct NotMutable <: MutableTrait end

"""
    mutability(T::Type, ::typeof(op), args::Type...)::MutableTrait

Return `IsMutable` to indicate that `op(a::T, ::args[1], ...)` returns `a`.
That is, the result of the operation is stored in `a` and then `a` is returned.
Equivalently, returns whether `op_impl` is supported.
"""
mutability(::Type, op, args::Type...) = NotMutable()
mutability(x, op, args...) = mutability(typeof(x), op, typeof.(args)...)

"""
    add_to!(a, b, c)

Return the sum of `b` and `c`, possibly modifying `a`.
"""
function add_to! end
function add_to!(a, b, c)
    add_to!(a, b, c, mutability(a, add_to!, b, c))
end
# generic fallbacks
add_to!(a, b, c, ::NotMutable) = b + c
add_to!(a, b, c, ::IsMutable) = add_to_impl!(a, b, c)


"""
    add!(a, b)

Return the sum of `a` and `b`, possibly modifying `a`.
"""
function add! end
add!(a, b) = add!(a, b, mutability(a, add!, b))
# generic fallbacks
add!(a, b, ::NotMutable) = a + b
add!(a, b, ::IsMutable) = add_impl!(a, b)
mutability(T::Type, ::typeof(add!), U::Type) = mutability(T, add_to!, T, U)

"""
    mul_to!(a, b, c)

Return the product of `b` and `c`, possibly modifying `a`.
"""
function mul_to! end
function mul_to!(a, b, c)
    mul_to!(a, b, c, mutability(a, mul_to!, b, c))
end
# generic fallbacks
mul_to!(a, b, c, ::NotMutable) = b * c
mul_to!(a, b, c, ::IsMutable) = mul_to_impl!(a, b, c)

"""
    mul!(a, b)

Return the product of `a` and `b`, possibly modifying `a`.
"""
function mul! end
mul!(a, b) = mul!(a, b, mutability(a, mul!, b))
# generic fallbacks
mul!(a, b, ::NotMutable) = a * b
mul!(a, b, ::IsMutable) = mul_impl!(a, b)
mutability(T::Type, ::typeof(mul!), U::Type) = mutability(T, mul_to!, T, U)

"""
    muladd_to!(a, b, c, d)

Return `muladd(c, d, b)`, possibly modifying `a`.
"""
function muladd_to! end
function muladd_to!(a, b, c, d)
    muladd_to!(a, b, c, d, mutability(a, muladd_to!, b, c, d))
end
# generic fallbacks
muladd_to!(a, b, c, d, ::NotMutable) = muladd(c, d, b)
muladd_to!(a, b, c, d, ::IsMutable) = muladd_to_impl!(a, b, c, d)
function mutability(S::Type, ::typeof(muladd_to!), T::Type, U::Type, V::Type)
    return mutability(S, add_to!, T, typeof(zero(U) * zero(V)))
end

"""
    muladd!(a, b, c)

Return `muladd(b, c, a)`, possibly modifying `a`.
"""
function muladd! end
function muladd!(a, b, c)
    muladd!(a, b, c, mutability(a, muladd!, b, c))
end
# generic fallbacks
muladd!(a, b, c, ::NotMutable) = muladd(b, c, a)
muladd!(a, b, c, ::IsMutable) = muladd_impl!(a, b, c)
function mutability(S::Type, ::typeof(muladd!), T::Type, U::Type)
    return mutability(S, add!, typeof(zero(T) * zero(U)))
end


"""
    muladd_buf!(buf, a, b, c)

Return `muladd(b, c, a)`, possibly modifying `a` and `buf`.
"""
function muladd_buf! end
function muladd_buf!(buf, a, b, c)
    muladd_buf!(buf, a, b, c, mutability(a, muladd!, b, c))
end
# generic fallbacks
muladd_buf!(buf, a, b, c, ::NotMutable) = muladd(b, c, a)
muladd_buf!(buf, a, b, c, ::IsMutable) = muladd_buf_impl!(buf, a, b, c)
function mutability(S::Type, ::typeof(muladd_buf!), T::Type, U::Type, V::Type)
    return mutability(S, add_to!, T, typeof(zero(U) * zero(V)))
end

"""
    zero!(a)

Return `zero(a)`, possibly modifying `a`.
"""
function zero! end
zero!(x) = zero!(x, mutability(x, zero!))
# fallback
zero!(x, ::NotMutable) = zero(x)
zero!(x, ::IsMutable) = zero_impl!(x)

"""
    one!(a)

Return `one(a)`, possibly modifying `a`.
"""
function one! end
one!(x) = one!(x, mutability(x, one!))
# fallback
one!(x, ::NotMutable) = one(x)
one!(x, ::IsMutable) = one_impl!(x)

function mutability(::Type{<:Vector}, ::typeof(mul_to!),
                    ::Type{<:AbstractVecOrMat}, ::Type{<:AbstractVector})
    return IsMutable() # Assume the element type of the first vector is of correct type which is the case if it is called from `mul`
end
function mul_to_impl!(C::Vector, A::AbstractVecOrMat, B::AbstractVector)
    if mutability(eltype(C), muladd!, eltype(A), eltype(B)) isa NotMutable
        return LinearAlgebra.mul!(C, A, B)
    end
    # If `mutability(S, muladd!, T, U)` is `NotMutable`, we might as well redirect to `LinearAlgebra.mul!(C, A, B)`
    # in which case we can do `muladd_buf_impl!(mul_buffer, A[aoffs + i], b, C[i])` here instead of
    # `A[aoffs + i] = muladd_buf!(mul_buffer, A[aoffs + i], b, C[i])`
    mB = length(B)
    mA, nA = (size(A, 1), size(A, 2)) # lapack_size is not exposed.
    if mB != nA
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA), vector B has length $mB"))
    end
    if mA != length(C)
        throw(DimensionMismatch("result C has length $(length(C)), needs length $mA"))
    end

    Astride = size(A, 1)
    @inbounds begin
    for i = 1:mA
        C[i] = zero!(C[i])
    end

    # We need a buffer to hold the intermediate multiplication.
    mul_buffer = zero(zero(eltype(A)) * zero(eltype(B)))
    for k = 1:mB
        aoffs = (k-1)*Astride
        b = B[k]
        for i = 1:mA
            # `C[i] = muladd_buf!(mul_buffer, C[i], A[aoffs + i], b)`
            muladd_buf_impl!(mul_buffer, C[i], A[aoffs + i], b)
        end
    end
    end # @inbounds
    return C
end

function mul(A::AbstractVecOrMat{T}, B::AbstractVector{S}) where {T, S}
    TS = Base.promote_op(LinearAlgebra.matprod, T, S)
    C = similar(B, TS, axes(A,1))
    # C now contains only undefined values, we need to fill this with actual zeros
    for i in eachindex(C)
        @inbounds C[i] = zero(TS)
    end
    return mul_to!(C, A, B)
end

end # module
