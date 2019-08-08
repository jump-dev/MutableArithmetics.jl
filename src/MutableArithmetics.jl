#  Copyright 2019, Gilles Peiffer, Benoît Legat, Sascha Timme, and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################

module MutableArithmetics

import LinearAlgebra

# Define Tratis
abstract type MutableTrait end
struct IsMutable <: MutableTrait end
struct NotMutable <: MutableTrait end


"""
    mutability(::Type)::MutableTrait

Return `IsMutable` to indicate that the type supports the MutableArithmetic API.
"""
mutability(::Type) = NotMutable()
mutability(x) = mutability(typeof(x))

"""
    add!([a::T], b::T, c::T) where T

Inplace addition of `b` and `c`, overwriting `a`. If `a` is not provided `c` is overwritten.
"""
function add! end
function add_impl! end
add!(b::T, c::T) where T = add!(c, b, c)
add!(a::T, b::T, c::T) where T = add!(a, b, c, mutability(T))
# generic fallbacks
add!(a, b, c, ::NotMutable) = b + c
add!(a, b, c, ::IsMutable) = add_impl!(a, b, c)

"""
    mul!([a::T], b::T, c::T) where T

Inplace multiplication of `b` and `c`, overwriting `a`. If `a` is not provided `c` is overwritten.
"""
function mul! end
function mul_impl! end

mul!(b::T, c::T) where T = mul!(c, b, c)
mul!(a::T, b::T, c::T) where T = mul!(a, b, c, mutability(T))

# generic fallbacks
mul!(a, b, c, ::NotMutable) = b * c
mul!(a, b, c, ::IsMutable) = mul_impl!(a, b, c)

"""
    mul!([u::T], x::T, y::T, z::T, b::T) where T

Inplace version of `muladd(x,y,z)` overwriting `u`. If `a` is not provided `z` is overwritten.
An intermediate value needs to be written therefore it is necesarsy to provide a buffer `b`.
"""
muladd!(x::T, y::T, z::T, b::T) where T = muladd!(z, x, y, z, b)
muladd!(u::T, x::T, y::T, z::T, b::T) where T = muladd!(u, x, y, z, b, mutability(T))
# fallback
muladd!(u, x, y, z, buffer, ::NotMutable) = muladd(x, y, z)
function muladd!(u, x, y, z, b, ::IsMutable)
    mul_impl!(b, x, y)
    add_impl!(u, z, b)
    u
end

function zero_impl! end
zero!(x) = zero!(x, mutability(x))
# fallback
zero!(x, ::NotMutable) = zero(x)
zero!(x, ::IsMutable) = zero_impl!(x)

function one_impl! end
one!(x) = one(x, mutability(x))
# fallback
one!(x, ::NotMutable) = one(x)
one!(x, ::IsMutable) = one_impl!(x)


"""
    mul!(C::AbstractVector{T}, A::AbstractVecOrMat{T}, B::AbstractVector{T})

Computes the product between a matrix of `AbstractMutable`s and a vector of `AbstractMutable`s.
"""
function mul!(C::AbstractVector{T}, A::AbstractVecOrMat{T}, B::AbstractVector{T}) where T
    mul!(C, A, B, mutability(T))
end
function mul!(C::AbstractVector, A::AbstractVecOrMat, B::AbstractVector, ::NotMutable)
    LinearAlgebra.mul!(C, A, B)
end
function mul!(C::AbstractVector{T}, A::AbstractVecOrMat{T}, B::AbstractVector{T}, ::IsMutable) where T
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
    mul_buffer = zero(T)
    for k = 1:mB
        aoffs = (k-1)*Astride
        b = B[k]
        for i = 1:mA
            muladd!(A[aoffs + i], b, C[i], mul_buffer)
        end
    end
    end # @inbounds
    C
end

function mul(A::AbstractVecOrMat{T}, B::AbstractVector{T}) where T
    C = similar(B, eltype(A), axes(A,1))
    # C now contains only undefined values, we need to fill this with actual zeros
    for i in eachindex(C)
        C[i] = zero(eltype(A))
    end
    mul!(C, A, B)
end

end # module
