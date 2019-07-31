#  Copyright 2019, Gilles Peiffer, Benoît Legat, Sascha Timme, and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################

module MutableArithmetics

using LinearAlgebra, Base.GMP.MPZ

export DummyBigInt
export set_zero!, add_prod!


"""
    AbstractMutable

Generic supertype for types
implementing the MutableArithmetics interface.
"""
abstract type AbstractMutable end

"""
    DummyBigInt

Mutable wrapper type around `BigInt`.
The goal of this type is to allow the package to test itself;
hence its name.
"""
struct DummyBigInt <: AbstractMutable
    data::BigInt
end

# For convenience.
Base.zero(::Type{DummyBigInt}) = DummyBigInt(0)
Base.convert(::Type{DummyBigInt}, x::Int64) = DummyBigInt(x)

# Arithmetics
Base.:*(a::DummyBigInt, b::DummyBigInt) = DummyBigInt(a.data * b.data)
Base.:+(a::DummyBigInt, b::DummyBigInt) = DummyBigInt(a.data + b.data)
Base.:(==)(a::DummyBigInt, b::DummyBigInt) = a.data == b.data

"""
    add_prod!(x, args...)

Returns the result of `x + (*)(args...)` and possibly destroys
`x` (i.e., reusing its storage for the result, if possible).
"""
function add_prod! end

add_prod!(x, args...) = x + (*)(args...)

function add_prod!(x::BigInt, a::BigInt, b::BigInt)
    MPZ.add!(x, a*b)
    x
end

function add_prod!(x::BigInt, a::BigInt, b::BigInt, buf::BigInt)
    MPZ.mul!(buf, a, b)
    MPZ.add!(x, buf)
    x
end

function add_prod!(x::DummyBigInt, a::DummyBigInt, b::DummyBigInt)
    add_prod!(x.data, a.data, b.data)
    x
end

function add_prod!(x::DummyBigInt, a::DummyBigInt, b::DummyBigInt, buf::DummyBigInt)
    add_prod!(x.data, a.data, b.data, buf.data)
    x
end

"""
    set_zero!(x)

Set the value of `x` to zero.
"""
function set_zero! end

function set_zero!(x)
    x = zero(x)
    x
end

function set_zero!(x::BigInt)
    MPZ.set_si!(x, 0)
    x
end

function set_zero!(x::DummyBigInt)
    set_zero!(x.data)
    x
end

"""
    LinearAlgebra.mul!(C::AbstractVector, A::AbstractVecOrMat{T}, B::AbstractVector{T}) where {T <: AbstractMutable}

Computes the product between a matrix of `AbstractMutable`s and a vector of `AbstractMutable`s.
"""
function LinearAlgebra.mul!(C::AbstractVector, A::AbstractVecOrMat{T}, B::AbstractVector{T}) where {T<:AbstractMutable}
    # require_one_based_indexing is not exposed.
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
        set_zero!(C[i])
    end

    # We need a buffer to hold the intermediate multiplication.
    mul_buffer = zero(T)
    for k = 1:mB
        aoffs = (k-1)*Astride
        b = B[k]
        for i = 1:mA
            add_prod!(C[i], A[aoffs + i], b, mul_buffer)
        end
    end
    end # @inbounds
    C
end

function mul(A::AbstractVecOrMat{T}, B::AbstractVector{T}) where {T<:AbstractMutable}
    C = similar(B, eltype(A), axes(A,1))
    # C now contains only undefined values, we need to fill this with actual zeros
    for i in eachindex(C)
        C[i] = zero(eltype(A))
    end
    mul!(C, A, B)
end

Base.:*(A::AbstractVecOrMat{T}, B::AbstractVector{T}) where {T<:AbstractMutable} = mul(A, B)

end # module
