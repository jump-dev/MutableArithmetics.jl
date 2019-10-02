# This example contains a full implementation of the MutableArithmetics API for the BigInt datatype.

using MutableArithmetics
const MA = MutableArithmetics

# zero!
MA.mutability(::Type{BigInt}, ::typeof(MA.zero!)) = MA.IsMutable()
MA.zero_impl!(x::BigInt) = Base.GMP.MPZ.set_si!(x, 0)

# one!
MA.mutability(::Type{BigInt}, ::typeof(MA.one!)) = MA.IsMutable()
MA.one_impl!(x::BigInt) = Base.GMP.MPZ.set_si!(x, 1)

# add_to! / add!
MA.mutability(::Type{BigInt}, ::typeof(MA.add_to!), ::Type{BigInt}, ::Type{BigInt}) = MA.IsMutable()
MA.add_to_impl!(x::BigInt, a::BigInt, b::BigInt) = Base.GMP.MPZ.add!(x, a, b)
MA.add_impl!(a::BigInt, b::BigInt) = Base.GMP.MPZ.add!(a, a, b)

# mul_to! / mul!
MA.mutability(::Type{BigInt}, ::typeof(MA.mul_to!), ::Type{BigInt}, ::Type{BigInt}) = MA.IsMutable()
MA.mul_to_impl!(x::BigInt, a::BigInt, b::BigInt) = Base.GMP.MPZ.mul!(x, a, b)
MA.mul_impl!(a::BigInt, b::BigInt) = Base.GMP.MPZ.mul!(a, a, b)

# muladd_to! / muladd! / muladd_buf!
MA.muladd_to_impl!(dest::BigInt, b::BigInt, c::BigInt, d::BigInt) = Base.GMP.MPZ.add!(dest, Base.GMP.MPZ.mul!(BigInt(), c, d), b)
MA.muladd_impl!(a::BigInt, b::BigInt, c::BigInt) = Base.GMP.MPZ.add!(a, a, Base.GMP.MPZ.mul!(BigInt(), b, c))
MA.muladd_buf_impl!(buf::BigInt, a::BigInt, b::BigInt, c::BigInt) = Base.GMP.MPZ.add!(a, Base.GMP.MPZ.mul!(buf, b, c))
