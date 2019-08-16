@testset "Matrix multiplication" begin
	@testset "matrix-vector product" begin
		A = BigInt[1 1 1; 1 1 1; 1 1 1]
		x = BigInt[1; 1; 1]
		y = BigInt[0; 0; 0]

		@test MA.mul(A, x) == BigInt[3; 3; 3]
		@test MA.mul_to!(y, A, x) == BigInt[3; 3; 3] && y == BigInt[3; 3; 3]

		MA.mutability(::Type{BigInt}, ::typeof(MA.zero!)) = MA.IsMutable()
		MA.zero_impl!(x::BigInt) = Base.GMP.MPZ.set_si!(x, 0)
		MA.mutability(::Type{BigInt}, ::typeof(MA.mul_to!), ::Type{BigInt}, ::Type{BigInt}) = MA.IsMutable()
		MA.mul_to_impl!(x::BigInt, a::BigInt, b::BigInt) = Base.GMP.MPZ.mul!(x, a, b)
		MA.mutability(::Type{BigInt}, ::typeof(MA.add_to!), ::Type{BigInt}, ::Type{BigInt}) = MA.IsMutable()
		MA.add_to_impl!(x::BigInt, a::BigInt, b::BigInt) = Base.GMP.MPZ.add!(x, a, b)
		MA.muladd_buf_impl!(buf::BigInt, a::BigInt, b::BigInt, c::BigInt) = Base.GMP.MPZ.add!(a, Base.GMP.MPZ.mul!(buf, b, c))

		A = BigInt[1 1 1; 1 1 1; 1 1 1]
		x = BigInt[1; 1; 1]
		y = BigInt[0; 0; 0]

		@test MA.mutability(y, MA.mul_to!, A, x) isa MA.IsMutable
		@test MA.mul(A, x) == BigInt[3; 3; 3]
		@test MA.mul_to!(y, A, x) == BigInt[3; 3; 3] && y == BigInt[3; 3; 3]
		@test_throws DimensionMismatch MA.mul(BigInt[1 1; 1 1], BigInt[])
		@test_throws DimensionMismatch MA.mul_to!(BigInt[], BigInt[1 1; 1 1], BigInt[1; 1])
	end
end
