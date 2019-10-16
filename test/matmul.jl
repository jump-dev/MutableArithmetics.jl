@testset "Matrix multiplication" begin
	@testset "matrix-vector product" begin
		A = BigInt[1 1 1; 1 1 1; 1 1 1]
		x = BigInt[1; 1; 1]
		y = BigInt[0; 0; 0]

		@test MA.mul(A, x) == BigInt[3; 3; 3]
		@test MA.mul_to!(y, A, x) == BigInt[3; 3; 3] && y == BigInt[3; 3; 3]

		A = BigInt[1 1 1; 1 1 1; 1 1 1]
		x = BigInt[1; 1; 1]
		y = BigInt[0; 0; 0]

		@test MA.mutability(y, *, A, x) isa MA.IsMutable
		@test MA.mul(A, x) == BigInt[3; 3; 3]
		@test MA.mul_to!(y, A, x) == BigInt[3; 3; 3] && y == BigInt[3; 3; 3]
		@test_throws DimensionMismatch MA.mul(BigInt[1 1; 1 1], BigInt[])
		@test_throws DimensionMismatch MA.mul_to!(BigInt[], BigInt[1 1; 1 1], BigInt[1; 1])
	end
end
