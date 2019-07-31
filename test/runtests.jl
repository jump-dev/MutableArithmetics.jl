using Test
using MutableArithmetics

using LinearAlgebra, Base.GMP.MPZ

@testset "MutableArithmetics" begin
    @testset "matrix-vector product" begin
        @test DummyBigInt[1 2; 3 4]*DummyBigInt[1; 1] == DummyBigInt[3; 7]
        @test_throws DimensionMismatch DummyBigInt[0 0; 0 0]*DummyBigInt[]
        @test_throws DimensionMismatch LinearAlgebra.mul!(DummyBigInt[], DummyBigInt[0 0; 0 0], DummyBigInt[0; 0])
    end
end
