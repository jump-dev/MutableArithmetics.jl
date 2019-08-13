using Test
using MutableArithmetics

using LinearAlgebra, Base.GMP.MPZ

@testset "MutableArithmetics" begin
    @testset "DummyBigInt arithmetics" begin
        @test DummyBigInt(2) + DummyBigInt(3) == DummyBigInt(5)
        @test DummyBigInt(2) * DummyBigInt(3) == DummyBigInt(6)
    end

    @testset "add_prod!" begin
        buf = zero(BigInt)
        dummy_buf = zero(DummyBigInt)
        @test add_prod!(1, 2, 3) == 7
        @test add_prod!(BigInt(1), BigInt(2), BigInt(3)) == BigInt(7)
        @test add_prod!(BigInt(1), BigInt(2), BigInt(3), buf) == BigInt(7)
        @test add_prod!(DummyBigInt(1), DummyBigInt(2), DummyBigInt(3)) == DummyBigInt(7)
        @test add_prod!(DummyBigInt(1), DummyBigInt(2), DummyBigInt(3), dummy_buf) == DummyBigInt(7)
    end

    @testset "set_zero!" begin
        @test set_zero!(1) == 0
        @test set_zero!(BigInt(1)) == BigInt(0)
        @test set_zero!(DummyBigInt(1)) == DummyBigInt(0)
    end

    @testset "matrix-vector product" begin
        @test DummyBigInt[1 2; 3 4]*DummyBigInt[1; 1] == DummyBigInt[3; 7]
        @test_throws DimensionMismatch DummyBigInt[0 0; 0 0]*DummyBigInt[]
        @test_throws DimensionMismatch LinearAlgebra.mul!(DummyBigInt[], DummyBigInt[0 0; 0 0], DummyBigInt[0; 0])
    end
end
