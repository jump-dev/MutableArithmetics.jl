using Test
import MutableArithmetics
const MA = MutableArithmetics

include("utilities.jl")

struct CustomArray{T, N} <: AbstractArray{T, N} end

import LinearAlgebra

@testset "Scaling convert" begin
    @test MA.scaling_convert(LinearAlgebra.UniformScaling{Int}, LinearAlgebra.I) isa LinearAlgebra.UniformScaling
    @test MA.scaling_convert(Int, LinearAlgebra.I) === 1
    @test MA.scaling_convert(Int, 1) === 1
end

@testset "Errors" begin
    @testset "`promote_op` error" begin
        AT = CustomArray{Int, 3}
        err = ErrorException("`promote_operation(+, CustomArray{Int64,3}, CustomArray{Int64,3})` not implemented yet, please report this.")
        @test_throws err MA.promote_operation(+, AT, AT)
    end

    @testset "Dimension mismatch" begin
        A = zeros(1, 1)
        B = zeros(2, 2)
        err = DimensionMismatch("Cannot sum matrices of size `(1, 1)` and size `(2, 2)`, the size of the two matrices must be equal.")
        @test_throws err MA.@rewrite A + B
    end
end

@testset "Matrix multiplication" begin
    @testset "matrix-vector product" begin
        A = [1 1 1; 1 1 1; 1 1 1]
        x = [1; 1; 1]
        y = [0; 0; 0]

        @test MA.mul(A, x) == [3; 3; 3]
        @test MA.mul_to!(y, A, x) == [3; 3; 3] && y == [3; 3; 3]

        A = BigInt[1 1 1; 1 1 1; 1 1 1]
        x = BigInt[1; 1; 1]
        y = BigInt[0; 0; 0]

        @test MA.mutability(y, *, A, x) isa MA.IsMutable
        @test MA.mul(A, x) == BigInt[3; 3; 3]
        @test MA.mul_to!(y, A, x) == BigInt[3; 3; 3] && y == BigInt[3; 3; 3]
        @test_throws DimensionMismatch MA.mul(BigInt[1 1; 1 1], BigInt[])
        @test_throws DimensionMismatch MA.mul_to!(BigInt[], BigInt[1 1; 1 1], BigInt[1; 1])

        @testset "mutability" begin
            alloc_test(() -> MA.promote_operation(*, typeof(A), typeof(x)), 0)
            alloc_test(() -> MA.promote_operation(+, typeof(y), MA.promote_operation(*, typeof(A), typeof(x))), 0)
            alloc_test(() -> MA.promote_operation(MA.add_mul, typeof(y), typeof(A), typeof(x)), 0)
            alloc_test(() -> MA.mutability(typeof(y), MA.add_mul, typeof(y), typeof(A), typeof(x)), 0)
            alloc_test(() -> MA.mutability(y, MA.add_mul, y, A, x), 0)
        end

        # 40 bytes to create the buffer
        # 8 bytes in the double for loop. FIXME: figure out why
        alloc_test(() -> MA.add_mul!(y, A, x), 48)
        alloc_test(() -> MA.operate_fallback!(MA.IsMutable(), MA.add_mul, y, A, x), 48)
        alloc_test(() -> MA.operate!(MA.add_mul, y, A, x), 48)
        alloc_test(() -> MA.mutable_operate!(MA.add_mul, y, A, x), 48)
    end
    @testset "matrix-matrix product" begin
        A = [1 2 3; 4 5 6; 6 8 9]
        B = [1 -1 2; -2 3 1; 2 -3 1]
        C = [one(Int) for i in 1:3, j in 1:3]

        D = [3 -4 7; 6 -7 19; 8 -9 29]
        @test MA.mul(A, B) == D
        @test MA.mul_to!(C, A, B) == D
        @test C == D

        A = BigInt[1 2 3; 4 5 6; 6 8 9]
        B = BigInt[1 -1 2; -2 3 1; 2 -3 1]
        C = [one(BigInt) for i in 1:3, j in 1:3]

        D = BigInt[3 -4 7; 6 -7 19; 8 -9 29]
        @test MA.mul(A, B) == D
        @test MA.mul_to!(C, A, B) == D
        @test C == D

        @test MA.mutability(C, *, A, B) isa MA.IsMutable
        @test_throws DimensionMismatch MA.mul(BigInt[1 1; 1 1], zeros(BigInt, 1, 1))
        @test_throws DimensionMismatch MA.mul_to!(zeros(BigInt, 1, 1), BigInt[1 1; 1 1], zeros(BigInt, 2, 1))

        @testset "mutability" begin
            alloc_test(() -> MA.promote_operation(*, typeof(A), typeof(B)), 0)
            alloc_test(() -> MA.promote_operation(+, typeof(C), MA.promote_operation(*, typeof(A), typeof(B))), 0)
            alloc_test(() -> MA.promote_operation(MA.add_mul, typeof(C), typeof(A), typeof(B)), 0)
            alloc_test(() -> MA.mutability(typeof(C), MA.add_mul, typeof(C), typeof(A), typeof(B)), 0)
            alloc_test(() -> MA.mutability(C, MA.add_mul, C, A, B), 0)
        end

        # 40 bytes to create the buffer
        # 8 bytes in the double for loop. FIXME: figure out why
        alloc_test(() -> MA.add_mul!(C, A, B), 48)
        alloc_test(() -> MA.operate!(MA.add_mul, C, A, B), 48)
        alloc_test(() -> MA.mutable_operate!(MA.add_mul, C, A, B), 48)
    end
end
