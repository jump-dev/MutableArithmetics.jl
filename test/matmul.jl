using Test
import MutableArithmetics
const MA = MutableArithmetics

include("utilities.jl")

struct CustomArray{T,N} <: AbstractArray{T,N} end

import LinearAlgebra

function dot_test(x, y)
    @test MA.operate(LinearAlgebra.dot, x, y) == LinearAlgebra.dot(x, y)
    @test MA.operate(LinearAlgebra.dot, y, x) == LinearAlgebra.dot(y, x)
    @test MA.operate(*, x', y) == x' * y
    @test MA.operate(*, y', x) == y' * x
    @test MA.operate(*, LinearAlgebra.transpose(x), y) == LinearAlgebra.transpose(x) * y
    @test MA.operate(*, LinearAlgebra.transpose(y), x) == LinearAlgebra.transpose(y) * x
end

@testset "dot" begin
    x = [1im]
    y = [1]
    A = reshape(x, 1, 1)
    B = reshape(y, 1, 1)
    dot_test(x, x)
    dot_test(y, y)
    dot_test(A, A)
    dot_test(B, B)
    dot_test(x, y)
    dot_test(x, A)
    dot_test(x, B)
    dot_test(y, A)
    dot_test(y, B)
    dot_test(A, B)
end

@testset "promote_operation" begin
    x = [1]
    @test MA.promote_operation(*, typeof(x'), typeof(x)) == Int
    @test MA.promote_operation(*, typeof(transpose(x)), typeof(x)) == Int
end

@testset "convert" begin
    @test MA.scaling_convert(LinearAlgebra.UniformScaling{Int}, LinearAlgebra.I) isa
          LinearAlgebra.UniformScaling
    @test MA.scaling_convert(Int, LinearAlgebra.I) === 1
    @test MA.scaling_convert(Int, 1) === 1
    @test MA.operate(convert, LinearAlgebra.UniformScaling{Int}, LinearAlgebra.I) isa
          LinearAlgebra.UniformScaling
    @test MA.operate(convert, Int, LinearAlgebra.I) === 1
    @test MA.operate(convert, Int, 1) === 1
end

@testset "Errors" begin
    @testset "`promote_op` error" begin
        AT = CustomArray{Int,3}
        err = ErrorException(
            "`promote_operation(+, $(CustomArray{Int,3}), $(CustomArray{Int,3}))` not implemented yet, please report this.",
        )
        @test_throws err MA.promote_operation(+, AT, AT)
    end

    @testset "Dimension mismatch" begin
        A = zeros(1, 1)
        B = zeros(2, 2)
        err = DimensionMismatch(
            "Cannot sum matrices of size `(1, 1)` and size `(2, 2)`, the size of the two matrices must be equal.",
        )
        @test_throws err MA.@rewrite A + B
        x = ones(1)
        y = ones(2)
        err = DimensionMismatch(
            "first array has length 1 which does not match the length of the second, 2.",
        )
        @test_throws err MA.operate(*, x', y)
        @test_throws err MA.operate(*, LinearAlgebra.transpose(x), y)
        err = DimensionMismatch("matrix A has dimensions (2,2), vector B has length 1")
        @test_throws err MA.operate(*, x', B)
        a = zeros(0)
        @test iszero(@inferred MA.operate(LinearAlgebra.dot, a, a))
        @test iszero(@inferred MA.operate(*, a', a))
        @test iszero(@inferred MA.operate(*, LinearAlgebra.transpose(a), a))
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
            alloc_test(
                () -> MA.promote_operation(
                    +,
                    typeof(y),
                    MA.promote_operation(*, typeof(A), typeof(x)),
                ),
                0,
            )
            alloc_test(
                () -> MA.promote_operation(MA.add_mul, typeof(y), typeof(A), typeof(x)),
                0,
            )
            alloc_test(
                () -> MA.mutability(typeof(y), MA.add_mul, typeof(y), typeof(A), typeof(x)),
                0,
            )
            alloc_test(() -> MA.mutability(y, MA.add_mul, y, A, x), 0)
        end

        # 40 bytes to create the buffer
        # 8 bytes in the double for loop. FIXME: figure out why
        # Half size on 32-bit.
        n = Sys.WORD_SIZE == 64 ? 48 : 24
        alloc_test(() -> MA.add_mul!(y, A, x), n)
        alloc_test(() -> MA.operate_fallback!(MA.IsMutable(), MA.add_mul, y, A, x), n)
        alloc_test(() -> MA.operate!(MA.add_mul, y, A, x), n)
        alloc_test(() -> MA.mutable_operate!(MA.add_mul, y, A, x), n)
    end
    @testset "matrix-matrix product" begin
        A = [1 2 3; 4 5 6; 6 8 9]
        B = [1 -1 2; -2 3 1; 2 -3 1]
        C = [one(Int) for i = 1:3, j = 1:3]

        D = [3 -4 7; 6 -7 19; 8 -9 29]
        @test MA.mul(A, B) == D
        @test MA.mul_to!(C, A, B) == D
        @test C == D

        A = BigInt[1 2 3; 4 5 6; 6 8 9]
        B = BigInt[1 -1 2; -2 3 1; 2 -3 1]
        C = [one(BigInt) for i = 1:3, j = 1:3]

        D = BigInt[3 -4 7; 6 -7 19; 8 -9 29]
        @test MA.mul(A, B) == D
        @test MA.mul_to!(C, A, B) == D
        @test C == D

        @test MA.mutability(C, *, A, B) isa MA.IsMutable
        @test_throws DimensionMismatch MA.mul(BigInt[1 1; 1 1], zeros(BigInt, 1, 1))
        @test_throws DimensionMismatch MA.mul_to!(
            zeros(BigInt, 1, 1),
            BigInt[1 1; 1 1],
            zeros(BigInt, 2, 1),
        )

        @testset "mutability" begin
            alloc_test(() -> MA.promote_operation(*, typeof(A), typeof(B)), 0)
            alloc_test(
                () -> MA.promote_operation(
                    +,
                    typeof(C),
                    MA.promote_operation(*, typeof(A), typeof(B)),
                ),
                0,
            )
            alloc_test(
                () -> MA.promote_operation(MA.add_mul, typeof(C), typeof(A), typeof(B)),
                0,
            )
            alloc_test(
                () -> MA.mutability(typeof(C), MA.add_mul, typeof(C), typeof(A), typeof(B)),
                0,
            )
            alloc_test(() -> MA.mutability(C, MA.add_mul, C, A, B), 0)
        end

        # 40 bytes to create the buffer on 64-bit.
        # 8 bytes in the double for loop. FIXME: figure out why
        # Half size on 32-bit.
        n = Sys.WORD_SIZE == 64 ? 48 : 24
        alloc_test(() -> MA.add_mul!(C, A, B), n)
        alloc_test(() -> MA.operate!(MA.add_mul, C, A, B), n)
        alloc_test(() -> MA.mutable_operate!(MA.add_mul, C, A, B), n)
    end
end

@testset "matrix multiplication" begin
    X = ones(BigInt, 1, 1)
    M = ones(1, 1)
    C = X * M
    D = MA.operate!(MA.add_mul, C, X, M)
    @test D == X * M + X * M
end
