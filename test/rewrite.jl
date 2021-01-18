using LinearAlgebra, SparseArrays, Test
import MutableArithmetics
const MA = MutableArithmetics

@testset "Zero" begin
    z = MA.Zero()
    #@test zero(MA.Zero) isa MA.Zero
    @test z + z isa MA.Zero
    @test z + 1 == 1
    @test 1 + z == 1
    @test z * z isa MA.Zero
    @test z * 1 isa MA.Zero
    @test 1 * z isa MA.Zero
end

# Test that the macro call `m` throws an error exception during pre-compilation
macro test_macro_throws(error, m)
    # See https://discourse.julialang.org/t/test-throws-with-macros-after-pr-23533/5878
    return quote
        @test_throws $(esc(error)) try
            @eval $m
        catch catched_error
            throw(catched_error.error)
        end
    end
end

include("dummy.jl")

function error_test(x, y, z)
    # $(:(y[j=1])) does not print the same on Julia v1.3 or Julia 1.4
    err = ErrorException("Unexpected assignment in expression `$(:(y[j=1]))`.")
    @test_macro_throws err MA.@rewrite y[j = 1]
    err = ErrorException("Unexpected assignment in expression `$(:(x[i=1]))`.")
    @test_macro_throws err MA.@rewrite y + x[i = 1] + z
    err = ErrorException(
        "The curly syntax (sum{},prod{},norm2{}) is no longer supported. Expression: `sum{(i for i = 1:2)}`.",
    )
    @test_macro_throws err MA.@rewrite sum{i for i = 1:2}
    err = ErrorException("Unexpected comparison in expression `z >= y`.")
    @test_macro_throws err MA.@rewrite z >= y
    err = ErrorException("Unexpected comparison in expression `y <= z`.")
    @test_macro_throws err MA.@rewrite y <= z
    err = ErrorException("Unexpected comparison in expression `x <= y <= z`.")
    @test_macro_throws err MA.@rewrite x <= y <= z
end

using LinearAlgebra
using OffsetArrays

@testset "@rewrite with $T" for (T, supports_float) in [
    (Int, true),
    (Float64, true),
    (BigInt, true),
    (BigFloat, true),
    (DummyBigInt, false),
]
    @testset "Error" begin
        error_test(T(1), T(2), T(3))
    end
    @testset "Scalar" begin
        MA.Test.scalar_test(T(2))
        MA.Test.scalar_test(T(3))
    end
    @testset "Quadratic" begin
        exclude = T == DummyBigInt ? ["quadratic_division"] : String[]
        MA.Test.quadratic_test(T(1), T(2), T(3), T(4), exclude = exclude)
    end
    @testset "Sparse" begin
        MA.Test.sparse_test(T(4), T(5), T[8 1 9; 4 3 1; 2 0 8])
    end
    exclude = T == DummyBigInt ? ["broadcast_division", "matrix_vector_division"] : String[]
    @testset "Vector" begin
        MA.Test.array_test(T[-7, 1, 4], exclude = exclude)
        MA.Test.array_test(T[3, 2, 6], exclude = exclude)
    end
    @testset "Matrix" begin
        MA.Test.array_test(T[1 2; 2 4], exclude = exclude)
        MA.Test.array_test(T[2 4; -1 3], exclude = exclude)
        MA.Test.array_test(T[0 -4; 6 -5], exclude = exclude)
        MA.Test.array_test(T[5 1 9; -7 2 4; -2 -7 5], exclude = exclude)
    end
    @testset "Tensor" begin
        S = zeros(T, 2, 2, 2)
        S[1, :, :] = T[5 -8; 3 -7]
        S[2, :, :] = T[-2 8; 8 -1]
        MA.Test.array_test(S, exclude = exclude)
    end
    @testset "Non-array" begin
        x = T[2, 4, 3]
        MA.Test.non_array_test(x, OffsetArray(x, -length(x)))
    end
end

@testset "generator with not sum" begin
    @test MA.@rewrite(minimum(j^2 for j in 2:3)) == 4
    @test MA.@rewrite(maximum(j^2 for j in 2:3)) == 9
end

@testset "issue_76_trailing_dimensions" begin
    @testset "(4,) + (4,1)" begin
        X = [1, 2, 3, 4]
        Y = reshape([1, 2, 3, 4], (4, 1))
        @test MA.@rewrite(X + Y) == X + Y
        @test_broken MA.@rewrite(Y + X) == Y + X
    end
    @testset "(4,) + (4,1,1)" begin
        X = [1, 2, 3, 4]
        Y = reshape([1, 2, 3, 4], (4, 1, 1))
        @test MA.@rewrite(X + Y) == X + Y
        @test_broken MA.@rewrite(Y + X) == Y + X
    end
    @testset "(2,2) + (2,2,1)" begin
        X = [1 2; 3 4]
        Y = reshape([1, 2, 3, 4], (2, 2, 1))
        @test MA.@rewrite(X + Y) == X + Y
        @test_broken MA.@rewrite(Y + X) == Y + X
    end
    @testset "(4,) - (4,1)" begin
        X = [1, 3, 5, 7]
        Y = reshape([1, 2, 3, 4], (4, 1))
        @test MA.@rewrite(X - Y) == X - Y
        @test_broken MA.@rewrite(Y - X) == Y - X
    end
    @testset "(4,) - (4,1,1)" begin
        X = [1, 3, 5, 7]
        Y = reshape([1, 2, 3, 4], (4, 1, 1))
        @test MA.@rewrite(X - Y) == X - Y
        @test_broken MA.@rewrite(Y - X) == Y - X
    end
    @testset "(2,2) - (2,2,1)" begin
        X = [1 3; 5 7]
        Y = reshape([1, 2, 3, 4], (2, 2, 1))
        @test MA.@rewrite(X - Y) == X - Y
        @test_broken MA.@rewrite(Y - X) == Y - X
    end
end
