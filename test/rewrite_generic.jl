# Copyright (c) 2019 MutableArithmetics.jl contributors
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v.2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.

module TestRewriteGeneric

using Test

import MutableArithmetics

const MA = MutableArithmetics

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

macro test_rewrite(expr)
    return quote
        esc(
            @test MA.isequal_canonical(
                MA.@rewrite($expr, assume_sums_are_linear = false),
                $expr,
            )
        )
    end
end

function test_rewrite()
    x, expr = MA.rewrite(1 + 1, assume_sums_are_linear = false)
    @test x isa Symbol
    @test Meta.isexpr(expr, :(=), 2)
    return
end

function test_rewrite_basic_ops()
    @test_rewrite 1 + 2
    @test_rewrite 1 - 2
    @test_rewrite 1 * 2
    @test_rewrite 1 / 2
    @test_rewrite 1 + 2 - 3
    @test_rewrite 1 - 2 + 4
    @test_rewrite 1 * 2 / 3
    @test_rewrite 1 / 2 * 3
    return
end

function test_rewrite_not_call()
    x = [1, 2, 3]
    for i in 1:3
        @test MA.@rewrite(x[i], assume_sums_are_linear = false) == i
    end
    return
end

function test_rewrite_zero_argument()
    @test_rewrite Dict()
    return
end

function test_rewrite_sum_to_add_mul()
    @test_rewrite +1
    @test_rewrite +(-2)
    x = [1.2]
    @test MA.@rewrite(+x, assume_sums_are_linear = false) == x
    @test MA.@rewrite(x + x, assume_sums_are_linear = false) == 2 * x
    @test MA.@rewrite(+(x, x, x), assume_sums_are_linear = false) == 3 * x
    @test MA.@rewrite(+(x, x, x, x), assume_sums_are_linear = false) == 4 * x
    return
end

function test_rewrite_prod_to_add_mul()
    @test_rewrite 1 * 1
    @test_rewrite 2 * -2
    A = [1.0 2.0; 3.0 4.0]
    x = [5.0, 6.0]
    @test MA.@rewrite(A * x, assume_sums_are_linear = false) == A * x
    return
end

function test_rewrite_nonconcrete_vector()
    x = [5.0, 6.0]
    y = Vector{Union{Float64,String}}(x)
    @test MA.@rewrite(x' * y, assume_sums_are_linear = false) == x' * y
    @test MA.@rewrite(x .+ y, assume_sums_are_linear = false) == x .+ y
    # Reproducing buggy behavior in MA.@rewrite.
    @test_broken MA.@rewrite(x + y, assume_sums_are_linear = false) == x + x
    return
end

function test_rewrite_minus_to_add_mul()
    @test_rewrite -1
    @test_rewrite -(+2)
    @test_rewrite -(-2)
    x = [1.2]
    @test MA.@rewrite(-x, assume_sums_are_linear = false) == -x
    @test MA.@rewrite(x - x, assume_sums_are_linear = false) == [0.0]
    @test MA.@rewrite(-(x, x, x), assume_sums_are_linear = false) == -1 * x
    @test MA.@rewrite(-(x, x, x, x), assume_sums_are_linear = false) == -2 * x
    return
end

function test_rewrite_sum()
    @test MA.@rewrite(sum(i for i in 1:0), assume_sums_are_linear = false) ==
          MA.Zero()
    @test_rewrite sum(i for i in 1:10)
    @test_rewrite sum(i + i^2 for i in 1:10)
    @test_rewrite sum(i * i for i in 1:10)
    return
end

function test_rewrite_unknown_generators()
    @test_rewrite prod(i for i in 1:10)
    @test_rewrite minimum(i for i in 1:10)
    @test_rewrite maximum(i for i in 1:10)
    return
end

function test_rewrite_generator()
    # Unfiltered generators
    # Univariate
    @test_rewrite sum(i for i in 1:2)
    @test_rewrite sum(j^2 for j in 2:3)
    # Binary
    @test_rewrite sum(i + j^2 for i in 1:2, j in 2:3)
    @test_rewrite sum(i + j^2 for i in 1:2 for j in 2:3)
    # Tertiary
    @test_rewrite sum(i + j^2 for i in 1:2, j in 2:3, k in 3:4)
    @test_rewrite sum(i + j^2 for i in 1:2, j in 2:3 for k in i:j)
    @test_rewrite sum(i + j^2 for i in 1:2 for j in 2:3, k in 3:4)
    @test_rewrite sum(i + j^2 for i in 1:2 for j in 2:3 for k in 3:4)
    # Generators with dependent variables
    # This syntax is unsupported by Julia!
    @test MA.@rewrite(
        sum(i + j^2 for i in 1:2, j in i:3),
        assume_sums_are_linear = false,
    ) == 34
    @test MA.@rewrite(
        sum(i + j^2 for i in 1:2, j in 2:3, k in i:j),
        assume_sums_are_linear = false,
    ) == 68
    # Unnivariate generators with an if statement
    @test_rewrite sum(i for i in 1:2 if i >= 1)
    @test_rewrite sum(j^2 for j in 2:3 if j <= 3)
    @test_rewrite sum(i for i in 1:2 if isodd(i))
    @test_rewrite sum(j^2 for j in 2:3 if isodd(j))
    # Binary generators with an if statement
    @test_rewrite sum(i + j^2 for i in 1:2, j in 2:3 if i + j >= 0)
    @test_rewrite sum(i + j^2 for i in 1:2, j in 2:3 if i <= 1)
    @test_rewrite sum(i + j^2 for i in 1:2, j in 2:3 if j <= 2)
    # Flatten with generator on first key
    @test_rewrite sum(i for i in 1:2 if isodd(i) for j in 1:3)
    @test_rewrite sum(i for i in 1:3 if i <= 2 for j in 1:2)
    # Flatten with generator on second key
    @test_rewrite sum(i for i in 1:2 for j in 1:i if isodd(j))
    @test_rewrite sum(i for i in 1:2 for j in 1:i if i != j)
    # Generator with two filters
    @test_rewrite sum(i for i in 1:3 if i <= 2 for j in 1:2 if j >= 1)
    return
end

function test_rewrite_linear_algebra()
    @test_rewrite [1 2; 3 4] * [5.0, 6.0]
    @test_rewrite [1 2; 3 4] * [5, 6]
    @test_rewrite [1 2; 3 4] * [1 2; 3 4]
    @test_rewrite [1 2; 3 4] * Float64[1 2; 3 4]
    @test_rewrite [5, 6]' * [1 2; 3 4] * [5, 6]
    @test_rewrite 2 * [1 2; 3 4]
    @test_rewrite [1 2; 3 4] * 2
    @test_rewrite [1, 2] * 2.0
    @test_rewrite 2.0 * [1, 2]
    x = [1, 2]
    A = [1 2; 3 4]
    y = reshape(x, (1, length(x))) * A * x .- 1
    @test MA.@rewrite(
        reshape(x, (1, length(x))) * A * x .- 1,
        assume_sums_are_linear = false,
    ) == y
    return
end

function test_rewrite_broadcast_add_mul()
    @test_rewrite 1 .+ 2
    @test_rewrite [1, 2] .+ [2, 3]
    @test_rewrite [5.0, 6.0] .+ [1 2; 3 4] * [5.0, 6.0]
    @test_rewrite .+(BigInt(1))

    return
end

function test_rewrite_broadcast_sub_mul()
    @test_rewrite 1 .* 2
    @test_rewrite [1, 2] .* [2, 3]
    @test_rewrite [5.0, 6.0] .- [1 2; 3 4] * [5.0, 6.0]
    @test_rewrite .-(BigInt(1))
    return
end

function test_rewrite_repeated_sums()
    n = 10
    x = BigInt[2 for _ in 1:n]
    @test MA.@rewrite(sum(x[i] for i in 1:n) + sum(x[i] for i in 1:n)) == 4 * n
    @test MA.@rewrite(sum(x[i] for i in 1:n) - sum(x[i] for i in 1:n)) == 0
    @test MA.@rewrite(sum(x[i]^2 for i in 1:n) - sum(x[i] for i in 1:n)) ==
          sum(x .^ 2) - sum(x)
    @test MA.@rewrite(sum(x[i]^2 for i in 1:n) + sum(x[i] for i in 1:n)) ==
          sum(x .^ 2) + sum(x)
    @test MA.@rewrite(sum(x[i] for i in 1:n) - sum(x[i]^2 for i in 1:n)) ==
          sum(x) - sum(x .^ 2)
    @test MA.@rewrite(sum(x[i] for i in 1:n) + sum(x[i]^2 for i in 1:n)) ==
          sum(x) + sum(x .^ 2)
    return
end

function test_multi_prod()
    @test_rewrite *(sum([BigInt(5)] for _ in 1:2), 2)
    @test_rewrite *(sum([BigInt(5)] for _ in 1:2), BigInt(2))
    @test_rewrite *(sum([BigInt(5)] for _ in 1:2)', [2])
    @test_rewrite *(sum(BigInt(5) for _ in 1:2), BigInt(2), 3, BigInt(4))
    return
end

end  # module

TestRewriteGeneric.runtests()
