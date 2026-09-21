# Copyright (c) 2026 MutableArithmetics.jl contributors
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v.2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.

module TestReduce

using Test
import LinearAlgebra
import MutableArithmetics as MA

@testset "Fused reductions with an initial accumulator" begin
    for T in (Int, BigInt), op in (MA.add_mul, MA.add_dot)
        a, b = T[1, 2, 3], T[4, 5, 6]
        originals = MA.mutable_copy.((a, b))
        init = T(7)
        result = @inferred MA.fused_map_reduce(op, a, b; init)
        @test result == 39
        @test (a, b) == originals
        if T === BigInt
            @test result === init
        end
        @test (@inferred MA.fused_map_reduce(op, a, b)) == 32
        @test iszero(MA.fused_map_reduce(op, T[], T[]))
        init = T(7)
        @test MA.fused_map_reduce(op, T[], T[]; init) === init
        @test_throws DimensionMismatch MA.fused_map_reduce(op, a, b[1:1]; init)
        @test init == 7
    end
    @test (@inferred MA.fused_map_reduce(MA.add_mul, [2], [3]; init = 0.5)) ==
          6.5
    @test MA.fused_map_reduce(
        MA.add_mul,
        BigInt[2, 3],
        BigInt[4, 5];
        init = 0,
    ) == 23
    init = big(7)
    @test MA.fused_map_reduce(MA.add_mul, Any[], Any[]; init) === init
    @test MA.fused_map_reduce(MA.sub_mul, [2, 3], [4, 5]; init = 30) == 7
    a, b = [1 + 2im, 3 - im], [2 - im, 1 + im]
    @test MA.fused_map_reduce(MA.add_dot, a, b; init = 2im) ==
          2im + LinearAlgebra.dot(a, b)
    @test MA.fused_map_reduce(MA.add_mul, Matrix{Int}[], Float64[]) isa MA.Zero
end

function test_allocations()
    a, b = [1, 2, 3], [4, 5, 6]
    @test (@inferred MA.fused_map_reduce(MA.add_mul, a, b; init = 7)) == 39
    @test (@allocated MA.fused_map_reduce(MA.add_mul, a, b; init = 7)) == 0
    return
end

@testset "Fused reduction allocations" begin
    test_allocations()
end

end
