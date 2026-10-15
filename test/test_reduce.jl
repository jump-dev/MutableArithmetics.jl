# Copyright (c) 2026 MutableArithmetics.jl contributors
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v.2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.

module TestReduce

using Test
import LinearAlgebra
import MutableArithmetics as MA
import SparseArrays

struct UnreadableInput end
function Base.iterate(::UnreadableInput)
    return error("The type-based zero must not read inputs")
end

@testset "Zeros from inputs" begin
    for T in (Int, Float64, BigInt)
        a = @inferred MA.zero_for(T, UnreadableInput())
        b = @inferred MA.zero_for(T, UnreadableInput())
        @test iszero(a) && iszero(b)
        @test typeof(a) === T
        if T === BigInt
            @test a !== b
            MA.operate!(+, a, 1)
            @test iszero(b)
        end
        @test (@inferred MA.operate(sum, T[1, 2, 3])) == 6
        @test iszero(MA.operate(sum, T[]))
    end
    @test MA.zero_for(Matrix{Int}, UnreadableInput()) isa MA.Zero
end

@testset "Sparse reductions preserve adjoints and inputs" begin
    for values in ([1, 0, 2], BigInt[1, 0, 2], [1 + 2im, 0, 3 - im])
        a = SparseArrays.sparse(values)
        b = [2, 3, 4]
        @test (@inferred MA.operate(sum, a)) == sum(values)
        @test (@inferred MA.operate(sum, transpose(a))) == sum(values)
        @test (@inferred MA.operate(sum, adjoint(a))) == conj(sum(values))
        @test (@inferred MA.fused_map_reduce(MA.add_dot, a, b)) ==
              LinearAlgebra.dot(values, b)
        @test a == values
    end
end

@testset "Fused reductions" begin
    for T in (Int, BigInt), op in (MA.add_mul, MA.add_dot)
        a, b = T[1, 2, 3], T[4, 5, 6]
        originals = MA.mutable_copy.((a, b))
        @test (@inferred MA.fused_map_reduce(op, a, b)) == 32
        @test (@inferred MA.fused_map_reduce(MA.sub_mul, a, b)) == -32
        @test iszero(MA.fused_map_reduce(op, T[], T[]))
        @test_throws DimensionMismatch MA.fused_map_reduce(op, a, b[1:1])
        @test (a, b) == originals
    end
    a, b = [1 + 2im, 3 - im], [2 - im, 1 + im]
    @test (@inferred MA.fused_map_reduce(MA.add_dot, a, b)) ==
          LinearAlgebra.dot(a, b)
    @test MA.fused_map_reduce(MA.add_mul, Matrix{Int}[], Float64[]) isa MA.Zero
end

function test_allocations()
    a, b = [1, 2, 3], [4, 5, 6]
    @test (@inferred MA.fused_map_reduce(MA.add_mul, a, b)) == 32
    @test (@allocated MA.fused_map_reduce(MA.add_mul, a, b)) == 0
    @test (@inferred MA.fused_map_reduce(MA.add_dot, a, b)) == 32
    @test (@allocated MA.fused_map_reduce(MA.add_dot, a, b)) == 0
    @test (@inferred MA.operate(sum, a)) == 6
    @test (@allocated MA.operate(sum, a)) == 0
    @test (@inferred MA.zero_for(Int, UnreadableInput())) == 0
    @test (@allocated MA.zero_for(Int, UnreadableInput())) == 0
    return
end

@testset "Fused reduction allocations" begin
    test_allocations()
end

function test_bigint_allocations()
    a, b = BigInt[1, 2, 3], BigInt[4, 5, 6]
    MA.fused_map_reduce(MA.add_mul, a, b)
    MA.fused_map_reduce(MA.add_dot, a, b)
    # Preserve the existing budget: result plus multiplication workspace.
    budget = 3 * @allocated(BigInt(1))
    @test (@allocated MA.fused_map_reduce(MA.add_mul, a, b)) <= budget
    @test (@allocated MA.fused_map_reduce(MA.add_dot, a, b)) <= budget
    return
end

@testset "BigInt reduction allocations" begin
    test_bigint_allocations()
end

end
