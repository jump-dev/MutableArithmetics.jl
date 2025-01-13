# Copyright (c) 2019 MutableArithmetics.jl contributors
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v.2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.

module TestReduce

using Test

import MutableArithmetics as MA
using LinearAlgebra

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

function _test_is_zero(x, ::Type{T}) where {T}
    @test iszero(x)
    @test x isa T
    return
end

function test_empty_dot()
    _test_is_zero(MA.operate(dot, Int[], Int[]), Int)
    _test_is_zero(MA.operate(dot, BigInt[], Int[]), BigInt)
    _test_is_zero(MA.operate(dot, Int[], Float64[]), Float64)
    _test_is_zero(MA.operate(dot, Matrix{Int}[], Matrix{Float64}[]), Float64)
    @test MA.fused_map_reduce(MA.add_mul, Matrix{Int}[], Float64[]) isa MA.Zero
    @test MA.fused_map_reduce(MA.add_mul, Float64[], Matrix{Int}[]) isa MA.Zero
    return
end

function test_add_mul_matrix()
    A = [1 2; 3 4]
    B = [3 1; 1 2]
    @test MA.fused_map_reduce(MA.add_mul, [A, B], [2, -1]) == 2A - B
    @test MA.fused_map_reduce(MA.add_mul, [-1, 3], [A, B]) == 3B - A
    return
end

end  # module

TestReduce.runtests()
