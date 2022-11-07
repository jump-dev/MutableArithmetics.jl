# Copyright (c) 2019 MutableArithmetics.jl contributors
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v.2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.

using LinearAlgebra

# Tests that the calls are correctly redirected to the mutable calls
# by checking allocations
function dispatch_tests(::Type{T}) where {T}
    x = convert.(T, [1, 2, 3])
    # Need to allocate 1 BigInt for the result and one for the buffer, plus two
    # zero(T) for determining the eltype of the result.
    cost = Sys.WORD_SIZE == 64 ? (2 * 48 + 2 * 40) : (2 * 24 + 2 * 20)
    alloc_test(() -> MA.fused_map_reduce(MA.add_mul, x, x), cost)
    alloc_test(() -> MA.fused_map_reduce(MA.add_dot, x, x), cost)
    if T <: MA.AbstractMutable
        alloc_test(() -> x'x, cost)
        alloc_test(() -> transpose(x) * x, cost)
        alloc_test(() -> LinearAlgebra.dot(x, x), cost)
    end
end

@testset "Dispatch tests" begin
    dispatch_tests(BigInt)
    # On `DummyBigInt` allocates more on previous releases of Julia
    # as it's dynamically allocated
    dispatch_tests(DummyBigInt)

    @testset "dot non-concrete vector" begin
        x = [5.0, 6.0]
        y = Vector{Union{Float64,String}}(x)
        @test MA.operate(LinearAlgebra.dot, x, y) == LinearAlgebra.dot(x, y)
        @test MA.operate(*, x', y) == x' * y
    end

    @testset "dot vector of vectors" begin
        x = [5.0, 6.0]
        z = [x, x]
        @test MA.operate(LinearAlgebra.dot, z, z) == LinearAlgebra.dot(z, z)
    end
end
