# Copyright (c) 2019 MutableArithmetics.jl contributors
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v.2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.

using LinearAlgebra

# Tests that the calls are correctly redirected to the mutable calls
# by checking allocations
function dispatch_tests(::Type{T}) where {T}
    buffer = zero(T)
    a = one(T)
    b = one(T)
    c = one(T)
    x = convert.(T, [1, 2, 3])
    # Need to allocate 1 BigInt for the result and one for the buffer
    alloc_test(() -> MA.fused_map_reduce(MA.add_mul, x, x), 2BIGINT_ALLOC)
    alloc_test(() -> MA.fused_map_reduce(MA.add_dot, x, x), 2BIGINT_ALLOC)
    if T <: MA.AbstractMutable
        alloc_test(() -> x'x, 2BIGINT_ALLOC)
        alloc_test(() -> transpose(x) * x, 2BIGINT_ALLOC)
        alloc_test(() -> LinearAlgebra.dot(x, x), 2BIGINT_ALLOC)
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

@testset "*(::Real, ::Union{Hermitian,Symmetric})" begin
    A = DummyBigInt[1 2; 2 3]
    B = DummyBigInt[2 4; 4 6]
    @test MA.isequal_canonical(2 * A, B)
    C = LinearAlgebra.Symmetric(B)
    @test MA.isequal_canonical(2 * LinearAlgebra.Symmetric(A, :U), C)
    @test MA.isequal_canonical(2 * LinearAlgebra.Symmetric(A, :L), C)
    D = LinearAlgebra.Hermitian(B)
    @test all(MA.isequal_canonical.(2 * LinearAlgebra.Hermitian(A, :L), D))
    @test all(MA.isequal_canonical.(2 * LinearAlgebra.Hermitian(A, :U), D))
end
