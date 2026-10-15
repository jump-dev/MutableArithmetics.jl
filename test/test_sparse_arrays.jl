# Copyright (c) 2019 MutableArithmetics.jl contributors
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v.2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.

module TestInterfaceSparseArrays

using Test

import LinearAlgebra
import MutableArithmetics as MA
import Random
import SparseArrays

function runtests()
    is_test(name::Symbol) = startswith("$name", "test_")
    @testset "$name" for name in filter(is_test, names(@__MODULE__; all = true))
        getfield(@__MODULE__, name)()
    end
    return
end

function test_mutable_copy_sparse_vector()
    for T in (Int, BigInt)
        v = SparseArrays.SparseVector(1_000_000, Int32[2, 7, 13], T[1, 0, 3])
        w = MA.mutable_copy(v)
        @test typeof(w) === typeof(v)
        @test length(w) == length(v)
        @test SparseArrays.nonzeroinds(w) == Int32[2, 7, 13]
        @test SparseArrays.nonzeros(w) == T[1, 0, 3]
        @test SparseArrays.nonzeroinds(w) !== SparseArrays.nonzeroinds(v)
        @test SparseArrays.nonzeros(w) !== SparseArrays.nonzeros(v)
        if T === BigInt
            MA.operate!(+, SparseArrays.nonzeros(w)[1], 1)
            @test w[2] == 2
            @test v[2] == 1
        end
        w[7] = 4
        @test v[7] == 0
    end
    v = SparseArrays.SparseVector(10, Int32[], BigInt[])
    w = MA.mutable_copy(v)
    @test typeof(w) === typeof(v)
    @test length(w) == 10
    @test SparseArrays.nnz(w) == 0
    return
end

function test_spmatmul()
    Random.seed!(1234)
    for m in [1, 2, 3, 5, 11]
        for n in [1, 2, 3, 5, 11]
            A = SparseArrays.sprand(Float64, m, n, 0.5)
            B = SparseArrays.sprand(Float64, n, m, 0.5)
            ret = SparseArrays.spzeros(Float64, m, m)
            MA.operate!(MA.add_mul, ret, A, B)
            @test ret ≈ A * B
            MA.operate!(MA.add_mul, ret, A, B)
            @test ret ≈ 2 * (A * B)
            ret = SparseArrays.spzeros(Float64, m, m)
            MA.operate!(MA.add_mul, ret, A, A')
            @test ret ≈ A * A'
            ret = SparseArrays.spzeros(Float64, m, m)
            MA.operate!(MA.add_mul, ret, A, B, 2.0)
            @test ret ≈ A * B * 2.0
            ret = SparseArrays.spzeros(Float64, m, m)
            MA.operate!(MA.add_mul, ret, A, B, 2.0, 1.5)
            @test ret ≈ A * B * 2.0 * 1.5
        end
    end
    return
end

function test_spmatmul_prefer_sort()
    Random.seed!(1234)
    m = n = 100
    p = 0.01
    A = SparseArrays.sprand(Float64, m, n, p)
    B = SparseArrays.sprand(Float64, n, m, p)
    ret = SparseArrays.spzeros(Float64, m, m)
    MA.operate!(MA.add_mul, ret, A, B)
    @test ret ≈ A * B
    MA.operate!(MA.add_mul, ret, A, B)
    @test ret ≈ 2 * (A * B)
    ret = SparseArrays.spzeros(Float64, m, m)
    MA.operate!(MA.add_mul, ret, A, B, 2.0)
    @test ret ≈ A * B * 2.0
    ret = SparseArrays.spzeros(Float64, m, m)
    MA.operate!(MA.add_mul, ret, A, B, 2.0, 1.5)
    @test ret ≈ A * B * 2.0 * 1.5
    return
end

function test_sparse_accumulator()
    for T in (Float64, BigInt, Complex{Int})
        a = T[1 0 2; 0 3 0; 4 0 0]
        b = T[0 5 0; 6 0 0; 0 0 7]
        if T === Complex{Int}
            a[1, 3] += im
            b[3, 3] -= im
        end
        A, B = SparseArrays.sparse(a), SparseArrays.sparse(b)
        for left in (identity, transpose, adjoint),
            right in (identity, transpose, adjoint)

            initial = T[1 0 0; 0 0 2; 0 3 0]
            output = SparseArrays.sparse(MA.mutable_copy(initial))
            expected = left(a) * right(b)
            @test (@inferred MA.operate!(
                MA.add_mul,
                output,
                left(A),
                right(B),
            )) === output
            @test output == initial + expected
            @test (@inferred MA.operate!(
                MA.add_mul,
                output,
                left(A),
                right(B),
                T(2),
                T(3),
            )) === output
            @test output == initial + 7expected
            @test A == a
            @test B == b
        end
        output = SparseArrays.sparse(T[1 0 0; 0 0 2; 0 3 0])
        empty = SparseArrays.spzeros(T, 3, 3)
        @test MA.operate!(MA.add_mul, output, empty, B) === output
        @test output == T[1 0 0; 0 0 2; 0 3 0]
        @test_throws DimensionMismatch MA.operate!(
            MA.add_mul,
            output,
            A,
            SparseArrays.spzeros(T, 2, 3),
        )
        @test output == T[1 0 0; 0 0 2; 0 3 0]
    end
    return
end

function test_sparse_addmul_matmat()
    Random.seed!(1234)
    m = 10
    ret = SparseArrays.spzeros(Float64, m, m)
    A, B = rand(m, m), rand(m, m)
    MA.operate!(MA.add_mul, ret, A, B)
    @test ret ≈ A * B
    MA.operate!(MA.add_mul, ret, A, B)
    @test ret ≈ A * B + A * B
    return
end

function test_sparse_dense_mul()
    for T in (Float64, BigInt, Complex{Int})
        a = T[1 0 2; 0 3 0; 4 0 0]
        b = T[0 5 0; 6 0 0; 0 0 7]
        if T === Complex{Int}
            a[1, 3] += im
            b[3, 3] -= im
        end
        A, B = SparseArrays.sparse(a), SparseArrays.sparse(b)
        for left in (identity, transpose, adjoint),
            right in (identity, transpose, adjoint)

            initial = T[1 0 0; 0 0 2; 0 3 0]
            output = MA.mutable_copy(initial)
            expected = left(a) * right(b)
            @test (@inferred MA.operate!(
                MA.add_mul,
                output,
                left(A),
                right(B),
            )) === output
            @test output == initial + expected
            @test (@inferred MA.operate!(
                MA.add_mul,
                output,
                left(A),
                right(B),
                T(2),
                T(3),
            )) === output
            @test output == initial + 7expected
            @test A == a
            @test B == b
        end
    end
    return
end

function test_sparse_addmul_mat_adjspmat()
    Random.seed!(1234)
    m = 10
    ret = zeros(m, m)
    A = rand(m, m)
    B = SparseArrays.sprand(Float64, m, m, 0.5)'
    MA.operate!(MA.add_mul, ret, A, B)
    @test ret ≈ A * B
    MA.operate!(MA.add_mul, ret, A, B, 3.0)
    @test ret ≈ A * B + A * B * 3.0
    return
end

function test_spmatmul_adj()
    Random.seed!(1234)
    for m in [1, 2, 3, 5, 11]
        for f in (identity, LinearAlgebra.adjoint)
            for g in (identity, LinearAlgebra.adjoint)
                A = f(SparseArrays.sprand(Float64, m, m, 0.5))
                B = g(SparseArrays.sprand(Float64, m, m, 0.5))
                ret = SparseArrays.spzeros(Float64, m, m)
                MA.operate!(MA.add_mul, ret, A, B)
                @test ret ≈ A * B
                ret = SparseArrays.spzeros(Float64, m, m)
                MA.operate!(MA.add_mul, ret, A, A')
                @test ret ≈ A * A'
                ret = SparseArrays.spzeros(Float64, m, m)
                MA.operate!(MA.add_mul, ret, A, B, 2.0)
                @test ret ≈ A * B * 2.0
                ret = SparseArrays.spzeros(Float64, m, m)
                MA.operate!(MA.add_mul, ret, A, B, 2.0, 1.5)
                @test ret ≈ A * B * 2.0 * 1.5
            end
        end
    end
    return
end

function test_sparse_broadcast()
    Random.seed!(1234)
    x = [BigInt(1) for _ in 1:10]
    y = SparseArrays.sprand(Float64, 10, 0.5)
    @test MA.broadcast!!(+, x, y) ≈ x .+ y
    Random.seed!(1234)
    X = [BigInt(1) for _ in 1:10, _ in 1:10]
    Y = SparseArrays.sprand(Float64, 10, 10, 0.5)
    @test MA.broadcast!!(+, X, Y) ≈ X .+ Y
    return
end

function test_zero()
    v = SparseArrays.sparsevec([1, 2], [3.0, 4.0])
    MA.operate!(zero, v)
    @test isempty(SparseArrays.nonzeroinds(v))
    @test isempty(SparseArrays.nonzeros(v))
    A = SparseArrays.sprand(Float64, 2, 2, 0.5)
    MA.operate!(zero, A)
    @test iszero(SparseArrays.nnz(A))
    @test isempty(SparseArrays.nonzeros(A))
    return
end

function test_broadcast_sparse_dense_type()
    for T in (Float64, BigInt)
        X = SparseArrays.sparse([1, 2], [1, 3], T[1, 2], 3, 3)
        Xd = Matrix(X)
        x = SparseArrays.sparsevec([1, 3], T[1, 2], 3)
        xd = Vector(x)
        for (A, B) in ((Xd, X), (X, Xd), (X, X), (xd, x), (x, xd), (X, 2))
            @test MA.@rewrite(A .+ B) isa typeof(A .+ B)
            @test MA.@rewrite(A .- B) isa typeof(A .- B)
            @test MA.@rewrite(A .+ B) == A .+ B
            @test MA.@rewrite(A .- B) == A .- B
            for (op, base_op) in
                ((+, +), (-, -), (MA.add_mul, +), (MA.sub_mul, -))
                expected = broadcast(base_op, A, B)
                # Test a dense destination even when A is sparse:
                # `broadcast!!` skips promotion for `IsNotMutable` inputs.
                @test MA.broadcast_mutability(
                    Array{eltype(expected),ndims(expected)},
                    op,
                    typeof(A),
                    typeof(B),
                ) == MA.mutability(typeof(expected))
                C = MA.copy_if_mutable(A)
                D = MA.broadcast!!(op, C, B)
                @test D isa typeof(expected)
                @test D == expected
            end
        end
    end
    return
end

end  # module

TestInterfaceSparseArrays.runtests()
