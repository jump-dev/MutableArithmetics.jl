# Copyright (c) 2019 MutableArithmetics.jl contributors
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v.2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.

module TestInterfaceSparseArrays

using Test

import MutableArithmetics
import Random
import SparseArrays

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

function test_spmatmul()
    Random.seed!(1234)
    for m in [1, 2, 3, 5, 11]
        for n in [1, 2, 3, 5, 11]
            A = SparseArrays.sprand(Float64, m, n, 0.5)
            B = SparseArrays.sprand(Float64, n, m, 0.5)
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
    ret = SparseArrays.spzeros(Float64, m, m)
    MA.operate!(MA.add_mul, ret, A, B, 2.0)
    @test ret ≈ A * B * 2.0
    ret = SparseArrays.spzeros(Float64, m, m)
    MA.operate!(MA.add_mul, ret, A, B, 2.0, 1.5)
    @test ret ≈ A * B * 2.0 * 1.5
    return
end

end  # module

TestInterfaceSparseArrays.runtests()
