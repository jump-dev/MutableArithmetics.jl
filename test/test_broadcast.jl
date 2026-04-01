# Copyright (c) 2019 MutableArithmetics.jl contributors
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v.2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.

module TestBroadcast

using Test

import MutableArithmetics as MA

function runtests()
    is_test(name::Symbol) = startswith("$name", "test_")
    @testset "$name" for name in filter(is_test, names(@__MODULE__; all = true))
        getfield(@__MODULE__, name)()
    end
    return
end

function alloc_test(f::F, expected_upper_bound::Integer) where {F<:Function}
    f() # compile
    measured_allocations = @allocated f()
    @test measured_allocations <= expected_upper_bound
    return
end

function test_Int()
    a = [1, 2]
    b = 3
    c = [3, 4]
    @test MA.broadcast!!(+, a, b) == [4, 5]
    @test a == [4, 5]
    # Need to have immutable structs that contain references to be allocated on
    # the stack: https://github.com/JuliaLang/julia/pull/33886
    alloc_test(() -> MA.broadcast!!(+, a, b), 0)
    alloc_test(() -> MA.broadcast!!(+, a, c), 0)
    return
end

function test_BigInt()
    x = BigInt(1)
    y = BigInt(2)
    a = [x, y]
    b = 3
    c = [2x, 3y]
    @test MA.broadcast!!(+, a, b) == [4, 5]
    @test a == [4, 5]
    @test x == 4
    @test y == 5
    # FIXME This should not allocate but I couldn't figure out where these
    #       allocations come from.
    n = 6 * @allocated(BigInt(1))
    alloc_test(() -> MA.broadcast!!(+, a, b), n)
    alloc_test(() -> MA.broadcast!!(+, a, c), 0)
    return
end

function test_broadcast_issue_158()
    x, y = BigInt[2 3], BigInt[2 3; 3 4]
    @test MA.@rewrite(x .+ y) == x .+ y
    @test MA.@rewrite(x .- y) == x .- y
    @test MA.@rewrite(y .+ x) == y .+ x
    @test MA.@rewrite(y .- x) == y .- x
    @test MA.@rewrite(y .* x) == y .* x
    @test MA.@rewrite(x .* y) == x .* y
    return
end

struct Struct221 <: AbstractArray{Int,1} end

struct BroadcastStyle221 <: Broadcast.BroadcastStyle end

Base.BroadcastStyle(::Type{Struct221}) = BroadcastStyle221()

function test_promote_broadcast_for_new_style()
    @test MA.promote_broadcast(MA.add_mul, Vector{Int}, Struct221) === Any
    return
end

function test_broadcast_length_1_dimensions()
    A = rand(2, 1, 3)
    B = rand(2, 3)
    @test MA.broadcast!!(MA.sub_mul, A, B) ≈ A .- B
    @test MA.broadcast!!(MA.sub_mul, B, A) ≈ B .- A
    return
end

end  # module

TestBroadcast.runtests()
