# Copyright (c) 2019 MutableArithmetics.jl contributors
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v.2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.

using Test
import MutableArithmetics
const MA = MutableArithmetics

@testset "Int" begin
    a = [1, 2]
    b = 3
    c = [3, 4]
    @test MA.broadcast!!(+, a, b) == [4, 5]
    @test a == [4, 5]
    # Need to have immutable structs that contain references to be allocated on
    # the stack: https://github.com/JuliaLang/julia/pull/33886
    alloc_test(() -> MA.broadcast!!(+, a, b), 0)
    alloc_test(() -> MA.broadcast!!(+, a, c), 0)
end
@testset "BigInt" begin
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
    #       240 come from.
    alloc_test(() -> MA.broadcast!!(+, a, b), 30 * sizeof(Int))
    alloc_test(() -> MA.broadcast!!(+, a, c), 0)
end
