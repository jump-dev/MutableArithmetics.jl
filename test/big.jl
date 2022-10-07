# Copyright (c) 2019 MutableArithmetics.jl contributors
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v.2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.

function allocation_test(op, T, short, short_to, n; a = T(2), b = T(3), c = T(4))
    @test MA.promote_operation(op, T, T) == T
    alloc_test(() -> promote_operation(op, T, T), 0)
    if op != div && op != -
        @test MA.promote_operation(op, T, T, T) == T
        alloc_test(() -> MA.promote_operation(op, T, T, T), 0)
    end
    g = op(a, b)
    @test c === short_to(c, a, b)
    @test g == c
    @test a === short(a, b)
    @test g == a
    alloc_test_le(() -> short(a, b), n)
    return alloc_test_le(() -> short_to(c, a, b), n)
end

@testset "$T" for T in [BigInt, BigFloat, Rational{BigInt}]
    MA.Test.int_test(T)
    @testset "Allocation" begin
        allocation_test(+, T, MA.add!!, MA.add_to!!, T <: Rational ? 168 : 0)
        allocation_test(-, T, MA.sub!!, MA.sub_to!!, T <: Rational ? 168 : 0)
        allocation_test(*, T, MA.mul!!, MA.mul_to!!, T <: Rational ? 240 : 0)
        if T <: Rational # https://github.com/jump-dev/MutableArithmetics.jl/issues/167
            allocation_test(+, T, MA.add!!, MA.add_to!!, 168, a = T(1//2), b = T(3//2), c = T(5//2))
            allocation_test(-, T, MA.sub!!, MA.sub_to!!, 168, a = T(1//2), b = T(3//2), c = T(5//2))
        end
        # Requires https://github.com/JuliaLang/julia/commit/3f92832df042198b2daefc1f7ca609db38cb8173
        # for `gcd` to be defined on `Rational`.
        if T == BigInt
            allocation_test(div, T, MA.div!!, MA.div_to!!, 0)
        end
        if T == BigInt || T == Rational{BigInt}
            allocation_test(gcd, T, MA.gcd!!, MA.gcd_to!!, 0)
            allocation_test(lcm, T, MA.lcm!!, MA.lcm_to!!, 0)
        end
    end
end
