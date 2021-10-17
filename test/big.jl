function allocation_test(op, T, short, short_to, n)
    a = T(2)
    b = T(3)
    c = T(4)
    @test MA.promote_operation(op, T, T) == T
    @test MA.promote_operation(op, T, T, T) == T
    g = op(a, b)
    @test c === short_to(c, a, b)
    @test g == c
    @test a === short(a, b)
    @test g == a
    alloc_test_le(() -> short(a, b), n)
    alloc_test_le(() -> short_to(c, a, b), n)
end

@testset "$T" for T in [BigInt, BigFloat, Rational{BigInt}]
    MA.Test.int_test(T)
    @testset "Allocation" begin
        allocation_test(+, T, MA.add!!, MA.add_to!!, T <: Rational ? 168 : 0)
        allocation_test(*, T, MA.mul!!, MA.mul_to!!, T <: Rational ? 240 : 0)
        # Requires https://github.com/JuliaLang/julia/commit/3f92832df042198b2daefc1f7ca609db38cb8173
        # for `gcd` to be defined on `Rational`.
        if T == BigInt || (T == Rational{BigInt} && VERSION >= v"1.4.0-DEV.606")
            allocation_test(gcd, T, MA.gcd!!, MA.gcd_to!!, 0)
            allocation_test(lcm, T, MA.lcm!!, MA.lcm_to!!, 0)
        end
    end
end
