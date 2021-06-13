function allocation_test(op, T, short, short_to)
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
    alloc_test(() -> short(a, b), 0)
    alloc_test(() -> short_to(c, a, b), 0)
end

@testset "$T" for T in [BigInt, BigFloat]
    MA.Test.int_test(T)
    @testset "Allocation" begin
        allocation_test(+, T, MA.add!, MA.add_to!)
        allocation_test(*, T, MA.mul!, MA.mul_to!)
        if T == BigInt
            allocation_test(gcd, T, MA.gcd!, MA.gcd_to!)
        end
    end
end
