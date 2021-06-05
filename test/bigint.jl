MA.Test.int_test(BigInt)

@testset "Allocation" begin
    a = BigInt(2)
    b = BigInt(3)
    c = BigInt(4)
    alloc_test(() -> MA.add!(a, b), 0)
    alloc_test(() -> MA.add_to!(c, a, b), 0)
    alloc_test(() -> MA.mul!(a, b), 0)
    alloc_test(() -> MA.mul_to!(c, a, b), 0)
    g = gcd(a, b)
    @test c === MA.gcd_to!(c, a, b)
    @test g == c
    @test a === MA.gcd!(a, b)
    @test g == a
    alloc_test(() -> MA.gcd!(a, b), 0)
    alloc_test(() -> MA.gcd_to!(c, a, b), 0)
end
