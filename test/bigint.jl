MA.Test.int_test(BigInt)

@testset "Allocation" begin
    a = BigInt(2)
    b = BigInt(3)
    c = BigInt(4)
    alloc_test(() -> MA.add!(a, b), 0)
    alloc_test(() -> MA.add_to!(c, a, b), 0)
end
