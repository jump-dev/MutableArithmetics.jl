MA.Test.int_test(BigFloat)

@testset "Allocation" begin
    a = BigFloat(2)
    b = BigFloat(3)
    c = BigFloat(4)
    alloc_test(() -> MA.add!(a, b), 0)
    alloc_test(() -> MA.add_to!(c, a, b), 0)
    alloc_test(() -> MA.mul!(a, b), 0)
    alloc_test(() -> MA.mul_to!(c, a, b), 0)
end
