using LinearAlgebra

# Tests that the calls are correctly redirected to the mutable calls
# by checking allocations
function dispatch_tests(::Type{T}) where {T}
    buffer = zero(T)
    a = one(T)
    b = one(T)
    c = one(T)
    x = convert.(T, [1, 2, 3])
    # Need to allocate 1 BigInt for the result and one for the buffer
    alloc_test(() -> MA.fused_map_reduce(MA.add_mul, x, x), 2BIGINT_ALLOC)
    alloc_test(() -> MA.fused_map_reduce(MA.add_dot, x, x), 2BIGINT_ALLOC)
    if T <: MA.AbstractMutable
        alloc_test(() -> x'x, 2BIGINT_ALLOC)
        alloc_test(() -> transpose(x) * x, 2BIGINT_ALLOC)
        alloc_test(() -> LinearAlgebra.dot(x, x), 2BIGINT_ALLOC)
    end
end

@testset "Dispatch tests" begin
    dispatch_tests(BigInt)
    if VERSION >= v"1.5"
        # On `DummyBigInt` allocates more on previous releases of Julia
        # as it's dynamically allocated
        dispatch_tests(DummyBigInt)
    end
end
