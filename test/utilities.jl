include("dummy.jl")

# Allocating size for allocating a `BigInt`.
# Half size on 32-bit.
const BIGINT_ALLOC = Sys.WORD_SIZE == 64 ? 48 : 24

function alloc_test(f, n)
    f() # compile
    @test n == @allocated f()
end
