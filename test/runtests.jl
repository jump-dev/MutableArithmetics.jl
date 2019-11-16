using Test
import MutableArithmetics
const MA = MutableArithmetics

function alloc_test(f, n)
    f() # compile
    @test n == @allocated f()
end

@testset "Int" begin
    include("int.jl")
end
@testset "BigInt" begin
    include("bigint.jl")
end
include("matmul.jl")
