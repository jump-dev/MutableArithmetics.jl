using Test
import MutableArithmetics
const MA = MutableArithmetics

@testset "BigInt" begin
    include("bigint.jl")
end
include("matmul.jl")
