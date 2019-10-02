using Test
import MutableArithmetics
const MA = MutableArithmetics

@testset "Int" begin
    include("int.jl")
end
@testset "BigInt" begin
    include("bigint.jl")
end
include("matmul.jl")
