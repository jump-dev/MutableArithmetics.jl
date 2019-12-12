using Test
import MutableArithmetics
const MA = MutableArithmetics

include("utilities.jl")

include("interface.jl")

@testset "Int" begin
    include("int.jl")
end
@testset "BigInt" begin
    include("bigint.jl")
end
@testset "BigFloat" begin
    include("bigfloat.jl")
end
@testset "Broadcast" begin
    include("broadcast.jl")
end
include("matmul.jl")
include("rewrite.jl")
include("hygiene.jl")
