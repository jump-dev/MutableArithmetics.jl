using Test
import MutableArithmetics
const MA = MutableArithmetics

include("utilities.jl")

include("interface.jl")

include("range.jl")

@testset "Int" begin
    include("int.jl")
end
@testset "BigInt" begin
    include("big.jl")
end
@testset "Broadcast" begin
    include("broadcast.jl")
end
include("matmul.jl")
include("dispatch.jl")
include("rewrite.jl")

# It is easy to introduce macro scoping issues into MutableArithmetics,
# particularly ones that rely on `MA` or `MutableArithmetics` being present in
# the current scope. To work around that, include the "hygiene" script in a
# clean module with no other scope.

function _include_sandbox(filename)
    mod = @eval module $(gensym()) end
    return Base.include(mod, filename)
end

_include_sandbox("hygiene.jl")
