module Test

import MutableArithmetics
const MA = MutableArithmetics

using LinearAlgebra, SparseArrays, Test

include("config.jl")

include("int.jl")
include("generic.jl")
include("scalar.jl")
include("quadratic.jl")
include("array.jl")
include("sparse.jl")

end # module
