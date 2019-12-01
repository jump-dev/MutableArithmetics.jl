#  Copyright 2019, Gilles Peiffer, Benoît Legat, Sascha Timme, and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################

module MutableArithmetics

# Performance note:
# We use `Vararg` instead of splatting `...` as using `where N` forces Julia to
# specialize in the number of arguments `N`. Otherwise, we get allocations and
# slowdown because it compiles something that works for any `N`. See
# https://github.com/JuliaLang/julia/issues/32761 for details.

"""
    add_mul(a, args...)

Return `a + *(args...)`. Note that `add_mul(a, b, c) = muladd(b, c, a)`.
"""
function add_mul end
add_mul(a, b) = a + b
add_mul(a, b, c) = muladd(b, c, a)
add_mul(a, b, c::Vararg{Any, N}) where {N} = add_mul(a, b *(c...))

include("interface.jl")
include("shortcuts.jl")
include("broadcast.jl")

# Implementation of the interface for Base types
import LinearAlgebra
const Scaling = Union{Number, LinearAlgebra.UniformScaling}
scaling(x::Scaling) = x
include("bigint.jl")
include("linear_algebra.jl")
include("sparse_arrays.jl")

isequal_canonical(a, b) = a == b

include("rewrite.jl")
include("dispatch.jl")

# Test that can be used to test an implementation of the interface
include("Test/Test.jl")


end # module
