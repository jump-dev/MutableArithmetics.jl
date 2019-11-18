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

include("interface.jl")
include("shortcuts.jl")

# Test that can be used to test an implementation of the interface
include("Test/Test.jl")

# Implementation of the interface for Base types
include("bigint.jl")
include("linear_algebra.jl")

isequal_canonical(a, b) = a == b

include("rewrite.jl")

include("dispatch.jl")

end # module
