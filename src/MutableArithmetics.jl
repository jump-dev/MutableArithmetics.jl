#  Copyright 2019, Gilles Peiffer, Benoît Legat, Sascha Timme, and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################

module MutableArithmetics

include("interface.jl")

# Test that can be used to test an implementation of the interface
include("Test/Test.jl")

# Implementation of the interface for Base types
include("bigint.jl")
include("linear_algebra.jl")

end # module
