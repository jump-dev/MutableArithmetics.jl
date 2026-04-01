# Copyright (c) 2019 MutableArithmetics.jl contributors
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v.2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.

using Test

is_test(f) = startswith(f, "test_") && endswith(f, ".jl")

@testset "$file" for file in filter(is_test, readdir(@__DIR__))
    include(joinpath(@__DIR__, file))
end
