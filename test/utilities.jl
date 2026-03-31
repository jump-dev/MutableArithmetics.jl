# Copyright (c) 2019 MutableArithmetics.jl contributors
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v.2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.

include("dummy.jl")

function alloc_test(f::F, expected_upper_bound::Integer) where {F<:Function}
    f() # compile
    measured_allocations = @allocated f()
    @test measured_allocations <= expected_upper_bound
    return
end
