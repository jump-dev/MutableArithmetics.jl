# Copyright (c) 2019 MutableArithmetics.jl contributors
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v.2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.

include("dummy.jl")

# Allocating size for allocating a `BigInt`. Half size on 32-bit.
# v1.12.5 off by 8
_big_int_alloc() = sizeof(Int) + @allocated(BigInt(1))
const BIGINT_ALLOC = _big_int_alloc()
@show BIGINT_ALLOC

function alloc_test(f, n)
    f() # compile
    y = @allocated f()
    @test n == y
    return
end

function alloc_test_le(f, n)
    f() # compile
    y = @allocated f()
    @test n >= y
    return
end
