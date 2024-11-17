# Copyright (c) 2019 MutableArithmetics.jl contributors
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v.2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.

include("dummy.jl")

# Allocating size for allocating a `BigInt`.
# Half size on 32-bit.
const BIGINT_ALLOC = Sys.WORD_SIZE == 64 ? 48 : 24

function alloc_test(f::F, n) where {F}
    f() # compile
    @test n == @allocated f()
end

function alloc_test_le(f::F, n) where {F}
    f() # compile
    @test n >= @allocated f()
end
