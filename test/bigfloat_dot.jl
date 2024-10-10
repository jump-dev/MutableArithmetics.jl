# Copyright (c) 2023 MutableArithmetics.jl contributors
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v.2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.

function rand_dot_rel_err(size::Int, bias::Real)
    x = rand(BigFloat, size) .- bias
    y = rand(BigFloat, size) .- bias
    backup = (MA.copy_if_mutable(x), MA.copy_if_mutable(y))
    buf = MA.buffer_for(LinearAlgebra.dot, Vector{BigFloat}, Vector{BigFloat})
    output = BigFloat()
    MA.buffered_operate_to!!(buf, output, LinearAlgebra.dot, x, y)
    @test (x, y) == backup
    accurate = setprecision(BigFloat, 8 * precision(BigFloat)) do
        return LinearAlgebra.dot(x, y)
    end
    return abs(accurate - output) / abs(accurate)
end

@testset "prec:$prec size:$size bias:$bias" for (prec, size, bias) in
                                                Iterators.product(
    # These precisions (in bits) are most probably smaller than what
    # users would set in their code, but they are relevant anyway
    # because the compensated summation algorithm used for computing
    # the dot product is accurate independently of the machine
    # precision (except when vector lengths are really huge with
    # respect to the precision).
    (32, 64),
    # Compensated summation should be accurate even for very large
    # input vectors, so test that.
    (10000,),
    # The zero "bias" signifies that the input will be entirely
    # nonnegative (drawn from the interval [0, 1]), while a positive
    # bias shifts that interval towards negative infinity. We want to
    # test a few different values here, but we do not want to set the
    # bias to 0.5, because the expected value is then zero and there's
    # no guarantee on the relative error in that case.
    (0.0, 2^-2, 2^-2 + 2^-3 + 2^-4),
)
    err = setprecision(BigFloat, prec) do
        return mapreduce(max, 1:10) do _
            return rand_dot_rel_err(size, bias) / eps(BigFloat)
        end
    end
    @test 0 <= err < 1
end

function alloc_test_helper(
    buf::B,
    output::BigFloat,
    x::Vector{BigFloat},
    y::Vector{BigFloat},
) where {B<:Any}
    let b = buf, o = output, x = x, y = y
        () -> MA.buffered_operate_to!!(b, o, LinearAlgebra.dot, x, y)
    end
end

@testset "alloc" begin
    x = rand(BigFloat, 1000)
    y = rand(BigFloat, 1000)
    V = Vector{BigFloat}
    alloc_test(
        alloc_test_helper(
            MA.buffer_for(LinearAlgebra.dot, V, V),
            BigFloat(),
            x,
            y,
        ),
        0,
    )
end
