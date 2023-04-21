# Copyright (c) 2023 MutableArithmetics.jl contributors
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v.2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.

backup_bigfloats(v::AbstractVector{BigFloat}) = map(MA.copy_if_mutable, v)

absolute_error(accurate::Real, approximate::Real) = abs(accurate - approximate)

function relative_error(accurate::Real, approximate::Real)
    return absolute_error(accurate, approximate) / abs(accurate)
end

function dotter(x::V, y::V) where {V<:AbstractVector{<:Real}}
    let x = x, y = y
        () -> LinearAlgebra.dot(x, y)
    end
end

function reference_dot(x::V, y::V) where {F<:Real,V<:AbstractVector{F}}
    return setprecision(dotter(x, y), F, 8 * precision(F))
end

function dot_test_relative_error(x::V, y::V) where {V<:AbstractVector{BigFloat}}
    buf = MA.buffer_for(LinearAlgebra.dot, V, V)

    input = (x, y)
    backup = map(backup_bigfloats, input)

    output = BigFloat()

    MA.buffered_operate_to!!(buf, output, LinearAlgebra.dot, input...)

    @test input == backup

    return relative_error(reference_dot(input...), output)
end

subtracter(s::Real) =
    let s = s
        x -> x - s
    end

our_rand(n::Int, bias::Real) = map(subtracter(bias), rand(BigFloat, n))

function rand_dot_rel_err(size::Int, bias::Real)
    x = our_rand(size, bias)
    y = our_rand(size, bias)
    return dot_test_relative_error(x, y)
end

function max_rand_dot_rel_err(size::Int, bias::Real, iter_cnt::Int)
    max_rel_err = zero(BigFloat)
    for i in 1:iter_cnt
        rel_err = rand_dot_rel_err(size, bias)
        <(max_rel_err, rel_err) && (max_rel_err = rel_err)
    end
    return max_rel_err
end

function max_rand_dot_ulps(size::Int, bias::Real, iter_cnt::Int)
    return max_rand_dot_rel_err(size, bias, iter_cnt) / eps(BigFloat)
end

function ulper(size::Int, bias::Real, iter_cnt::Int)
    let s = size, b = bias, c = iter_cnt
        () -> max_rand_dot_ulps(s, b, c)
    end
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
    iter_cnt = 10
    err = setprecision(ulper(size, bias, iter_cnt), BigFloat, prec)
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
