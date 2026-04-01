# Copyright (c) 2019 MutableArithmetics.jl contributors
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v.2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.

module TestBig

using Test

import LinearAlgebra
import MutableArithmetics as MA

function runtests()
    is_test(name::Symbol) = startswith("$name", "test_")
    @testset "$name" for name in filter(is_test, names(@__MODULE__; all = true))
        getfield(@__MODULE__, name)()
    end
    return
end

function alloc_test(f::F, expected_upper_bound::Integer) where {F<:Function}
    f() # compile
    measured_allocations = @allocated f()
    @test measured_allocations <= expected_upper_bound
    return
end

function _test_allocation(
    op,
    T,
    short,
    short_to,
    n;
    a = T(2),
    b = T(3),
    c = T(4),
)
    @test MA.promote_operation(op, T, T) == T
    alloc_test(() -> MA.promote_operation(op, T, T), 0)
    if op != div && op != -
        @test MA.promote_operation(op, T, T, T) == T
        alloc_test(() -> MA.promote_operation(op, T, T, T), 0)
    end
    g = op(a, b)
    @test c === short_to(c, a, b)
    @test g == c
    A = MA.copy_if_mutable(a)
    @test A === short(A, b)
    @test g == A
    alloc_test(() -> short(A, b), n)
    alloc_test(() -> short_to(c, a, b), n)
    @test g == MA.buffered_operate!(nothing, op, MA.copy_if_mutable(a), b)
    @test g == MA.buffered_operate_to!(nothing, c, op, a, b)
    buffer = MA.buffer_for(op, typeof(a), typeof(b))
    @test g == MA.buffered_operate_to!(buffer, c, op, a, b)
    alloc_test(() -> MA.buffered_operate_to!(buffer, c, op, a, b), 0)
    return
end

function add_sub_mul_test(op, T; a = T(2), b = T(3), c = T(4))
    g = op(a, b, c)
    @test g == MA.buffered_operate!(nothing, op, MA.copy_if_mutable(a), b, c)
    buffer = MA.buffer_for(op, typeof(a), typeof(b), typeof(c))
    return alloc_test(() -> MA.buffered_operate!(buffer, op, a, b, c), 0)
end

test_int_test_BigInt() = MA.Test.int_test(BigInt)

test_int_test_BigFloat() = MA.Test.int_test(BigFloat)

test_int_test_Rational_BigInt() = MA.Test.int_test(Rational{BigInt})

test_allocation_BigInt() = _test_allocation(BigInt)

test_allocation_BigFloat() = _test_allocation(BigFloat)

test_allocation_Rational_BigInt() = _test_allocation(Rational{BigInt})

function _test_allocation(::Type{T}) where {T}
    MAX_ALLOC = T <: Rational ? 280 : 0
    _test_allocation(+, T, MA.add!!, MA.add_to!!, MAX_ALLOC)
    _test_allocation(-, T, MA.sub!!, MA.sub_to!!, MAX_ALLOC)
    _test_allocation(*, T, MA.mul!!, MA.mul_to!!, MAX_ALLOC)
    add_sub_mul_test(MA.add_mul, T)
    add_sub_mul_test(MA.sub_mul, T)
    if T <: Rational # https://github.com/jump-dev/MutableArithmetics.jl/issues/167
        _test_allocation(
            +,
            T,
            MA.add!!,
            MA.add_to!!,
            MAX_ALLOC,
            a = T(1 // 2),
            b = T(3 // 2),
            c = T(5 // 2),
        )
        _test_allocation(
            -,
            T,
            MA.sub!!,
            MA.sub_to!!,
            MAX_ALLOC,
            a = T(1 // 2),
            b = T(3 // 2),
            c = T(5 // 2),
        )
    end
    # Requires https://github.com/JuliaLang/julia/commit/3f92832df042198b2daefc1f7ca609db38cb8173
    # for `gcd` to be defined on `Rational`.
    if T == BigInt
        _test_allocation(div, T, MA.div!!, MA.div_to!!, 0)
    end
    if T == BigInt || T == Rational{BigInt}
        _test_allocation(gcd, T, MA.gcd!!, MA.gcd_to!!, 0)
        _test_allocation(lcm, T, MA.lcm!!, MA.lcm_to!!, 0)
    end
    return
end

function test_unary_neg()
    for x in BigFloat[-1.3, -1.0, -0.7, -0.0, 0.0, 0.3, 1.0, 1.7]
        backup = MA.copy_if_mutable(x)
        @test -(backup) == MA.operate!(-, x) == x
    end
    return
end

function test_unary_abs()
    for x in BigFloat[-1.3, -1.0, -0.7, -0.0, 0.0, 0.3, 1.0, 1.7]
        backup = MA.copy_if_mutable(x)
        @test abs(backup) == MA.operate!(abs, x) == x
    end
    return
end

function _test_fma_output_values(x::F, y::F, z::F) where {F<:BigFloat}
    two_roundings_reference = x * y + z
    one_rounding_reference = fma(x, y, z)
    @test one_rounding_reference != two_roundings_reference
    @testset "fma $op output values" for op in (MA.operate!, MA.operate!!)
        (a, b, c) = map(MA.copy_if_mutable, (x, y, z))
        @inferred op(fma, a, b, c)
        @test one_rounding_reference == a
        @test y == b
        @test z == c
    end
    @testset "fma $op output values" for op in (MA.operate_to!, MA.operate_to!!)
        (a, b, c) = map(MA.copy_if_mutable, (x, y, z))
        out = F()
        @inferred op(out, fma, a, b, c)
        @test one_rounding_reference == out
        @test x == a
        @test y == b
        @test z == c
    end
    return
end

function _test_fma_output_values(x::F, y::F, z::F) where {F<:Float64}
    return _test_fma_output_values(map(BigFloat, (x, y, z))...)
end

function _test_fma_output_values_func(x::F, y::F, z::F) where {F<:Float64}
    return let x = x, y = y, z = z
        () -> _test_fma_output_values(x, y, z)
    end
end

function test_fma_output_values()
    for exp_x in (-3):3, exp_y in (-3):3, sign_x in (-1, 1), sign_y in (-1, 1)
        _test_set_precision(exp_x, exp_y, sign_x, sign_y)
    end
    return
end

function _test_set_precision(exp_x, exp_y, sign_x, sign_y)
    # Assuming a two-bit mantissa
    significand_length = 2
    sign_z = -sign_x * sign_y
    exp_z = exp_x + exp_y
    base = 2.0
    bit_0 = 2.0^0
    bit_1 = 2.0^-1
    x = sign_x * base^exp_x * (bit_0 + bit_1)
    y = sign_y * base^exp_y * (bit_0 + bit_1)
    z = sign_z * base^exp_z * (bit_0 + bit_1)
    setprecision(
        _test_fma_output_values_func(x, y, z),
        BigFloat,
        significand_length,
    )
    return
end

function test_muladd_operate_to!!_type_inferred()
    m1 = BigFloat(-1.0)
    out = BigFloat()
    @test iszero(@inferred MA.operate_to!!(out, Base.muladd, m1, m1, m1))
    return
end

function test_muladd_operate!!_type_inferred()
    x = BigFloat(-1.0)
    y = BigFloat(-1.0)
    z = BigFloat(-1.0)
    @test iszero(@inferred MA.operate!!(Base.muladd, x, y, z))
    return
end

function test_fma_operate_to!_doesnt_allocate()
    alloc_test(let o = big"1.3", x = big"1.3"
        () -> MA.operate_to!(o, Base.fma, x, x, x)
    end, 0)
    return
end

function test_fma_operate_to!!_doesnt_allocate()
    alloc_test(let o = big"1.3", x = big"1.3"
        () -> MA.operate_to!!(o, Base.fma, x, x, x)
    end, 0)
    return
end

function test_fma_operate!_doesnt_allocate()
    alloc_test(let x = big"1.3", y = big"1.3", z = big"1.3"
        () -> MA.operate!(Base.fma, x, y, z)
    end, 0)
    return
end

function test_fma_operate!!_doesnt_allocate()
    alloc_test(let x = big"1.3", y = big"1.3", z = big"1.3"
        () -> MA.operate!!(Base.fma, x, y, z)
    end, 0)
    return
end

function test_precision()
    iters = Iterators.product(
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
    @testset "prec:$prec size:$size bias:$bias" for (prec, size, bias) in iters
        _test_precision(prec, size, bias)
    end
    return
end

function _test_precision(prec, size, bias)
    err = setprecision(BigFloat, prec) do
        maximum_relative_error = mapreduce(max, 1:10) do _
            # Generate some random vectors for dot(x, y) input.
            x = rand(BigFloat, size) .- bias
            y = rand(BigFloat, size) .- bias
            # Copy x and y so that we can check we haven't mutated them after
            # the fact.
            old_x, old_y = MA.copy_if_mutable(x), MA.copy_if_mutable(y)
            # Compute output = dot(x, y)
            buf = MA.buffer_for(
                LinearAlgebra.dot,
                Vector{BigFloat},
                Vector{BigFloat},
            )
            output = BigFloat()
            MA.buffered_operate_to!!(buf, output, LinearAlgebra.dot, x, y)
            # Check that we haven't mutated x or y
            @test old_x == x
            @test old_y == y
            # Compute dot(x, y) in larger precision. This will be used to
            # compare with our `dot`.
            accurate = setprecision(BigFloat, 8 * precision(BigFloat)) do
                    return LinearAlgebra.dot(x, y)
                end
            # Compute the relative error
            return abs(accurate - output) / abs(accurate)
        end
        # Return estimate for ULP
        return maximum_relative_error / eps(BigFloat)
    end
    @test 0 <= err < 1
    return
end

function _alloc_test_helper(
    buf::B,
    output::BigFloat,
    x::Vector{BigFloat},
    y::Vector{BigFloat},
) where {B<:Any}
    let b = buf, o = output, x = x, y = y
        () -> MA.buffered_operate_to!!(b, o, LinearAlgebra.dot, x, y)
    end
end

function test_alloc()
    x = rand(BigFloat, 1000)
    y = rand(BigFloat, 1000)
    V = Vector{BigFloat}
    buffer = MA.buffer_for(LinearAlgebra.dot, V, V)
    alloc_test(_alloc_test_helper(buffer, BigFloat(), x, y), 0)
    return
end

end  # TestBig

TestBig.runtests()
