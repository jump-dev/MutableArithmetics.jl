# Copyright (c) 2023 MutableArithmetics.jl contributors
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v.2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.

module TestBasics

using Test

import MutableArithmetics as MA

include("dummy.jl")

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

function _test_copy(::Type{T}) where {T}
    @test MA.operate!!(copy, T(2)) == 2
    @test MA.operate_to!!(T(3), copy, T(2)) == 2
    if MA.mutability(T, copy, T) != MA.IsMutable()
        return
    end
    @testset "correctness" begin
        x = T(2)
        y = T(3)
        @test MA.operate!(copy, x) === x == 2
        @test MA.operate_to!(y, copy, x) === y == 2
    end
    @testset "alloc" begin
        f = let x = T(2)
            () -> MA.operate!(copy, x)
        end
        g = let x = T(2), y = T(3)
            () -> MA.operate_to!(y, copy, x)
        end
        alloc_test(f, 0)
        alloc_test(g, 0)
    end
    return
end

test_copy_Float64() = _test_copy(Float64)

test_copy_BigFloat() = _test_copy(BigFloat)

test_copy_Int() = _test_copy(Int)

test_copy_BigInt() = _test_copy(BigInt)

test_copy_Rational_Int() = _test_copy(Rational{Int})

test_copy_Rational_BigInt() = _test_copy(Rational{BigInt})

function _test_mutating_step_range(::Type{T}) where {T}
    r = MA.MutatingStepRange(T(2), T(3), T(9))
    expected = MA.mutability(T) isa MA.IsMutable ? 8 * ones(T, 3) : T[2, 5, 8]
    @test collect(r) == expected
    @test reduce(MA.add!!, r) == T(15)
    return
end

test_mutating_step_range_Int() = _test_mutating_step_range(Int)

test_mutating_step_range_BigInt() = _test_mutating_step_range(BigInt)

function test_Zero()
    z = MA.Zero()
    @test zero(MA.Zero) isa MA.Zero
    @test z + z isa MA.Zero
    @test z + 1 == 1
    @test 1 + z == 1
    @test z * z isa MA.Zero
    @test z * 1 isa MA.Zero
    @test 1 * z isa MA.Zero
    @test -z isa MA.Zero
    @test +z isa MA.Zero
    @test *(z) isa MA.Zero
    @test iszero(z)
    @test !isone(z)
    return
end

# *(::Complex, ::Hermitian)
function test_mult_Complex_Hermitian()
    A = BigInt[1 2; 2 3]
    B = LinearAlgebra.Hermitian(DummyBigInt.(A))
    C = 2im * A
    @test 2im * B == C
    @test C isa Matrix{Complex{BigInt}}
    return
end

function test_operate_to!_Array_AbstractMutable_Array()
    for A in ([1 2; 5 3], DummyBigInt[1 2; 5 3])
        for x in (2, DummyBigInt(2))
            # operate_to!(::Array, *, ::AbstractMutable, ::Array)
            B = x * A
            C = zero(B)
            D = MA.operate_to!(C, *, x, A)
            @test C === D
            @test typeof(B) == typeof(C)
            @test MA.isequal_canonical(B, C)
            # operate_to!(::Array, *, ::Array, ::AbstractMutable)
            B = A * x
            C = zero(B)
            D = MA.operate_to!(C, *, A, x)
            @test C === D
            @test typeof(B) == typeof(C)
            @test MA.isequal_canonical(B, C)
        end
    end
    return
end

function non_mutable_sum_pr306(x)
    y = zero(eltype(x))
    for xi in x
        y += xi
    end
    return y
end

function test_sum_with_init()
    x = convert(Vector{DummyBigInt}, 1:100)
    # compilation
    @allocated sum(x)
    @allocated sum(x; init = DummyBigInt(0))
    @allocated non_mutable_sum_pr306(x)
    # now test actual allocations
    no_init = @allocated sum(x)
    with_init = @allocated sum(x; init = DummyBigInt(0))
    no_ma = @allocated non_mutable_sum_pr306(x)
    # There's an additional 16 bytes for kwarg version. Upper bound by 40 to be
    # safe between Julia versions
    @test with_init <= no_init + 40
    # MA is at least 10-times better than no MA for this example
    @test 10 * with_init < no_ma
end

function test_sum_with_init_and_dims()
    x = reshape(convert(Vector{DummyBigInt}, 1:12), 3, 4)
    X = reshape(1:12, 3, 4)
    for dims in (1, 2, :, 1:2, (1, 2))
        # Without (; init)
        @test MA.isequal_canonical(sum(x; dims), DummyBigInt.(sum(X; dims)))
        # With (; init)
        y = sum(x; init = DummyBigInt(0), dims)
        @test MA.isequal_canonical(y, DummyBigInt.(sum(X; dims)))
    end
    return
end

function test_issue_336()
    A = DummyBigInt[1 2; 3 4]
    B = Any[A[1], A[2]]
    @test MA.isequal_canonical(A * B, A * A[1:2])
    @test_throws DimensionMismatch A * Any[A[1]]
    @test_throws DimensionMismatch MA.operate(*, A, Any[A[1]])
    return
end

function test_dispatch_dot()
    # Symmetric
    x = DummyBigInt[1 2; 2 3]
    y = LinearAlgebra.Symmetric(x)
    @test MA.isequal_canonical(
        LinearAlgebra.dot(x, y),
        MA.operate(LinearAlgebra.dot, x, y),
    )
    a = @allocated LinearAlgebra.dot(x, y)
    b = @allocated MA.operate(LinearAlgebra.dot, x, y)
    @test a == b
    # Symmetric
    x = DummyBigInt[1 2; 2 3]
    y = LinearAlgebra.Hermitian(x)
    @test MA.isequal_canonical(
        LinearAlgebra.dot(x, y),
        MA.operate(LinearAlgebra.dot, x, y),
    )
    a = @allocated LinearAlgebra.dot(x, y)
    b = @allocated MA.operate(LinearAlgebra.dot, x, y)
    @test a == b
    # AbstractArray
    x = DummyBigInt[1 2; 2 3]
    y = x
    @test MA.isequal_canonical(
        LinearAlgebra.dot(x, y),
        MA.operate(LinearAlgebra.dot, x, y),
    )
    a = @allocated LinearAlgebra.dot(x, y)
    b = @allocated MA.operate(LinearAlgebra.dot, x, y)
    @test a == b
    return
end

# Tests that the calls are correctly redirected to the mutable calls
# by checking allocations
function _test_dispatch(::Type{T}) where {T}
    buffer = zero(T)
    a = one(T)
    b = one(T)
    c = one(T)
    x = convert.(T, [1, 2, 3])
    # Need to allocate 1 BigInt for the result and one for the buffer
    nalloc = 3 * @allocated(BigInt(1))
    alloc_test(() -> MA.fused_map_reduce(MA.add_mul, x, x), nalloc)
    alloc_test(() -> MA.fused_map_reduce(MA.add_dot, x, x), nalloc)
    if T <: MA.AbstractMutable
        alloc_test(() -> x'x, nalloc)
        alloc_test(() -> transpose(x) * x, nalloc)
        alloc_test(() -> LinearAlgebra.dot(x, x), nalloc)
    end
    return
end

test_dispatch_BigInt() = _test_dispatch(BigInt)

test_dispatch_DummyBigInt() = _test_dispatch(DummyBigInt)

function test_dot_non_concrete_vector()
    x = [5.0, 6.0]
    y = Vector{Union{Float64,String}}(x)
    @test MA.operate(LinearAlgebra.dot, x, y) == LinearAlgebra.dot(x, y)
    @test MA.operate(*, x', y) == x' * y
    return
end

function test_dot_vector_of_vectors()
    x = [5.0, 6.0]
    z = [x, x]
    @test MA.operate(LinearAlgebra.dot, z, z) == LinearAlgebra.dot(z, z)
    return
end

# *(::Real, ::Hermitian)
function test_mult_Real_Hermitian()
    A = DummyBigInt[1 2; 2 3]
    B = DummyBigInt[2 4; 4 6]
    D = LinearAlgebra.Hermitian(B)
    for s in (:L, :U)
        Ah = LinearAlgebra.Hermitian(A, s)
        @test all(MA.isequal_canonical.(2 * Ah, D))
        @test all(MA.isequal_canonical.(Ah * 2, D))
    end
    return
end

# *(::AbstractMutable, ::Symmetric)
function test_mult_AbstractMutable_Symmetric()
    for A in ([1 2; 5 3], DummyBigInt[1 2; 5 3])
        for x in (2, DummyBigInt(2))
            for s in (:L, :U)
                # *(::AbstractMutable, ::Symmetric)
                B = LinearAlgebra.Symmetric(A, s)
                C = LinearAlgebra.Symmetric(x * A, s)
                D = x * B
                @test D isa LinearAlgebra.Symmetric
                @test MA.isequal_canonical(D, C)
                @test MA.isequal_canonical(D + D, 2 * D)
                # *(::Symmetric, ::AbstractMutable)
                B = LinearAlgebra.Symmetric(A, s)
                C = LinearAlgebra.Symmetric(A * x, s)
                D = B * x
                @test D isa LinearAlgebra.Symmetric
                @test MA.isequal_canonical(D, C)
                @test MA.isequal_canonical(D + D, 2 * D)
            end
        end
    end
end

function test_issue_271_mutability()
    a = 1
    x = [1; 2;;]
    y = [1 2; 3 4]
    z = [1 2; 3 4; 5 6]
    @test MA.mutability(x, *, x, x') == MA.IsNotMutable()
    @test MA.mutability(x, *, x', x) == MA.IsNotMutable()
    @test MA.mutability(x, *, x, a, x') == MA.IsNotMutable()
    @test MA.mutability(x, *, x', a, x) == MA.IsNotMutable()
    @test MA.mutability(y, *, y, y) == MA.IsMutable()
    @test MA.mutability(y, *, y, y, y) == MA.IsMutable()
    @test MA.mutability(y, *, y, a, y') == MA.IsMutable()
    @test MA.mutability(y, *, y', a, y) == MA.IsMutable()
    @test MA.mutability(y, *, a, a, y) == MA.IsMutable()
    @test MA.mutability(y, *, y, z', z) == MA.IsMutable()
    @test MA.mutability(z, *, z, z) == MA.IsNotMutable()
    @test MA.mutability(z, *, z, z, y) == MA.IsNotMutable()
    return
end

function test_issue_316_SubArray()
    y = reshape([1.0], 1, 1, 1)
    Y = view(y,:,:,1)
    ret = reshape([1.0], 1, 1)
    ret = MA.operate!!(MA.add_mul, ret, 2.0, Y)
    @test ret == reshape([3.0], 1, 1)
    @test y == reshape([1.0], 1, 1, 1)
    return
end

function test_issue_318_neutral_element()
    a = rand(3)
    A = [rand(2, 2) for _ in 1:3]
    @test_throws DimensionMismatch MA.operate(LinearAlgebra.dot, a, A)
    y = a' * A
    @test isapprox(MA.fused_map_reduce(MA.add_mul, a', A), y)
    z = MA.operate(LinearAlgebra.dot, Int[], Int[])
    @test iszero(z) && z isa Int
    z = MA.operate(LinearAlgebra.dot, BigInt[], Int[])
    @test iszero(z) && z isa BigInt
    z = MA.operate(LinearAlgebra.dot, Int[], Float64[])
    @test iszero(z) && z isa Float64
    z = MA.operate(LinearAlgebra.dot, Matrix{Int}[], Matrix{Float64}[])
    @test iszero(z) && z isa Float64
    @test MA.fused_map_reduce(MA.add_mul, Matrix{Int}[], Float64[]) isa MA.Zero
    @test MA.fused_map_reduce(MA.add_mul, Float64[], Matrix{Int}[]) isa MA.Zero
    return
end

function test_add_mul_for_BitArray()
    x = BigInt[0, 0]
    MA.operate!!(MA.add_mul, x, big(2), trues(2))
    @test x == BigInt[2, 2]
    MA.operate!!(MA.add_mul, x, big(3), BitVector([true, false]))
    @test x == BigInt[5, 2]
    x = BigInt[0 0; 0 0]
    MA.operate!!(MA.add_mul, x, big(2), trues(2, 2))
    @test x == BigInt[2 2; 2 2]
    MA.operate!!(MA.add_mul, x, big(3), BitArray([true false; true true]))
    @test x == BigInt[5 2; 5 5]
    return
end

function test_similar_array_type()
    @test MA.similar_array_type(BitArray{2}, Int) == Array{Int,2}
    @test MA.similar_array_type(BitArray{2}, Bool) == BitArray{2}
    return
end

function test_similar_array_type_Diagonal()
    z = zeros(2, 2)
    y = MA.operate!!(MA.add_mul, z, big(1), LinearAlgebra.I(2))
    @test y == BigFloat[1 0; 0 1]
    y = MA.operate!!(MA.add_mul, z, 2.4, LinearAlgebra.I(2))
    @test y === z
    @test y == Float64[2.4 0; 0 2.4]
    z = zeros(2, 2)
    y = MA.operate!!(MA.add_mul, z, 2.4, LinearAlgebra.Diagonal(1:2))
    @test y == LinearAlgebra.Diagonal(2.4 * (1:2))
    return
end

function test_Errors()
    err = ArgumentError(
        "Cannot call `operate_to!(::$Int, +, ::$Int, ::$Int)` as objects of type `$Int` cannot be modifed to equal the result of the operation. Use `operate_to!!` instead which returns the value of the result (possibly modifying the first argument) to write generic code that also works when the type cannot be modified.",
    )
    @test_throws err MA.operate_to!(0, +, 0, 0)
    err = ArgumentError(
        "Cannot call `operate!(+, ::$Int, ::$Int)` as objects of type `$Int` cannot be modifed to equal the result of the operation. Use `operate!!` instead which returns the value of the result (possibly modifying the first argument) to write generic code that also works when the type cannot be modified.",
    )
    @test_throws err MA.operate!(+, 0, 0)
    err = ArgumentError(
        "Cannot call `buffered_operate_to!(::$Int, ::$Int, +, ::$Int, ::$Int)` as objects of type `$Int` cannot be modifed to equal the result of the operation. Use `buffered_operate_to!!` instead which returns the value of the result (possibly modifying the first argument) to write generic code that also works when the type cannot be modified.",
    )
    @test_throws err MA.buffered_operate_to!(0, 0, +, 0, 0)
    err = ArgumentError(
        "Cannot call `buffered_operate!(::$Int, +, ::$Int, ::$Int)` as objects of type `$Int` cannot be modifed to equal the result of the operation. Use `buffered_operate!!` instead which returns the value of the result (possibly modifying the first argument) to write generic code that also works when the type cannot be modified.",
    )
    @test_throws err MA.buffered_operate!(0, +, 0, 0)
    x = DummyMutable()
    err = ErrorException(
        "`operate_to!(::$DummyMutable, +, ::$DummyMutable, ::$DummyMutable)` is not implemented yet.",
    )
    @test_throws err MA.operate_to!(x, +, x, x)
    err = ErrorException(
        "`operate!(+, ::$DummyMutable, ::$DummyMutable)` is not implemented yet.",
    )
    @test_throws err MA.operate!(+, x, x)
    err = ErrorException(
        "`buffered_operate_to!(::$DummyMutable, ::$DummyMutable, +, ::$DummyMutable, ::$DummyMutable)` is not implemented.",
    )
    @test_throws err MA.buffered_operate_to!(x, x, +, x, x)
    err = ErrorException(
        "`buffered_operate!(::$DummyMutable, +, ::$DummyMutable, ::$DummyMutable)` is not implemented.",
    )
    @test_throws err MA.buffered_operate!(x, +, x, x)
    return
end

function test_issue_240_error_free_mutability()
    for op in (+, -, *, /, div)
        for T in
            (Float64, BigFloat, Int, BigInt, Rational{Int}, Rational{BigInt})
            @test_nowarn MA.mutability(T, op, T, T) # should run without error
        end
    end
    return
end

function test_unary_op()
    @testset "unary op(::$T)" for T in (
        Float64,
        BigFloat,
        Int,
        BigInt,
        Rational{Int},
        Rational{BigInt},
    )
        @test MA.operate!!(+, T(7)) == 7
        @test MA.operate!!(*, T(7)) == 7
        @test MA.operate!!(-, T(7)) == -7
        @test MA.operate_to!!(T(6), +, T(7)) == 7
        @test MA.operate_to!!(T(6), *, T(7)) == 7
        @test MA.operate_to!!(T(6), -, T(7)) == -7
        @test MA.operate!!(abs, T(7)) == 7
        @test MA.operate!!(abs, T(-7)) == 7
        @test MA.operate_to!!(T(6), abs, T(7)) == 7
        @test MA.operate_to!!(T(6), abs, T(-7)) == 7
    end
    return
end

# Theodorus' constant √3 (deined in global scope due to const)
Base.@irrational theodorus 1.73205080756887729353 sqrt(big(3))

function test_issue_164()
    iπ() = MA.promote_operation(+, Int, typeof(π))
    @test iπ() == Float64
    alloc_test(iπ, 0)
    ℯbf() = MA.promote_operation(+, typeof(ℯ), BigFloat)
    @test ℯbf() == BigFloat
    # TODO this allocates as it creates the `BigFloat`
    #alloc_test(ℯbf, 0)
    bγ() = MA.promote_operation(/, Bool, typeof(Base.MathConstants.γ))
    @test bγ() == Float64
    alloc_test(bγ, 0)
    φf32() = MA.promote_operation(*, typeof(Base.MathConstants.φ), Float32)
    @test φf32() == Float32
    alloc_test(φf32, 0)
    # test user-defined Irrational
    i_theodorus() = MA.promote_operation(+, Int, typeof(theodorus))
    @test i_theodorus() == Float64
    alloc_test(i_theodorus, 0)
    # test _instantiate(::Type{S}) where {S<:Irrational} return value
    @test MA._instantiate(typeof(π)) == π
    @test MA._instantiate(typeof(MathConstants.catalan)) ==
          MathConstants.catalan
    @test MA._instantiate(typeof(theodorus)) == theodorus
    return
end

function test_promote_operation_complex()
    for op in [real, imag]
        @test MA.promote_operation(op, ComplexF64) == Float64
        @test MA.promote_operation(op, Complex{Real}) == Real
    end
    return
end

function test_promote_operation()
    @test MA.promote_operation(/, Rational{Int}, Rational{Int}) == Rational{Int}
    @test MA.promote_operation(-, DummyMutable, DummyMutable) == DummyMutable
    @test MA.promote_operation(/, DummyMutable, DummyMutable) == DummyMutable
    return
end

function _test_unary_op(::Type{T}, op) where {T}
    x = T(7)
    a = op(x)
    b = MA.operate(op, x)
    @test a == b
    if MA.mutability(T, op, T) == MA.IsMutable()
        @test a !== b
    end
    return
end

function _test_binary_op(::Type{T}, op) where {T}
    x = T(7)
    a = op(x, x)
    b = MA.operate(op, x, x)
    @test a == b
    if MA.mutability(T, op, T, T) == MA.IsMutable()
        @test a !== b
    end
    return
end

function _test_quaternary_op(::Type{T}, op) where {T}
    x = T(7)
    a = op(x, x, x, x)
    b = MA.operate(op, x, x, x, x)
    @test a == b
    if MA.mutability(T, op, T, T, T, T) == MA.IsMutable()
        @test a !== b
    end
    return
end

function _test_operate_dot(::Type{T}, ::Type{U}) where {T,U}
    x = T(7)
    y = U(5)
    a = LinearAlgebra.dot(x, y)
    b = MA.operate(LinearAlgebra.dot, [x], [y])
    @test a == b
    if MA.mutability(
        promote_type(T, U),
        LinearAlgebra.dot,
        Vector{T},
        Vector{U},
    ) == MA.IsMutable()
        @test a !== b
    end
end

function test_operate()
    @testset "$T" for T in (Int, BigInt, Rational{Int})
        @testset "1-ary $op" for op in [+, *, gcd, lcm, copy, abs]
            _test_unary_op(T, op)
        end
        ops = [+, *, MA.add_mul, MA.sub_mul, MA.add_dot, gcd, lcm]
        @testset "4-ary $op" for op in ops
            _test_quaternary_op(T, op)
        end
        @testset "2-ary $op" for op in [-, /, div]
            _test_binary_op(T, op)
        end
        @testset "dot" for U in (Int, BigInt, Rational{Int}, Rational{BigInt})
            _test_operate_dot(T, U)
        end
    end
    return
end

function test_op_divide()
    @test MA.Zero() / DummyBigInt(1) == MA.Zero()
    @test_throws DivideError MA.Zero() / DummyBigInt(0)
    return
end

end  # TestBasics

TestBasics.runtests()
