# Copyright (c) 2019 MutableArithmetics.jl contributors
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v.2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.

using Test
import MutableArithmetics
const MA = MutableArithmetics

struct DummyMutable end

function MA.promote_operation(
    ::typeof(+),
    ::Type{DummyMutable},
    ::Type{DummyMutable},
)
    return DummyMutable
end
# Test that this does not triggers any ambiguity even if there is a fallback specific to `/`.
function MA.promote_operation(
    ::Union{typeof(-),typeof(/)},
    ::Type{DummyMutable},
    ::Type{DummyMutable},
)
    return DummyMutable
end
MA.mutability(::Type{DummyMutable}) = MA.IsMutable()

# Theodorus' constant √3 (deined in global scope due to const)
Base.@irrational theodorus 1.73205080756887729353 sqrt(big(3))
@testset "promote_operation" begin
    @test MA.promote_operation(/, Rational{Int}, Rational{Int}) == Rational{Int}
    @test MA.promote_operation(-, DummyMutable, DummyMutable) == DummyMutable
    @test MA.promote_operation(/, DummyMutable, DummyMutable) == DummyMutable
    @testset "Issue #164" begin
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
    end
end

@testset "Errors" begin
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
        "`operate_to!(::DummyMutable, +, ::DummyMutable, ::DummyMutable)` is not implemented yet.",
    )
    @test_throws err MA.operate_to!(x, +, x, x)
    err = ErrorException(
        "`operate!(+, ::DummyMutable, ::DummyMutable)` is not implemented yet.",
    )
    @test_throws err MA.operate!(+, x, x)
    err = ErrorException(
        "`buffered_operate_to!(::DummyMutable, ::DummyMutable, +, ::DummyMutable, ::DummyMutable)` is not implemented.",
    )
    @test_throws err MA.buffered_operate_to!(x, x, +, x, x)
    err = ErrorException(
        "`buffered_operate!(::DummyMutable, +, ::DummyMutable, ::DummyMutable)` is not implemented.",
    )
    @test_throws err MA.buffered_operate!(x, +, x, x)
end
