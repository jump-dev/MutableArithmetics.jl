# Copyright (c) 2019 MutableArithmetics.jl contributors
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v.2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.

module TestRewrite

using Test

import MutableArithmetics as MA
import OffsetArrays

include(joinpath(@__DIR__, "dummy.jl"))

function runtests()
    is_test(name::Symbol) = startswith("$name", "test_")
    @testset "$name" for name in filter(is_test, names(@__MODULE__; all = true))
        @testset "$T" for T in (
            Int,
            Float64,
            BigInt,
            BigFloat,
            Rational{Int},
            Rational{BigInt},
            DummyBigInt,
        )
            getfield(@__MODULE__, name)(T)
        end
    end
    return
end

# Test that the macro call `m` throws an error exception during pre-compilation
macro test_macro_throws(error, m)
    # See https://discourse.julialang.org/t/test-throws-with-macros-after-pr-23533/5878
    return quote
        @test_throws $(esc(error)) try
            @eval $m
        catch catched_error
            throw(catched_error.error)
        end
    end
end

function test_is_supported_test(::Type{T}) where {T}
    @test MA.Test._is_supported(*, T, T)
    @test MA.Test._is_supported(+, T, T)
    return
end

function test_Error(::Type{T}) where {T}
    x, y, z = T(1), T(2), T(3)
    err = ErrorException(
        "The curly syntax (sum{},prod{},norm2{}) is no longer supported. Expression: `sum{(i for i = 1:2)}`.",
    )
    @test_macro_throws err MA.@rewrite sum{i for i in 1:2}
    err = ErrorException("Unexpected comparison in expression `z >= y`.")
    @test_macro_throws err MA.@rewrite z >= y
    err = ErrorException("Unexpected comparison in expression `y <= z`.")
    @test_macro_throws err MA.@rewrite y <= z
    err = ErrorException("Unexpected comparison in expression `x <= y <= z`.")
    @test_macro_throws err MA.@rewrite x <= y <= z
    return
end

function test_Scalar(::Type{T}) where {T}
    MA.Test.scalar_test(T(2))
    MA.Test.scalar_test(T(3))
    return
end

function test_Quadratic(::Type{T}) where {T}
    exclude = T == DummyBigInt ? ["quadratic_division"] : String[]
    MA.Test.quadratic_test(T(1), T(2), T(3), T(4); exclude)
    return
end

function test_Sparse(::Type{T}) where {T}
    return MA.Test.sparse_test(T(4), T(5), T[8 1 9; 4 3 1; 2 0 8])
end

_exclude(::Type{DummyBigInt}) = ["broadcast_division", "matrix_vector_division"]

_exclude(::Type{<:Rational}) = ["broadcast_division"]

_exclude(::Type) = String[]

function test_Vector(::Type{T}) where {T}
    exclude = _exclude(T)
    MA.Test.array_test(T[-7, 1, 4]; exclude)
    MA.Test.array_test(T[3, 2, 6]; exclude)
    return
end

function test_Square_Matrix(::Type{T}) where {T}
    exclude = _exclude(T)
    MA.Test.array_test(T[1 2; 2 4]; exclude)
    MA.Test.array_test(T[2 4; -1 3]; exclude)
    MA.Test.array_test(T[0 -4; 6 -5]; exclude)
    MA.Test.array_test(T[5 1 9; -7 2 4; -2 -7 5]; exclude)
    return
end

function test_non_square_matrix(::Type{T}) where {T}
    exclude = _exclude(T)
    MA.Test.array_test(T[5 1 9; -7 2 4]; exclude)
    return
end

function test_tensor(::Type{T}) where {T}
    exclude = _exclude(T)
    S = zeros(T, 2, 2, 2)
    S[1, :, :] = T[5 -8; 3 -7]
    S[2, :, :] = T[-2 8; 8 -1]
    MA.Test.array_test(S; exclude)
    return
end

function test_offset_array(::Type{T}) where {T}
    x = T[2, 4, 3]
    MA.Test.non_array_test(x, OffsetArrays.OffsetArray(x, -length(x)))
    return
end

end  # TestRewrite

TestRewrite.runtests()
