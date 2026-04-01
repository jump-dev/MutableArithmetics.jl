# Copyright (c) 2023 MutableArithmetics.jl contributors
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v.2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.

module TestEvalPoly

using Test

import MutableArithmetics as MA

function runtests()
    is_test(name::Symbol) = startswith("$name", "test_")
    @testset "$name" for name in filter(is_test, names(@__MODULE__; all = true))
        getfield(@__MODULE__, name)()
    end
    return
end

_is_op_to(::Any) = false

_is_op_to(::Union{typeof(MA.operate_to!),typeof(MA.operate_to!!)}) = true

function _should_test(
    ::Union{typeof(MA.operate!),typeof(MA.operate_to!)},
    ::Type{F},
) where {F}
    return MA.mutability(F) == MA.IsMutable()
end

_should_test(op, ::Type{F}) where {F} = true

function test_evalpoly()
    for F in (BigFloat, Rational{Int}, Float64, Float32, Int)
        for op in (
            MA.operate,
            MA.operate!,
            MA.operate_to!,
            MA.operate!!,
            MA.operate_to!!,
        )
            if _should_test(op, F)
                _test_evalpoly(op, F, "vector")
                _test_evalpoly(op, F, "tuple")
            end
        end
    end
    return
end

function _test_evalpoly(op, ::Type{F}, collection_type) where {F}
    if MA.mutability(F) == MA.IsMutable()
        out = one(F)
        x = one(F)
        backup = MA.copy_if_mutable(x)
        coefs = collection_type == "tuple" ? () : F[]
        if _is_op_to(op)
            @test iszero(@inferred op(out, evalpoly, x, coefs))
        else
            @test iszero(@inferred op(evalpoly, x, coefs))
        end
        if op == MA.operate || _is_op_to(op)
            @test backup == x
        end
    end
    for degree in 0:4, x_int in -5:5
        coefs_int = rand(-6:6, degree + 1)
        coefs = map(F, coefs_int)
        x = F(x_int)
        backup = MA.copy_if_mutable(x)
        reference = evalpoly(x, coefs)
        @test reference == evalpoly(x_int, coefs_int)
        out = zero(F)
        coefs_arg = collection_type == "vector" ? coefs : (coefs...,)
        if _is_op_to(op)
            @test reference == @inferred op(out, evalpoly, x, coefs_arg)
        else
            @test reference == @inferred op(evalpoly, x, coefs_arg)
        end
        if op == MA.operate || _is_op_to(op)
            @test x == backup
        end
        @test coefs == map(F, coefs_int)
    end
    byte_cnt = _is_op_to(op) ? 0 : @allocated(zero(F))
    coefs = if collection_type == "vector"
        F[0, 1, 0, 1, 1]
    else
        F.((0, 1, 0, 1, 1))
    end
    let o = one(F), x = one(F), coefs = coefs
        if _is_op_to(op)
            @test @allocated(op(o, evalpoly, x, coefs)) <= byte_cnt
        else
            @test @allocated(op(evalpoly, x, coefs)) <= byte_cnt
        end
    end
    return
end

end  # TestEvalPoly

TestEvalPoly.runtests()
