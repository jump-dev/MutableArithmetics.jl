# Copyright (c) 2019 MutableArithmetics.jl contributors
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v.2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.

# This file contains methods to implement the MutableArithmetics API for
# Base.BigFloat.

mutability(::Type{BigFloat}) = IsMutable()

# Copied from `deepcopy_internal` implementation in Julia:
# https://github.com/JuliaLang/julia/blob/7d41d1eb610cad490cbaece8887f9bbd2a775021/base/mpfr.jl#L1041-L1050
function mutable_copy(x::BigFloat)
    d = x._d
    d′ = GC.@preserve d unsafe_string(pointer(d), sizeof(d)) # creates a definitely-new String
    return Base.MPFR._BigFloat(x.prec, x.sign, x.exp, d′)
end

const _MPFRRoundingMode = Base.MPFR.MPFRRoundingMode

# zero

promote_operation(::typeof(zero), ::Type{BigFloat}) = BigFloat

function _set_si!(x::BigFloat, value)
    ccall(
        (:mpfr_set_si, :libmpfr),
        Int32,
        (Ref{BigFloat}, Clong, _MPFRRoundingMode),
        x,
        value,
        Base.MPFR.ROUNDING_MODE[],
    )
    return x
end
operate!(::typeof(zero), x::BigFloat) = _set_si!(x, 0)

# one

promote_operation(::typeof(one), ::Type{BigFloat}) = BigFloat

operate!(::typeof(one), x::BigFloat) = _set_si!(x, 1)

# +

function promote_operation(::typeof(+), ::Type{BigFloat}, ::Type{BigFloat})
    return BigFloat
end

function operate_to!(output::BigFloat, ::typeof(+), a::BigFloat, b::BigFloat)
    ccall(
        (:mpfr_add, :libmpfr),
        Int32,
        (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, _MPFRRoundingMode),
        output,
        a,
        b,
        Base.MPFR.ROUNDING_MODE[],
    )
    return output
end

# -

promote_operation(::typeof(-), ::Vararg{Type{BigFloat},N}) where {N} = BigFloat

function operate_to!(output::BigFloat, ::typeof(-), a::BigFloat, b::BigFloat)
    ccall(
        (:mpfr_sub, :libmpfr),
        Int32,
        (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, _MPFRRoundingMode),
        output,
        a,
        b,
        Base.MPFR.ROUNDING_MODE[],
    )
    return output
end

function operate!(::typeof(-), x::BigFloat)
    ccall(
        (:mpfr_neg, :libmpfr),
        Int32,
        (Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode),
        x,
        x,
        Base.MPFR.ROUNDING_MODE[],
    )
    return x
end

# Base.abs

function operate!(::typeof(Base.abs), x::BigFloat)
    ccall(
        (:mpfr_abs, :libmpfr),
        Int32,
        (Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode),
        x,
        x,
        Base.MPFR.ROUNDING_MODE[],
    )
    return x
end

# *

promote_operation(::typeof(*), ::Type{BigFloat}, ::Type{BigFloat}) = BigFloat

function operate_to!(output::BigFloat, ::typeof(*), a::BigFloat, b::BigFloat)
    ccall(
        (:mpfr_mul, :libmpfr),
        Int32,
        (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, _MPFRRoundingMode),
        output,
        a,
        b,
        Base.MPFR.ROUNDING_MODE[],
    )
    return output
end

function operate_to!(
    output::BigFloat,
    op::Union{typeof(+),typeof(-),typeof(*)},
    a::BigFloat,
    b::BigFloat,
    c::Vararg{BigFloat,N},
) where {N}
    operate_to!(output, op, a, b)
    return operate!(op, output, c...)
end

function operate!(op::Function, x::BigFloat, args::Vararg{Any,N}) where {N}
    return operate_to!(x, op, x, args...)
end

# add_mul and sub_mul

# Buffer to hold the product
buffer_for(::AddSubMul, args::Vararg{Type{BigFloat},N}) where {N} = BigFloat()

function operate_to!(
    output::BigFloat,
    op::AddSubMul,
    x::BigFloat,
    y::BigFloat,
    z::BigFloat,
    args::Vararg{BigFloat,N},
) where {N}
    return buffered_operate_to!(BigFloat(), output, op, x, y, z, args...)
end

function buffered_operate_to!(
    buffer::BigFloat,
    output::BigFloat,
    op::AddSubMul,
    a::BigFloat,
    x::BigFloat,
    y::BigFloat,
    args::Vararg{BigFloat,N},
) where {N}
    operate_to!(buffer, *, x, y, args...)
    return operate_to!(output, add_sub_op(op), a, buffer)
end

function buffered_operate!(
    buffer::BigFloat,
    op::AddSubMul,
    x::BigFloat,
    args::Vararg{Any,N},
) where {N}
    return buffered_operate_to!(buffer, x, op, x, args...)
end

_scaling_to_bigfloat(x) = _scaling_to(BigFloat, x)

function operate_to!(
    output::BigFloat,
    op::Union{typeof(+),typeof(-),typeof(*)},
    args::Vararg{Scaling,N},
) where {N}
    return operate_to!(output, op, _scaling_to_bigfloat.(args)...)
end

function operate_to!(
    output::BigFloat,
    op::AddSubMul,
    x::Scaling,
    y::Scaling,
    z::Scaling,
    args::Vararg{Scaling,N},
) where {N}
    return operate_to!(
        output,
        op,
        _scaling_to_bigfloat(x),
        _scaling_to_bigfloat(y),
        _scaling_to_bigfloat(z),
        _scaling_to_bigfloat.(args)...,
    )
end

# Called for instance if `args` is `(v', v)` for a vector `v`.
function operate_to!(
    output::BigFloat,
    op::AddSubMul,
    x,
    y,
    z,
    args::Vararg{Any,N},
) where {N}
    return operate_to!(output, add_sub_op(op), x, *(y, z, args...))
end

struct DotBuffer{F<:Real}
    compensation::F
    summation_temp::F
    multiplication_temp::F
    inner_temp::F

    DotBuffer{F}() where {F<:Real} = new{F}(ntuple(i -> F(), Val{4}())...)
end

function buffer_for(
    ::typeof(LinearAlgebra.dot),
    ::Type{V},
    ::Type{V},
) where {V<:AbstractVector{BigFloat}}
    return DotBuffer{BigFloat}()
end

# Dot product using Kahan-Babuška-Neumaier compensated summation.
#
# Currently restricted to BigFloat, but easily extendable to other
# similar types, if any such types appear in the ecosystem.
#
# Neumaier's 1974 paper is in German and not currently available from
# Google Scholar, but a scanned version is here:
# https://www.mat.univie.ac.at/~neum/scan/01.pdf
#
# Neumaier 1974:
#   * Rundungsfehleranalyse einiger Verfahren zur Summation endlicher
#     Summen
#   * English name: Rounding Error Analysis of Some Methods for Summing
#     Finite Sums
#   * DOI: https://doi.org/10.1002%2Fzamm.19740540106
#
# The paper has the pseudocode on page two of the PDF, in section 2,
# subsection "IV. Verbessertes Kahan-Babuška-Verfahren".
#
# Wikipedia page and section:
# https://en.wikipedia.org/w/index.php?title=Kahan_summation_algorithm&oldid=1114844162#Further_enhancements
#
# TODO: further improvement: Ogita, Rump and Oishi have a compensated
#       algorithm specifically tailored for dot product, so it would
#       be better to use that (I found out only after writing the code
#       here)
#
# Pseudocode as in Neumaier's 1974 paper:
#
#   s_0 := 0
#
#   # A running compensation for lost low-order bits.
#   w_0 := 0
#
#   for m ∈ 1:n
#     s_m := a_m + s_(m-1)
#
#     if abs(a_m) ≤ abs(s_(m-1))
#       w_m := w_(m-1) + (a_m + (s_(m-1) - s_m))
#     else
#       w_m := w_(m-1) + (s_(m-1) + (a_m - s_m))
#     end
#   end
#
#   # The result, with the correction only applied once in the very
#   # end.
#   s := s_n + w_n
#
# Pseudocode as in the Wikipedia page on Kahan summation (equivalent to
# Neumaier, but closer to the implementation):
#
#   function KahanBabushkaNeumaierSum(input)
#       sum = 0.0
#
#       # A running compensation for lost low-order bits.
#       c = 0.0
#
#       for i ∈ eachindex(input)
#           t = sum + input[i]
#
#           if abs(input[i]) ≤ abs(sum)
#               c += (sum - t) + input[i]
#           else
#               c += (input[i] - t) + sum
#           end
#
#           sum = t
#       end
#
#       # The result, with the correction only applied once in the very
#       # end.
#       sum + c
#   end
function buffered_operate_to!(
    buf::DotBuffer{F},
    sum::F,
    ::typeof(LinearAlgebra.dot),
    x::AbstractVector{F},
    y::AbstractVector{F},
) where {F<:BigFloat}
    local set! = function (out::F, in::F)
        ccall(
            (:mpfr_set, :libmpfr),
            Int32,
            (Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode),
            out,
            in,
            Base.MPFR.ROUNDING_MODE[],
        )
        return nothing
    end

    local swap! = function (x::BigFloat, y::BigFloat)
        ccall((:mpfr_swap, :libmpfr), Cvoid, (Ref{BigFloat}, Ref{BigFloat}), x, y)
        return nothing
    end

    # Returns abs(x) <= abs(y) without allocating.
    local abs_lte_abs = function (x::F, y::F)
        local x_is_neg = signbit(x)
        local y_is_neg = signbit(y)

        local x_neg = x_is_neg != y_is_neg

        x_neg && operate!(-, x)

        local ret = if y_is_neg
            y <= x
        else
            x <= y
        end

        x_neg && operate!(-, x)

        return ret
    end

    operate!(zero, sum)
    operate!(zero, buf.compensation)

    for i in 0:(length(x)-1)
        set!(buf.multiplication_temp, x[begin+i])
        operate!(*, buf.multiplication_temp, y[begin+i])

        operate!(zero, buf.summation_temp)
        operate_to!(buf.summation_temp, +, buf.multiplication_temp, sum)

        if abs_lte_abs(buf.multiplication_temp, sum)
            set!(buf.inner_temp, sum)
            operate!(-, buf.inner_temp, buf.summation_temp)
            operate!(+, buf.inner_temp, buf.multiplication_temp)
        else
            set!(buf.inner_temp, buf.multiplication_temp)
            operate!(-, buf.inner_temp, buf.summation_temp)
            operate!(+, buf.inner_temp, sum)
        end

        operate!(+, buf.compensation, buf.inner_temp)

        swap!(sum, buf.summation_temp)
    end

    operate!(+, sum, buf.compensation)

    return sum
end
