# Copyright (c) 2019 MutableArithmetics.jl contributors
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v.2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.

# This file contains methods to implement the MutableArithmetics API for
# Base.BigFloat.

mutability(::Type{BigFloat}) = IsMutable()

# These methods are copied from `deepcopy_internal` in `base/mpfr.jl`. We don't
# use `mutable_copy(x) = deepcopy(x)` because this creates an empty `IdDict()`
# which costs some extra allocations. We don't need the IdDict case because we
# never call `mutable_copy` recursively.
@static if VERSION >= v"1.12.0-DEV.1343"
    mutable_copy(x::BigFloat) = Base.MPFR._BigFloat(copy(getfield(x, :d)))
else
    function mutable_copy(x::BigFloat)
        d = x._d
        GC.@preserve d begin
            d′ = unsafe_string(pointer(d), sizeof(d))
            return Base.MPFR._BigFloat(x.prec, x.sign, x.exp, d′)
        end
    end
end

const _MPFRRoundingMode = Base.MPFR.MPFRRoundingMode
const _MPFRMachineSigned = Base.GMP.ClongMax
const _MPFRMachineUnsigned = Base.GMP.CulongMax
const _MPFRMachineInteger = Union{_MPFRMachineSigned,_MPFRMachineUnsigned}
const _MPFRMachineFloat = Base.GMP.CdoubleMax

make_mpfr_error() = error("Invalid use of @make_mpfr")
function make_mpfr_impl(
    fn::Expr,
    mod::Module,
    rounding_mode::Bool,
    rest::Expr...,
)
    pre = :()
    post = :()
    for restᵢ in rest
        restᵢ isa Expr && restᵢ.head === :(=) && length(restᵢ.args) == 2 ||
            make_mpfr_error()
        if restᵢ.args[1] === :pre
            pre = restᵢ.args[2]
        elseif restᵢ.args[1] === :post
            post = restᵢ.args[2]
        else
            make_mpfr_error()
        end
    end
    fn.head === :(->) || make_mpfr_error()
    if fn.args[2] isa Expr
        # Julia likes to insert a line number node
        (fn.args[2].head === :block && fn.args[2].args[1] isa LineNumberNode) ||
            make_mpfr_error()
        fn_name = fn.args[2].args[end]
    else
        fn_name = fn.args[2]
    end
    if fn_name isa Expr
        fn_name.head === :tuple
        surplus_args = fn_name.args[2:end]
        fn_name = fn_name.args[1]
    else
        surplus_args = []
    end
    fn_name isa Symbol || make_mpfr_error()
    fn = fn.args[1]::Expr
    if fn.head === :(::)
        return_type = mod.eval(fn.args[2])
        return_type <: Tuple{Vararg{BigFloat}} || make_mpfr_error()
        fn = fn.args[1]::Expr
    else
        return_type = BigFloat
    end
    fn.head === :call || make_mpfr_error()
    ju_name = fn.args[1]
    try
        mod.eval(ju_name) isa Base.Callable || make_mpfr_error()
    catch
        return :() # some functions may not be known in all Julia versions - this is not an error
    end
    args = sizehint!(Any[], length(fn.args) - 1)
    argnames = sizehint!(Symbol[], length(fn.args) - 1)
    types = sizehint!(
        Any[],
        length(fn.args) +
        (return_type === BigFloat ? 0 : fieldcount(return_type) - 1),
    )
    if return_type === BigFloat
        push!(types, Ref{BigFloat})
    else
        append!(
            types,
            Iterators.repeated(Ref{BigFloat}, fieldcount(return_type)),
        )
    end
    for (i, argᵢ) in enumerate(Iterators.drop(fn.args, 1))
        argᵢ isa Expr && argᵢ.head === :(::) || make_mpfr_error()
        argtype = mod.eval(argᵢ.args[end])
        # singleton types may be used for method disambiguation - we only need them on the Julia side
        if Base.issingletontype(argtype)
            push!(args, Expr(:(::), argtype))
            continue
        end
        argname =
            length(argᵢ.args) == 2 ? argᵢ.args[1]::Symbol : Symbol(:arg, i)
        push!(args, Expr(:(::), argname, argtype))
        push!(argnames, argname)
        if isbitstype(argtype)
            push!(types, argtype)
        elseif argtype isa Union
            push!(types, promote_type(Base.uniontypes(argtype)...))
        else
            push!(types, Ref{argtype})
        end
    end
    return quote
        function promote_operation(
            ::typeof($ju_name),
            $((:(::Type{<:$(arg.args[end])}) for arg in args)...),
        )
            return $return_type
        end

        function operate_to!(out::$return_type, ::typeof($ju_name), $(args...))
            $pre
            ccall(
                ($(QuoteNode(fn_name)), :libmpfr),
                Int32,
                (
                    $(types...),
                    $((typeof(s) for s in surplus_args)...),
                    $((rounding_mode ? (:($_MPFRRoundingMode),) : ())...),
                ),
                $(
                    (
                        return_type <: Tuple ?
                        (:(out[$i]) for i in 1:fieldcount(return_type)) :
                        (:out,)
                    )...
                ),
                $(argnames...),
                $(surplus_args...),
                $(
                    (
                        rounding_mode ?
                        (:($(Base.Rounding.rounding_raw)($BigFloat)),) : ()
                    )...
                ),
            )
            $post
            return out
        end
    end
end

macro make_mpfr(fn::Expr, rest::Expr...)
    return esc(make_mpfr_impl(fn, __module__, true, rest...))
end

macro make_mpfr_noround(fn::Expr, rest::Expr...)
    return esc(make_mpfr_impl(fn, __module__, false, rest...))
end

# copy

@make_mpfr copy(::BigFloat) -> mpfr_set
@make_mpfr copy(::_MPFRMachineSigned) -> mpfr_set_si
@make_mpfr copy(::_MPFRMachineUnsigned) -> mpfr_set_ui
# the Julia MPFR library does not come with the set_sj/set_uj functions
@make_mpfr copy(::Float32) -> mpfr_set_flt
@make_mpfr copy(::Float64) -> mpfr_set_d
@make_mpfr copy(x::Float16) -> mpfr_set_flt pre=(x = Float32(x))

operate!(::typeof(copy), x::BigFloat) = x

# zero

promote_operation(::typeof(zero), ::Type{BigFloat}) = BigFloat

operate!(::typeof(zero), x::BigFloat) = operate_to!(x, copy, 0)

# one

promote_operation(::typeof(one), ::Type{BigFloat}) = BigFloat

operate!(::typeof(one), x::BigFloat) = operate_to!(x, copy, 1)

# ldexp

@make_mpfr ldexp(::_MPFRMachineSigned, ::_MPFRMachineSigned) -> mpfr_set_si_2exp
@make_mpfr ldexp(::_MPFRMachineUnsigned, ::_MPFRMachineSigned) ->
    mpfr_set_ui_2exp
@make_mpfr ldexp(::BigFloat, ::_MPFRMachineSigned) -> mpfr_mul_2si
@make_mpfr ldexp(::BigFloat, ::_MPFRMachineUnsigned) -> mpfr_mul_2ui

# +

@make_mpfr +(::BigFloat, ::BigFloat) -> mpfr_add
@make_mpfr +(::BigFloat, ::_MPFRMachineSigned) -> mpfr_add_si
@make_mpfr +(::BigFloat, ::_MPFRMachineUnsigned) -> mpfr_add_ui
@make_mpfr +(::BigFloat, ::_MPFRMachineFloat) -> mpfr_add_d

operate_to!(out::BigFloat, ::typeof(+), a::Real) = operate_to!(out, copy, a)

operate!(::typeof(+), a::BigFloat) = a

# -

@make_mpfr -(::BigFloat, ::BigFloat) -> mpfr_sub
@make_mpfr -(::BigFloat, ::_MPFRMachineSigned) -> mpfr_sub_si
@make_mpfr -(::BigFloat, ::_MPFRMachineUnsigned) -> mpfr_sub_ui
@make_mpfr -(::BigFloat, ::_MPFRMachineFloat) -> mpfr_sub_d
@make_mpfr -(::_MPFRMachineSigned, ::BigFloat) -> mpfr_si_sub
@make_mpfr -(::_MPFRMachineUnsigned, ::BigFloat) -> mpfr_ui_sub
@make_mpfr -(::_MPFRMachineFloat, ::BigFloat) -> mpfr_d_sub

promote_operation(::typeof(-), ::Type{BigFloat}) = BigFloat

function operate!(::typeof(-), x::BigFloat)
    x.sign = -x.sign
    return x
end

function operate_to!(o::BigFloat, ::typeof(-), x::Real)
    operate_to!(o, copy, x)
    return operate!(-, o)
end

# abs

function operate!(::typeof(abs), x::BigFloat)
    x.sign = abs(x.sign)
    return x
end

function operate_to!(o::BigFloat, ::typeof(abs), x::Real)
    operate_to!(o, copy, x)
    return operate!(abs, o)
end

# *

@make_mpfr *(::BigFloat, ::BigFloat) -> mpfr_mul
@make_mpfr *(::BigFloat, ::_MPFRMachineSigned) -> mpfr_mul_si
@make_mpfr *(::BigFloat, ::_MPFRMachineUnsigned) -> mpfr_mul_ui
@make_mpfr *(::BigFloat, ::_MPFRMachineFloat) -> mpfr_mul_d

operate_to!(out::BigFloat, ::typeof(*), a::Real) = operate_to!(out, copy, a)

operate!(::typeof(*), a::BigFloat) = a

# /

@make_mpfr /(::BigFloat, ::BigFloat) -> mpfr_div
@make_mpfr /(::BigFloat, ::_MPFRMachineSigned) -> mpfr_div_si
@make_mpfr /(::BigFloat, ::_MPFRMachineUnsigned) -> mpfr_div_ui
@make_mpfr /(::BigFloat, ::_MPFRMachineFloat) -> mpfr_div_d
@make_mpfr /(::_MPFRMachineSigned, ::BigFloat) -> mpfr_si_div
@make_mpfr /(::_MPFRMachineUnsigned, ::BigFloat) -> mpfr_ui_div
@make_mpfr /(::_MPFRMachineFloat, ::BigFloat) -> mpfr_d_div

# roots
@make_mpfr sqrt(x::BigFloat) -> mpfr_sqrt pre=begin
    isnan(x) && return operate_to!(out, copy, x)
end post=begin
    isnan(out) && throw(DomainError(x, "NaN result for non-NaN input"))
end
@make_mpfr sqrt(::_MPFRMachineUnsigned) -> mpfr_sqrt_ui
@make_mpfr cbrt(::BigFloat) -> mpfr_cbrt
@make_mpfr fourthroot(::BigFloat) -> (mpfr_rootn_ui, 0x00000004)

# factorial

@make_mpfr factorial(::_MPFRMachineUnsigned) -> mpfr_fac_ui

# Base.fma

@make_mpfr fma(::BigFloat, ::BigFloat, ::BigFloat) -> mpfr_fma

function operate!(::typeof(fma), x::F, y::F, z::F) where {F<:BigFloat}
    return operate_to!(x, fma, x, y, z)
end

# hypot

@make_mpfr hypot(::BigFloat, ::BigFloat) -> mpfr_hypot

# log

@make_mpfr log(::_MPFRMachineUnsigned) -> mpfr_log_ui
for f in (:log, :log2, :log10)
    @eval @make_mpfr $f(x::BigFloat) -> $(Symbol(:mpfr_, f)) pre=begin
        if x < 0
            throw(
                DomainError(
                    x,
                    string(
                        $f,
                        " was called with a negative real argument but ",
                        "will only return a complex result if called ",
                        "with a complex argument. Try ",
                        $f,
                        "(complex(x)).",
                    ),
                ),
            )
        end
    end
end
@make_mpfr log1p(x::BigFloat) -> mpfr_log1p pre=begin
    if x < -1
        throw(
            DomainError(
                x,
                string(
                    "log1p was called with a real argument < -1 but ",
                    "will only return a complex result if called ",
                    "with a complex argument. Try log1p(complex(x)).",
                ),
            ),
        )
    end
end

# exp

@make_mpfr exp(::BigFloat) -> mpfr_exp
@make_mpfr exp2(::BigFloat) -> mpfr_exp2
@make_mpfr exp10(::BigFloat) -> mpfr_exp10
@make_mpfr expm1(::BigFloat) -> mpfr_expm1

# ^
@make_mpfr ^(::BigFloat, ::BigFloat) -> mpfr_pow
@make_mpfr ^(::BigFloat, ::_MPFRMachineSigned) -> mpfr_pow_si
@make_mpfr ^(::BigFloat, ::_MPFRMachineUnsigned) -> mpfr_pow_ui

# trigonometric
# Functions for which NaN results are converted to DomainError, following Base
for f in (
    :sin,
    :cos,
    :tan,
    :cot,
    :sec,
    :csc,
    :acos,
    :asin,
    :atan,
    :acosh,
    :asinh,
    :atanh,
    (VERSION ≥ v"1.10" ? (:sinpi, :cospi, :tanpi) : ())...,
)
    @eval @make_mpfr $f(x::BigFloat) -> $(Symbol(:mpfr_, f)) pre=begin
        isnan(x) && return operate_to!(out, copy, x)
    end post=begin
        isnan(out) && throw(DomainError(x, "NaN result for non-NaN input."))
    end
end
@make_mpfr sincos(::BigFloat)::Tuple{BigFloat,BigFloat} -> mpfr_sin_cos
VERSION ≥ v"1.10" && for f in (:sin, :cos, :tan)
    @eval begin
        @make_mpfr $(Symbol(f, :d))(x::BigFloat) ->
            ($(Symbol(:mpfr_, f, :u)), 0x00000168) pre=begin
            isnan(x) && return operate_to!(out, copy, x)
        end post=begin
            isnan(out) && throw(DomainError(x, "NaN result for non-NaN input."))
        end

        @make_mpfr $(Symbol(:a, f, :d))(x::BigFloat) ->
            ($(Symbol(:mpfr_a, f, :u)), 0x00000168) pre=begin
            isnan(x) && return operate_to!(out, copy, x)
        end post=begin
            isnan(out) && throw(DomainError(x, "NaN result for non-NaN input."))
        end
    end
end
@make_mpfr atan(::BigFloat, ::BigFloat) -> mpfr_atan2
VERSION ≥ v"1.10" &&
    @make_mpfr atand(::BigFloat, ::BigFloat) -> (mpfr_atan2u, 0x00000168)

# hyperbolic
@make_mpfr cosh(::BigFloat) -> mpfr_cosh
@make_mpfr sinh(::BigFloat) -> mpfr_sinh
@make_mpfr tanh(::BigFloat) -> mpfr_tanh
@make_mpfr sech(::BigFloat) -> mpfr_sech
@make_mpfr csch(::BigFloat) -> mpfr_csch
@make_mpfr coth(::BigFloat) -> mpfr_coth

# integer/remainder
@make_mpfr_noround round(::BigFloat, ::RoundingMode{:Nearest}) -> mpfr_roundeven
@make_mpfr_noround round(::BigFloat, ::RoundingMode{:Up}) -> mpfr_ceil
@make_mpfr_noround round(::BigFloat, ::RoundingMode{:Down}) -> mpfr_floor
@make_mpfr_noround round(::BigFloat, ::RoundingMode{:ToZero}) -> mpfr_trunc
@make_mpfr_noround round(::BigFloat, ::RoundingMode{:NearestTiesAway}) ->
    mpfr_round

@make_mpfr modf(::BigFloat)::Tuple{BigFloat,BigFloat} -> mpfr_modf pre=(
    out = reverse(out)
) post=(out = reverse(out))
@make_mpfr rem(::BigFloat, ::BigFloat) -> mpfr_fmod
@make_mpfr rem(::BigFloat, ::BigFloat, ::RoundingMode{:Nearest}) ->
    mpfr_remainder

# miscellaneous
@make_mpfr min(::BigFloat, ::BigFloat) -> mpfr_min
@make_mpfr max(::BigFloat, ::BigFloat) -> mpfr_max
@make_mpfr copysign(::BigFloat, ::BigFloat) -> mpfr_copysign

# Base.muladd

function promote_operation(
    ::typeof(Base.muladd),
    ::Type{F},
    ::Type{F},
    ::Type{F},
) where {F<:BigFloat}
    return F
end

function operate_to!(
    output::F,
    ::typeof(Base.muladd),
    x::F,
    y::F,
    z::F,
) where {F<:BigFloat}
    return operate_to!(output, Base.fma, x, y, z)
end

function operate!(::typeof(Base.muladd), x::F, y::F, z::F) where {F<:BigFloat}
    return operate!(Base.fma, x, y, z)
end

function operate_to!(
    output::BigFloat,
    op::Union{typeof(+),typeof(-),typeof(*)},
    a::BigFloat,
    b::Real,
    c::Vararg{Real,N},
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
    c::F
    t::F
    input::F
    tmp::F

    DotBuffer{F}() where {F<:Real} = new{F}(zero(F), zero(F), zero(F), zero(F))
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
#       # A running compensation for lost low-order bits.
#       c = 0.0
#       for i ∈ eachindex(input)
#           t = sum + input[i]
#           if abs(input[i]) ≤ abs(sum)
#               tmp = (sum - t) + input[i]
#           else
#               tmp = (input[i] - t) + sum
#           end
#           c += tmp
#           sum = t
#       end
#       # The result, with the correction only applied once in the very end.
#       sum + c
#   end

# Returns abs(x) <= abs(y) without allocating.
function _abs_lte_abs(x::BigFloat, y::BigFloat)
    x_is_neg, y_is_neg = signbit(x), signbit(y)
    if x_is_neg != y_is_neg
        operate!(-, x)
    end
    ret = y_is_neg ? y <= x : x <= y
    if x_is_neg != y_is_neg
        operate!(-, x)
    end
    return ret
end

function buffered_operate_to!(
    buf::DotBuffer{BigFloat},
    sum::BigFloat,
    ::typeof(LinearAlgebra.dot),
    x::AbstractVector{BigFloat},
    y::AbstractVector{BigFloat},
)                                                   # See pseudocode description
    operate!(zero, sum)                             # sum = 0
    operate!(zero, buf.c)                           # c = 0
    for (xi, yi) in zip(x, y)                       # for i in eachindex(input)
        operate_to!(buf.input, copy, xi)            # input = x[i]
        operate!(*, buf.input, yi)                  # input = x[i] * y[i]
        operate_to!(buf.t, +, sum, buf.input)       # t = sum + input
        if _abs_lte_abs(buf.input, sum)             # if |input| < |sum|
            operate_to!(buf.tmp, copy, sum)         # tmp = sum
            operate!(-, buf.tmp, buf.t)             # tmp = sum - t
            operate!(+, buf.tmp, buf.input)         # tmp = (sum - t) + input
        else
            operate_to!(buf.tmp, copy, buf.input)   # tmp = input
            operate!(-, buf.tmp, buf.t)             # tmp = input - t
            operate!(+, buf.tmp, sum)               # tmp = (input - t) + sum
        end
        operate!(+, buf.c, buf.tmp)                 # c += tmp
        operate_to!(sum, copy, buf.t)               # sum = t
    end
    operate!(+, sum, buf.c)                         # sum += c
    return sum
end

# Base.evalpoly

function operate!(
    op::typeof(Base.evalpoly),
    out::F,
    coefs::Union{Tuple,AbstractVector},
) where {F<:BigFloat}
    return operate_to!(out, op, mutable_copy(out), coefs)
end

function operate(
    op::typeof(Base.evalpoly),
    x::F,
    coefs::Union{Tuple,AbstractVector},
) where {F<:BigFloat}
    return operate_to!(zero(x), op, x, coefs)
end
