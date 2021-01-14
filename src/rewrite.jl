# Heavily inspired from `JuMP/src/parse_expr.jl` code.

export @rewrite

"""
    @macro(expr)

Return the value of `expr` exploiting the mutability of the temporary
expressions created for the computation of the result.

## Examples

The expression
```julia
MA.@rewrite(x + y * z + u * v * w)
```
is rewritten into
```julia
MA.add_mul!(MA.add_mul!(MA.copy_if_mutable(x),
                        y, z),
            u, v, w)
```
"""
macro rewrite(expr)
    return rewrite_and_return(expr)
end

struct Zero end
## We need to copy `x` as it will be used as might be given by the user and be
## given as first argument of `operate!`.
#Base.:(+)(zero::Zero, x) = copy_if_mutable(x)
## `add_mul(zero, ...)` redirects to `muladd(..., zero)` which calls `... + zero`.
#Base.:(+)(x, zero::Zero) = copy_if_mutable(x)
function operate(::typeof(add_mul), ::Zero, args::Vararg{Any,N}) where {N}
    return operate(*, args...)
end
function operate(::typeof(sub_mul), ::Zero, x)
    # `operate(*, x)` would redirect to `copy_if_mutable(x)` which would be a
    # useless copy.
    return operate(-, x)
end
function operate(::typeof(sub_mul), ::Zero, x, y, args::Vararg{Any,N}) where {N}
    return operate(-, operate(*, x, y, args...))
end
broadcast!(::Union{typeof(add_mul),typeof(+)}, ::Zero, x) = copy_if_mutable(x)
broadcast!(::typeof(add_mul), ::Zero, x, y) = x * y

# Needed in `@rewrite(1 .+ sum(1 for i in 1:0) * 1^2)`
Base.:*(z::Zero, ::Any) = z
Base.:*(::Any, z::Zero) = z
Base.:*(z::Zero, ::Zero) = z
Base.:+(::Zero, x::Any) = x
Base.:+(x::Any, ::Zero) = x
Base.:+(z::Zero, ::Zero) = z
Base.:-(::Zero, x::Any) = -x
Base.:-(x::Any, ::Zero) = x
Base.:-(z::Zero, ::Zero) = z
function Base.:/(z::Zero, x::Any)
    if iszero(x)
        throw(DivideError())
    else
        return z
    end
end

# Needed by `@rewrite(BigInt(1) .+ sum(1 for i in 1:0) * 1^2)`
# since we don't require mutable type to support Zero in
# `mutable_operate!`.
_any_zero() = false
_any_zero(::Any, args::Vararg{Any,N}) where {N} = _any_zero(args...)
_any_zero(::Zero, ::Vararg{Any,N}) where {N} = true
function operate!(
    op::Union{typeof(add_mul),typeof(sub_mul)},
    x,
    args::Vararg{Any,N},
) where {N}
    if _any_zero(args...)
        return x
    else
        return operate_fallback!(mutability(x, op, x, args...), op, x, args...)
    end
end

# Needed for `@rewrite(BigInt(1) .+ sum(1 for i in 1:0) * 1^2)`
Base.broadcastable(z::Zero) = Ref(z)
Base.ndims(::Type{Zero}) = 0
Base.length(::Zero) = 1
Base.iterate(z::Zero) = (z, nothing)
Base.iterate(::Zero, ::Nothing) = nothing

using Base.Meta

# See `JuMP._try_parse_idx_set`
function _try_parse_idx_set(arg::Expr)
    # [i=1] and x[i=1] parse as Expr(:vect, Expr(:(=), :i, 1)) and
    # Expr(:ref, :x, Expr(:kw, :i, 1)) respectively.
    if arg.head === :kw || arg.head === :(=)
        @assert length(arg.args) == 2
        return true, arg.args[1], arg.args[2]
    elseif isexpr(arg, :call) && arg.args[1] === :in
        # FIXME It seems not to be called anymore on Julia v1.0 as `i in idx`
        #       is parsed as `i = idx`.
        return true, arg.args[2], arg.args[3]
    else
        # FIXME When is this called ?
        return false, nothing, nothing
    end
end

function _parse_idx_set(arg::Expr)
    parse_done, idxvar, idxset = _try_parse_idx_set(arg)
    if parse_done
        return idxvar, idxset
    end
    error("Invalid syntax: $arg")
end

"""
    rewrite_generator(expr::Expr, inner::Function)

Rewrites the generator statements `expr` and returns a properly nested for loop
with nested filters as specified.

# Examples

```jldoctest
julia> using MutableArithmetics

julia> MutableArithmetics.rewrite_generator(:(i for i in 1:2 if isodd(i)), i -> :(\$i + 1))
:(for \$(Expr(:escape, :(i = 1:2)))
      if \$(Expr(:escape, :(isodd(i))))
          i + 1
      end
  end)
```
"""
function rewrite_generator(ex, inner)
    # `i + j for i in 1:2 for j in 1:2` is a `flatten` expression
    if isexpr(ex, :flatten)
        return rewrite_generator(ex.args[1], inner)
    end
    if !isexpr(ex, :generator)
        return inner(ex)
    end
    # `i + j for i in 1:2, j in 1:2` is a `generator` expression
    function itrsets(sets)
        if isa(sets, Expr)
            return sets
        elseif length(sets) == 1
            return sets[1]
        else
            return Expr(:block, sets...)
        end
    end

    idxvars = []
    if isexpr(ex.args[2], :filter) # if condition
        loop = Expr(
            :for,
            esc(itrsets(ex.args[2].args[2:end])),
            Expr(:if, esc(ex.args[2].args[1]), rewrite_generator(ex.args[1], inner)),
        )
        for idxset in ex.args[2].args[2:end]
            idxvar, s = _parse_idx_set(idxset)
            push!(idxvars, idxvar)
        end
    else
        loop =
            Expr(:for, esc(itrsets(ex.args[2:end])), rewrite_generator(ex.args[1], inner))
        for idxset in ex.args[2:end]
            idxvar, s = _parse_idx_set(idxset)
            push!(idxvars, idxvar)
        end
    end
    return loop
end

# See `JuMP._is_sum`
_is_sum(s::Symbol) = (s == :sum) || (s == :∑) || (s == :Σ)

function _parse_generator(
    vectorized::Bool,
    minus::Bool,
    inner_factor::Expr,
    current_sum::Union{Nothing,Symbol},
    left_factors,
    right_factors,
    new_var = gensym(),
)
    @assert isexpr(inner_factor, :call)
    @assert length(inner_factor.args) > 1
    @assert isexpr(inner_factor.args[2], :generator) ||
            isexpr(inner_factor.args[2], :flatten)
    header = inner_factor.args[1]
    if _is_sum(header)
        _parse_generator_sum(
            vectorized,
            minus,
            inner_factor.args[2],
            current_sum,
            left_factors,
            right_factors,
            new_var,
        )
    else
        error("Expected `sum` outside generator expression; got `$header`.")
    end
end

function _parse_generator_sum(
    vectorized::Bool,
    minus::Bool,
    inner_factor::Expr,
    current_sum::Union{Nothing,Symbol},
    left_factors,
    right_factors,
    new_var,
)
    # We used to preallocate the expression at the lowest level of the loop.
    # When rewriting this some benchmarks revealed that it actually doesn't
    # seem to help anymore, so might as well keep the code simple.
    return _start_summing(
        current_sum,
        current_sum -> begin
            code = rewrite_generator(
                inner_factor,
                t -> _rewrite(
                    vectorized,
                    minus,
                    t,
                    current_sum,
                    left_factors,
                    right_factors,
                    current_sum,
                )[2],
            )
            return Expr(:block, code, :($new_var = $current_sum))
        end,
    )
end

_is_complex_expr(ex) = isa(ex, Expr) && !isexpr(ex, :ref)
function _is_decomposable_with_factors(ex)
    # `.+` and `.-` do not support being decomposed if `left_factors` or
    # `right_factors` are not empty. Otherwise, for instance
    # `I * (x .+ 1)` would be rewritten into `(I * x) .+ (I * 1)` which is
    # incorrect.
    return _is_complex_expr(ex) &&
           (isempty(ex.args) || (ex.args[1] != :.+ && ex.args[1] != :.-))
end

"""
    rewrite(x)

Rewrite the expression `x` as specified in [`@rewrite`](@ref).
Return a variable name as `Symbol` and the rewritten expression assigning the
value of the expression `x` to the variable.
"""
function rewrite(x)
    variable = gensym()
    code = rewrite_and_return(x)
    return variable, :($variable = $code)
end

"""
    rewrite_and_return(x)

Rewrite the expression `x` as specified in [`@rewrite`](@ref).
Return the rewritten expression returning the result.
"""
function rewrite_and_return(x)
    output_variable, code = _rewrite(false, false, x, nothing, [], [])
    # We need to use `let` because `rewrite(:(sum(i for i in 1:2))`
    return quote
        let
            $code
            $output_variable
        end
    end
end

function _is_comparison(ex::Expr)
    if isexpr(ex, :comparison)
        # Range comparison `_ <= _ <= _`.
        return true
    elseif isexpr(ex, :call)
        # Binary comparison `_ <= _`.
        if ex.args[1] in (:<=, :≤, :>=, :≥, :(==))
            return true
        else
            return false
        end
    else
        return false
    end
end

# `x[i = 1]` is a somewhat common user error. Catch it here.
function _has_assignment_in_ref(ex::Expr)
    if isexpr(ex, :ref)
        return any(x -> isexpr(x, :kw), ex.args)
    else
        return any(_has_assignment_in_ref, ex.args)
    end
end
_has_assignment_in_ref(other) = false

function rewrite_sum(
    vectorized::Bool,
    minus::Bool,
    terms,
    current_sum::Union{Nothing,Symbol},
    left_factors::Vector,
    right_factors::Vector,
    output::Symbol,
    block = Expr(:block),
)
    var = current_sum
    for term in terms[1:(end-1)]
        var, code = _rewrite(vectorized, minus, term, var, left_factors, right_factors)
        push!(block.args, code)
    end
    new_output, code =
        _rewrite(vectorized, minus, terms[end], var, left_factors, right_factors, output)
    @assert new_output == output
    push!(block.args, code)
    return output, block
end

function _start_summing(current_sum::Nothing, first_term::Function)
    variable = gensym()
    return Expr(:block, :($variable = MutableArithmetics.Zero()), first_term(variable))
end
function _start_summing(current_sum::Symbol, first_term::Function)
    return first_term(current_sum)
end

function _write_add_mul(
    vectorized,
    minus,
    current_sum,
    left_factors,
    inner_factors,
    right_factors,
    new_var::Symbol,
)
    if vectorized
        f = :(MutableArithmetics.broadcast!)
    else
        f = :(MutableArithmetics.operate!)
    end
    op = minus ? :(MutableArithmetics.sub_mul) : :(MutableArithmetics.add_mul)
    return _start_summing(
        current_sum,
        current_sum -> begin
            call_expr = Expr(
                :call,
                f,
                op,
                current_sum,
                left_factors...,
                inner_factors...,
                reverse(right_factors)...,
            )
            return :($new_var = $call_expr)
        end,
    )
end

"""
    _rewrite(
        vectorized::Bool,
        minus::Bool,
        inner_factor,
        current_sum::Union{Symbol, Nothing},
        left_factors::Vector,
        right_factors::Vector,
        new_var::Symbol = gensym(),
    )

Return `new_var, code` such that `code` is equivalent to
```julia
new_var = prod(left_factors) * inner_factor * prod(reverse(right_factors))
```

If `current_sum` is `nothing`, and is
```julia
new_var = current_sum op prod(left_factors) * inner_factor * prod(reverse(right_factors))
```
otherwise where `op` is `+` if `!vectorized & !minus`, `.+` if
`vectorized & !minus`, `-` if `!vectorized & minus` and `.-` if
`vectorized & minus`.
"""
function _rewrite(
    vectorized::Bool,
    minus::Bool,
    inner_factor,
    current_sum::Union{Symbol,Nothing},
    left_factors::Vector,
    right_factors::Vector,
    new_var::Symbol = gensym(),
)
    if isexpr(inner_factor, :call)
        if (
            inner_factor.args[1] == :+ ||
            inner_factor.args[1] == :- ||
            (
                current_sum === nothing &&
                isempty(left_factors) &&
                isempty(right_factors) &&
                (inner_factor.args[1] == :.+ || inner_factor.args[1] == :.-)
            )
        )
            # There are three cases here:
            #   1. scalar addition      : +(args...)
            #   2. scalar subtraction   : -(args...)
            #   3. broadcast addition or subtraction.
            # For case (3), we need to verify that current_sum, left_factors,
            # and right_factors are empty, otherwise we are unsure that the
            # elements in the containers have been copied, e.g., in
            # `I + (x .+ 1)`, the offdiagonal entries of `I + x` are the same as
            # `x` so we cannot do `broadcast!(add_mul, I + x, 1)`.
            code = Expr(:block)
            if length(inner_factor.args) == 2
                # Unary addition or subtraction.
                next_sum = current_sum
                start = 2
            else
                next_sum, new_code = _rewrite(
                    vectorized,
                    minus,
                    inner_factor.args[2],
                    current_sum,
                    left_factors,
                    right_factors,
                )
                push!(code.args, new_code)
                start = 3
            end
            if inner_factor.args[1] == :- || inner_factor.args[1] == :.-
                minus = !minus
            end
            vectorized =
                (vectorized || inner_factor.args[1] == :.+ || inner_factor.args[1] == :.-)
            return rewrite_sum(
                vectorized,
                minus,
                inner_factor.args[start:end],
                next_sum,
                left_factors,
                right_factors,
                new_var,
                code,
            )
        elseif inner_factor.args[1] == :* && !vectorized
            # A multiplication expression *(args...). We need `!vectorized`
            # otherwise `x .+ A * b` would be rewritten
            # `broadcast!(add_mul, x, A, b)`.
            # We might need to recurse on multiple arguments, e.g., (x+y)*(x+y).
            # As a special case, only recurse on one argument and don't create
            # temporary objects
            if (
                isone(mapreduce(_is_complex_expr, +, inner_factor.args)) &&
                isone(mapreduce(_is_decomposable_with_factors, +, inner_factor.args))
            )
                # `findfirst` return the index in `2:...` so we need to add `1`.
                which_idx = 1 + findfirst(2:length(inner_factor.args)) do i
                    _is_decomposable_with_factors(inner_factor.args[i])
                end
                return _rewrite(
                    vectorized,
                    minus,
                    inner_factor.args[which_idx],
                    current_sum,
                    vcat(left_factors, [esc(inner_factor.args[i]) for i = 2:(which_idx-1)]),
                    vcat(
                        right_factors,
                        [
                            esc(inner_factor.args[i]) for
                            i = length(inner_factor.args):-1:(which_idx+1)
                        ],
                    ),
                    new_var,
                )
            else
                code = Expr(:block)
                for i = 2:length(inner_factor.args)
                    arg = inner_factor.args[i]
                    if _is_complex_expr(arg)  # `arg` needs rewriting.
                        new_arg, new_arg_code = rewrite(arg)
                        push!(code.args, new_arg_code)
                        inner_factor.args[i] = new_arg
                    else
                        inner_factor.args[i] = esc(arg)
                    end
                end
                push!(
                    code.args,
                    _write_add_mul(
                        vectorized,
                        minus,
                        current_sum,
                        left_factors,
                        inner_factor.args[2:end],
                        right_factors,
                        new_var,
                    ),
                )
                return new_var, code
            end
        elseif (
            inner_factor.args[1] == :^ &&
            _is_complex_expr(inner_factor.args[2]) &&
            !vectorized
        )
            # An expression like `base ^ exponent`, where the `base` is a
            # non-trivial expression that also needs to be re-written. We need
            # `!vectorized` otherwise `A .+ (A + A)^2` would be rewritten as
            # `broadcast!(add_mul, x, AA, AA)` where `AA` is `A + A`.
            MulType = :(MA.promote_operation(
                *,
                typeof($(inner_factor.args[2])),
                typeof($(inner_factor.args[2])),
            ))
            if inner_factor.args[3] == 0
                # If the exponent is 0, rewrite
                #    new_var = base^0
                # as
                #    new_var = 1
                return _rewrite(
                    vectorized,
                    minus,
                    :(one($MulType)),
                    current_sum,
                    left_factors,
                    right_factors,
                    new_var,
                )
            elseif inner_factor.args[3] == 1
                # If the exponent is 1, rewrite
                #    new_var = base^1
                # as
                #    new_var = base
                return _rewrite(
                    vectorized,
                    minus,
                    :(convert($MulType, $(inner_factor.args[2]))),
                    current_sum,
                    left_factors,
                    right_factors,
                    new_var,
                )
            elseif inner_factor.args[3] == 2
                # If the exponent is 2, rewrite
                #    new_var = base^2
                # as
                #    new_base = base_rewrite
                #    new_var = base_rewrite * base_rewrite
                new_var_, parsed = rewrite(inner_factor.args[2])
                square_expr = _write_add_mul(
                    vectorized,
                    minus,
                    current_sum,
                    left_factors,
                    (new_var_, new_var_),
                    right_factors,
                    new_var,
                )
                return new_var, Expr(:block, parsed, square_expr)
            else
                # In the general case, rewrite
                #    new_var = base^exponent
                # as
                #    new_base = base_rewrite
                #    new_var = base_rewrite^(exponent)
                new_base, base_rewrite = rewrite(inner_factor.args[2])
                new_expr = _write_add_mul(
                    vectorized,
                    minus,
                    current_sum,
                    left_factors,
                    (Expr(:call, :^, new_base, esc(inner_factor.args[3])),),
                    right_factors,
                    new_var,
                )
                return new_var, Expr(:block, base_rewrite, new_expr)
            end
        elseif inner_factor.args[1] == :/ && !vectorized
            # Rewrite
            #   new_var = numerator / denominator
            # as
            #   new_var = numerator * (1 / denominator)
            @assert length(inner_factor.args) == 3
            return _rewrite(
                vectorized,
                minus,
                inner_factor.args[2],
                current_sum,
                left_factors,
                vcat(esc(:(1 / $(inner_factor.args[3]))), right_factors),
                new_var,
            )
        elseif (
            length(inner_factor.args) >= 2 && (
                isexpr(inner_factor.args[2], :generator) ||
                isexpr(inner_factor.args[2], :flatten)
            ) &&
            _is_sum(inner_factor.args[1])
        )
            # A generator statement.
            code = _parse_generator(
                vectorized,
                minus,
                inner_factor,
                current_sum,
                left_factors,
                right_factors,
                new_var,
            )
            return new_var, code
        end
    end
    if isexpr(inner_factor, :curly)
        error(
            "The curly syntax (sum{},prod{},norm2{}) is no longer supported. " *
            "Expression: `$inner_factor`.",
        )
    elseif isa(inner_factor, Expr) && _is_comparison(inner_factor)
        error("Unexpected comparison in expression `$inner_factor`.")
    elseif isa(inner_factor, Expr) && _has_assignment_in_ref(inner_factor)
        error("Unexpected assignment in expression `$inner_factor`.")
    end
    # None of the special cases were hit! This probably means we are vectorized.
    code = _write_add_mul(
        vectorized,
        minus,
        current_sum,
        left_factors,
        (esc(inner_factor),),
        right_factors,
        new_var,
    )
    return new_var, code
end
