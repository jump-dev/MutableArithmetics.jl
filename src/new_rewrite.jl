# Copyright (c) 2019 MutableArithmetics.jl contributors
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v.2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.

module MutableArithmetics2

import ..MutableArithmetics

const MA = MutableArithmetics

"""
    @rewrite(expr)

Rewrites the expression `expr` to use mutable arithmetics.

For a non-macro version, see [`rewrite_and_return`](@ref).
"""
macro rewrite(expr)
    return rewrite_and_return(expr)
end

"""
    rewrite_and_return(expr) -> Expr

Rewrites the expression `expr` to use mutable arithmetics.
"""
function rewrite_and_return(expr)
    stack = quote end
    root, _ = _rewrite(stack, expr)
    return quote
        let
            $stack
            $root
        end
    end
end

"""
    rewrite(expr) -> Tuple{Symbol,Expr}

Rewrites the expression `expr` to use mutable arithmetics. Returns
`(variable, code)` comprised of a `gensym`'d variable equivalent to `expr` and
the code necessary to create the variable.
"""
function rewrite(expr)
    variable = gensym()
    code = rewrite_and_return(expr)
    return variable, :($variable = $code)
end

"""
    _rewrite(stack::Expr, expr::T)::Tuple{Any,Bool}

This method is the heart of the rewrite logic. It converts `expr` into a mutable
equivalent, places any intermediate calculations onto `stack`, and returns a
tuple containing the return value---which is either `expr` or a `gensym`ed
variable equivalent to `expr`---and a boolean flag that indicates whether the
return value can be mutated by future callers.
"""
function _rewrite end
"""
    _rewrite(::Expr, x)

A generic fallback. Given a type `x` we return it without mutation. In addition,
this type should not be mutated by future callers.
"""
_rewrite(::Expr, x) = esc(x), false

"""
    _rewrite(::Expr, x::Number)

If `x` is a `Number` at macro expansion time, it _must_ be a constant literal.
We return `x` without mutation, but we return `true` because other callers may
mutate the value without fear. Put aother way, they don't need to wrap the value
in `copy_if_mutable(x)` before using it as the first argument to `operate!!`.

This most commonly happens in situations like `x^2`.
"""
_rewrite(::Expr, x::Number) = x, true

"""
    _rewrite(stack::Expr, expr::Expr)

This method is the heart of the rewrite logic. It converts `expr` into a mutable
equivalent.
"""
function _rewrite(stack::Expr, expr::Expr)
    if !Meta.isexpr(expr, :call)
        # In situations like `x[i]`, we do not attempt to rewrite. Return `expr`
        # and don't let future callers mutate.
        return esc(expr), false
    elseif Meta.isexpr(expr, :call, 1)
        # A zero-argument function
        return esc(expr), false
    elseif Meta.isexpr(expr, :call, 2) && (
        Meta.isexpr(expr.args[2], :generator) ||
        Meta.isexpr(expr.args[2], :flatten)
    )
        # This is a generator expression like `sum(i for i in args)`. Generators
        # come in two forms: `sum(i for i=I, j=J)` or `sum(i for i=I for j=J)`.
        # The latter is a `:flatten` expression and needs additional handling,
        # but we delay this complexity for _rewrite_generator.
        if expr.args[1] in (:sum, :Σ, :∑)
            # Summations use :+ as the reduction operator.
            return _rewrite_generator(stack, :+, expr.args[2])
        end
        # We don't know what this is. Return the expression and don't let
        # future callers mutate.
        return esc(expr), false
    end
    # At this point, we have an expression like `op(args...)`. We can either
    # choose to convert the operation to it's mutable equivalent, or return the
    # non-mutating operation.
    if expr.args[1] == :+
        # +(args...) => add_mul(add_mul(arg1, arg2), arg3)
        @assert length(expr.args) > 1
        if length(expr.args) == 2  # +(arg)
            return _rewrite(stack, expr.args[2])
        end
        return _rewrite_to_nested_op(stack, expr, MA.add_mul)
    elseif expr.args[1] == :-
        # -(args...) => sub_mul(sub_ul(arg1, arg2), arg3)
        @assert length(expr.args) > 1
        if length(expr.args) == 2  # -(arg)
            return _rewrite(stack, Expr(:call, :*, -1, expr.args[2]))
        end
        return _rewrite_to_nested_op(stack, expr, MA.sub_mul)
    elseif expr.args[1] == :*
        # *(args...) => *(*(arg1, arg2), arg3)
        @assert length(expr.args) > 2
        return _rewrite_to_nested_op(stack, expr, *)
    else
        # Use the non-mutating call.
        result = Expr(:call, esc(expr.args[1]))
        for i in 2:length(expr.args)
            arg, _ = _rewrite(stack, expr.args[i])
            push!(result.args, arg)
        end
        root = gensym()
        push!(stack.args, Expr(:(=), root, result))
        # This value isn't safe to mutate, because it might be a reference to
        # another object.
        return root, false
    end
end

function _rewrite_to_nested_op(stack, expr, op)
    root, is_mutable = _rewrite(stack, expr.args[2])
    if !is_mutable
        arg = Expr(:call, MA.copy_if_mutable, root)
        root = gensym()
        push!(stack.args, Expr(:(=), root, arg))
    end
    for i in 3:length(expr.args)
        arg, _ = _rewrite(stack, expr.args[i])
        rhs = Expr(:call, MA.operate!!, op, root, arg)
        root = gensym()
        push!(stack.args, Expr(:(=), root, rhs))
    end
    return root, true
end

_is_call(expr, op) = Meta.isexpr(expr, :call) && expr.args[1] == op

"""
    _rewrite_generator(stack::Expr, op::Symbol, expr::Expr)

Special handling for generator expressions.

`op` is one of `:+` or `:*`, and `expr` is a `:generator` or `:flatten`
expression.
"""
function _rewrite_generator(stack::Expr, op::Symbol, expr::Expr, root = nothing)
    is_flatten = Meta.isexpr(expr, :flatten)
    if is_flatten
        expr = expr.args[1]
    end
    # The value we're going to mutate. Start it off at `Zero`.
    if root === nothing
        root = gensym()
        push!(stack.args, Expr(:(=), root, MA.Zero()))
    end
    # We need a new stack to go inside our for-loops since we want to
    # recursively rewrite the inner part as well.
    new_stack = quote end
    if _is_call(expr.args[1], op)
        # Optimization time! Instead of operate!!(op, root, op(args...)),
        # rewrite as operate!!(op, root, arg) for arg in args
        for arg in expr.args[1].args[2:end]
            value, _ = _rewrite(new_stack, arg)
            push!(
                new_stack.args,
                Expr(:(=), root, Expr(:call, MA.operate!!, op, root, value)),
            )
        end
    elseif op == :+ && _is_call(expr.args[1], :*)
        # Optimization time! Instead of operate!!(+, root, *(args...)), rewrite
        # this as operate!!(add_mul, root, args...)
        rhs = Expr(:call, MA.operate!!, MA.add_mul, root)
        for arg in expr.args[1].args[2:end]
            value, _ = _rewrite(new_stack, arg)
            push!(rhs.args, value)
        end
        push!(new_stack.args, Expr(:(=), root, rhs))
    elseif is_flatten
        # The first argument is itself a generator
        _rewrite_generator(new_stack, op, expr.args[1], root)
    else
        # expr.args[1] is the inner part of the loop. Rewrite it. We don't care
        # if it is mutable because we need a new value every iteration.
        inner, _ = _rewrite(new_stack, expr.args[1])
        # Now build up the summation or product part of the inner loop. It's
        # always safe to mutate because we're going to start with `root=Zero()`.
        push!(
            new_stack.args,
            Expr(:(=), root, Expr(:call, MA.operate!!, op, root, inner)),
        )
    end
    # This is a little complicated: walk back out of the generator statements
    # wrapping each level in a for loop and the over-writing the `new_stack`
    # variable.
    #
    # !!! warning
    #     The Julia syntax sum(i for i in 1:2, j in 1:i) is incorrect, but we
    #     handle it anyway! Because the user will write dependencies from left
    #     to right, we need to wrap from right to left.
    for i in length(expr.args):-1:2
        new_stack = _iterable_condition(new_stack, expr.args[i])
    end
    # Finally, push our new_stack onto the old `stack`...
    push!(stack.args, new_stack)
    # and return the `root`. We can mutate this in future because it started off
    # as `Zero`.
    return root, true
end

function _iterable_condition(new_stack, expr)
    if !Meta.isexpr(expr, :filter)
        return Expr(:for, esc(expr), new_stack)
    end
    body = quote
        if $(esc(expr.args[1]))
            $new_stack
        end
    end
    # A filter might be over multiple index sets
    for i in length(expr.args):-1:2
        body = Expr(:for, esc(expr.args[i]), body)
    end
    return body
end

end  # module
