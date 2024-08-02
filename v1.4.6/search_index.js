var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = MutableArithmetics\nDocTestSetup = quote\n    using MutableArithmetics\nend","category":"page"},{"location":"#MutableArithmetics.jl","page":"Home","title":"MutableArithmetics.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [MutableArithmetics]","category":"page"},{"location":"#MutableArithmetics.IsMutable","page":"Home","title":"MutableArithmetics.IsMutable","text":"struct IsMutable <: MutableTrait end\n\nWhen this is returned by mutability, it means that object of the given type can always be mutated to equal the result of the operation.\n\n\n\n\n\n","category":"type"},{"location":"#MutableArithmetics.IsNotMutable","page":"Home","title":"MutableArithmetics.IsNotMutable","text":"struct IsNotMutable <: MutableTrait end\n\nWhen this is returned by mutability, it means that object of the given type cannot be mutated to equal the result of the operation.\n\n\n\n\n\n","category":"type"},{"location":"#MutableArithmetics.MutableTrait","page":"Home","title":"MutableArithmetics.MutableTrait","text":"abstract type MutableTrait end\n\nAbstract type for IsMutable and IsNotMutable that are returned by mutability.\n\n\n\n\n\n","category":"type"},{"location":"#MutableArithmetics._broadcasted_type-Tuple{Base.Broadcast.BroadcastStyle, Base.HasShape, Type}","page":"Home","title":"MutableArithmetics._broadcasted_type","text":"This method is a generic fallback for array types that are not DefaultArrayStyle. Because we can't tell the container from a generic broadcast style, we fallback to Any, which is always a valid super type (just not a helpful one).\n\nIn MutableArithmetics, _broadcasted_type appears only in promote_broadcast, which itself appears only in broadcast_mutability, and so types hitting this method will fallback to the IsNotMutable() branch, which is the expected outcome.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics._rewrite","page":"Home","title":"MutableArithmetics._rewrite","text":"_rewrite(\n    vectorized::Bool,\n    minus::Bool,\n    inner_factor,\n    current_sum::Union{Symbol, Nothing},\n    left_factors::Vector,\n    right_factors::Vector,\n    new_var::Symbol = gensym(),\n)\n\nReturn new_var, code such that code is equivalent to\n\nnew_var = prod(left_factors) * inner_factor * prod(reverse(right_factors))\n\nIf current_sum is nothing, and is\n\nnew_var = current_sum op prod(left_factors) * inner_factor * prod(reverse(right_factors))\n\notherwise where op is + if !vectorized & !minus, .+ if vectorized & !minus, - if !vectorized & minus and .- if vectorized & minus.\n\n\n\n\n\n","category":"function"},{"location":"#MutableArithmetics._rewrite_generic","page":"Home","title":"MutableArithmetics._rewrite_generic","text":"_rewrite_generic(stack::Expr, expr::T)::Tuple{Any,Bool}\n\nThis method is the heart of the rewrite logic. It converts expr into a mutable equivalent, places any intermediate calculations onto stack, and returns a tuple containing the return value–-which is either expr or a gensymed variable equivalent to expr–-and a boolean flag that indicates whether the return value can be mutated by future callers.\n\n\n\n\n\n","category":"function"},{"location":"#MutableArithmetics._rewrite_generic-Tuple{Expr, Any}","page":"Home","title":"MutableArithmetics._rewrite_generic","text":"_rewrite_generic(::Expr, x)\n\nA generic fallback. Given a type x we return it without mutation. In addition, this type should not be mutated by future callers.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics._rewrite_generic-Tuple{Expr, Expr}","page":"Home","title":"MutableArithmetics._rewrite_generic","text":"_rewrite_generic(stack::Expr, expr::Expr)\n\nThis method is the heart of the rewrite logic. It converts expr into a mutable equivalent.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics._rewrite_generic-Tuple{Expr, Number}","page":"Home","title":"MutableArithmetics._rewrite_generic","text":"_rewrite_generic(::Expr, x::Number)\n\nIf x is a Number at macro expansion time, it must be a constant literal. We return x without mutation, but we return true because other callers may mutate the value without fear. Put aother way, they don't need to wrap the value in copy_if_mutable(x) before using it as the first argument to operate!!.\n\nThis most commonly happens in situations like x^2.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics._rewrite_generic_generator","page":"Home","title":"MutableArithmetics._rewrite_generic_generator","text":"_rewrite_generic_generator(stack::Expr, op::Symbol, expr::Expr)\n\nSpecial handling for generator expressions.\n\nop is :+ and expr is a :generator or :flatten expression.\n\n\n\n\n\n","category":"function"},{"location":"#MutableArithmetics.add!!-Union{Tuple{Vararg{Any, N}}, Tuple{N}} where N","page":"Home","title":"MutableArithmetics.add!!","text":"add!!(a, b, ...)\n\nReturn the sum of a, b, ..., possibly modifying a.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.add_dot-Union{Tuple{N}, Tuple{Any, Any, Any, Vararg{Any, N}}} where N","page":"Home","title":"MutableArithmetics.add_dot","text":"add_dot(a, args...)\n\nReturn a + dot(args...).\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.add_mul","page":"Home","title":"MutableArithmetics.add_mul","text":"add_mul(a, args...)\n\nReturn a + *(args...). Note that add_mul(a, b, c) = muladd(b, c, a).\n\n\n\n\n\n","category":"function"},{"location":"#MutableArithmetics.add_mul!!-Union{Tuple{Vararg{Any, N}}, Tuple{N}} where N","page":"Home","title":"MutableArithmetics.add_mul!!","text":"add_mul!!(args...)\n\nReturn add_mul(args...), possibly modifying args[1].\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.add_mul_buf!!-Union{Tuple{N}, Tuple{Any, Vararg{Any, N}}} where N","page":"Home","title":"MutableArithmetics.add_mul_buf!!","text":"add_mul_buf!!(buffer, args...)\n\nReturn add_mul(args...), possibly modifying args[1] and buffer.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.add_mul_buf_to!!-Union{Tuple{N}, Tuple{Any, Any, Vararg{Any, N}}} where N","page":"Home","title":"MutableArithmetics.add_mul_buf_to!!","text":"add_mul_buf_to!!(buffer, output, args...)\n\nReturn add_mul(args...), possibly modifying output and buffer.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.add_mul_to!!-Union{Tuple{N}, Tuple{Any, Vararg{Any, N}}} where N","page":"Home","title":"MutableArithmetics.add_mul_to!!","text":"add_mul_to!!(output, args...)\n\nReturn add_mul(args...), possibly modifying output.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.add_to!!-Union{Tuple{N}, Tuple{Any, Vararg{Any, N}}} where N","page":"Home","title":"MutableArithmetics.add_to!!","text":"add_to!!(a, b, c)\n\nReturn the sum of b and c, possibly modifying a.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.broadcast!","page":"Home","title":"MutableArithmetics.broadcast!","text":"broadcast!(op::Function, args...)\n\nModify the value of args[1] to be equal to the value of broadcast(op, args...).\n\nThis method can only be called if mutability(args[1], op, args...) returns IsMutable.\n\n\n\n\n\n","category":"function"},{"location":"#MutableArithmetics.broadcast!!-Union{Tuple{N}, Tuple{F}, Tuple{F, Vararg{Any, N}}} where {F<:Function, N}","page":"Home","title":"MutableArithmetics.broadcast!!","text":"broadcast!!(op::Function, args...)\n\nReturns the value of broadcast(op, args...), possibly modifying args[1].\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.broadcast_mutability-Union{Tuple{N}, Tuple{Type, Any, Vararg{Type, N}}} where N","page":"Home","title":"MutableArithmetics.broadcast_mutability","text":"broadcast_mutability(T::Type, ::typeof(op), args::Type...)::MutableTrait\n\nReturn IsMutable to indicate an object of type T can be modified to be equal to broadcast(op, args...).\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.buffered_operate!","page":"Home","title":"MutableArithmetics.buffered_operate!","text":"buffered_operate!(buffer, op::Function, args...)\n\nModify the value of args[1] to be equal to the value of op(args...), possibly modifying buffer. Can only be called if mutability(args[1], op, args...) returns IsMutable.\n\n\n\n\n\n","category":"function"},{"location":"#MutableArithmetics.buffered_operate!!-Union{Tuple{N}, Tuple{F}, Tuple{Any, F, Vararg{Any, N}}} where {F<:Function, N}","page":"Home","title":"MutableArithmetics.buffered_operate!!","text":"buffered_operate!!(buffer, op::Function, args...)\n\nReturns the value of op(args...), possibly modifying buffer.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.buffered_operate_to!!-Union{Tuple{N}, Tuple{F}, Tuple{Any, Any, F, Vararg{Any, N}}} where {F<:Function, N}","page":"Home","title":"MutableArithmetics.buffered_operate_to!!","text":"buffered_operate_to!(buffer, output, op::Function, args...)\n\nReturns the value of op(args...), possibly modifying buffer and output.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.buffered_operate_to!-Union{Tuple{N}, Tuple{F}, Tuple{Any, Any, F, Vararg{Any, N}}} where {F<:Function, N}","page":"Home","title":"MutableArithmetics.buffered_operate_to!","text":"buffered_operate_to!(buffer, output, op::Function, args...)\n\nModify the value of output to be equal to the value of op(args...), possibly modifying buffer. Can only be called if mutability(output, op, args...) returns IsMutable.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.copy_if_mutable-Tuple{Any}","page":"Home","title":"MutableArithmetics.copy_if_mutable","text":"copy_if_mutable(x)\n\nReturn a copy of x that can be mutated with MultableArithmetics's API without altering x. If mutability(x) is IsNotMutable then x is returned as none of x can be mutated. Otherwise, it redirects to mutable_copy. Mutable types should not implement a method for this function but should implement a method for mutable_copy instead.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.div!!-Tuple{Any, Any}","page":"Home","title":"MutableArithmetics.div!!","text":"div!!(a, b)\n\nReturn div(a, b) possibly modifying a.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.div_to!!-Tuple{Any, Any, Any}","page":"Home","title":"MutableArithmetics.div_to!!","text":"div_to!!(output, a, b)\n\nReturn div(a, b) possibly modifying output.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.gcd!!-Union{Tuple{Vararg{Any, N}}, Tuple{N}} where N","page":"Home","title":"MutableArithmetics.gcd!!","text":"gcd!!(a, b, ...)\n\nReturn the gcd of a, b, ..., possibly modifying a.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.gcd_to!!-Union{Tuple{N}, Tuple{Any, Vararg{Any, N}}} where N","page":"Home","title":"MutableArithmetics.gcd_to!!","text":"gcd_to!!(a, b, c, ...)\n\nReturn the gcd of b, c, ..., possibly modifying a.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.isequal_canonical-Tuple{Any, Any}","page":"Home","title":"MutableArithmetics.isequal_canonical","text":"isequal_canonical(a, b)\n\nReturn whether a and b represent a same object, even if their representations differ.\n\nExamples\n\nThe terms in two MathOptInterface affine functions may not match but once the duplicates are merged, the zero terms are removed and the terms are sorted in ascending order of variable indices (i.e. their canonical representation), the equality of the representation is equivalent to the equality of the objects begin represented.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.iszero!!-Tuple{Any}","page":"Home","title":"MutableArithmetics.iszero!!","text":"iszero!!(x)\n\nReturn a Bool indicating whether x is zero, possibly modifying x.\n\nExamples\n\nIn MathOptInterface, a ScalarAffineFunction may contain duplicate terms. In Base.iszero, duplicate terms need to be merged but the function is left with duplicates as it cannot be modified. If iszero!! is called instead, the function will be canonicalized in addition for checking whether it is zero.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.lcm!!-Union{Tuple{Vararg{Any, N}}, Tuple{N}} where N","page":"Home","title":"MutableArithmetics.lcm!!","text":"lcm!!(a, b, ...)\n\nReturn the lcm of a, b, ..., possibly modifying a.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.lcm_to!!-Union{Tuple{N}, Tuple{Any, Vararg{Any, N}}} where N","page":"Home","title":"MutableArithmetics.lcm_to!!","text":"lcm_to!!(a, b, c, ...)\n\nReturn the lcm of b, c, ..., possibly modifying a.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.mul!!-Union{Tuple{Vararg{Any, N}}, Tuple{N}} where N","page":"Home","title":"MutableArithmetics.mul!!","text":"mul!!(a, b, ...)\n\nReturn the product of a, b, ..., possibly modifying a.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.mul-Union{Tuple{Vararg{Any, N}}, Tuple{N}} where N","page":"Home","title":"MutableArithmetics.mul","text":"mul(a, b, ...)\n\nShortcut for operate(*, a, b, ...), see operate.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.mul_to!!-Union{Tuple{N}, Tuple{Any, Vararg{Any, N}}} where N","page":"Home","title":"MutableArithmetics.mul_to!!","text":"mul_to!!(a, b, c, ...)\n\nReturn the product of b, c, ..., possibly modifying a.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.mutability-Union{Tuple{N}, Tuple{Type, Any, Vararg{Type, N}}} where N","page":"Home","title":"MutableArithmetics.mutability","text":"mutability(T::Type, ::typeof(op), args::Type...)::MutableTrait\n\nReturn either IsMutable to indicate an object of type T can be modified to be equal to op(args...) or IsNotMutable otherwise.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.mutable_copy","page":"Home","title":"MutableArithmetics.mutable_copy","text":"mutable_copy(x)\n\nReturn a copy of x that can be mutated with MultableArithmetics's API without altering x.\n\nExamples\n\nThe copy of a JuMP affine expression does not copy the underlying model as it cannot be modified though the MultableArithmetics's API, however, it calls copy_if_mutable on the coefficients and on the constant as they could be mutated.\n\n\n\n\n\n","category":"function"},{"location":"#MutableArithmetics.one!!-Tuple{Any}","page":"Home","title":"MutableArithmetics.one!!","text":"one!!(a)\n\nReturn one(a), possibly modifying a.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.operate","page":"Home","title":"MutableArithmetics.operate","text":"operate(op::Function, args...)\n\nReturn an object equal to the result of op(args...) that can be mutated through the MultableArithmetics API without affecting the arguments.\n\nBy default:\n\noperate(+, x) and operate(+, x) redirect to copy_if_mutable(x) so a mutable type T can return the same instance from unary operators +(x::T) = x and *(x::T) = x.\noperate(+, args...) (resp. operate(-, args...) and operate(*, args...)) redirect to +(args...) (resp. -(args...) and *(args...)) if length(args) is at least 2 (or the operation is -).\n\nNote that when op is a Base function whose implementation can be improved for mutable arguments, operate(op, args...) may have an implementation in this package relying on the MutableArithmetics API instead of redirecting to op(args...). This is the case for instance:\n\nfor Base.sum,\nfor LinearAlgebra.dot and\nfor matrix-matrix product and matrix-vector product.\n\nTherefore, for mutable arguments, there may be a performance advantage to call operate(op, args...) instead of op(args...).\n\nExample\n\nIf for a mutable type T, the following is defined:\n\nfunction Base.:*(a::Bool, x::T)\n    return a ? x : zero(T)\nend\n\nthen operate(*, a, x) will return the instance x whose modification will affect the argument of operate. Therefore, the following method need to be implemented\n\nfunction MA.operate(::typeof(*), a::Bool, x::T)\n    return a ? MA.mutable_copy(x) : zero(T)\nend\n\n\n\n\n\n","category":"function"},{"location":"#MutableArithmetics.operate!!-Union{Tuple{N}, Tuple{F}, Tuple{F, Vararg{Any, N}}} where {F<:Function, N}","page":"Home","title":"MutableArithmetics.operate!!","text":"operate!!(op::Function, args...)\n\nReturns the value of op(args...), possibly modifying args[1].\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.operate!-Union{Tuple{N}, Tuple{F}, Tuple{F, Vararg{Any, N}}} where {F<:Function, N}","page":"Home","title":"MutableArithmetics.operate!","text":"operate!(op::Function, args...)\n\nModify the value of args[1] to be equal to the value of op(args...). Can only be called if mutability(args[1], op, args...) returns IsMutable.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.operate_to!!-Union{Tuple{N}, Tuple{F}, Tuple{Any, F, Vararg{Any, N}}} where {F<:Function, N}","page":"Home","title":"MutableArithmetics.operate_to!!","text":"operate_to!(output, op::Function, args...)\n\nReturns the value of op(args...), possibly modifying output.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.operate_to!-Union{Tuple{N}, Tuple{F}, Tuple{Any, F, Vararg{Any, N}}} where {F<:Function, N}","page":"Home","title":"MutableArithmetics.operate_to!","text":"operate_to!(output, op::Function, args...)\n\nModify the value of output to be equal to the value of op(args...). Can only be called if mutability(output, op, args...) returns IsMutable.\n\nIf output === args[i] for some i, this function may throw an error. Use operate!! or operate! instead.\n\nFor example, in DynamicPolynomials, operate_to!(p, +, p, q) throws an error because otherwise, the algorithm would fill p while iterating over the terms of p and q hence it will never terminate. On the other hand operate!(+, p, q) uses a different algorithm that efficiently inserts the terms of q in the sorted list of terms of p with minimal displacement.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.promote_operation-Union{Tuple{N}, Tuple{F}, Tuple{F, Vararg{Type, N}}} where {F<:Function, N}","page":"Home","title":"MutableArithmetics.promote_operation","text":"promote_operation(op::Function, ArgsTypes::Type...)\n\nReturns the type returned to the call operate(op, args...) where the types of the arguments args are ArgsTypes.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.rewrite-Tuple{Any}","page":"Home","title":"MutableArithmetics.rewrite","text":"rewrite(expr; move_factors_into_sums::Bool = true) -> Tuple{Symbol,Expr}\n\nRewrites the expression expr to use mutable arithmetics.\n\nReturns (variable, code) comprised of a gensym'd variable equivalent to expr and the code necessary to create the variable.\n\nmove_factors_into_sums\n\nIf move_factors_into_sums = true, some terms are rewritten based on the assumption that summations produce a linear function.\n\nFor example, if move_factors_into_sums = true, then y * sum(x[i] for i in 1:2) is rewritten to:\n\nvariable = MA.Zero()\nfor i in 1:2\n    variable = MA.operate!!(MA.add_mul, result, y, x[i])\nend\n\nIf move_factors_into_sums = false, it is rewritten to:\n\nterm = MA.Zero()\nfor i in 1:2\n    term = MA.operate!!(MA.add_mul, term, x[i])\nend\nvariable = MA.operate!!(*, y, term)\n\nThe latter can produce an additional allocation if there is an efficient fallback for add_mul and not for *(y, term).\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.rewrite_and_return-Tuple{Any}","page":"Home","title":"MutableArithmetics.rewrite_and_return","text":"rewrite_and_return(expr; move_factors_into_sums::Bool = true) -> Expr\n\nRewrite the expression expr using mutable arithmetics and return an expression in which the last statement is equivalent to expr.\n\nSee rewrite for an explanation of the keyword argument.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.rewrite_generator-Tuple{Any, Any}","page":"Home","title":"MutableArithmetics.rewrite_generator","text":"rewrite_generator(expr::Expr, inner::Function)\n\nRewrites the generator statements expr and returns a properly nested for loop with nested filters as specified.\n\nExamples\n\njulia> using MutableArithmetics\n\njulia> MutableArithmetics.rewrite_generator(:(i for i in 1:2 if isodd(i)), i -> :($i + 1))\n:(for $(Expr(:escape, :(i = 1:2)))\n      if $(Expr(:escape, :(isodd(i))))\n          i + 1\n      end\n  end)\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.sub!!-Tuple{Any, Any}","page":"Home","title":"MutableArithmetics.sub!!","text":"sub!!(a, b)\n\nReturn a - b, possibly modifying a.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.sub_mul","page":"Home","title":"MutableArithmetics.sub_mul","text":"sub_mul(a, args...)\n\nReturn a - *(args...).\n\n\n\n\n\n","category":"function"},{"location":"#MutableArithmetics.sub_mul!!-Union{Tuple{Vararg{Any, N}}, Tuple{N}} where N","page":"Home","title":"MutableArithmetics.sub_mul!!","text":"sub_mul!!(args...)\n\nReturn sub_mul(args...), possibly modifying args[1].\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.sub_mul_buf!!-Union{Tuple{N}, Tuple{Any, Vararg{Any, N}}} where N","page":"Home","title":"MutableArithmetics.sub_mul_buf!!","text":"sub_mul_buf!!(buffer, args...)\n\nReturn sub_mul(args...), possibly modifying args[1] and buffer.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.sub_mul_buf_to!!-Union{Tuple{N}, Tuple{Any, Any, Vararg{Any, N}}} where N","page":"Home","title":"MutableArithmetics.sub_mul_buf_to!!","text":"sub_mul_buf_to!!(buffer, output, args...)\n\nReturn sub_mul(args...), possibly modifying output and buffer.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.sub_mul_to!!-Union{Tuple{N}, Tuple{Any, Vararg{Any, N}}} where N","page":"Home","title":"MutableArithmetics.sub_mul_to!!","text":"sub_mul_to!!(output, args...)\n\nReturn sub_mul(args...), possibly modifying output.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.sub_to!!-Tuple{Any, Any, Any}","page":"Home","title":"MutableArithmetics.sub_to!!","text":"sub_to!!(output, a, b)\n\nReturn the a - b, possibly modifying output.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.zero!!-Tuple{Any}","page":"Home","title":"MutableArithmetics.zero!!","text":"zero!!(a)\n\nReturn zero(a), possibly modifying a.\n\n\n\n\n\n","category":"method"},{"location":"#MutableArithmetics.@rewrite-Tuple","page":"Home","title":"MutableArithmetics.@rewrite","text":"@rewrite(expr, move_factors_into_sums = true)\n\nReturn the value of expr, exploiting the mutability of the temporary expressions created for the computation of the result.\n\nIf you have an Expr as input, use rewrite_and_return instead.\n\nSee rewrite for an explanation of the keyword argument.\n\ninfo: Info\nPassing move_factors_into_sums after a ; is not supported. Use a , instead.\n\n\n\n\n\n","category":"macro"}]
}
