function broadcasted_type(
    ::Broadcast.DefaultArrayStyle{N},
    ::Base.HasShape{N},
    ::Type{Eltype},
) where {N,Eltype}
    return Array{Eltype,N}
end
function broadcasted_type(
    ::Broadcast.DefaultArrayStyle{N},
    ::Base.HasShape{N},
    ::Type{Bool},
) where {N}
    return BitArray{N}
end

# Same as `Base.Broadcast.combine_styles` but with types as argument.
combine_styles() = Broadcast.DefaultArrayStyle{0}()
combine_styles(c::Type) = Broadcast.result_style(Broadcast.BroadcastStyle(c))
combine_styles(c1::Type, c2::Type) =
    Broadcast.result_style(combine_styles(c1), combine_styles(c2))
@inline combine_styles(c1::Type, c2::Type, cs::Vararg{Type,N}) where {N} =
    Broadcast.result_style(combine_styles(c1), combine_styles(c2, cs...))

combine_shapes(s) = s
combine_2_shapes(s1::Base.HasShape{N}, s2::Base.HasShape{M}) where {N,M} =
    Base.HasShape{max(N, M)}()
combine_shapes(s1, s2, args::Vararg{Any,N}) where {N} =
    combine_shapes(combine_2_shapes(s1, s2), args...)
_shape(T) = Base.HasShape{ndims(T)}()
combine_sizes(args::Vararg{Any,N}) where {N} = combine_shapes(_shape.(args)...)

function promote_broadcast(op::Function, args::Vararg{Any,N}) where {N}
    # FIXME we could use `promote_operation` instead as
    # `combine_eltypes` uses `return_type` hence it may return a non-concrete type
    # and we do not handle that case.
    T = Base.Broadcast.combine_eltypes(op, args)
    return broadcasted_type(combine_styles(args...), combine_sizes(args...), T)
end

"""
    broadcast_mutability(T::Type, ::typeof(op), args::Type...)::MutableTrait

Return `IsMutable` to indicate an object of type `T` can be modified to be
equal to `broadcast(op, args...)`.
"""
function broadcast_mutability(T::Type, op, args::Vararg{Type,N}) where {N}
    if mutability(T) isa IsMutable && promote_broadcast(op, args...) == T
        return IsMutable()
    else
        return NotMutable()
    end
end
broadcast_mutability(x, op, args::Vararg{Any,N}) where {N} =
    broadcast_mutability(typeof(x), op, typeof.(args)...)
broadcast_mutability(::Type) = NotMutable()

"""
    mutable_broadcast!(op::Function, args...)

Modify the value of `args[1]` to be equal to the value of `broadcast(op, args...)`. Can
only be called if `mutability(args[1], op, args...)` returns `true`.
"""
function mutable_broadcast! end

function mutable_broadcasted(broadcasted::Broadcast.Broadcasted{S}) where {S}
    function f(args::Vararg{Any,N}) where {N}
        return operate!(broadcasted.f, args...)
    end
    return Broadcast.Broadcasted{S}(f, broadcasted.args, broadcasted.axes)
end

# If A is `Symmetric`, we cannot do that as we might modify the same entry twice.
# See https://github.com/jump-dev/JuMP.jl/issues/2102
function mutable_broadcast!(op::Function, A::Array, args::Vararg{Any,N}) where {N}
    bc = Broadcast.broadcasted(op, A, args...)
    instantiated = Broadcast.instantiate(bc)
    return copyto!(A, mutable_broadcasted(instantiated))
end

"""
    broadcast!(op::Function, args...)

Returns the value of `broadcast(op, args...)`, possibly modifying `args[1]`.
"""
function broadcast!(op::Function, args::Vararg{Any,N}) where {N}
    # TODO use traits instead
    if any(x -> x isa LinearAlgebra.UniformScaling, args)
        return broadcast_with_uniform_scaling!(op, args...)
    else
        return broadcast_fallback!(broadcast_mutability(args[1], op, args...), op, args...)
    end
end
function broadcast_with_uniform_scaling!(op::Function, args::Vararg{Any,N}) where {N}
    return op(args...)
end

function broadcast_fallback!(::NotMutable, op::Function, args::Vararg{Any,N}) where {N}
    return broadcast(op, args...)
end
function broadcast_fallback!(::IsMutable, op::Function, args::Vararg{Any,N}) where {N}
    return mutable_broadcast!(op, args...)
end
