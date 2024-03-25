mutability(::Type) = IsNotMutable()
function mutability(
    T::Type,
    op::Function,
    args::Type...,
)
    if mutability(T) isa IsMutable &&
        T == promote_operation(op, args...)
        return IsMutable()
    else
        return IsNotMutable()
    end
end
