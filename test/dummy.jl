# It does not support operation with floats on purpose to test that
# MutableArithmetics does not convert to float when it shouldn't.
struct DummyBigInt <: MA.AbstractMutable
    data::BigInt
end
DummyBigInt(J::UniformScaling) = DummyBigInt(J.λ)

# Broadcast
Base.ndims(::Type{DummyBigInt}) = 0
Base.broadcastable(x::DummyBigInt) = Ref(x)

Base.promote_rule(::Type{DummyBigInt}, ::Type{<:Union{Integer, UniformScaling{<:Integer}}}) = DummyBigInt
# `copy` on BigInt returns the same instance anyway
Base.copy(x::DummyBigInt) = x
MA.mutable_copy(x::DummyBigInt) = DummyBigInt(MA.mutable_copy(x.data))
LinearAlgebra.symmetric_type(::Type{DummyBigInt}) = DummyBigInt
LinearAlgebra.symmetric(x::DummyBigInt, ::Symbol) = x
LinearAlgebra.dot(x::DummyBigInt, y::DummyBigInt) = x * y
LinearAlgebra.dot(x::DummyBigInt, y::Union{Integer, UniformScaling{<:Integer}}) = x * y
LinearAlgebra.dot(x::Union{Integer, UniformScaling{<:Integer}}, y::DummyBigInt) = x * y
LinearAlgebra.transpose(x::DummyBigInt) = x
LinearAlgebra.adjoint(x::DummyBigInt) = x
MA.mutability(::Type{DummyBigInt}) = MA.IsMutable()
MA.promote_operation(::typeof(zero), ::Type{DummyBigInt}) = DummyBigInt
MA.promote_operation(::typeof(one), ::Type{DummyBigInt}) = DummyBigInt
MA.promote_operation(::typeof(+), ::Type{DummyBigInt}, ::Type{DummyBigInt}) = DummyBigInt
_data(x) = x
_data(x::DummyBigInt) = x.data
MA.scaling(x::DummyBigInt) = x

MA.mutable_operate_to!(x::DummyBigInt, op::Function, args...) = DummyBigInt(MA.mutable_operate_to!(x.data, op, _data.(args)...))
function MA.mutable_operate!(op::Function, x::DummyBigInt, args::Vararg{Any, N}) where N
    MA.mutable_operate_to!(x, op, x, args...)
end

function MA.mutable_operate!(op::Union{typeof(zero), typeof(one)}, x::DummyBigInt)
    return DummyBigInt(MA.mutable_operate!(op, x.data))
end

MA.promote_operation(::typeof(*), ::Type{DummyBigInt}, ::Type{DummyBigInt}) = DummyBigInt
Base.convert(::Type{DummyBigInt}, x::Int) = DummyBigInt(x)
Base.:(==)(x::DummyBigInt, y::DummyBigInt) = x.data == y.data
Base.zero(::Union{DummyBigInt, Type{DummyBigInt}}) = DummyBigInt(zero(BigInt))
Base.one(::Union{DummyBigInt, Type{DummyBigInt}}) = DummyBigInt(one(BigInt))
Base.:+(x::DummyBigInt) = DummyBigInt(+x.data)
Base.:+(x::DummyBigInt, y::DummyBigInt) = DummyBigInt(x.data + y.data)
Base.:+(x::DummyBigInt, y::Union{Integer, UniformScaling{<:Integer}}) = DummyBigInt(x.data + y)
Base.:+(x::Union{Integer, UniformScaling{<:Integer}}, y::DummyBigInt) = DummyBigInt(x + y.data)
Base.:-(x::DummyBigInt) = DummyBigInt(-x.data)
Base.:-(x::DummyBigInt, y::DummyBigInt) = DummyBigInt(x.data - y.data)
Base.:-(x::Union{Integer, UniformScaling{<:Integer}}, y::DummyBigInt) = DummyBigInt(x - y.data)
Base.:-(x::DummyBigInt, y::Union{Integer, UniformScaling{<:Integer}}) = DummyBigInt(x.data - y)
Base.:*(x::DummyBigInt) = x
Base.:*(x::DummyBigInt, y::DummyBigInt) = DummyBigInt(x.data * y.data)
Base.:*(x::Union{Integer, UniformScaling{<:Integer}}, y::DummyBigInt) = DummyBigInt(x * y.data)
Base.:*(x::DummyBigInt, y::Union{Integer, UniformScaling{<:Integer}}) = DummyBigInt(x.data * y)
Base.:^(x::DummyBigInt, y::Union{Integer, UniformScaling{<:Integer}}) = DummyBigInt(x.data ^ y)
