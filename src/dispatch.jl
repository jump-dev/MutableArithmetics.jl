# TODO: Intercepting "externally owned" method calls by dispatching on type parameters
# (rather than outermost wrapper type) is generally bad practice, but refactoring this code
# to use a different mechanism would be a lot of work. In the future, this interception code
# would be more easily/robustly replaced by using a tool like
# https://github.com/jrevels/Cassette.jl.

abstract type AbstractMutable end

function Base.sum(a::AbstractArray{<:AbstractMutable})
    return mapreduce(identity, add!, a, init = zero(promote_operation(+, eltype(a), eltype(a))))
end

LinearAlgebra.dot(lhs::AbstractArray{<:AbstractMutable}, rhs::AbstractArray) = _dot(lhs, rhs)
LinearAlgebra.dot(lhs::AbstractArray, rhs::AbstractArray{<:AbstractMutable}) = _dot(lhs, rhs)
LinearAlgebra.dot(lhs::AbstractArray{<:AbstractMutable}, rhs::AbstractArray{<:AbstractMutable}) = _dot(lhs, rhs)

function _dot(x::AbstractArray, y::AbstractArray)
    lx = length(x)
    if lx != length(y)
        throw(DimensionMismatch("first array has length $(lx) which does not match the length of the second, $(length(y))."))
    end
    if iszero(lx)
        return LinearAlgebra.dot(zero(eltype(x)), zero(eltype(y)))
    end

    # We need a buffer to hold the intermediate multiplication.
    mul_buffer = buffer_for(add_mul, eltype(x), eltype(y))

    s = zero(promote_operation(add_mul, eltype(x), eltype(x), eltype(y)))

    for (Ix, Iy) in zip(eachindex(x), eachindex(y))
        s = @inbounds buffered_operate!(mul_buffer, add_mul, s, x[Ix], y[Iy])
    end

    return s
end

# Special-case because the the base version wants to do fill!(::Array{AbstractVariableRef}, zero(GenericAffExpr{Float64,eltype(x)}))
_one_indexed(A) = all(x -> isa(x, Base.OneTo), axes(A))
function LinearAlgebra.diagm(x::AbstractVector{<:AbstractMutable})
    @assert _one_indexed(x) # `LinearAlgebra.diagm` doesn't work for non-one-indexed arrays in general.
    ZeroType = promote_operation(zero, eltype(x))
    return LinearAlgebra.diagm(0 => copyto!(similar(x, ZeroType), x))
end

###############################################################################
# Interception of Base's matrix/vector arithmetic machinery

# Redirect calls with `eltype(ret) <: AbstractMutable` to `_mul!` to
# replace it with an implementation more efficient than `generic_matmatmul!` and
# `generic_matvecmul!` since it takes into account the mutability of the arithmetic.
# We need `args...` because SparseArrays` also gives `α` and `β` arguments.

function _mul!(output, A, B, α, β)
    # See SparseArrays/src/linalg.jl
    if !isone(β)
        if iszero(β)
            mutable_operate!(zero, output)
        else
            rmul!(output, β)
        end
    end
    return mutable_operate!(add_mul, output, A, B, α)
end
function _mul!(output, A, B, α)
    mutable_operate!(zero, output)
    return mutable_operate!(add_mul, output, A, B, α)
end
function _mul!(output, A, B)
    mutable_operate!(zero, output)
    return mutable_operate!(add_mul, output, A, B)
end

function LinearAlgebra.mul!(ret::AbstractMatrix{<:AbstractMutable},
                            A::AbstractVecOrMat, B::AbstractVecOrMat, args::Vararg{Any, N}) where N
    _mul!(ret, A, B, args...)
end
function LinearAlgebra.mul!(ret::AbstractVector{<:AbstractMutable},
                            A::AbstractVecOrMat, B::AbstractVector, args...)
    _mul!(ret, A, B, args...)
end
function LinearAlgebra.mul!(ret::AbstractVector{<:AbstractMutable},
                            A::LinearAlgebra.Transpose{<:Any,<:AbstractVecOrMat},
                            B::AbstractVector, args...)
    _mul!(ret, A, B, args...)
end
function LinearAlgebra.mul!(ret::AbstractVector{<:AbstractMutable},
                            A::LinearAlgebra.Adjoint{<:Any,<:AbstractVecOrMat},
                            B::AbstractVector, args...)
    _mul!(ret, A, B, args...)
end
function LinearAlgebra.mul!(ret::AbstractMatrix{<:AbstractMutable},
                            A::LinearAlgebra.Transpose{<:Any,<:AbstractVecOrMat},
                            B::AbstractMatrix, args...)
    _mul!(ret, A, B, args...)
end
function LinearAlgebra.mul!(ret::AbstractMatrix{<:AbstractMutable},
                            A::LinearAlgebra.Adjoint{<:Any,<:AbstractVecOrMat},
                            B::AbstractMatrix, args...)
    _mul!(ret, A, B, args...)
end
function LinearAlgebra.mul!(ret::AbstractMatrix{<:AbstractMutable},
                            A::AbstractMatrix,
                            B::LinearAlgebra.Transpose{<:Any,<:AbstractVecOrMat}, args...)
    _mul!(ret, A, B, args...)
end
function LinearAlgebra.mul!(ret::AbstractMatrix{<:AbstractMutable},
                            A::AbstractMatrix,
                            B::LinearAlgebra.Adjoint{<:Any,<:AbstractVecOrMat}, args...)
    _mul!(ret, A, B, args...)
end

# SparseArrays promotes the element types of `A` and `B` to the same type
# which always produce quadratic expressions for JuMP even if only one of them
# was affine and the other one constant. Moreover, it does not always go through
# `LinearAlgebra.mul!` which prevents us from using mutability of the arithmetic.
# For this reason we intercept the calls and redirect them to `mul`.

# A few are overwritten below but many more need to be redirected to `mul` in
# `linalg.jl`.

Base.:*(A::SparseMat{<:AbstractMutable}, x::StridedVector) = mul(A, x)
Base.:*(A::SparseMat, x::StridedVector{<:AbstractMutable}) = mul(A, x)
Base.:*(A::SparseMat{<:AbstractMutable}, x::StridedVector{<:AbstractMutable}) = mul(A, x)

Base.:*(A::SparseMat{<:AbstractMutable}, B::SparseMat{<:AbstractMutable}) = mul(A, B)
Base.:*(A::SparseMat{<:Any}, B::SparseMat{<:AbstractMutable}) = mul(A, B)
Base.:*(A::SparseMat{<:AbstractMutable}, B::SparseMat{<:Any}) = mul(A, B)

Base.:*(A::SparseMat{<:AbstractMutable}, B::LinearAlgebra.Adjoint{<:AbstractMutable, <:SparseMat}) = mul(A, B)
Base.:*(A::SparseMat{<:Any}, B::LinearAlgebra.Adjoint{<:AbstractMutable, <:SparseMat}) = mul(A, B)
Base.:*(A::SparseMat{<:AbstractMutable}, B::LinearAlgebra.Adjoint{<:Any, <:SparseMat}) = mul(A, B)

Base.:*(A::LinearAlgebra.Adjoint{<:AbstractMutable, <:SparseMat}, B::SparseMat{<:AbstractMutable}) = mul(A, B)
Base.:*(A::LinearAlgebra.Adjoint{<:Any, <:SparseMat}, B::SparseMat{<:AbstractMutable}) = mul(A, B)
Base.:*(A::LinearAlgebra.Adjoint{<:AbstractMutable, <:SparseMat}, B::SparseMat{<:Any}) = mul(A, B)

Base.:*(A::StridedMatrix{<:AbstractMutable}, B::SparseMat{<:AbstractMutable}) = mul(A, B)
Base.:*(A::StridedMatrix{<:Any}, B::SparseMat{<:AbstractMutable}) = mul(A, B)
Base.:*(A::StridedMatrix{<:AbstractMutable}, B::SparseMat{<:Any}) = mul(A, B)

Base.:*(A::SparseMat{<:AbstractMutable}, B::StridedMatrix{<:AbstractMutable}) = mul(A, B)
Base.:*(A::SparseMat{<:Any}, B::StridedMatrix{<:AbstractMutable}) = mul(A, B)
Base.:*(A::SparseMat{<:AbstractMutable}, B::StridedMatrix{<:Any}) = mul(A, B)

Base.:*(A::LinearAlgebra.Adjoint{<:AbstractMutable, <:SparseMat}, B::StridedMatrix{<:AbstractMutable}) = mul(A, B)
Base.:*(A::LinearAlgebra.Adjoint{<:Any, <:SparseMat}, B::StridedMatrix{<:AbstractMutable}) = mul(A, B)
Base.:*(A::LinearAlgebra.Adjoint{<:AbstractMutable, <:SparseMat}, B::StridedMatrix{<:Any}) = mul(A, B)

# Base doesn't define efficient fallbacks for sparse array arithmetic involving
# non-`<:Number` scalar elements, so we define some of these for `<:AbstractMutable` scalar
# elements here.

function Base.:*(A::Scaling, B::SparseMat{<:AbstractMutable})
    return SparseMat(B.m, B.n, copy(B.colptr), copy(SparseArrays.rowvals(B)), A .* SparseArrays.nonzeros(B))
end
# Fix ambiguity with Base method
function Base.:*(A::Number, B::SparseMat{<:AbstractMutable})
    return SparseMat(B.m, B.n, copy(B.colptr), copy(SparseArrays.rowvals(B)), A .* SparseArrays.nonzeros(B))
end

function Base.:*(A::SparseMat{<:AbstractMutable}, B::Scaling)
    return SparseMat(A.m, A.n, copy(A.colptr), copy(SparseArrays.rowvals(A)), SparseArrays.nonzeros(A) .* B)
end
# Fix ambiguity with Base method
function Base.:*(A::SparseMat{<:AbstractMutable}, B::Number)
    return SparseMat(A.m, A.n, copy(A.colptr), copy(SparseArrays.rowvals(A)), SparseArrays.nonzeros(A) .* B)
end

function Base.:*(A::AbstractMutable, B::SparseMat)
    return SparseMat(B.m, B.n, copy(B.colptr), copy(SparseArrays.rowvals(B)), A .* SparseArrays.nonzeros(B))
end

function Base.:*(A::SparseMat, B::AbstractMutable)
    return SparseMat(A.m, A.n, copy(A.colptr), copy(SparseArrays.rowvals(A)), SparseArrays.nonzeros(A) .* B)
end

function Base.:/(A::SparseMat{<:AbstractMutable}, B::Scaling)
    return SparseMat(A.m, A.n, copy(A.colptr), copy(SparseArrays.rowvals(A)), SparseArrays.nonzeros(A) ./ B)
end

# +(::SparseMatrixCSC) is not defined for generic types in Base.
Base.:+(A::AbstractArray{<:AbstractMutable}) = A

# Fix https://github.com/JuliaLang/julia/issues/32374 as done in
# https://github.com/JuliaLang/julia/pull/32375. This hack should
# be removed once we drop Julia v1.0.
function Base.:-(A::LinearAlgebra.Symmetric{<:AbstractMutable})
    return LinearAlgebra.Symmetric(-parent(A), LinearAlgebra.sym_uplo(A.uplo))
end
function Base.:-(A::LinearAlgebra.Hermitian{<:AbstractMutable})
    return LinearAlgebra.Hermitian(-parent(A), LinearAlgebra.sym_uplo(A.uplo))
end
