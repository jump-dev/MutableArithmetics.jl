import SparseArrays

const SparseMat = SparseArrays.SparseMatrixCSC

function mutable_operate!(::typeof(zero), A::SparseMat)
    for i in eachindex(A.colptr)
        A.colptr[i] = one(A.colptr[i])
    end
    empty!(A.rowval)
    empty!(A.nzval)
    return A
end

function promote_operation(op::Union{typeof(+), typeof(-)}, ::Type{<:SparseArrays.AbstractSparseArray{Tv, Ti, N}}, ::Type{Array{T, N}}) where {Tv, Ti, T, N}
    return Array{promote_operation(op, Tv, T), N}
end
function promote_operation(op::Union{typeof(+), typeof(-)}, ::Type{Array{T, N}}, ::Type{<:SparseArrays.AbstractSparseArray{Tv, Ti, N}}) where {Tv, Ti, T, N}
    return Array{promote_operation(op, Tv, T), N}
end
function _mutable_operate!(op::Union{typeof(+), typeof(-)}, A::Matrix, B::SparseMat, left_factors::Tuple, right_factors::Tuple)
    B_nonzeros = SparseArrays.nonzeros(B)
    B_rowvals = SparseArrays.rowvals(B)
    for col in 1:size(B, 2)
        for k ∈ SparseArrays.nzrange(B, col)
            row = B_rowvals[k]
            A[row, col] = operate!(mul_rhs(op), A[row, col], left_factors..., B_nonzeros[k], right_factors...)
        end
    end
    return A
end

similar_array_type(::Type{SparseArrays.SparseVector{Tv, Ti}}, ::Type{T}) where {T, Tv, Ti} = SparseArrays.SparseVector{T, Ti}
similar_array_type(::Type{SparseMat{Tv, Ti}}, ::Type{T}) where {T, Tv, Ti} = SparseMat{T, Ti}

const TransposeOrAdjoint{T, MT} = Union{LinearAlgebra.Transpose{T, MT}, LinearAlgebra.Adjoint{T, MT}}
_mirror_transpose_or_adjoint(x, ::LinearAlgebra.Transpose) = LinearAlgebra.transpose(x)
_mirror_transpose_or_adjoint(x, ::LinearAlgebra.Adjoint) = LinearAlgebra.adjoint(x)

# `SparseArrays/src/linalg.jl` sometimes create a sparse matrix to contain the result.
# For instance with `Matrix * Adjoint{SparseMatrixCSC}` and then uses `generic_matmatmul!`
# which looks quite inefficient as it does not exploit the sparsity of the result matrix and the rhs.
# The approach used here should be more efficient as we redirect to a method that exploits the sparsity of the rhs and `copyto!` should be faster to write the result matrix.
function mutable_operate!(::typeof(add_mul), output::SparseMat{T},
                          A::AbstractMatrix, B::AbstractMatrix) where T
    C = Matrix{T}(undef, size(output)...)
    mutable_operate!(zero, C)
    mutable_operate!(add_mul, C, A, B)
    copyto!(output, C)
    return output
end

function mutable_operate!(::typeof(add_mul), ret::VecOrMat{T},
                          adjA::TransposeOrAdjoint{<:Any, <:SparseMat},
                          B::AbstractVecOrMat,
                          α::Vararg{Union{T, Scaling}, N}) where {T, N}
    _dim_check(ret, adjA, B)
    λ = scaling.(α)
    A = parent(adjA)
    A_nonzeros = SparseArrays.nonzeros(A)
    A_rowvals = SparseArrays.rowvals(A)
    for k ∈ 1:size(ret, 2)
        for col ∈ 1:A.n
            cur = ret[col, k]
            # TODO replace by nzrange
            for j ∈ A.colptr[col]:(A.colptr[col + 1] - 1)
                A_val = _mirror_transpose_or_adjoint(A_nonzeros[j], adjA)
                mutable_operate!(add_mul, cur, A_val, B[A_rowvals[j], k], λ...)
            end
        end
    end
    return ret
end
function mutable_operate!(::typeof(add_mul), ret::VecOrMat{T},
                          A::SparseMat, B::AbstractVecOrMat,
                          α::Vararg{Union{T, Scaling}, N}) where {T, N}
    _dim_check(ret, A, B)
    λ = scaling.(α)
    A_nonzeros = SparseArrays.nonzeros(A)
    A_rowvals = SparseArrays.rowvals(A)
    for col ∈ 1:size(A, 2)
        for k ∈ 1:size(ret, 2)
            αxj = *(B[col,k], λ...)
            for j ∈ SparseArrays.nzrange(A, col)
                mutable_operate!(add_mul, ret[A_rowvals[j], k], A_nonzeros[j], αxj)
            end
        end
    end
    return ret
end
function mutable_operate!(::typeof(add_mul), ret::Matrix{T},
                          A::AbstractMatrix, B::SparseMat,
                          α::Vararg{Union{T, Scaling}, N}) where {T, N}
    _dim_check(ret, A, B)
    λ = scaling.(α)
    rowval = SparseArrays.rowvals(B)
    B_nonzeros = SparseArrays.nonzeros(B)
    for multivec_row in 1:size(A, 1)
        for col ∈ 1:size(B, 2)
            cur = ret[multivec_row, col]
            for k ∈ SparseArrays.nzrange(B, col)
                mutable_operate!(add_mul, cur, A[multivec_row, rowval[k]], B_nonzeros[k], λ...)
            end
        end
    end
    return ret
end
function mutable_operate!(::typeof(add_mul), ret::Matrix{T},
                          A::SparseMat, B::SparseMat,
                          α::Vararg{Union{T, Scaling}, N}) where {T, N}
    # Resolve ambiguity (detected on Julia v1.3) with two methods above.
    # TODO adapt implementation of `SparseArray.spmatmul`
    mutable_operate!(add_mul, ret, Matrix{Base.promote_op(zero, eltype(A))}(A), B, α...)
end
function mutable_operate!(::typeof(add_mul), ret::Matrix{T},
                          A::AbstractMatrix,
                          adjB::TransposeOrAdjoint{<:Any, <:SparseMat},
                          α::Vararg{Union{T, Scaling}, N}) where {T, N}
    _dim_check(ret, A, adjB)
    λ = scaling.(α)
    B = parent(adjB)
    B_rowvals = SparseArrays.rowvals(B)
    B_nonzeros = SparseArrays.nonzeros(B)
    for B_col ∈ 1:size(B, 2), k ∈ SparseArrays.nzrange(B, B_col)
        B_row = B_rowvals[k]
        B_val = _mirror_transpose_or_adjoint(B_nonzeros[k], adjB)
        αB_val = *(B_val, λ...)
        for A_row in 1:size(A, 1)
            mutable_operate!(add_mul, ret[A_row, B_row], A[A_row, B_col], αB_val)
        end
    end
    return ret
end
function mutable_operate!(::typeof(add_mul), ret::Matrix{T},
                          A::SparseMat, B::TransposeOrAdjoint{<:Any, <:SparseMat},
                          α::Vararg{Union{T, Scaling}, N}) where {T, N}
    # Resolve ambiguity (detected on Julia v1.3) with two methods above.
    # TODO adapt implementation of `SparseArray.spmatmul`
    mutable_operate!(add_mul, ret, Matrix{Base.promote_op(zero, eltype(A))}(A), B, α...)
end
function mutable_operate!(::typeof(add_mul), ret::Matrix{T},
                          A::TransposeOrAdjoint{<:Any, <:SparseMat}, B::SparseMat,
                          α::Vararg{Union{T, Scaling}, N}) where {T, N}
    # Resolve ambiguity (detected on Julia v1.3) with two methods above.
    # TODO adapt implementation of `SparseArray.spmatmul`
    mutable_operate!(add_mul, ret, Matrix{Base.promote_op(zero, eltype(A))}(A), B, α...)
end

# TODO
#function _densify_with_jump_eltype(x::SparseMat{V}) where {V <: AbstractVariableRef}
#    return convert(Matrix{GenericAffExpr{Float64, V}}, x)
#end
#_densify_with_jump_eltype(x::AbstractMatrix) = convert(Matrix, x)
#
## TODO: Implement sparse * sparse code as in base/sparse/linalg.jl (spmatmul).
#function _mul!(ret::AbstractMatrix{<:AbstractMutable},
#               A::SparseMat,
#               B::SparseMat)
#    return mul!(ret, A, _densify_with_jump_eltype(B))
#end
#
## TODO: Implement sparse * sparse code as in base/sparse/linalg.jl (spmatmul).
#function _mul!(ret::AbstractMatrix{<:AbstractMutable},
#               A::TransposeOrAdjoint{<:Any, <:SparseMat},
#               B::SparseMat)
#    return mul!(ret, A, _densify_with_jump_eltype(B))
#end
#
## TODO: Implement sparse * sparse code as in base/sparse/linalg.jl (spmatmul).
#function _mul!(ret::AbstractMatrix{<:AbstractMutable},
#               A::SparseMat,
#               B::TransposeOrAdjoint{<:Any, <:SparseMat})
#    return mul!(ret, _densify_with_jump_eltype(A), B)
#end
