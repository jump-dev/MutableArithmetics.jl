import SparseArrays
similar_array_type(::Type{SparseArrays.SparseVector{Tv, Ti}}, ::Type{T}) where {T, Tv, Ti} = SparseArrays.SparseVector{T, Ti}
similar_array_type(::Type{SparseArrays.SparseMatrixCSC{Tv, Ti}}, ::Type{T}) where {T, Tv, Ti} = SparseArrays.SparseMatrixCSC{T, Ti}
