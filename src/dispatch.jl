abstract type AbstractMutable end

# Special-case because the the base version wants to do fill!(::Array{AbstractVariableRef}, zero(GenericAffExpr{Float64,eltype(x)}))
_one_indexed(A) = all(x -> isa(x, Base.OneTo), axes(A))
function LinearAlgebra.diagm(x::AbstractVector{<:AbstractMutable})
    @assert _one_indexed(x) # Base.diagm doesn't work for non-one-indexed arrays in general.
    ZeroType = promote_operation(zero, eltype(x))
    return diagm(0 => copyto!(similar(x, ZeroType), x))
end
