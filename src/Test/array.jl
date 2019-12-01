function matrix_vector_division_test(x)
    if size(x) == (3, 3)
        A = [2 1 0
             1 2 1
             0 1 2]
        @test_rewrite A * x / 2
    end
end

function _xAx_test(x::AbstractVector, A::AbstractMatrix)
    @test_rewrite(x' * A)
    # Complex expression
    @test_rewrite(x' * ones(Int, size(A)...))
    @test_rewrite(x' * A * x)
    # Complex expression
    @test_rewrite(x' * ones(Int, size(A)...) * x)
    @test_rewrite reshape(x, (1, length(x))) * A * x .- 1
    @test_rewrite x' * A * x .- 1
    @test_rewrite x' * A * x - 1
end
function _xABx_test(x::AbstractVector, A::AbstractMatrix, B::AbstractMatrix)
    @test_rewrite (x'A)' + 2B * x
    @test_rewrite (x'A)' + 2B * x .- 1
    @test_rewrite (x'A)' + 2B * x .- [length(x):-1:1;]
    @test_rewrite (x'A)' + 2B * x - [length(x):-1:1;]
end

function _matrix_vector_test(x::AbstractVector, A::AbstractMatrix)
    _xAx_test(x, A)
    B = sparse(A)
    _xAx_test(x, B)

    @test MA.isequal_canonical(A * x, B * x)
    @test MA.isequal_canonical(A * x, MA.@rewrite(B * x))
    @test MA.isequal_canonical(MA.@rewrite(A * x), MA.@rewrite(B * x))
    @test MA.isequal_canonical(x' * A, x' * B)
    @test MA.isequal_canonical(x' * A, MA.@rewrite(x' * B))
    @test MA.isequal_canonical(MA.@rewrite(x' * A), MA.@rewrite(x' * B))
    @test MA.isequal_canonical(x'A*x, x'*B*x)
    @test MA.isequal_canonical(x'*A*x, MA.@rewrite(x'*B*x))
    @test MA.isequal_canonical(MA.@rewrite(x'*A*x), MA.@rewrite(x'*B*x))

    _xABx_test(x, A, A)
    _xABx_test(x, A, B)
    _xABx_test(x, B, A)
    _xABx_test(x, B, B)
end

function matrix_vector_test(x)
    add_test(x, x)
    if size(x) != (3,)
        return
    end
    A = [2 1 0
         1 2 1
         0 1 2]
    @test MA.isequal_canonical(-x, [-x[1], -x[2], -x[3]])
    xAx = 2x[1]*x[1] + 2x[1]*x[2] + 2x[2]*x[2] + 2x[2]*x[3] + 2x[3]*x[3]
    @test MA.isequal_canonical(x' * A * x, xAx)
    y = A * x
    @test MA.isequal_canonical(x' * y, xAx)
    @test MA.isequal_canonical(y, [
        2x[1] +  x[2]
        2x[2] +  x[1] + x[3]
         x[2] + 2x[3]])
    @test MA.isequal_canonical(-y, [
        -2x[1] -  x[2]
         -x[1] - 2x[2] -  x[3]
                 -x[2] - 2x[3]])
    @test MA.isequal_canonical(x' * A, [
        2x[1] +  x[2];
         x[1] + 2x[2] +  x[3];
                 x[2] + 2x[3]]')

    @test MA.isequal_canonical(y .+ 1, [
        2x[1] +  x[2]         + 1
         x[1] + 2x[2] +  x[3] + 1
                 x[2] + 2x[3] + 1])
    @test MA.isequal_canonical(y .- 1, [
        2x[1] +  x[2]         - 1
         x[1] + 2x[2] +  x[3] - 1
                 x[2] + 2x[3] - 1])
    @test MA.isequal_canonical(y .+ 2ones(Int, 3), [
        2x[1] +  x[2]         + 2
         x[1] + 2x[2] +  x[3] + 2
                 x[2] + 2x[3] + 2])
    @test MA.isequal_canonical(y .- 2ones(Int, 3), [
        2x[1] +  x[2]         - 2
         x[1] + 2x[2] +  x[3] - 2
                 x[2] + 2x[3] - 2])
    @test MA.isequal_canonical(2ones(Int, 3) .+ y, [
        2x[1] +  x[2]         + 2
         x[1] + 2x[2] +  x[3] + 2
                 x[2] + 2x[3] + 2])
    @test MA.isequal_canonical(2ones(Int, 3) .- y, [
        -2x[1] -  x[2]         + 2
         -x[1] - 2x[2] -  x[3] + 2
                 -x[2] - 2x[3] + 2])
    @test MA.isequal_canonical(y .+ x, [
        3x[1] +  x[2]
         x[1] + 3x[2] +  x[3]
                 x[2] + 3x[3]])
    @test MA.isequal_canonical(x .+ y, [
        3x[1] +  x[2]
         x[1] + 3x[2] +  x[3]
                 x[2] + 3x[3]])
    @test MA.isequal_canonical(2y .+ 2x, [
        6x[1] + 2x[2]
        2x[1] + 6x[2] + 2x[3]
                2x[2] + 6x[3]])
    @test MA.isequal_canonical(y .- x, [
        x[1] + x[2]
        x[1] + x[2] + x[3]
               x[2] + x[3]])
    @test MA.isequal_canonical(x .- y, [
        -x[1] - x[2]
        -x[1] - x[2] - x[3]
        -x[2] - x[3]])
    @test MA.isequal_canonical(y .+ x[:], [
        3x[1] +  x[2]
         x[1] + 3x[2] +  x[3]
                 x[2] + 3x[3]])
    @test MA.isequal_canonical(x[:] .+ y, [
        3x[1] +  x[2]
         x[1] + 3x[2] +  x[3]
                 x[2] + 3x[3]])

    _matrix_vector_test(x, A)

    A = [1 2 3
         0 4 5
         6 0 7]
    _matrix_vector_test(x, A)
end

_constant(x) = reshape(collect(1:length(x)), size(x)...)

function non_array_test(x, x2)
    # This is needed to compare arrays that have nonstandard indexing
    elements_equal(A::AbstractArray{T, N}, B::AbstractArray{T, N}) where {T, N} = all(a == b for (a, b) in zip(A, B))

    @test elements_equal(+x, +x2)
    @test elements_equal(-x, -x2)
    @test elements_equal(x .+ first(x), x2 .+ first(x2))
    @test elements_equal(x .- first(x), x2 .- first(x2))
    @test elements_equal(first(x) .- x, first(x2) .- x2)
    @test elements_equal(first(x) .+ x, first(x2) .+ x2)
    @test elements_equal(2 .* x, 2 .* x2)
    @test elements_equal(first(x) .+ x2, first(x2) .+ x)
    @test sum(x) == sum(x2)
    if !MA._one_indexed(x2)
        @test_throws DimensionMismatch x + x2
    end
    if x isa AbstractVector
        # `diagm` only define with `Pair` in Julia v1.0 and v1.1
        @testset "diagm" begin
            if !MA._one_indexed(x2) && eltype(x2) <: MA.AbstractMutable
                @test_throws AssertionError diagm(x2)
            else
                @test diagm(0 => x) == diagm(0 => x2)
                if VERSION >= v"1.2"
                    @test diagm(x) == diagm(x2)
                end
            end
        end
    end
end

function non_array_test(x)
    non_array_test(x, x)
    if x isa AbstractVector
        non_array_test(x, view(x, :))
    end
    if x isa AbstractVector || x isa AbstractMatrix
        non_array_test(x, sparse(x))
    end
end

function dot_test(x)
    @test_rewrite dot(x[1], x[1])
    @test_rewrite dot(2, x[1])
    @test_rewrite dot(x[1], 2)

    A = _constant(x)
    @test_rewrite dot(A, x)
    @test_rewrite dot(x, A)

    y = repeat(x, outer = (one.(size(x))..., size(x, 1)))
    @test_rewrite dot(x, _constant(x)) - dot(y, _constant(y))
end

function sum_test(matrix)
    @test_rewrite sum(matrix)
    if matrix isa AbstractMatrix
        @test_rewrite sum([2matrix[i, j] for i in 1:size(matrix, 1), j in 1:size(matrix, 2)])
        @test_rewrite sum(2matrix[i, j] for i in 1:size(matrix, 1), j in 1:size(matrix, 2))
        @test_rewrite sum([2matrix[i, j]^2 for i in 1:size(matrix, 1), j in 1:size(matrix, 2)])
        @test_rewrite sum(2matrix[i, j]^2 for i in 1:size(matrix, 1), j in 1:size(matrix, 2))
    end
end

function transpose_test(x)
    if x isa Vector
        y = reshape(x, 1, length(x))
        @test MA.isequal_canonical(x', y)
        @test MA.isequal_canonical(copy(transpose(x)), y)
    end
    if x isa AbstractMatrix
        y = [x[i, j] for j in 1:size(x, 2), i in 1:size(x, 1)]
        @test MA.isequal_canonical(x', y)
        @test MA.isequal_canonical(copy(transpose(x)), y)
    end
    if x isa AbstractVector || x isa AbstractMatrix
        @test_rewrite(x')
        @test_rewrite(transpose(x))
        @test (x')' == x
        @test transpose(transpose(x)) == x
    end
end

function _broadcast_test(x, A)
    B = sparse(A)
    y = SparseMatrixCSC(size(x)..., copy(B.colptr), copy(B.rowval), vec(x))

    @test MA.isequal_canonical(A .+ x, B .+ x)
    @test MA.isequal_canonical(A .+ x, A .+ y)
    @test MA.isequal_canonical(A .+ y, B .+ y)
    @test MA.isequal_canonical(x .+ A, x .+ B)
    @test MA.isequal_canonical(x .+ A, y .+ A)
    @test MA.isequal_canonical(y .+ A, y .+ B)

    @test MA.isequal_canonical(A .- x, B .- x)
    @test MA.isequal_canonical(A .- x, A .- y)
    @test MA.isequal_canonical(A .- y, B .- y)
    @test MA.isequal_canonical(x .- A, x .- B)
    @test MA.isequal_canonical(x .- A, y .- A)
    @test MA.isequal_canonical(y .- A, y .- B)

    @test MA.isequal_canonical(A .* x, B .* x)
    @test MA.isequal_canonical(A .* x, A .* y)
    @test MA.isequal_canonical(A .* y, B .* y)
    @test MA.isequal_canonical(x .* A, x .* B)
    @test MA.isequal_canonical(x .* A, y .* A)
    @test MA.isequal_canonical(y .* A, y .* B)
end
function broadcast_test(x)
    if !(x isa AbstractMatrix)
        return
    end
    A = reshape(1:length(x), size(x)...)
    if size(x) == (2, 2)
        @test MA.isequal_canonical(A .+ x, [
            1+x[1,1]  3+x[1,2];
            2+x[2,1]  4+x[2,2]])
        @test MA.isequal_canonical(x .+ A, [
            1+x[1,1]  3+x[1,2];
            2+x[2,1]  4+x[2,2]])
        @test MA.isequal_canonical(x .+ x, [2x[1,1] 2x[1,2]; 2x[2,1] 2x[2,2]])

        @test MA.isequal_canonical(A .- x, [
            1-x[1,1]  3-x[1,2];
            2-x[2,1]  4-x[2,2]])
        @test MA.isequal_canonical(x .- x, [zero(x[1] - x[1]) for _1 in 1:2, _2 in 1:2])
        @test MA.isequal_canonical(x .- A, [
            -1+x[1,1]  -3+x[1,2];
            -2+x[2,1]  -4+x[2,2]])

        @test MA.isequal_canonical(A .* x, [
            1*x[1,1]  3*x[1,2];
            2*x[2,1]  4*x[2,2]])
        @test MA.isequal_canonical(x .* A, [
            1*x[1,1]  3*x[1,2];
            2*x[2,1]  4*x[2,2]])
        @test MA.isequal_canonical(x .* x, [x[1,1]^2 x[1,2]^2; x[2,1]^2 x[2,2]^2])

        # TODO: Refactor to avoid calling the internal JuMP function
        # `_densify_with_jump_eltype`.
        #z = JuMP._densify_with_jump_eltype((2 .* y) ./ 3)
        #@test MA.isequal_canonical((2 .* x) ./ 3, z)
        #z = JuMP._densify_with_jump_eltype(2 * (y ./ 3))
        #@test MA.isequal_canonical(2 .* (x ./ 3), z)
        #z = JuMP._densify_with_jump_eltype((x[1,1],) .* B)
        #@test MA.isequal_canonical((x[1,1],) .* A, z)
    end
    _broadcast_test(x, A)
end

function _broadcast_division_test(x, A)
    B = sparse(A)
    y = SparseMatrixCSC(size(x)..., copy(B.colptr), copy(B.rowval), vec(x))

    @test MA.isequal_canonical(x ./ A, x ./ B)
    @test MA.isequal_canonical(x ./ A, y ./ A)
end
function broadcast_division_test(x)
    if !(x isa AbstractMatrix)
        return
    end
    A = reshape(1:length(x), size(x)...)
    if size(x) == (2, 2)
        @test MA.isequal_canonical(x ./ A, [
            x[1,1] / 1  x[1,2] / 3;
            x[2,1] / 2  x[2,2] / 4])
    end
    _broadcast_division_test(x, A)
end

function symmetric_unary_test(x)
    if x isa AbstractMatrix
        unary_test(LinearAlgebra.Symmetric(x))
    end
end

function matrix_uniform_scaling_test(x)
    if !(x isa AbstractMatrix)
        return
    end
    add_test(x, I)
    @test_rewrite (x .+ 1) + I
    @test_rewrite (x .- 1) - I
    @test_rewrite I + (x .+ 1)
    @test_rewrite I - (x .- 1)
    @test_rewrite I * x
    @test_rewrite I * (x .+ 1)
    @test_rewrite (x .+ 1) * I
    @test_rewrite (x .+ 1) + I * I
    @test_rewrite (x .+ 1) + 2 * I
    @test_rewrite (x .+ 1) + I * 2
end

function symmetric_matrix_uniform_scaling_test(x)
    if x isa AbstractMatrix
        matrix_uniform_scaling_test(LinearAlgebra.Symmetric(x))
    end
end

const matrix_tests = Dict(
    "matrix_vector_division" => matrix_vector_division_test,
    "non_array" => non_array_test,
    "matrix_vector" => matrix_vector_test,
    "dot" => dot_test,
    "sum" => sum_test,
    "transpose" => transpose_test,
    "broadcast" => broadcast_test,
    "broadcast_division" => broadcast_division_test,
    "unary" => unary_test,
    "symmetric_unary" => symmetric_unary_test,
    "matrix_uniform_scaling" => matrix_uniform_scaling_test,
    "symmetric_matrix_uniform_scaling" => symmetric_matrix_uniform_scaling_test
)

@test_suite matrix
