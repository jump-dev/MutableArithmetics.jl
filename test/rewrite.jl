using SparseArrays, Test
import MutableArithmetics
const MA = MutableArithmetics

macro test_rewrite(expr)
    esc(quote
        @test MA.isequal_canonical(MA.@rewrite($expr), $expr)
    end)
end

function basic_operators_test(w, x, y, z)
    aff = @inferred 7.1 * x + 2.5
    @test_rewrite 7.1 * x + 2.5
    aff2 = @inferred 1.2 * y + 1.2
    @test_rewrite 1.2 * y + 1.2
    q = @inferred 2.5 * y * z + aff
    @test_rewrite 2.5 * y * z + aff
    q2 = @inferred 8 * x * z + aff2
    @test_rewrite 8 * x * z + aff2
    @test_rewrite 2 * x * x + 1 * y * y + z + 3

    @testset "Comparison" begin
        @testset "iszero" begin
            @test !iszero(x)
            @test !iszero(aff)
            @test iszero(zero(aff))
            @test !iszero(q)
            @test iszero(zero(q))
        end

        @testset "isequal_canonical" begin
            @test MA.isequal_canonical((@inferred 3w + 2y), @inferred 2y + 3w)
            @test !MA.isequal_canonical((@inferred 3w + 2y + 1), @inferred 3w + 2y)
            @test !MA.isequal_canonical((@inferred 3w + 2y), @inferred 3y + 2w)
            @test !MA.isequal_canonical((@inferred 3w + 2y), @inferred 3w + y)

            @test !MA.isequal_canonical(aff, aff2)
            @test !MA.isequal_canonical(aff2, aff)

            @test  MA.isequal_canonical(q, @inferred 2.5z*y + aff)
            @test !MA.isequal_canonical(q, @inferred 2.5y*z + aff2)
            @test !MA.isequal_canonical(q, @inferred 2.5x*z + aff)
            @test !MA.isequal_canonical(q, @inferred 2.5y*x + aff)
            @test !MA.isequal_canonical(q, @inferred 1.5y*z + aff)
            @test  MA.isequal_canonical(q2, @inferred 8z*x + aff2)
            @test !MA.isequal_canonical(q2, @inferred 8x*z + aff)
            @test !MA.isequal_canonical(q2, @inferred 7x*z + aff2)
            @test !MA.isequal_canonical(q2, @inferred 8x*y + aff2)
            @test !MA.isequal_canonical(q2, @inferred 8y*z + aff2)
        end
    end

    # Different objects that must all interact:
    # 1. Number
    # 2. Variable
    # 3. AffExpr
    # 4. QuadExpr

    # 1. Number tests
    @testset "Number--???" begin
        # 1-1 Number--Number - nope!
        # 1-2 Number--Variable
        @test_rewrite 4.13 + w
        @test_rewrite 3.16 - w
        @test_rewrite 5.23 * w
        # 1-3 Number--AffExpr
        @test_rewrite 1.5 + aff
        @test_rewrite 1.5 - aff
        @test_rewrite 2 * aff
        # 1-4 Number--QuadExpr
        @test_rewrite 1.5 + q
        @test_rewrite 1.5 - q
        @test_rewrite 2 * q
    end

    # 2. Variable tests
    @testset "Variable--???" begin
        # 2-0 Variable unary
        @test (+x) === x
        @test_rewrite -x
        # 2-1 Variable--Number
        @test_rewrite w + 4.13
        @test_rewrite w - 4.13
        @test_rewrite w * 4.13
        @test_rewrite w / 2.00
        @test w == w
        @test_rewrite x*y - 1
        @test_rewrite x^2
        @test_rewrite x^1
        @test_rewrite x^0
        # 2-2 Variable--Variable
        @test_rewrite w + x
        @test_rewrite w - x
        @test_rewrite w * x
        @test_rewrite x - x
        @test_rewrite y*z - x
        # 2-3 Variable--AffExpr
        @test_rewrite z + aff
        @test_rewrite z - aff
        @test_rewrite z * aff
        @test_rewrite 7.1 * x - aff
        # 2-4 Variable--QuadExpr
        @test_rewrite w + q
        @test_rewrite w - q
    end

    # 3. AffExpr tests
    @testset "AffExpr--???" begin
        # 3-0 AffExpr unary
        @test_rewrite +aff
        @test_rewrite -aff
        # 3-1 AffExpr--Number
        @test_rewrite aff + 1.5
        @test_rewrite aff - 1.5
        @test_rewrite aff * 2
        @test_rewrite aff / 2
        @test aff == aff
        @test_rewrite aff - 1
        @test_rewrite aff^2
        @test_rewrite (7.1*x + 2.5)^2
        @test_rewrite aff^1
        @test_rewrite (7.1*x + 2.5)^1
        @test_rewrite aff^0
        @test_rewrite (7.1*x + 2.5)^0
        # 3-2 AffExpr--Variable
        @test_rewrite aff + z
        @test_rewrite aff - z
        @test_rewrite aff * z
        @test_rewrite aff - 7.1 * x
        # 3-3 AffExpr--AffExpr
        @test_rewrite aff + aff2
        @test_rewrite aff - aff2
        @test_rewrite aff * aff2
        @test string((x+x)*(x+3)) == string((x+3)*(x+x))  # Issue #288
        @test_rewrite aff-aff
        # 4-4 AffExpr--QuadExpr
        @test_rewrite aff2 + q
        @test_rewrite aff2 - q
    end

    # 4. QuadExpr
    # TODO: This test block and others above should be rewritten to be
    # self-contained. The definitions of q, w, and aff2 are too far to
    # easily check correctness of the tests.
    @testset "QuadExpr--???" begin
        # 4-0 QuadExpr unary
        @test_rewrite +q
        @test_rewrite -q
        # 4-1 QuadExpr--Number
        @test_rewrite q + 1.5
        @test_rewrite q - 1.5
        @test_rewrite q * 2
        @test_rewrite q / 2
        @test q == q
        @test_rewrite aff2 - q
        # 4-2 QuadExpr--Variable
        @test_rewrite q + w
        @test_rewrite q - w
        # 4-3 QuadExpr--AffExpr
        @test_rewrite q + aff2
        @test_rewrite q - aff2
        # 4-4 QuadExpr--QuadExpr
        @test_rewrite q + q2
        @test_rewrite q - q2
    end
end

function sum_test(matrix)
    @testset "sum(j::DenseAxisArray{Variable})" begin
        @test_rewrite sum(matrix)
    end
    @testset "sum(affs::Array{AffExpr})" begin
        @test_rewrite sum([2matrix[i, j] for i in 1:size(matrix, 1), j in 1:size(matrix, 2)])
    end
    @testset "sum(quads::Array{QuadExpr})" begin
        @test_rewrite sum([2matrix[i, j]^2 for i in 1:size(matrix, 1), j in 1:size(matrix, 2)])
    end
end

function dot_test(x, y, z)
    @test_rewrite dot(x[1], x[1])
    @test_rewrite dot(2, x[1])
    @test_rewrite dot(x[1], 2)

    c = vcat(1:3)
    @test_rewrite dot(c, x)
    @test_rewrite dot(x, c)

    A = [1 3 ; 2 4]
    @test_rewrite dot(A, y)
    @test_rewrite dot(y, A)

    B = ones(2, 2, 2)
    @test_rewrite dot(B, z)
    @test_rewrite dot(z, B)

    @test_rewrite dot(x, ones(3)) - dot(y, ones(2,2))
end

# JuMP issue #656
function issue_656(x)
    floats = Float64[i for i in 1:2]
    anys   = Array{Any}(undef, 2)
    anys[1] = 10
    anys[2] = 20 + x
    @test dot(floats, anys) == 10 + 40 + 2x
end

function transpose_test(x, y, z)
    @test MA.isequal_canonical(x', [x[1] x[2] x[3]])
    @test MA.isequal_canonical(copy(transpose(x)), [x[1] x[2] x[3]])
    @test MA.isequal_canonical(y', [y[1,1] y[2,1]
                      y[1,2] y[2,2]
                      y[1,3] y[2,3]])
    @test MA.isequal_canonical(copy(transpose(y)),
                 [y[1,1] y[2,1]
                  y[1,2] y[2,2]
                  y[1,3] y[2,3]])
    @test (z')' == z
    @test transpose(transpose(z)) == z
end

function vectorized_test(x, X11, X23, Xd)
    A = [2 1 0
         1 2 1
         0 1 2]
    B = sparse(A)
    X = sparse([1, 2], [1, 3], [X11, X23], 3, 3) # for testing Variable
    # FIXME
    #@test MA.isequal_canonical([X11 0. 0.; 0. 0. X23; 0. 0. 0.], @inferred MA._densify_with_jump_eltype(X))
    Y = sparse([1, 2], [1, 3], [2X11, 4X23], 3, 3) # for testing GenericAffExpr
    Yd = [2X11 0    0
          0    0 4X23
          0    0    0]
    Z = sparse([1, 2], [1, 3], [X11^2, 2X23^2], 3, 3) # for testing GenericQuadExpr
    Zd = [X11^2 0      0
          0     0 2X23^2
          0     0      0]
    v = [4, 5, 6]

    @testset "Sum of matrices" begin
        @test_rewrite(x - x)
        @test_rewrite(x + x)
        @test_rewrite(x + 2x)
        @test_rewrite(x - 2x)
        @test_rewrite(x + x * 2)
        @test_rewrite(x - x * 2)
        @test_rewrite(x + 2x * 2)
        @test_rewrite(x - 2x * 2)
        @test_rewrite(Xd + Yd)
        @test_rewrite(Xd - Yd)
        @test_rewrite(Xd + 2Yd)
        @test_rewrite(Xd - 2Yd)
        @test_rewrite(Xd + Yd * 2)
        @test_rewrite(Xd - Yd * 2)
        @test_rewrite(Yd + Xd)
        @test_rewrite(Yd + 2Xd)
        @test_rewrite(Yd + Xd * 2)
        @test_rewrite(Yd + Zd)
        @test_rewrite(Yd + 2Zd)
        @test_rewrite(Yd + Zd * 2)
        @test_rewrite(Zd + Yd)
        @test_rewrite(Zd + 2Yd)
        @test_rewrite(Zd + Yd * 2)
        @test_rewrite(Zd + Xd)
        @test_rewrite(Zd + 2Xd)
        @test_rewrite(Zd + Xd * 2)
        @test_rewrite(Xd + Zd)
        @test_rewrite(Xd + 2Zd)
        @test_rewrite(Xd + Zd * 2)
    end

    @test_rewrite(x')
    @test_rewrite(x' * A)
    # Complex expression
    @test_rewrite(x' * ones(3, 3))
    @test_rewrite(x' * A * x)
    # Complex expression
    @test_rewrite(x' * ones(3, 3) * x)

    @test MA.isequal_canonical(A*x, [2x[1] +  x[2]
                       2x[2] +  x[1] + x[3]
                        x[2] + 2x[3]])
    @test MA.isequal_canonical(A*x, B*x)
    @test MA.isequal_canonical(A*x, MA.@rewrite(B*x))
    @test MA.isequal_canonical(MA.@rewrite(A*x), MA.@rewrite(B*x))
    @test MA.isequal_canonical(x'*A, [2x[1]+x[2]; 2x[2]+x[1]+x[3]; x[2]+2x[3]]')
    @test MA.isequal_canonical(x'*A, x'*B)
    @test MA.isequal_canonical(x'*A, MA.@rewrite(x'*B))
    @test MA.isequal_canonical(MA.@rewrite(x'*A), MA.@rewrite(x'*B))
    @test MA.isequal_canonical(x'*A*x, 2x[1]*x[1] + 2x[1]*x[2] + 2x[2]*x[2] + 2x[2]*x[3] + 2x[3]*x[3])
    @test MA.isequal_canonical(x'A*x, x'*B*x)
    @test MA.isequal_canonical(x'*A*x, MA.@rewrite(x'*B*x))
    @test MA.isequal_canonical(MA.@rewrite(x'*A*x), MA.@rewrite(x'*B*x))

    y = A*x
    @test MA.isequal_canonical(-x, [-x[1], -x[2], -x[3]])
    @test MA.isequal_canonical(-y, [-2x[1] -  x[2]
                       -x[1] - 2x[2] -  x[3]
                               -x[2] - 2x[3]])
    @test MA.isequal_canonical(y .+ 1, [2x[1] +  x[2]         + 1
                          x[1] + 2x[2] +  x[3] + 1
                          x[2] + 2x[3] + 1])
    @test MA.isequal_canonical(y .- 1, [
        2x[1] +  x[2]         - 1
         x[1] + 2x[2] +  x[3] - 1
                 x[2] + 2x[3] - 1])
    @test MA.isequal_canonical(y .+ 2ones(3), [2x[1] +  x[2]         + 2
                                 x[1] + 2x[2] +  x[3] + 2
                                 x[2] + 2x[3] + 2])
    @test MA.isequal_canonical(y .- 2ones(3), [2x[1] +  x[2]         - 2
                                 x[1] + 2x[2] +  x[3] - 2
                                 x[2] + 2x[3] - 2])
    @test MA.isequal_canonical(2ones(3) .+ y, [2x[1] +  x[2]         + 2
                                 x[1] + 2x[2] +  x[3] + 2
                                 x[2] + 2x[3] + 2])
    @test MA.isequal_canonical(2ones(3) .- y, [-2x[1] -  x[2]         + 2
                                 -x[1] - 2x[2] -  x[3] + 2
                                 -x[2] - 2x[3] + 2])
    @test MA.isequal_canonical(y .+ x, [3x[1] +  x[2]
                          x[1] + 3x[2] +  x[3]
                                  x[2] + 3x[3]])
    @test MA.isequal_canonical(x .+ y, [3x[1] +  x[2]
                          x[1] + 3x[2] +  x[3]
                          x[2] + 3x[3]])
    @test MA.isequal_canonical(2y .+ 2x, [6x[1] + 2x[2]
                           2x[1] + 6x[2] + 2x[3]
                           2x[2] + 6x[3]])
    @test MA.isequal_canonical(y .- x, [ x[1] + x[2]
                          x[1] + x[2] + x[3]
                                 x[2] + x[3]])
    @test MA.isequal_canonical(x .- y, [-x[1] - x[2]
                         -x[1] - x[2] - x[3]
                         -x[2] - x[3]])
    @test MA.isequal_canonical(y .+ x[:], [3x[1] +  x[2]
                             x[1] + 3x[2] +  x[3]
                                     x[2] + 3x[3]])
    @test MA.isequal_canonical(x[:] .+ y, [3x[1] +  x[2]
                             x[1] + 3x[2] +  x[3]
                                     x[2] + 3x[3]])

    @test MA.isequal_canonical(MA.@rewrite(A*x/2), A*x/2)
    @test MA.isequal_canonical(X*v,  [4X11; 6X23; 0])
    @test MA.isequal_canonical(v'*X,  [4X11  0   5X23])
    @test MA.isequal_canonical(copy(transpose(v))*X, [4X11  0   5X23])
    @test MA.isequal_canonical(X'*v,  [4X11;  0;  5X23])
    @test MA.isequal_canonical(copy(transpose(X))*v, [4X11; 0;  5X23])
    @test MA.isequal_canonical(X*A,  [2X11  X11  0
                        0     X23  2X23
                        0     0    0   ])
    @test MA.isequal_canonical(A*X,  [2X11  0    X23
                        X11   0    2X23
                        0     0    X23])
    @test MA.isequal_canonical(A*X', [2X11  0    0
                        X11   X23  0
                        0     2X23 0])
    @test MA.isequal_canonical(X'*A, [2X11  X11  0
                        0     0    0
                        X23   2X23 X23])
    @test MA.isequal_canonical(copy(transpose(X))*A, [2X11 X11  0
                         0    0    0
                         X23  2X23 X23])
    @test MA.isequal_canonical(A'*X, [2X11  0 X23
                        X11   0 2X23
                        0     0 X23])
    @test MA.isequal_canonical(copy(transpose(X))*A, X'*A)
    @test MA.isequal_canonical(copy(transpose(A))*X, A'*X)
    @test MA.isequal_canonical(X*A, X*B)
    @test MA.isequal_canonical(Y'*A, copy(transpose(Y))*A)
    @test MA.isequal_canonical(A*Y', A*copy(transpose(Y)))
    @test MA.isequal_canonical(Z'*A, copy(transpose(Z))*A)
    @test MA.isequal_canonical(Xd'*Y, copy(transpose(Xd))*Y)
    @test MA.isequal_canonical(Y'*Xd, copy(transpose(Y))*Xd)
    @test MA.isequal_canonical(Xd'*Xd, copy(transpose(Xd))*Xd)
    @test MA.isequal_canonical(A*X, B*X)
    @test MA.isequal_canonical(A*X', B*X')
    @test MA.isequal_canonical(A'*X, B'*X)

    A = [1 2 3
         0 4 5
         6 0 7]
    B = sparse(A)

    @test_rewrite reshape(x, (1, 3)) * A * x .- 1
    @test_rewrite x'*A*x .- 1
    @test_rewrite x'*B*x .- 1
    for (A1, A2) in [(A, A), (A, B), (B, A), (B, B)]
        @test_rewrite (x'A1)' + 2A2*x
        @test_rewrite (x'A1)' + 2A2*x .- 1
        @test_rewrite (x'A1)' + 2A2*x .- [3:-1:1;]
        @test_rewrite (x'A1)' + 2A2*x - [3:-1:1;]
    end
end

function broadcast_test(x)
    A = [1 2;
         3 4]
    B = sparse(A)
    y = SparseMatrixCSC(2, 2, copy(B.colptr), copy(B.rowval), vec(x))
    @test MA.isequal_canonical(A.+x, [1+x[1,1]  2+x[1,2];
                        3+x[2,1]  4+x[2,2]])
    @test MA.isequal_canonical(A.+x, B.+x)
    @test MA.isequal_canonical(A.+x, A.+y)
    @test MA.isequal_canonical(A.+y, B.+y)
    @test MA.isequal_canonical(x.+A, [1+x[1,1]  2+x[1,2];
                        3+x[2,1]  4+x[2,2]])
    @test MA.isequal_canonical(x.+A, x.+B)
    @test MA.isequal_canonical(x.+A, y.+A)
    @test MA.isequal_canonical(x .+ x, [2x[1,1] 2x[1,2]; 2x[2,1] 2x[2,2]])
    @test MA.isequal_canonical(y.+A, y.+B)
    @test MA.isequal_canonical(A.-x, [1-x[1,1]  2-x[1,2];
                        3-x[2,1]  4-x[2,2]])
    @test MA.isequal_canonical(A.-x, B.-x)
    @test MA.isequal_canonical(A.-x, A.-y)
    @test MA.isequal_canonical(x .- x, [zero(x[1] - x[1]) for _1 in 1:2, _2 in 1:2])
    @test MA.isequal_canonical(A.-y, B.-y)
    @test MA.isequal_canonical(x.-A, [-1+x[1,1]  -2+x[1,2];
                        -3+x[2,1]  -4+x[2,2]])
    @test MA.isequal_canonical(x.-A, x.-B)
    @test MA.isequal_canonical(x.-A, y.-A)
    @test MA.isequal_canonical(y.-A, y.-B)
    @test MA.isequal_canonical(A.*x, [1*x[1,1]  2*x[1,2];
                        3*x[2,1]  4*x[2,2]])
    @test MA.isequal_canonical(A.*x, B.*x)
    @test MA.isequal_canonical(A.*x, A.*y)
    @test MA.isequal_canonical(A.*y, B.*y)

    @test MA.isequal_canonical(x.*A, [1*x[1,1]  2*x[1,2];
                        3*x[2,1]  4*x[2,2]])
    @test MA.isequal_canonical(x.*A, x.*B)
    @test MA.isequal_canonical(x.*A, y.*A)
    @test MA.isequal_canonical(y.*A, y.*B)

    @test MA.isequal_canonical(x .* x, [x[1,1]^2 x[1,2]^2; x[2,1]^2 x[2,2]^2])
    @test MA.isequal_canonical(x ./ A, [
        x[1,1] / 1  x[1,2] / 2;
        x[2,1] / 3  x[2,2] / 4])
    @test MA.isequal_canonical(x ./ A, x ./ B)
    @test MA.isequal_canonical(x ./ A, y ./ A)

    # TODO: Refactor to avoid calling the internal JuMP function
    # `_densify_with_jump_eltype`.
    #z = JuMP._densify_with_jump_eltype((2 .* y) ./ 3)
    #@test MA.isequal_canonical((2 .* x) ./ 3, z)
    #z = JuMP._densify_with_jump_eltype(2 * (y ./ 3))
    #@test MA.isequal_canonical(2 .* (x ./ 3), z)
    #z = JuMP._densify_with_jump_eltype((x[1,1],) .* B)
    #@test MA.isequal_canonical((x[1,1],) .* A, z)
end

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
    # `diagm` only define with `Pair` in Julia v1.0 and v1.1
    @testset "diagm" begin
        if !MA._one_indexed(x2) && eltype(x2) isa MA.AbstractMutable
            @test_throws AssertionError diagm(x2)
        else
            @test diagm(0 => x) == diagm(0 => x2)
            if VERSION >= v"1.2"
                @test diagm(x) == diagm(x2)
            end
        end
    end
end

function unary_matrix(Q)
    @test_rewrite 2Q
    # See https://github.com/JuliaLang/julia/issues/32374 for `Symmetric`
    @test_rewrite -Q
end

function scalar_uniform_scaling(x)
    @test_rewrite x + 2I
    @test_rewrite (x + 1) + I
    @test_rewrite x - 2I
    @test_rewrite (x - 1) - I
    @test_rewrite 2I + x
    @test_rewrite I + (x + 1)
    @test_rewrite 2I - x
    @test_rewrite I - (x - 1)
    @test_rewrite I * x
    @test_rewrite I * (x + 1)
    @test_rewrite (x + 1) * I
end

function matrix_uniform_scaling(x)
    @test_rewrite x + 2I
    @test_rewrite (x .+ 1) + I
    @test_rewrite x - 2I
    @test_rewrite (x .- 1) - I
    @test_rewrite 2I + x
    @test_rewrite I + (x .+ 1)
    @test_rewrite 2I - x
    @test_rewrite I - (x .- 1)
    @test_rewrite I * x
    @test_rewrite I * (x .+ 1)
    @test_rewrite (x .+ 1) * I
    @test_rewrite (x .+ 1) + I * I
    @test_rewrite (x .+ 1) + 2 * I
    @test_rewrite (x .+ 1) + I * 2
end

using LinearAlgebra
using OffsetArrays

@testset "@rewrite with $T" for T in [
        Int,
        Float64,
        BigInt
    ]
    basic_operators_test(T(1), T(2), T(3), T(4))
    sum_test(T[5 1 9; -7 2 4; -2 -7 5])
    S = zeros(T, 2, 2, 2)
    S[1, :, :] = T[5 -8; 3 -7]
    S[2, :, :] = T[-2 8; 8 -1]
    dot_test(T[-7, 1, 4], T[0 -4; 6 -5], S)
    issue_656(T(3))
    transpose_test(T[9, -3, 8], T[-4 4 1; 4 -8 -6], T[6, 9, 2, 4, -3])
    vectorized_test(T[3, 2, 6], T(4), T(5), T[8 1 9; 4 3 1; 2 0 8])
    broadcast_test(T[2 4; -1 3])
    x = T[2, 4, 3]
    non_array_test(x, x)
    non_array_test(x, OffsetArray(x, -length(x)))
    non_array_test(x, view(x, :))
    non_array_test(x, sparse(x))
    unary_matrix(T[1 2; 3 4])
    unary_matrix(Symmetric(T[1 2; 2 4]))
    scalar_uniform_scaling(T(3))
    matrix_uniform_scaling(T[1 2; 3 4])
    matrix_uniform_scaling(Symmetric(T[1 2; 2 4]))
end
