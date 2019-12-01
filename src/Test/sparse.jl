function _mat_mat_test(A, B)
    add_test(A, B)
    @test MA.isequal_canonical(copy(transpose(A)) * B, A' * B)
    @test MA.isequal_canonical(A * copy(transpose(B)), A * B')
    @test MA.isequal_canonical(copy(transpose(B)) * A, B' * A)
    @test MA.isequal_canonical(B * copy(transpose(A)), B * A')
end

function sparse_test(X11, X23, Xd)
    X = sparse([1, 2], [1, 3], [X11, X23], 3, 3)
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

    _mat_mat_test(Xd, Yd)
    _mat_mat_test(Yd, Zd)
    _mat_mat_test(Zd, Xd)
    _mat_mat_test(Xd, Y)
    _mat_mat_test(Xd, Xd)

    v = [4, 5, 6]
    @test MA.isequal_canonical(X * v,  [4X11; 6X23; 0])
    @test MA.isequal_canonical(v' * X,  [4X11  0   5X23])
    @test MA.isequal_canonical(copy(transpose(v)) * X, [4X11  0   5X23])
    @test MA.isequal_canonical(X' * v,  [4X11;  0;  5X23])
    @test MA.isequal_canonical(copy(transpose(X)) * v, [4X11; 0;  5X23])

    A = [2 1 0
         1 2 1
         0 1 2]

    _mat_mat_test(X, A)
    _mat_mat_test(Y, A)
    _mat_mat_test(Z, A)

    @test MA.isequal_canonical(X * A,  [
        2X11  X11  0
        0     X23  2X23
        0     0    0   ])
    @test MA.isequal_canonical(A * X,  [
        2X11  0    X23
         X11  0   2X23
         0    0    X23])
    @test MA.isequal_canonical(A * X', [
        2X11  0     0
         X11   X23  0
        0     2X23  0])
    @test MA.isequal_canonical(X' * A, [
        2X11  X11   0
         0     0    0
         X23   2X23 X23])
    @test MA.isequal_canonical(A' * X, [
        2X11  0  X23
         X11  0 2X23
         0    0  X23])

    B = sparse(A)
    @test MA.isequal_canonical(X * A, X * B)
    @test MA.isequal_canonical(A * X, B * X)
    @test MA.isequal_canonical(A * X', B * X')
    @test MA.isequal_canonical(A' * X, B' * X)
end
