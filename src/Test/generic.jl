macro test_rewrite(expr)
    esc(quote
        @test MA.isequal_canonical(MA.@rewrite($expr), $expr)
    end)
end

function _add_test(x, y)
    @test_rewrite(x + y)
    @test_rewrite(-x + y)
    @test_rewrite(x - y)
    @test_rewrite(-x - y)
    if !(x isa UniformScaling)
        # Unary `+` not defined for `UniformScaling`.
        @test_rewrite(+x - y)
        @test_rewrite(+x + y)
    end
    @test_rewrite(2x + y)
    @test_rewrite(2x - y)
    @test_rewrite(x + 2y)
    @test_rewrite(x - 2y)
    @test_rewrite(x + y * 2)
    @test_rewrite(x - y * 2)
    @test_rewrite(x + 2y * 2)
    @test_rewrite(x - 2y * 2)
end
function add_test(x, y)
    _add_test(x, y)
    _add_test(y, x)
end

function unary_test(x)
    @test_rewrite 2x
    @test_rewrite x * 2
    @test_rewrite 2x * 2
    @test_rewrite +x
    @test_rewrite -x
end
