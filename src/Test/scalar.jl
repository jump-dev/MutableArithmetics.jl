function iszero_test(x)
    x_copy = x

    @test iszero(x - x)
    @test iszero(MA.@rewrite(x - x))
    @test MA.iszero!(x - x)
    @test MA.iszero!(MA.@rewrite(x - x))

    @test iszero(0 * x)
    @test iszero(MA.@rewrite(0 * x))
    @test MA.iszero!(0 * x)
    @test MA.iszero!(MA.@rewrite(0 * x))

    @test iszero(x - 2x + x)
    @test iszero(MA.@rewrite(x - 2x + x))
    @test MA.iszero!(x - 2x + x)
    @test MA.iszero!(MA.@rewrite(x - 2x + x))

    @test MA.isequal_canonical(x_copy, x)
end

function empty_sum_test(x)
    @test MA.isequal_canonical(MA.@rewrite(x + sum(1 for i = 1:0) * sum(x for i = 1:0)), x)
    @test MA.isequal_canonical(MA.@rewrite(x + sum(x for i = 1:0) * sum(1 for i = 1:0)), x)
    @test MA.isequal_canonical(MA.@rewrite(x + sum(1 for i = 1:0) * 1^2), x)
    @test MA.isequal_canonical(MA.@rewrite(x + 1^2 * sum(1 for i = 1:0)), x)
    # `1^2` is considered as a complex expression by `@rewrite`.
    @test MA.isequal_canonical(
        MA.@rewrite(x + 1^2 * sum(1 for i = 1:0) * sum(x for i = 1:0)),
        x,
    )
    @test MA.isequal_canonical(
        MA.@rewrite(x + sum(1 for i = 1:0) * 1^2 * sum(x for i = 1:0)),
        x,
    )
    @test MA.isequal_canonical(
        MA.@rewrite(x + 1^2 * sum(1 for i = 1:0) * sum(x for i = 1:0) * 1^2),
        x,
    )
    # Fails because the sum is not rewritten because `*` is not rewritten when `vectorized` is `true`.
    #@test MA.isequal_canonical(MA.@rewrite(x .+ sum(1 for i in 1:0) * sum(x for i in 1:0)), x)
    #@test MA.isequal_canonical(MA.@rewrite(x .+ 1^2 * sum(1 for i in 1:0) * sum(x for i in 1:0) * 1^2), x)
    @test MA.isequal_canonical(MA.@rewrite(x .+ sum(1 for i = 1:0)), x)
    @test_broken MA.isequal_canonical(MA.@rewrite(x .+ sum(1 for i = 1:0) * 1^2), x + 0)
    @test MA.isequal_canonical(MA.@rewrite(sum(1 for i = 1:0) .+ x), x)
    @test MA.isequal_canonical(MA.@rewrite(sum(1 for i = 1:0) * 1^2 .+ x), x)
    @test MA.isequal_canonical(MA.@rewrite(x .- sum(1 for i = 1:0)), x)
    @test_broken MA.isequal_canonical(MA.@rewrite(x .- sum(1 for i = 1:0) * 1^2), x)
    @test MA.isequal_canonical(MA.@rewrite(sum(1 for i = 1:0) .- x), -x)
    @test MA.isequal_canonical(MA.@rewrite(sum(1 for i = 1:0) * 1^2 .- x), -x)
end

function cube_test(x)
    @test_rewrite x^3
    @test_rewrite (x + 1)^3
    @test_rewrite x^2 * x
    @test_rewrite (x + 1)^2 * x
    @test_rewrite x^2 * (x + 1)
    @test_rewrite (x + 1)^2 * (x + 1)
    @test_rewrite x * x^2
    @test_rewrite (x + 1) * x^2
    @test_rewrite x * (x + 1)^2
    @test_rewrite (x + 1) * (x + 1)^2
    @test_rewrite x * x * x
    @test_rewrite (x + 1) * x * x
    @test_rewrite x * (x + 1) * x
    @test_rewrite x * x * (x + 1)
end

function mul_scalar_array_test(x)
    for A in [[1, 2, 3], [1 2; 3 4]]
        @test_rewrite x * A
        @test_rewrite A * x
    end
end

# See JuMP issue #656
function scalar_in_any_test(x)
    ints = [i for i = 1:2]
    anys = Array{Any}(undef, 2)
    anys[1] = 10
    anys[2] = 20 + x
    @test MA.isequal_canonical(dot(ints, anys), 10 + 40 + 2x)
end

function scalar_uniform_scaling_test(x)
    add_test(x, I)
    @test_rewrite (x + 1) + I
    @test_rewrite (x - 1) - I
    @test_rewrite I + (x + 1)
    @test_rewrite I - (x - 1)
    @test_rewrite I * x
    @test_rewrite I * (x + 1)
    @test_rewrite (x + 1) * I
end

function convert_test(x)
    y = MA.operate(convert, typeof(x), x)
    MA.mutable_operate!(+, y, 1)
    @test MA.isequal_canonical(y, x + 1)
end

const scalar_tests = Dict(
    "mul_scalar_array" => mul_scalar_array_test,
    "empty_sum" => empty_sum_test,
    "cube" => cube_test,
    "iszero" => iszero_test,
    "scalar_in_any" => scalar_in_any_test,
    "scalar_uniform_scaling" => scalar_uniform_scaling_test,
)

@test_suite scalar
