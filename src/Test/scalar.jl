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

# See JuMP issue #656
function scalar_in_any_test(x)
    ints = [i for i in 1:2]
    anys = Array{Any}(undef, 2)
    anys[1] = 10
    anys[2] = 20 + x
    @test dot(ints, anys) == 10 + 40 + 2x
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

const scalar_tests = Dict(
    "cube" => cube_test,
    "scalar_in_any" => scalar_in_any_test,
    "scalar_uniform_scaling" => scalar_uniform_scaling_test
)

@test_suite scalar
