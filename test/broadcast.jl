using Test
import MutableArithmetics
const MA = MutableArithmetics

@testset "Int" begin
    a = [1, 2]
    b = 3
    @test MA.broadcast!(+, a, b) == [4, 5]
    @test a == [4, 5]
end
@testset "BigInt" begin
    x = BigInt(1)
    y = BigInt(2)
    a = [x, y]
    b = 3
    @test MA.broadcast!(+, a, b) == [4, 5]
    @test a == [4, 5]
    @test x == 4
    @test y == 5
end
