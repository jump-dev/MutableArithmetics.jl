using Test
import MutableArithmetics
const MA = MutableArithmetics

@testset "promote_operation" begin
    @test MA.promote_operation(MA.zero, Int) == Int
    @test MA.promote_operation(MA.one, Int) == Int
    @test MA.promote_operation(+, Int, Int) == Int
    @test MA.promote_operation(-, Int, Int) == Int
    @test MA.promote_operation(*, Int, Int) == Int
    @test MA.promote_operation(MA.add_mul, Int, Int, Int) == Int
    err = ErrorException(
        "Operation `+` between `$(Array{Int,1})` and `$Int` is not allowed. You should use broadcast.",
    )
    @test_throws err MA.promote_operation(+, Vector{Int}, Int)
    err = ErrorException(
        "Operation `+` between `$Int` and `$(Array{Int,1})` is not allowed. You should use broadcast.",
    )
    @test_throws err MA.promote_operation(+, Int, Vector{Int})
end
@testset "add_to! / add!" begin
    @test MA.mutability(Int, MA.add_to!, Int, Int) isa MA.NotMutable
    @test MA.mutability(Int, MA.add!, Int) isa MA.NotMutable
    a = 5
    b = 28
    c = 41
    @test MA.add_to!(a, b, c) == 69
    @test MA.add!(b, c) == 69
    a = 165
    b = 255
    @test MA.add!(a, b) == 420
end

@testset "mul_to! / mul!" begin
    @test MA.mutability(Int, MA.mul_to!, Int, Int) isa MA.NotMutable
    @test MA.mutability(Int, MA.mul!, Int) isa MA.NotMutable
    a = 5
    b = 23
    c = 3
    @test MA.mul_to!(a, b, c) == 69
    @test MA.mul!(b, c) == 69
    a = 15
    b = 28
    @test MA.mul!(a, b) == 420
end

@testset "add_mul_to! / add_mul! / add_mul_buf_to! /add_mul_buf!" begin
    @test MA.mutability(Int, MA.add_mul_to!, Int, Int, Int) isa MA.NotMutable
    @test MA.mutability(Int, MA.add_mul!, Int, Int) isa MA.NotMutable
    @test MA.mutability(Int, MA.add_mul_buf_to!, Int, Int, Int, Int) isa MA.NotMutable
    @test MA.mutability(Int, MA.add_mul_buf!, Int, Int, Int) isa MA.NotMutable
    a = 5
    b = 9
    c = 3
    d = 20
    buf = 24
    @test MA.add_mul_to!(a, b, c, d) == 69
    @test MA.add_mul!(b, c, d) == 69
    @test MA.add_mul_buf_to!(buf, a, b, c, d) == 69
    @test MA.add_mul_buf!(buf, b, c, d) == 69
    a = 148
    b = 16
    c = 17
    @test MA.add_mul!(a, b, c) == 420
    a = 148
    b = 16
    c = 17
    buf = 56
    @test MA.add_mul_buf!(buf, a, b, c) == 420
    a = 148
    b = 16
    c = 17
    d = 42
    buf = 56
    @test MA.add_mul_buf_to!(buf, d, a, b, c) == 420
end

@testset "zero!" begin
    @test MA.mutability(Int, MA.zero!) isa MA.NotMutable
    a = 5
    @test MA.zero!(a) == 0
end

@testset "one!" begin
    @test MA.mutability(Int, MA.one!) isa MA.NotMutable
    a = 5
    @test MA.one!(a) == 1
end

@testset "Zero / Int" begin
    @test MA.Zero() / 1 == MA.Zero()
    @test_throws DivideError MA.Zero() / 0
end
