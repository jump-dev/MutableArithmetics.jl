function int_add_test(::Type{T}) where T
    @testset "add_to! / add!" begin
        @test MA.mutability(T, +, T, T) isa MA.IsMutable

        t(n) = convert(T, n)
        a = t(5)
        b = t(28)
        c = t(41)
        expected = t(69)
        @test MA.add_to!(a, b, c) == expected
        @test a == expected
        @test MA.add!(b, c) == expected
        @test b == expected

        a = t(165)
        b = t(255)
        @test MA.add!(a, b) == t(420)
        @test a == t(420)
    end
end
function int_mul_test(::Type{T}) where T
    @testset "mul_to! / mul!" begin
        @test MA.mutability(T, *, T, T) isa MA.IsMutable

        t(n) = convert(T, n)
        a = t(5)
        b = t(23)
        c = t(3)
        @test MA.mul_to!(a, b, c) == t(69)
        @test a == t(69)
        @test MA.mul!(b, c) == t(69)
        @test b == t(69)

        a = t(15)
        b = t(28)
        @test MA.mul!(a, b) == t(420)
        @test a == t(420)
    end
end
function int_add_mul_test(::Type{T}) where T
    @testset "add_mul_to! / add_mul! / add_mul_buf_to! /add_mul_buf!" begin
        @test MA.mutability(T, MA.add_mul, T, T) isa MA.IsMutable
        @test MA.mutability(T, MA.add_mul, T, T, T) isa MA.IsMutable
        @test MA.mutability(T, MA.add_mul, T, T, T, T) isa MA.IsMutable

        t(n) = convert(T, n)
        a = t(5)
        b = t(9)
        c = t(3)
        d = t(20)
        buf = t(24)

        @test MA.add_mul_to!(a, b, c, d) == t(69)
        @test a == t(69)
        a = t(5)
        @test MA.add_mul!(b, c, d) == t(69)
        @test b == t(69)
        b = t(9)

        @test MA.add_mul_buf_to!(buf, a, b, c, d) == t(69)
        @test a == t(69)
        @test MA.add_mul_buf!(buf, b, c, d) == t(69)
        @test b == t(69)

        a = t(148)
        b = t(16)
        c = t(17)
        d = t(42)
        buf = t(56)
        @test MA.add_mul!(a, b, c) == t(420)
        @test a == t(420)
        a = t(148)
        @test MA.add_mul_buf_to!(buf, d, a, b, c) == t(420)
        @test d == t(420)
        @test MA.add_mul_buf!(buf, a, b, c) == t(420)
        @test a == t(420)
    end
end

function int_zero_test(::Type{T}) where T
    @testset "zero!" begin
        @test MA.mutability(T, zero, T) isa MA.IsMutable

        t(n) = convert(T, n)
        a = t(5)
        @test MA.zero!(a) == t(0)
        @test a == t(0)
        @test iszero(a)
    end
end

function int_one_test(::Type{T}) where T
    @testset "one!" begin
        @test MA.mutability(T, one, T) isa MA.IsMutable

        t(n) = convert(T, n)
        a = t(5)
        @test MA.one!(a) == t(1)
        @test a == t(1)
        @test isone(a)
    end
end

const int_tests = Dict(
    "int_add" => int_add_test,
    "int_mul" => int_mul_test,
    "int_add_mul" => int_add_mul_test,
    "int_zero" => int_zero_test,
    "int_one" => int_one_test
)

@test_suite int
