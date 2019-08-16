
@testset "Basic arithmetic" begin
    @testset "add_to! / add!" begin
        a = BigInt(5)
        b = BigInt(28)
        c = BigInt(41)

        @test MA.add_to!(a, b, c) == BigInt(69) && a == BigInt(5)
        @test MA.add!(b, c) == BigInt(69) && b == BigInt(28)
        @test MA.mutability(BigInt, MA.add_to!, BigInt, BigInt) isa MA.NotMutable
        @test MA.mutability(BigInt, MA.add!, BigInt) isa MA.NotMutable

        MA.mutability(::Type{BigInt}, ::typeof(MA.add_to!), ::Type{BigInt}, ::Type{BigInt}) = MA.IsMutable()
        MA.add_to_impl!(x::BigInt, y::BigInt, z::BigInt) = Base.GMP.MPZ.add!(x, y, z)

        @test MA.add_to!(a, b, c) == BigInt(69) && a == BigInt(69)
        @test MA.mutability(BigInt, MA.add_to!, BigInt, BigInt) isa MA.IsMutable
        @test MA.mutability(BigInt, MA.add!, BigInt) isa MA.IsMutable

        a = BigInt(165)
        b = BigInt(255)

        # Using the generic fallback.
        @test MA.add!(a, b) == BigInt(420) && a == BigInt(420)
        # Using a specific method.
        MA.add_impl!(x::BigInt, y::BigInt) = Base.GMP.MPZ.add!(x, x, y)
        a = BigInt(165)
        b = BigInt(255)

        @test MA.add!(a, b) == BigInt(420) && a == BigInt(420)
    end

    @testset "mul_to! / mul!" begin
        a = BigInt(5)
        b = BigInt(23)
        c = BigInt(3)

        @test MA.mul_to!(a, b, c) == BigInt(69) && a == BigInt(5)
        @test MA.mul!(b, c) == BigInt(69) && b == BigInt(23)
        @test MA.mutability(BigInt, MA.mul_to!, BigInt, BigInt) isa MA.NotMutable
        @test MA.mutability(BigInt, MA.mul!, BigInt) isa MA.NotMutable

        MA.mutability(::Type{BigInt}, ::typeof(MA.mul_to!), ::Type{BigInt}, ::Type{BigInt}) = MA.IsMutable()
        MA.mul_to_impl!(x::BigInt, y::BigInt, z::BigInt) = Base.GMP.MPZ.mul!(x, y, z)

        @test MA.mul_to!(a, b, c) == BigInt(69) && a == BigInt(69)
        @test MA.mutability(BigInt, MA.mul_to!, BigInt, BigInt) isa MA.IsMutable
        @test MA.mutability(BigInt, MA.mul!, BigInt) isa MA.IsMutable

        a = BigInt(15)
        b = BigInt(28)

        # Using the generic fallback.
        @test MA.mul!(a, b) == BigInt(420) && a == BigInt(420)
        # Using a specific method.
        MA.mul_impl!(x::BigInt, y::BigInt) = Base.GMP.MPZ.mul!(x, x, y)
        a = BigInt(15)
        b = BigInt(28)

        @test MA.mul!(a, b) == BigInt(420) && a == BigInt(420)
    end

    @testset "muladd_to! / muladd! / muladd_buf_to! /muladd_buf!" begin
        a = BigInt(5)
        b = BigInt(9)
        c = BigInt(3)
        d = BigInt(20)
        buf = BigInt(24)

        MA.mutability(::Type{BigInt}, ::typeof(MA.add_to!), ::Type{BigInt}, ::Type{BigInt}) = MA.NotMutable()

        @test MA.muladd_to!(a, b, c, d) == BigInt(69) && a == BigInt(5)
        @test MA.muladd!(b, c, d) == BigInt(69) && b == BigInt(9)
        @test MA.muladd_buf_to!(buf, a, b, c, d) == BigInt(69) && buf == BigInt(24) && a == BigInt(5)
        @test MA.muladd_buf!(buf, b, c, d) == BigInt(69) && b == BigInt(9) && buf == BigInt(24)
        @test MA.mutability(BigInt, MA.muladd_to!, BigInt, BigInt, BigInt) isa MA.NotMutable
        @test MA.mutability(BigInt, MA.muladd!, BigInt, BigInt) isa MA.NotMutable
        @test MA.mutability(BigInt, MA.muladd_buf_to!, BigInt, BigInt, BigInt, BigInt) isa MA.NotMutable
        @test MA.mutability(BigInt, MA.muladd_buf!, BigInt, BigInt, BigInt) isa MA.NotMutable

        MA.mutability(::Type{BigInt}, ::typeof(MA.add_to!), ::Type{BigInt}, ::Type{BigInt}) = MA.IsMutable()
        function MA.muladd_to_impl!(w::BigInt, x::BigInt, y::BigInt, z::BigInt)
            Base.GMP.MPZ.add!(w, x, Base.GMP.MPZ.mul!(BigInt(), y, z))
        end

        @test MA.muladd_to!(a, b, c, d) == BigInt(69) && a == BigInt(69)
        @test MA.mutability(BigInt, MA.muladd_to!, BigInt, BigInt, BigInt) isa MA.IsMutable
        @test MA.mutability(BigInt, MA.muladd!, BigInt, BigInt) isa MA.IsMutable
        @test MA.mutability(BigInt, MA.muladd_buf!, BigInt, BigInt, BigInt) isa MA.IsMutable

        a = BigInt(148)
        b = BigInt(16)
        c = BigInt(17)

        # Using the generic fallback.
        @test MA.muladd!(a, b, c) == BigInt(420) && a == BigInt(420)
        # Using a specific method.
        function MA.muladd_impl!(x::BigInt, y::BigInt, z::BigInt)
            Base.GMP.MPZ.add!(x, x, Base.GMP.MPZ.mul!(BigInt(), y, z))
        end
        a = BigInt(148)
        b = BigInt(16)
        c = BigInt(17)

        @test MA.muladd!(a, b, c) == BigInt(420) && a == BigInt(420)

        a = BigInt(148)
        b = BigInt(16)
        c = BigInt(17)
        buf = BigInt(56)

        MA.muladd_buf_impl!(buf::BigInt, x::BigInt, y::BigInt, z::BigInt) = Base.GMP.MPZ.add!(x, Base.GMP.MPZ.mul!(buf, y, z))

        @test MA.muladd_buf!(buf, a, b, c) == BigInt(420) && buf == BigInt(272) && a == BigInt(420)

        a = BigInt(148)
        b = BigInt(16)
        c = BigInt(17)
        d = BigInt(42)
        buf = BigInt(56)

        MA.mul_to_impl!(x::BigInt, y::BigInt, z::BigInt) = Base.GMP.MPZ.mul!(x, y, z)
        MA.add_to_impl!(x::BigInt, y::BigInt, z::BigInt) = Base.GMP.MPZ.add!(x, y, z)

        @test MA.muladd_buf_to!(buf, d, a, b, c) == BigInt(420) && buf == BigInt(272) && d == BigInt(420)
    end

    @testset "zero! / one!" begin
        a = BigInt(5)
        @test MA.zero!(a) == BigInt(0) && a == BigInt(5)
        @test MA.mutability(BigInt, MA.zero!) isa MA.NotMutable
        MA.mutability(::Type{BigInt}, ::typeof(MA.zero!)) = MA.IsMutable()
        MA.zero_impl!(x::BigInt) = Base.GMP.MPZ.set_si!(x, 0)
        @test MA.zero!(a) == BigInt(0) && a == BigInt(0)
        @test MA.mutability(BigInt, MA.zero!) isa MA.IsMutable

        a = BigInt(5)
        @test MA.one!(a) == BigInt(1) && a == BigInt(5)
        @test MA.mutability(BigInt, MA.one!) isa MA.NotMutable
        MA.mutability(::Type{BigInt}, ::typeof(MA.one!)) = MA.IsMutable()
        MA.one_impl!(x::BigInt) = Base.GMP.MPZ.set_si!(x, 1)
        @test MA.one!(a) == BigInt(1) && a == BigInt(1)
        @test MA.mutability(BigInt, MA.one!) isa MA.IsMutable
    end
end
