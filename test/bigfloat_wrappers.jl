@testset "MPFR wrappers" begin
    # call MA.operate_to!(out, op, args...) and compare with expected. If Base throws for
    # the inputs, assert MA throws the same kind of error.
    function check_op_matches_expected(
        op,
        args...;
        out = BigFloat(),
        expected = missing,
    )
        # preserve originals for mutation checks
        old_args = MA.copy_if_mutable.(args)

        # Test operate_to! into an output
        local result
        success = try
            result = MA.operate_to!(out, op, args...)
            true
        catch e
            @test_throws typeof(e) op(args...)
            false
        end
        if success
            @test result === out # check return value
            if ismissing(expected)
                success = try
                    expected = op(args...)
                    true
                catch e
                    @test false
                    false
                end
            end
            if success
                if out isa Real
                    @test out ≈ expected atol=1e-13
                else
                    @test length(out) == length(expected)
                    for (outᵢ, expectedᵢ) in zip(out, expected)
                        @test outᵢ ≈ expectedᵢ atol=1e-13
                    end
                end
            end
        end

        # Ensure operate_to! didn't mutate the input arguments
        @test old_args == args
    end

    setprecision(BigFloat, 64) do
        check_op_matches_expected(copy, big"12.")
        check_op_matches_expected(copy, -12)
        check_op_matches_expected(copy, UInt(12))
        check_op_matches_expected(copy, 12.0f0)
        check_op_matches_expected(copy, 12.0)

        check_op_matches_expected(ldexp, -5, 3; expected = ldexp(-5.0, 3))
        check_op_matches_expected(ldexp, UInt(5), 3; expected = ldexp(5.0, 3))
        check_op_matches_expected(ldexp, big"5.", -3)
        check_op_matches_expected(ldexp, big"5.", UInt(3))

        check_op_matches_expected(+, big"1.5", big"2.5")
        check_op_matches_expected(+, big"1.5", 10)
        check_op_matches_expected(+, big"1.5", 0x23)
        check_op_matches_expected(+, big"1.5", 2.5)
        check_op_matches_expected(+, big"1.5", 2.5f0)

        check_op_matches_expected(-, big"1.5", big"2.5")
        check_op_matches_expected(-, big"1.5", Int32(-42))
        check_op_matches_expected(-, big"1.5", 0x6525)
        check_op_matches_expected(-, big"1.5", 17.0)
        check_op_matches_expected(-, Int16(-62), big"1.5")
        check_op_matches_expected(-, 0x73764fa, big"1.5")
        check_op_matches_expected(-, 24.0, big"1.5")

        check_op_matches_expected(*, big"63.6", big"91.")
        check_op_matches_expected(*, big"63.6", Int32(-52))
        check_op_matches_expected(*, big"63.6", 0x53d2a)
        check_op_matches_expected(*, big"63.6", 8.0f3)

        for dividend in (58, 0)
            check_op_matches_expected(/, big"5352.5", BigFloat(dividend))
            check_op_matches_expected(/, big"5352.5", Int(dividend))
            check_op_matches_expected(/, big"5352.5", UInt(dividend))
            check_op_matches_expected(/, big"5352.5", Float64(dividend))
            check_op_matches_expected(/, 5352, BigFloat(dividend))
            check_op_matches_expected(/, 0x452, BigFloat(dividend))
            check_op_matches_expected(/, 5352.5, BigFloat(dividend))
        end

        check_op_matches_expected(sqrt, big"646.4")
        check_op_matches_expected(sqrt, 0x12)
        check_op_matches_expected(cbrt, big"652.6")
        VERSION ≥ v"1.10" && check_op_matches_expected(fourthroot, big"746.3")

        check_op_matches_expected(factorial, 0xc)

        check_op_matches_expected(fma, big"1.2", big"2.3", big"0.7")

        check_op_matches_expected(hypot, big"3.", big"4.")

        check_op_matches_expected(log, big"2.5")
        check_op_matches_expected(log, big"-53.")
        check_op_matches_expected(log, 0x623)
        check_op_matches_expected(log, 0x0)
        check_op_matches_expected(log2, big"8.0")
        check_op_matches_expected(log2, big"0.")
        check_op_matches_expected(log10, big"100.0")
        check_op_matches_expected(log10, big"-17.")
        check_op_matches_expected(log1p, big"0.001")
        check_op_matches_expected(log1p, big"-13.")

        check_op_matches_expected(exp, big"1.2")
        check_op_matches_expected(exp2, big"3.0")
        check_op_matches_expected(exp10, big"2.0")
        check_op_matches_expected(expm1, big"0.001")

        check_op_matches_expected(^, big"17.", big"13.")
        check_op_matches_expected(^, big"17.", 13)
        check_op_matches_expected(^, big"17.", 0x13)

        check_op_matches_expected(cos, big"0.5")
        check_op_matches_expected(sin, big"0.5")
        check_op_matches_expected(tan, big"0.5")
        if VERSION ≥ v"1.10"
            check_op_matches_expected(cospi, big"0.5")
            check_op_matches_expected(sinpi, big"0.5")
            check_op_matches_expected(tanpi, big"0.125")
            check_op_matches_expected(tanpi, big"0.5")
            check_op_matches_expected(cosd, big"60.0")
            check_op_matches_expected(sind, big"30.0")
            check_op_matches_expected(tand, big"30.0")
            check_op_matches_expected(tand, big"90.")
        end
        check_op_matches_expected(
            sincos,
            big"0.5";
            out = (BigFloat(), BigFloat()),
        )
        check_op_matches_expected(sec, big"0.7")
        check_op_matches_expected(csc, big"0.7")
        check_op_matches_expected(cot, big"0.7")
        check_op_matches_expected(acos, big"0.5")
        check_op_matches_expected(asin, big"0.5")
        check_op_matches_expected(atan, big"0.5")
        check_op_matches_expected(atan, big"0.5", big"0.9")
        if VERSION ≥ v"1.10"
            check_op_matches_expected(acosd, big"0.5")
            check_op_matches_expected(asind, big"0.5")
            check_op_matches_expected(atand, big"0.5")
            check_op_matches_expected(atand, big"0.5", big"0.9")
        end

        check_op_matches_expected(cosh, big"0.5")
        check_op_matches_expected(sinh, big"0.5")
        check_op_matches_expected(tanh, big"0.5")
        check_op_matches_expected(sech, big"0.5")
        check_op_matches_expected(csch, big"0.5")
        check_op_matches_expected(coth, big"0.5")
        check_op_matches_expected(acosh, big"2.0")
        check_op_matches_expected(asinh, big"0.5")
        check_op_matches_expected(atanh, big"0.3")

        for rm in (
            RoundNearest,
            RoundUp,
            RoundDown,
            RoundToZero,
            RoundNearestTiesAway,
        )
            check_op_matches_expected(round, big"24.", rm)
        end

        check_op_matches_expected(
            modf,
            big"3.1415";
            out = (BigFloat(), BigFloat()),
        )
        check_op_matches_expected(rem, big"7.5", big"2.3")
        check_op_matches_expected(rem, big"7.5", big"2.3", RoundNearest)
        check_op_matches_expected(min, big"1.2", big"2.3")
        check_op_matches_expected(max, big"1.2", big"2.3")
        return check_op_matches_expected(copysign, big"-1.2", big"2.0")
    end
end
