macro test_suite(setname, subsets=false)
    testname = Symbol(string(setname) * "_test")
    testdict = Symbol(string(testname) * "s")
    if subsets
        runtest = :( f(args...; exclude = exclude) )
    else
        runtest = :( f(args...) )
    end
    esc(:(
        function $testname(args...;
                           exclude::Vector{String} = String[]) where T
            for (name, f) in $testdict
                if name in exclude
                    continue
                end
                @testset "$name" begin
                    $runtest
                end
            end
        end
    ))
end
