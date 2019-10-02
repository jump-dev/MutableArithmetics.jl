macro test_suite(setname, subsets=false)
    testname = Symbol(string(setname) * "_test")
    testdict = Symbol(string(testname) * "s")
    if subsets
        runtest = :( f(T, exclude) )
    else
        runtest = :( f(T) )
    end
    esc(:(
        function $testname(::Type{T},
                           exclude::Vector{String} = String[]) where T
            for (name,f) in $testdict
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
