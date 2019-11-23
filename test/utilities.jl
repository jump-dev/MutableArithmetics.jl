function alloc_test(f, n)
    f() # compile
    @test n == @allocated f()
end
