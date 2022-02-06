for op in (+, -, *, //)
    for (a,b) in (
            (2//3, 5), (2, 3//5), (2//3, 5//7),
            (big(2)//3, 5), (big(2), 3//5), (big(2)//3, 5//7),
        )
        @test MA.operate_to!!(MA.copy_if_mutable(op(a,b)), op, a, b) == op(a, b)
        @test MA.operate_to!!(MA.copy_if_mutable(op(b,a)), op, b, a) == op(b, a)
    end
end

op = //
for (a,b) in (
    (2,3), (big(2), 3), (2, big(3))
    )
    @test MA.operate_to!!(MA.copy_if_mutable(op(a,b)), op, a, b) == op(a,b)
    @test MA.operate_to!!(MA.copy_if_mutable(op(b,a)), op, b, a) == op(b,a)
end
