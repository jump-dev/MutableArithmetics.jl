# hygiene.jl
# Make sure that our macros have good hygiene

module M
using Test
import MutableArithmetics

macro _rewrite(expr)
    variable, code = MutableArithmetics.rewrite(expr)
    return quote
        $code
        $variable
    end
end

# Don't include this in `@test` to make sure it is in global scope
x = MutableArithmetics.@rewrite sum(i for i = 1:10)
@test x == 55
x = @_rewrite sum(i for i = 1:10)
@test x == 55

x = MutableArithmetics.@rewrite sum(i for i in 1:10 if isodd(i))
@test x == 25
x = @_rewrite sum(i for i in 1:10 if isodd(i))
@test x == 25

x = MutableArithmetics.@rewrite sum(i * j for i = 1:4 for j ∈ 1:4 if i == j)
@test x == 30
x = @_rewrite sum(i * j for i = 1:4 for j ∈ 1:4 if i == j)
@test x == 30

end
