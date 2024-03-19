function mul!!(a::Rational{S}, b::Rational{T})
    if # S can be mutated to `*(::S, ::T)`
        mul!(a.num, b.num)
        mul!(a.den, b.den)
        return a
    else
        return a * b
    end
end
