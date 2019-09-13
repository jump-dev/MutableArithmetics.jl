# MutableArithmetics.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaOpt.github.io/MutableArithmetics.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaOpt.github.io/MutableArithmetics.jl/dev)
[![Build Status](https://travis-ci.com/JuliaOpt/MutableArithmetics.jl.svg?branch=master)](https://travis-ci.com/JuliaOpt/MutableArithmetics.jl)
[![Codecov](https://codecov.io/gh/JuliaOpt/MutableArithmetics.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaOpt/MutableArithmetics.jl)
[![Coveralls](https://coveralls.io/repos/github/JuliaOpt/MutableArithmetics.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaOpt/MutableArithmetics.jl?branch=master)

**MutableArithmetics** is a [Julia](http://julialang.org) package which allows:
*   for mutable types to implement mutable arithmetics;
*   for algorithms that could exploit mutable arithmetics to exploit them while still being completely generic.

While in some cases, similar features have been included in packages
idiosyncratically, the goal of this package is to provide a generic interface to
 allow anyone to make use of mutability when desired.

The package allows users to indicate when a mutable implementation of a certain
method is available through the use of so-called *traits*, as well as providing
a simple way forusers to make operations fall back to these implementations.
Examples of implementations of this interface are given in the `examples`
folder.

## Quick Example & Benchmark

```julia
using BenchmarkTools
using MutableArithmetics
const MA = MutableArithmetics

n = 200
A = rand(-10:10, n, n)
b = rand(-10:10, n)
c = rand(-10:10, n)

# MA.mul works for arbitrary types
MA.mul(A, b)

A2 = big.(A)
b2 = big.(b)
c2 = big.(c)


println("Default performance: ")
trial = @benchmark MA.mul_to!($c2, $A2, $b2)
display(trial)

# Define MutableArithmetics API for numbers
MA.mutability(::Type{BigInt}, ::typeof(MA.zero!)) = MA.IsMutable()
MA.zero_impl!(x::BigInt) = Base.GMP.MPZ.set_si!(x, 0)
MA.mutability(::Type{BigInt}, ::typeof(MA.one!)) = MA.IsMutable()
MA.one_impl!(x::BigInt) = Base.GMP.MPZ.set_si!(x, 1)
MA.mutability(::Type{BigInt}, ::typeof(MA.mul_to!), ::Type{BigInt}, ::Type{BigInt}) = MA.IsMutable()
MA.mul_to_impl!(x::BigInt, a::BigInt, b::BigInt) = Base.GMP.MPZ.mul!(x, a, b)
MA.mutability(::Type{BigInt}, ::typeof(MA.add_to!), ::Type{BigInt}, ::Type{BigInt}) = MA.IsMutable()
MA.add_to_impl!(x::BigInt, a::BigInt, b::BigInt) = Base.GMP.MPZ.add!(x, a, b)
MA.muladd_buf_impl!(buf::BigInt, a::BigInt, b::BigInt, c::BigInt) = Base.GMP.MPZ.add!(a, Base.GMP.MPZ.mul!(buf, b, c))

println("MA performance after defining the MA interface: ")
trial2 = @benchmark MA.mul_to!($c2, $A2, $b2)
display(trial2)
```

> This package started out as a GSoC '19 project.
