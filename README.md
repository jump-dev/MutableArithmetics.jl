# MutableArithmetics.jl

[![Stable][docs-stable-img]][docs-stable-url]
[![Dev][docs-latest-img]][docs-latest-url]
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
An example implementation of this interface is given in `src/bigint.jl`.

## Documentation

- [**STABLE**][docs-stable-url] &mdash; **most recently tagged version of the documentation.**
- [**LATEST**][docs-latest-url] &mdash; *in-development version of the documentation.*

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
```

The default implementation `LinearAlgebra.generic_matvecmul!` does not exploit
the mutability of `BigInt` is quite slow and allocates a lot:
```julia
using LinearAlgebra
trial = @benchmark LinearAlgebra.mul!($c2, $A2, $b2)
display(trial)

# output

BenchmarkTools.Trial:
  memory estimate:  3.67 MiB
  allocs estimate:  238775
  --------------
  minimum time:     6.116 ms (0.00% GC)
  median time:      6.263 ms (0.00% GC)
  mean time:        11.711 ms (27.72% GC)
  maximum time:     122.627 ms (70.45% GC)
  --------------
  samples:          429
  evals/sample:     1
```

In `MA.mutable_operate_to(::Vector, ::typeof(*), ::Matrix, ::Vector)`, we
exploit the mutability of `BigInt` through the MutableArithmetics API.
This provides a significant speedup and a drastic reduction of memory usage:
```julia
trial2 = @benchmark MA.mul_to!($c2, $A2, $b2)
display(trial2)

BenchmarkTools.Trial:
  memory estimate:  168 bytes
  allocs estimate:  9
  --------------
  minimum time:     928.306 μs (0.00% GC)
  median time:      933.144 μs (0.00% GC)
  mean time:        952.015 μs (0.00% GC)
  maximum time:     1.910 ms (0.00% GC)
  --------------
  samples:          5244
  evals/sample:     1
```

> This package started out as a GSoC '19 project.

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-stable-url]: https://juliaopt.github.io/MutableArithmetics.jl/stable
[docs-latest-url]: https://juliaopt.github.io/MutableArithmetics.jl/latest
