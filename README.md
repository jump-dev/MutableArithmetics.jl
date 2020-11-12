# MutableArithmetics.jl

[![Stable][docs-stable-img]][docs-stable-url]
[![Dev][docs-latest-img]][docs-latest-url]
[![Build Status](https://github.com/jump-dev/MutableArithmetics.jl/workflows/CI/badge.svg?branch=master)](https://github.com/jump-dev/MutableArithmetics.jl/actions?query=workflow%3ACI)
[![Codecov](https://codecov.io/gh/jump-dev/MutableArithmetics.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jump-dev/MutableArithmetics.jl)

**MutableArithmetics** (MA for short) is a [Julia](http://julialang.org) package which allows:
*   for mutable types to implement mutable arithmetics;
*   for algorithms that could exploit mutable arithmetics to exploit them while still being completely generic.

While in some cases, similar features have been included in packages
idiosyncratically, the goal of this package is to provide a generic interface to
allow anyone to make use of mutability when desired.

The package allows a given type to declare itself mutable through the
`MA.mutability` trait.
Then the user can use the `MA.operate!` function to write generic code
that works for arbitrary type while exploiting mutability of the type
if possible. More precisely:

* The `MA.operate!(op::Function, x, args...)` redirects to `op(x, args...)`
  if `x` is not mutable or if the result of the operation cannot be stored in `x`.
  Otherwise, it redirects to `MA.mutable_operate!(op, x, args...)`.
* `MA.mutable_operate!(op::Function, x, args...)` stores the result of the
  operation in `x`. It is a `MethodError` if `x` is not mutable or if the
  result of the operation cannot be stored in `x`.

So from a generic code, `MA.operate!` can be used when the value of `x` is not
used anywhere else to recycle it if possible. This allows the code to both
work for mutable and for non-mutable type.

When the type is known to be mutable, `MA.mutable_operate!` can be used to make
sure the operation is done in-place. If it is not possible, the `MethodError`
allows to easily fix the issue while `MA.operate!` would have silently fallen
back to the non-mutating function.

In conclusion, the distinction between `MA.operate!` and `MA.mutable_operate!`
allows to cover all use case while having an universal convention accross all
operations.

The following types implement the MutableArithmetics API:
* The API is implemented for `Base.BigInt` in `src/bigint.jl`.
* The API is implemented for `Base.BigFloat` in `src/bigfloat.jl`.
* The API is implemented for `Base.Array` in `src/linear_algebra.jl`.
* The interface for multivariate polynomials [MultivariatePolynomials](https://github.com/JuliaAlgebra/MultivariatePolynomials.jl)
  as well as its two implementations [DynamicPolynomials](https://github.com/JuliaAlgebra/DynamicPolynomials.jl)
  and [TypedPolynomials](https://github.com/JuliaAlgebra/TypedPolynomials.jl).
* The scalar and quadratic functions used to define an Optimization Program in
  [MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl).
* The scalar and quadratic expressions used to model optimization in
  [JuMP](https://github.com/jump-dev/JuMP.jl).

The algorithms from the following libraries use the MutableArithmetics API
to exploit the mutability of the type when possible:
* The multivariate polynomials implemented in [MultivariatePolynomials](https://github.com/JuliaAlgebra/MultivariatePolynomials.jl),
  [DynamicPolynomials](https://github.com/JuliaAlgebra/DynamicPolynomials.jl)
  and [TypedPolynomials](https://github.com/JuliaAlgebra/TypedPolynomials.jl)
  work with any type and exploit the mutability of the type through the MA API.

In addition, the implementation of the following functionalities available from
`Base` are reimplemented on top of the MA API:
* Matrix-matrix, matrix-vector and array-scalar multiplication including
  `SparseArrays.AbstractSparseArray`, `LinearAlgebra.Adjoint`,
  `LinearAlgebra.Transpose`, `LinearAlgebra.Symmetric`.
* `Base.sum`, `LinearAlgebra.dot` and `LinearAlgebra.diagm`.

These methods are reimplemented in this package for several reasons:
* The implementation in `Base` does not exploit the mutability of the type
  (except for `sum(::Vector{BigInt})` which has a specialized method) and
  are hence much slower.
* Some implementations in `Base` assume the following for the types `S`, `T` used satisfy:
  - `typeof(zero(T)) == T`, `typeof(one(T)) == T`, `typeof(S + T) == promote_type(S, T)`
    or `typeof(S * T) == promote_type(S, T)` which is not true for
    instance if `T` is a polynomial variable or the decision variable of an
    optimization model.
  - The multiplication between elements of type `S` and `T` is commutative which
    is not true for matrices or non-commutative polynomial variables.

The trait defined in this package cannot make the methods for the functions
defined in Base to be dispatched to the implementations of this package.
For these to be used for a given type, it needs to inherit from `MA.AbstractMutable`.
Not that subtypes of `MA.AbstractMutable` are not necessarily mutable,
for instance, polynomial variables and the decision variable of an optimization
model are subtypes of `MA.AbstractMutable` but are not mutable.
The only purpose of this abstract type is to have `Base` methods to be dispatched
to the implementations of this package. See `src/dispatch.jl` for more details.

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
  memory estimate:  48 bytes
  allocs estimate:  3
  --------------
  minimum time:     917.819 μs (0.00% GC)
  median time:      999.239 μs (0.00% GC)
  mean time:        1.042 ms (0.00% GC)
  maximum time:     2.319 ms (0.00% GC)
  --------------
  samples:          4791
  evals/sample:     1
```

> This package started out as a GSoC '19 project.

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-stable-url]: https://jump.dev/MutableArithmetics.jl/stable
[docs-latest-url]: https://jump.dev/MutableArithmetics.jl/latest
