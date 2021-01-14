using Documenter, MutableArithmetics

makedocs(;
    modules = [MutableArithmetics],
    # See https://github.com/JuliaDocs/Documenter.jl/issues/868
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        assets = String[],
    ),
    # See https://github.com/jump-dev/JuMP.jl/issues/1576
    strict = true,
    pages = ["Home" => "index.md"],
    repo = "https://github.com/jump-dev/MutableArithmetics.jl/blob/{commit}{path}#L{line}",
    sitename = "MutableArithmetics",
    authors = "Gilles Peiffer, Benoît Legat, and Sascha Timme",
)

deploydocs(; repo = "github.com/jump-dev/MutableArithmetics.jl.git")
