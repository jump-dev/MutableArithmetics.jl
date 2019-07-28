using Documenter, MutableArithmetics

makedocs(;
    modules=[MutableArithmetics],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/Peiffap/MutableArithmetics.jl/blob/{commit}{path}#L{line}",
    sitename="MutableArithmetics.jl",
    authors="Gilles Peiffer, Benoît Legat, Sascha Timme",
    assets=String[],
)

deploydocs(;
    repo="github.com/Peiffap/MutableArithmetics.jl",
)
