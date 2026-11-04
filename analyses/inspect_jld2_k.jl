#!/usr/bin/env julia
# Usage:
#   julia --project=~/software/ProductPartitionModels.jl/simulations inspect_jld2_k.jl FILE...
#
# Read-only: opens each JLD2 file and inspects its `sim` draws. Prints n,
# number of retained draws, modal K, and the full K distribution.

using JLD2, StatsBase

function chain_info(path::AbstractString)
    jldopen(path, "r") do io
        if !haskey(io, "sim")
            return (n=nothing, draws=0, modeK=nothing, dist=Dict{Int,Int}(),
                    note="no sim key; keys=$(collect(keys(io)))")
        end
        sim = io["sim"]
        ks = Int[]
        n = nothing
        for s in sim
            s isa AbstractDict || continue
            haskey(s, :C) || continue
            n === nothing && (n = length(s[:C]))
            push!(ks, length(unique(Int.(s[:C]))))
        end
        isempty(ks) && return (n=n, draws=0, modeK=nothing, dist=Dict{Int,Int}(),
                               note="sim has no :C draws")
        return (n=n, draws=length(ks), modeK=mode(ks),
                dist=sort(collect(countmap(ks))), note="")
    end
end

for path in ARGS
    try
        info = chain_info(path)
        println(path)
        println("  n=", info.n, "  draws=", info.draws,
                "  modeK=", info.modeK, "  dist=", info.dist)
        isempty(info.note) || println("  NOTE: ", info.note)
    catch e
        println(path, "\n  ERROR: ", sprint(showerror, e))
    end
end
