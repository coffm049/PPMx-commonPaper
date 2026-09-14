# Quick diagnostic: check what keys are in the v2.0 chain .jld2 files
# Run on HPC from analyses/ directory:
#   julia --project=../../simulations/ check_keys.jl

using JLD2

for fname in ["output/openTotalFull2-v2.jld2", "output/openTotalFull3-v2.jld2"]
    println("="^60)
    println("File: $fname")
    if !isfile(fname)
        println("  NOT FOUND")
        continue
    end
    data = jldopen(fname, "r") do file
        keys(file) |> collect
    end
    println("  Top-level keys: $data")
    
    # Load sim and check its structure
    sim = jldopen(fname, "r") do file
        file["sim"]
    end
    println("  sim type: $(typeof(sim))")
    println("  sim length: $(length(sim))")
    if length(sim) > 0
        println("  Keys in first sim element: $(collect(keys(sim[1])))")
        s1 = sim[1]
        for k in collect(keys(s1))
            println("    $k :: $(typeof(s1[k]))")
        end
    end
end

# Also check the full-data chains (from 01fullDat-ppmxTot)
for fname in ["output/openTotalFullDatamcmc1-v2.jld2", "output/openTotalFullDatamcmc2-v2.jld2"]
    println("="^60)
    println("File: $fname")
    if !isfile(fname)
        println("  NOT FOUND")
        continue
    end
    sim = jldopen(fname, "r") do file
        file["sim"]
    end
    println("  sim type: $(typeof(sim))")
    println("  sim length: $(length(sim))")
    if length(sim) > 0
        println("  Keys in first sim element: $(collect(keys(sim[1])))")
        s1 = sim[1]
        for k in collect(keys(s1))
            println("    $k :: $(typeof(s1[k]))")
        end
    end
end
