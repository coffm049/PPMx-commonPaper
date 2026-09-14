# Diagnostic: key structure across ALL draws in each v2.0 chain file.
# Run on HPC from analyses/:
#   julia --project=../../simulations/ check_draws.jl
# Paste the full output back.
using JLD2

files = [
    "output/openTotalFull2-v2.jld2",
    "output/openTotalFull3-v2.jld2",
    "output/openTotalFullDatamcmc1-v2.jld2",
    "output/openTotalFullDatamcmc2-v2.jld2",
    "output/stdPPmxTot.jld2",
]

for fname in files
    println("="^70)
    println("File: $fname")
    if !isfile(fname)
        println("  NOT FOUND")
        continue
    end
    local sim
    try
        sim = jldopen(fname, "r") do f
            f["sim"]
        end
    catch e
        println("  LOAD ERROR: $e")
        continue
    end
    println("  n draws: $(length(sim))")
    ks = Dict{Vector{Symbol},Int}()
    for s in sim
        k = sort(collect(keys(s)))
        ks[k] = get(ks, k, 0) + 1
    end
    for (k, n) in sort(collect(ks); by=x->x[2], rev=true)
        println("  $n draws with keys: $k")
    end
    println("  all have :prior_mean_beta: $(all(s -> haskey(s, :prior_mean_beta), sim))")
    println("  all have :C: $(all(s -> haskey(s, :C), sim))")
    i = findfirst(s -> haskey(s, :prior_mean_beta), sim)
    if i !== nothing
        b = sim[i][:prior_mean_beta]
        println("  :prior_mean_beta eltype: $(typeof(b)), length: $(length(b))")
    end
end
