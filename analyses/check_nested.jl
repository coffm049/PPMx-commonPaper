# Diagnostic: nested key structure inside saved draws.
# The top-level keys are Symbols and fine; postPred fails on NESTED lookups
# (sims[ii][:lik_params][k][:mu], sims[ii][:baseline][:mu0]).
# Run on HPC from analyses/:
#   julia --project=../../simulations/ check_nested.jl
# Paste the full output back.
using JLD2

files = [
    "output/openTotalFull2-v2.jld2",
    "output/openTotalFull3-v2.jld2",
    "output/openTotalFullDatamcmc1-v2.jld2",
    "output/openTotalFullDatamcmc2-v2.jld2",
]

for fname in files
    println("="^70)
    println("File: $fname")
    sim = jldopen(fname, "r") do f
        f["sim"]
    end
    full = filter(s -> s isa AbstractDict && haskey(s, :C) && haskey(s, :lik_params), sim)
    println("  n draws: $(length(sim)), full draws: $(length(full))")
    s = full[1]
    lp = s[:lik_params]
    println("  :lik_params type: $(typeof(lp)), length: $(length(lp))")
    println("  lik_params[1] type: $(typeof(lp[1]))")
    println("  lik_params[1] keys: $(collect(keys(lp[1])))")
    println("  lik_params[1] keytypes: $(unique(typeof.(collect(keys(lp[1])))))")
    b = s[:baseline]
    println("  :baseline type: $(typeof(b))")
    println("  baseline keys: $(collect(keys(b)))")
    println("  baseline keytypes: $(unique(typeof.(collect(keys(b)))))")
    n_mu = count(s -> (s[:lik_params] isa AbstractVector && length(s[:lik_params]) > 0 &&
                       s[:lik_params][1] isa AbstractDict && haskey(s[:lik_params][1], :mu)), full)
    n_mu0 = count(s -> (s[:baseline] isa AbstractDict && haskey(s[:baseline], :mu0)), full)
    println("  full draws with lik_params[1][:mu]: $n_mu / $(length(full))")
    println("  full draws with baseline[:mu0]: $n_mu0 / $(length(full))")
end
