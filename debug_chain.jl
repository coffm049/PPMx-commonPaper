using JLD2
using DataFrames
using CSV
using StatsBase

# Load the stdPPmx-female chain
data = load("analyses/output/stdPPmx-female/stdPPmxTot.jld2")
sim = data[:sim]
println("Number of draws: ", length(sim))

if length(sim) > 0
    first_draw = sim[1]
    println("First draw keys: ", collect(keys(first_draw)))
    println("Has :C: ", :C in keys(first_draw))
    println("Has :prior_mean_beta: ", :prior_mean_beta in keys(first_draw))
    println("Has :lik_params: ", :lik_params in keys(first_draw))
    
    if :C in keys(first_draw)
        C_vec = first_draw[:C]
        println("C vector length: ", length(C_vec))
        println("C vector first 10: ", C_vec[1:min(10, length(C_vec))])
        println("Unique C values: ", unique(C_vec))
    end
    
    if :lik_params in keys(first_draw)
        lp = first_draw[:lik_params]
        println("lik_params type: ", typeof(lp))
        println("lik_params keys: ", keys(lp))
        if :beta in keys(lp)
            beta = lp[:beta]
            println("beta type: ", typeof(beta))
            println("beta size: ", size(beta))
        end
    end
end

# Also check the other chain
data2 = load("analyses/output/stdPPmx-male/stdPPmxTot.jld2")
sim2 = data2[:sim]
println("\nMale chain - Num draws: ", length(sim2))

if length(sim2) > 0
    first_draw2 = sim2[1]
    println("Male first C: ", get(first_draw2, :C, "NOT FOUND"))
end

# Check the archive file
println("\n\nChecking archive file...")
try
    data3 = load("analyses/archive/subLista1.0b1.0m1.0_1.0.jld2")
    sim3 = data3[:sim]
    println("Archive - Num draws: ", length(sim3))
    if length(sim3) > 0
        first_draw3 = sim3[1]
        println("Archive first C: ", get(first_draw3, :C, "NOT FOUND"))
        println("Archive first keys: ", collect(keys(first_draw3)))
    end
catch e
    println("Error loading archive: ", e)
end