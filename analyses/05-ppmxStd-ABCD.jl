using Pkg
Pkg.activate("../../simulations/")
using Random
using DelimitedFiles
using LinearAlgebra
using JLD2
using KernelDensity
using FreqTables
using DataFrames
using TexTables
using StatsBase
using Statistics
using StatsPlots
using Plots
using StatsModels
using Clustering
using CSV
using GLM
using TidierData
using MCMCChains
using Revise
using ProductPartitionModels

include("../code/utilities.jl")
include("../code/loadNClean.jl")
include("../code/analysis.jl")
include("../code/ppmxBaselines.jl")

# ---- load and clean the data (same pipeline as 01fullDat-ppmxTot.jl) ----
fullDf, standardization_params = loadNclean()
transform!(fullDf, [:ADHD1, :ADHD2, :ADHD3, :ADHD4] => ByRow((a1, a2, a3, a4) -> begin
  if a4 == 1
    4
  elseif a3 == 1
    3
  elseif a2 == 1
    2
  elseif a1 == 1
    1
  else
    0
  end
end) => :adhdLevel)
fullDf  = CSV.read("output/sampledDF.csv", DataFrame)

# ---- partition data ----
A1train = innerjoin(fullDf, CSV.read("../data/a1TrFl.csv", DataFrame, header= ["IID"]), on = "IID")
A1test = innerjoin(fullDf, CSV.read("../data/a1TeFl.csv", DataFrame, header= ["IID"]), on = "IID")
A2train = innerjoin(fullDf, CSV.read("../data/a2TrFl.csv", DataFrame, header= ["IID"]), on = "IID")
A2test = innerjoin(fullDf, CSV.read("../data/a2TeFl.csv", DataFrame, header= ["IID"]), on = "IID")
train = vcat(A1train[completecases(A1train), :], A2train[completecases(A2train), :])
test = vcat(A1test[completecases(A1test), :], A2test[completecases(A2test), :])

modelVars = [:age, :female, :uPosUrg, :uLplanning, :uLpers, :uNegUrg, :bbRR, :bbFS, :bbSum]
Xtrain = convert(Matrix{Float64}, Matrix(train[:, modelVars]) .* 1.1)
Xtrain = hcat(ones(size(Xtrain)[1], 1), Xtrain)
Xtest = convert(Matrix{Float64}, Matrix(test[:, modelVars]) .* 1.1)
Xtest = hcat(ones(size(Xtest)[1], 1), Xtest)
ytrain = convert(Vector{Float64}, train[:, :nTotal]) .* 6
ytest = convert(Vector{Float64}, test[:, :nTotal]) .* 6

# ---- standard PPMx (no common-effect model) ----
# kmeans groupings to initialize partition, as in the PPMx-common script
kclust = argmin([kmeans(Xtrain', i).totalcost for i in 2:20])
kmodel = kmeans(Xtrain', kclust)
rindK1mean = Clustering.randindex(kmodel.assignments, train.adhdLevel)

sim, model = fit_standardPPMx(ytrain, Xtrain, kmodel.assignments;
                       nburn=10000, nmc=6000,
                       outfile="output/stdPPmxTot.jld2")

# ---- summaries ----
mkdir("output/stdPPmx")
stdBetas = perClusterBetas(sim)
nbClusters = maximum([maximum(s[:C]) for s in sim])
ns = [maximum(s[:C]) for s in sim]
Plots.histogram(ns, title="# clusters", label="PPMx (standard)")
Plots.vline!([kclust], label="kMeans")
Plots.savefig("output/stdPPmx/NumberofClusters.png")

# per-cluster beta medians (posterior mean of each coefficient, by cluster)
# only the mode-cluster count, mirroring checkCommon but without :prior_mean_beta
nc = mode(ns)
dims = size(Xtrain)[2]
stdBetaByCoef = [summarize_standardPPMx(sim, c, 0.9) for c in 2:dims]
CSV.write("output/stdPPmx/betas.csv",
          DataFrame(coef=string.(modelVars),
                    median=round.([b[1] for b in stdBetaByCoef], digits=3),
                    qlo=round.([b[2][1] for b in stdBetaByCoef], digits=3),
                    qhi=round.([b[2][2] for b in stdBetaByCoef], digits=3)))

# ---- out-of-sample prediction with standard PPMx ----
yPred, cPred = postPred(Xtest, model, sim)
# OOS RMSE / LPS and label consistency (FRF / ADHD) are computed in the
# comparison script 06-baselines-ABCD.jl, which loads this `-std.jld2`.

println("Done standard PPMx fit; #burnt=10000; saved output/stdPPmxTot.jld2")