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
using DPMM

include("../code/utilities.jl")
include("../code/loadNClean.jl")
include("../code/analysis.jl")
include("../code/analysis2.jl")
include("../code/ppmxBaselines.jl")

# ============================================================================
# 06-baselines-ABCD.jl
# FRF-total analysis: compare partition recovery (ARI vs the external FRF
# "community" labels) and out-of-sample prediction (RMSE) across
#   PPMx-common, standard PPMx (mixDPM=false), k-means, and DP Gaussian mixture.
# Outputs are written to output/baselines/.
# ============================================================================

commonFiles = [
    "output/openTotalFullDatamcmc1.jld2",
    "output/openTotalFullDatamcmc2.jld2",
]
stdFile = "output/stdPPmxTot.jld2"
dataDir = "/projects/standard/feczk001/shared/projects/FEZ_USERS/feczk001/UPPS_ABCD_FRF/code/jacob/"
frfFiles = [
    dataDir * "ADHDscores_list_ARMS1_merged.csv",
    dataDir * "ADHDscores_list_ARMS2_merged.csv",
]
outcome = :nTotal

# ---- load + clean, same as 03subTotStableInference.jl ----
fullDf = CSV.read("output/sampledDF.csv", DataFrame)
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

frfLabels = reduce(vcat, [CSV.read(f, DataFrame; select = ["subjectkey", "community"]) for f in frfFiles])
rename!(frfLabels, :subjectkey => :IID)
fullDf = leftjoin(fullDf, frfLabels, on = :IID)

A1train = innerjoin(fullDf, CSV.read("../data/a1TrFl.csv", DataFrame, header= ["IID"]), on = "IID")
A1test = innerjoin(fullDf, CSV.read("../data/a1TeFl.csv", DataFrame, header= ["IID"]), on = "IID")
A2train = innerjoin(fullDf, CSV.read("../data/a2TrFl.csv", DataFrame, header= ["IID"]), on = "IID")
A2test = innerjoin(fullDf, CSV.read("../data/a2TeFl.csv", DataFrame, header= ["IID"]), on = "IID")
train = vcat(A1train[completecases(A1train), :], A2train[completecases(A2train), :])
test = vcat(A1test[completecases(A1test), :], A2test[completecases(A2test), :])

modelVars = [:age, :female, :uPosUrg, :uLplanning, :uLpers, :uNegUrg, :bbRR, :bbFS, :bbSum]
predVars = [:age, :female, :uPosUrg, :uLplanning, :uLpers]
Xtrain = convert(Matrix{Float64}, Matrix(train[:, modelVars]) .* 1.1)
Xtrain = hcat(ones(size(Xtrain)[1], 1), Xtrain)
Xtest = convert(Matrix{Float64}, Matrix(test[:, modelVars]) .* 1.1)
Xtest = hcat(ones(size(Xtest)[1], 1), Xtest)
ytrain = convert(Vector{Float64}, train[:, outcome]) .* 6
ytest = convert(Vector{Float64}, test[:, outcome]) .* 6

# truth labels (external FRF community) on the subset where they are observed
trComm = findall(!ismissing, train.community)
teComm = findall(!ismissing, test.community)
commTr = Int.(vec(train[trComm, :community]))
commTe = Int.(vec(test[teComm, :community]))

# ============================================================================
# 1. k-means baseline (on covariate columns)
# ============================================================================
kclust = argmin([kmeans(Xtrain[:, 2:end]', i).totalcost for i in 2:20])
km = kmeans(Xtrain[:, 2:end]', kclust)
kmTr = km.assignments
kmTe = assign_to_centroids(Xtest[:, 2:end]', km.centers)

ariK_tr = Clustering.randindex(kmTr[trComm], commTr)[1]
ariK_te = Clustering.randindex(kmTe[teComm], commTe)[1]
# per-cluster interaction regression, OOS RMSE
train.kclust = string.(kmTr)
test.kclust = string.(kmTe)
formK = term(outcome) ~ reduce(+, [term(Symbol(v)) for v in predVars]) * term(:kclust)
ck = Dict(:kclust => EffectsCoding())
kmLm = lm(formK, train; contrasts=ck)
rmseK = sqrt(mean(((predict(kmLm, test)) .- test[!, outcome]) .^ 2))

# ============================================================================
# 2. DP Gaussian mixture baseline (module DPMM)
# ============================================================================
dpm = dpm_regression_compare(copy(train), copy(test), modelVars, predVars, outcome, :community;
                             alpha=1.0, iters=500, scale=1.1)
dpmTr = dpm.trainLabels
dpmTe = dpm.testLabels

# ============================================================================
# 3. PPMx-common  (mixDPM=true)
# ============================================================================
simC = Dict{Symbol,Any}[]
for f in commonFiles
    @load f sim modelC
    append!(simC, sim)
end
# training ARI from chain allocations (mode-number-of-clusters iterations)
cC_tr = [maximum(s[:C]) for s in simC]
ncC = mode(cC_tr)
ariC_tr = Clustering.randindex([s[:C] for s in simC if maximum(s[:C]) == ncC][end][trComm], commTr)[1]

yC, cC = postPred(Xtest, modelC, simC[1:100:end])
meanC = vec(mean(yC, dims = 1))
rmseC = sqrt(mean((meanC .- ytest) .^ 2))
ariC_te = Clustering.randindex(vec(mode.(eachcol(cC)))[teComm], commTe)[1]

# ============================================================================
# 4. Standard PPMx  (mixDPM=false)
# ============================================================================
@load stdFile simS modelS
cS_tr = [maximum(s[:C]) for s in simS]
ncS = mode(cS_tr)
ariS_tr = Clustering.randindex([s[:C] for s in simS if maximum(s[:C]) == ncS][end][trComm], commTr)[1]

yS, cS = postPred(Xtest, modelS, simS[1:100:end])
meanS = vec(mean(yS, dims = 1))
rmseS = sqrt(mean((meanS .- ytest) .^ 2))
ariS_te = Clustering.randindex(vec(mode.(eachcol(cS)))[teComm], commTe)[1]

# ============================================================================
# comparison table
# ============================================================================
comparison = DataFrame(
    model = ["PPMx-common", "PPMx (standard)", "k-means", "DP-GMM"],
    trainARI = [ariC_tr, ariS_tr, ariK_tr, Clustering.randindex(dpmTr[trComm], commTr)[1]],
    testARI = [ariC_te, ariS_te, ariK_te, Clustering.randindex(dpmTe[teComm], commTe)[1]],
    testRMSE = [rmseC, rmseS, rmseK, dpm.rmseoos],
    nclusters = [ncC, ncS, kclust, dpm.nclusts],
)
mkdir("output/baselines")
CSV.write("output/baselines/frftotalComparison.csv", comparison)
println(comparison)

# plots: ARI vs RMSE
scatter(comparison.testARI, comparison.testRMSE,
        group = comparison.model, legend = :bottomleft, xlabel = "test ARI (FRF)",
        ylabel = "test RMSE", title = "FRF-total: baselines vs PPMx")
Plots.savefig("output/baselines/frftotalComparison.png")

println("Done baselines comparison; outputs in output/baselines/")