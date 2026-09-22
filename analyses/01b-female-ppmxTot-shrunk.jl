using Pkg
# v2.0: absolute shared env (relative ../../simulations/ resolved to ~/papers/simulations, an empty project).
Pkg.activate(expanduser("~/software/ProductPartitionModels.jl/simulations"))
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
# v2.0: `using MCMCChains` removed (dead import; package not in shared env, nothing uses it).
using Revise
using ProductPartitionModels

include("../code/utilities.jl")
include("../code/loadNClean.jl")
include("../code/analysis.jl")

# ============================================================================
# 01b-female-ppmxTot-shrunk.jl
# Sex-stratified: FEMALE subjects only. PPMx-common MCMC chains.
# ============================================================================

# load and clean the data
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

# ---- FEMALE subjects only ----
fullDf = fullDf[fullDf.female .== 1, :]

# partition data
A1train = innerjoin(fullDf, CSV.read("../data/a1TrFl.csv", DataFrame, header= ["IID"]), on = "IID")
A1test = innerjoin(fullDf, CSV.read("../data/a1TeFl.csv", DataFrame, header= ["IID"]), on = "IID")
A2train = innerjoin(fullDf, CSV.read("../data/a2TrFl.csv", DataFrame, header= ["IID"]), on = "IID")
A2test = innerjoin(fullDf, CSV.read("../data/a2TeFl.csv", DataFrame, header= ["IID"]), on = "IID")
fullDf = vcat(
  A1train[completecases(A1train), :],
  A2test[completecases(A2test), :],
  A2train[completecases(A2train), :],
  A1test[completecases(A1test), :])

# ---- NO FEMALE in covariate set ----
modelVars= [:age, :uPosUrg, :uLplanning, :uLpers, :uNegUrg, :bbRR, :bbFS, :bbSum]

# ---- NO post-standardisation scaling ----
Xtrain = convert(Matrix{Float64}, Matrix(fullDf[:, modelVars]))
Xtrain = hcat(ones(size(Xtrain)[1], 1), Xtrain)

# kmeans
kclust = argmin([kmeans(Xtrain', i).totalcost for i in 2:20])
kmodel = kmeans(Xtrain', kclust)
rindK1mean = Clustering.randindex(kmodel.assignments, fullDf.adhdLevel)
y = convert(Vector{Float64}, fullDf[:, :nTotal])

# model
model = Model_PPMx(y, Xtrain, kmodel.assignments, similarity_type=:NN, sampling_model=:Reg, init_lik_rand=true)
# set priors for base measure sampling
prec = 0.01
alph=1.0
bet=1.0
dims = size(Xtrain)[2]
model.prior.base = Prior_base(
    repeat([0.0], dims),
    repeat([prec], dims), #1.0
    repeat([alph], dims), # 1.0
    repeat([bet], dims) # 1.0
)
model.prior.massParams = [1, 1] # 1e-3 for  common 10, inter 5
model.state.baseline.tau0 = 1e6
mcmc!(model, 10000; mixDPM=true)
sim = mcmc!(model, 6000; mixDPM=true)
mkpath("output/openTotalFullDatamcmc-female")
@save "output/openTotalFullDatamcmc-female/mcmc1.jld2" sim model
sim = mcmc!(model, 6000; mixDPM=true)
@save "output/openTotalFullDatamcmc-female/mcmc2.jld2" sim model

println("Done PPMx-common fit (female only)")
