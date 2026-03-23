using Pkg
Pkg.activate("../../simulations/")
using Random
using DelimitedFiles
using LinearAlgebra
using JLD2
using KernelDensity
using FreqTables
using DataFrames
using ProductPartitionModels
using TexTables
# using CategoricalArrays
using StatsBase
using Statistics
using StatsPlots
using Plots
# using UnicodePlots
using StatsModels
using ProductPartitionModels
using Clustering
using CSV
using GLM
using TidierData
using MCMCChains

include("../code/utilities.jl")
include("../code/loadNClean.jl")
include("../code/analysis.jl")

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
#CSV.write("output/sampledDF.csv", fullDf)
fullDf  = CSV.read("output/sampledDF.csv", DataFrame)

# combine(groupby(fullDf, :adhdLevel), [:nTotal, :nPicVocab] .=> (x -> mean(skipmissing(x))))

# fullDf = fullDf[fullDf.adhdLevel .∈ Ref([2,3,4]), :]
# fullDf.lTot= log.(0.001 .+ skipmissing(fullDf.nTotal .- minimum(skipmissing(fullDf.nTotal))))

#fullMod = lm(@formula(nPicVocab ~ age + female + uPosUrg + uLplanning + uLpers + bbRR + bbFS + bbSum + uNegUrg + ADHDcomposite), fullDf)
#selMod = lm(@formula(nTotal ~ age + female + uPosUrg + uLplanning + uLpers + bbRR + bbSum + ADHDcomposite), fullDf)
#

# partition data
A1train = innerjoin(fullDf, CSV.read("../data/a1TrFl.csv", DataFrame, header= ["IID"]), on = "IID")
A1test = innerjoin(fullDf, CSV.read("../data/a1TeFl.csv", DataFrame, header= ["IID"]), on = "IID")
A2train = innerjoin(fullDf, CSV.read("../data/a2TrFl.csv", DataFrame, header= ["IID"]), on = "IID")
A2test = innerjoin(fullDf, CSV.read("../data/a2TeFl.csv", DataFrame, header= ["IID"]), on = "IID")
A1train = A1train[completecases(A1train), :]
A2train = A2train[completecases(A2train), :]
A1test = A1test[completecases(A1test), :]
A2test = A2test[completecases(A2test), :]
train = vcat(A1train, A2train)
test = vcat(A1test, A2test)

# tabulate(A2Df, :ADHD3, :ADHD2)
modelVars= [:age, :female, :uPosUrg, :uLplanning, :uLpers, :uNegUrg, :bbRR, :bbFS, :bbSum]


Xtrain = convert(Matrix{Float64}, Matrix(train[:, modelVars]) .* 1.1)
Xtrain = hcat(ones(size(Xtrain)[1], 1), Xtrain)

Xtest = convert(Matrix{Float64}, Matrix(test[:, modelVars]) .* 1.1)
Xtest = hcat(ones(size(Xtest)[1], 1), Xtest)

# kmeans
kclust = argmin([kmeans(Xtrain', i).totalcost for i in 2:20])
kmodel = kmeans(Xtrain', kclust)
# train.kclust = kmodel.assignments
rindK1mean = Clustering.randindex(kmodel.assignments, train.adhdLevel)
y1 = convert(Vector{Float64}, train[:, :nTotal]) .* 6
y2 = convert(Vector{Float64}, test[:, :nTotal]) .* 6


# model
model = Model_PPMx(y1, Xtrain, kmodel.assignments, similarity_type=:NN, sampling_model=:Reg, init_lik_rand=true)
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
@save "output/openTotalFull2-shrunk.jld2" sim model
sim = mcmc!(model, 6000; mixDPM=true)
@save "output/openTotalFull3-shrunk.jld2" sim model
