using Pkg
# v2.0: absolute shared env (relative ../../simulations/ resolved to ~/papers/simulations, an empty project).
Pkg.activate(expanduser("~/software/ProductPartitionModels.jl/simulations"))
using Random
using LinearAlgebra
using JLD2
using DataFrames
using Statistics
using Clustering
using CSV
using ProductPartitionModels

include("../code/utilities.jl")
include("../code/loadNClean.jl")

# ----------------------------------------------------------------------------
# 08-cvCombined-ABCD.jl
# 5-fold cross-validation of the PPMx-common (combined) model on the full
# ABCD-ADHD sample, requested by Rev 1 C10 so the combined-model RMSE is
# comparable to the ARMS1/ARMS2 held-out RMSEs. Each fold re-fits the model on
# 4/5 of the data and predicts on the held-out 1/5; OOS RMSE is averaged.
# Output: output/baselines/combined5foldCV.csv
# ----------------------------------------------------------------------------

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
fullDf = CSV.read("output/sampledDF.csv", DataFrame)

A1train = innerjoin(fullDf, CSV.read("../data/a1TrFl.csv", DataFrame, header = ["IID"]), on = "IID")
A1test  = innerjoin(fullDf, CSV.read("../data/a1TeFl.csv", DataFrame, header = ["IID"]), on = "IID")
A2train = innerjoin(fullDf, CSV.read("../data/a2TrFl.csv", DataFrame, header = ["IID"]), on = "IID")
A2test  = innerjoin(fullDf, CSV.read("../data/a2TeFl.csv", DataFrame, header = ["IID"]), on = "IID")
fullDf = vcat(
  A1train[completecases(A1train), :],
  A2test[completecases(A2test), :],
  A2train[completecases(A2train), :],
  A1test[completecases(A1test), :])

modelVars = [:age, :female, :uPosUrg, :uLplanning, :uLpers, :uNegUrg, :bbRR, :bbFS, :bbSum]
outcome = :nTotal

rng = MersenneTwister(20240829)
n = nrow(fullDf)
folds = rand(rng, 1:5, n)

rmsePerFold = Float64[]
for f in 1:5
    testIdx  = findall(==(f), folds)
    trainIdx = findall(!=(f), folds)
    dfTr = fullDf[trainIdx, :]
    dfTe = fullDf[testIdx, :]

    Xtr = convert(Matrix{Float64}, Matrix(dfTr[:, modelVars]) .* 1.25)
    Xtr = hcat(ones(size(Xtr, 1), 1), Xtr)
    ytr = convert(Vector{Float64}, dfTr[:, outcome]) .* 8

    Xte = convert(Matrix{Float64}, Matrix(dfTe[:, modelVars]) .* 1.25)
    Xte = hcat(ones(size(Xte, 1), 1), Xte)
    yte = convert(Vector{Float64}, dfTe[:, outcome]) .* 8

    kclust = argmin([kmeans(Xtr', i).totalcost for i in 2:20])
    kmodel = kmeans(Xtr', kclust)

    model = Model_PPMx(ytr, Xtr, kmodel.assignments, similarity_type = :NN,
                       sampling_model = :Reg, init_lik_rand = true)
    prec = 0.01; alph = 1.0; bet = 1.0; dims = size(Xtr, 2)
    model.prior.base = Prior_base(repeat([0.0], dims), repeat([prec], dims),
                                  repeat([alph], dims), repeat([bet], dims))
    model.prior.massParams = [1, 1]
    model.state.baseline.tau0 = 1e6
    mcmc!(model, 10000; mixDPM = true)
    sim = mcmc!(model, 6000; mixDPM = true)

    yPred, cPred = postPred(Xte, model, sim)
    yhat = vec(mean(yPred, dims = 1))
    push!(rmsePerFold, sqrt(mean((yhat .- yte) .^ 2)))
    println("Fold $f RMSE = $(rmsePerFold[end])")
end

mkdir("output/baselines")
CSV.write("output/baselines/combined5foldCV.csv",
          DataFrame(fold = 1:5, rmse = rmsePerFold,
                    mean_rmse = fill(mean(rmsePerFold), 5)))
println("Combined 5-fold CV RMSE: ", mean(rmsePerFold))
