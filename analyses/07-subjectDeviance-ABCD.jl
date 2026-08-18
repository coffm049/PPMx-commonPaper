using Pkg
Pkg.activate("../../simulations/")
using Random
using DelimitedFiles
using LinearAlgebra
using JLD2
using KernelDensity
using FreqTables
using DataFrames
using StatsBase
using Statistics
using Clustering
using CSV
using ProductPartitionModels

include("../code/utilities.jl")
include("../code/loadNClean.jl")
include("../code/analysis.jl")
include("../code/analysis2.jl")

# ============================================================================
# 07-subjectDeviance-ABCD.jl
# Subject-level deviance of the PPMx-common / PPMx predictive distributions
# against two standard references:
#   (1) the empirical distribution of the observed outcome, and
#   (2) the average-subject predictive distribution (posterior predictive draws
#       pooled across all subjects).
# For each subject we report the predictive mean/SD, the signed and standardized
# deviation from each reference, and a log-predictive-density comparison
# (KDE-based) of the subject's predictive draws against each reference.
# Outputs are written to output/baselines/.
# ============================================================================

commonFiles = [
    "output/openTotalFullDatamcmc1.jld2",
    "output/openTotalFullDatamcmc2.jld2",
]
stdFile = "output/stdPPmxTot.jld2"
outcome = :nTotal

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

A1test = innerjoin(fullDf, CSV.read("../data/a1TeFl.csv", DataFrame, header= ["IID"]), on = "IID")
A2test = innerjoin(fullDf, CSV.read("../data/a2TeFl.csv", DataFrame, header= ["IID"]), on = "IID")
test = vcat(A1test[completecases(A1test), :], A2test[completecases(A2test), :])

modelVars = [:age, :female, :uPosUrg, :uLplanning, :uLpers, :uNegUrg, :bbRR, :bbFS, :bbSum]
Xtest = convert(Matrix{Float64}, Matrix(test[:, modelVars]) .* 1.1)
Xtest = hcat(ones(size(Xtest)[1], 1), Xtest)
ytest = convert(Vector{Float64}, test[:, outcome]) .* 6

nTest = length(ytest)

# posterior predictive draws per subject (n_sim x nTest)
simC = Dict{Symbol,Any}[]
for f in commonFiles
    @load f sim modelC
    append!(simC, sim)
end
yC, cC, meanC = postPred(Xtest, modelC, simC[1:100:end])

@load stdFile simS modelS
yS, cS, meanS = postPred(Xtest, modelS, simS[1:100:end])

# ----------------------------------------------------------------------------
# references
# ----------------------------------------------------------------------------
# (1) empirical distribution of the observed outcome
empKDE = kde(ytest)

# (2) average-subject predictive distribution = pooled predictive draws
poolC = vec(yC)
poolS = vec(yS)
poolKDE_C = kde(poolC)
poolKDE_S = kde(poolS)

logdens(k, x) = log(max(pdf(k, x), eps()))

function subject_metrics(Ypred, empKDE, poolKDE, pool)
    n_sim = size(Ypred, 1)
    rows = Vector{NamedTuple}(undef, nTest)
    poolMean = mean(pool)
    poolSD = std(pool)
    empMean = mean(ytest)
    empSD = std(ytest)
    for i in 1:nTest
        di = Ypred[:, i]
        mu_i = mean(di)
        sd_i = std(di)
        lpd_emp = mean(logdens(empKDE, d) for d in di)
        lpd_pool = mean(logdens(poolKDE, d) for d in di)
        rows[i] = (
            subject = i,
            predMean = mu_i,
            predSD = sd_i,
            dev_emp = mu_i - empMean,
            dev_emp_std = (mu_i - empMean) / empSD,
            dev_pool = mu_i - poolMean,
            dev_pool_std = (mu_i - poolMean) / poolSD,
            lpd_emp = lpd_emp,
            lpd_pool = lpd_pool,
            lpd_diff = lpd_emp - lpd_pool,
        )
    end
    return DataFrame(rows)
end

subC = subject_metrics(yC, empKDE, poolKDE_C, poolC)
subC.model = fill("PPMx-common", nTest)
subS = subject_metrics(yS, empKDE, poolKDE_S, poolS)
subS.model = fill("PPMx (standard)", nTest)
subDF = vcat(subC, subS)

mkdir("output/baselines")
CSV.write("output/baselines/subjectDeviance.csv", subDF)

# ----------------------------------------------------------------------------
# summary: mean subject deviance and mean log predictive density per model
# ----------------------------------------------------------------------------
summaryDF = DataFrame(
    model = ["PPMx-common", "PPMx (standard)"],
    meanAbsDevEmp = [mean(abs.(subC.dev_emp)), mean(abs.(subS.dev_emp))],
    meanAbsDevEmpStd = [mean(abs.(subC.dev_emp_std)), mean(abs.(subS.dev_emp_std))],
    meanAbsDevPool = [mean(abs.(subC.dev_pool)), mean(abs.(subS.dev_pool))],
    meanAbsDevPoolStd = [mean(abs.(subC.dev_pool_std)), mean(abs.(subS.dev_pool_std))],
    meanLPDemp = [mean(subC.lpd_emp), mean(subS.lpd_emp)],
    meanLPDpool = [mean(subC.lpd_pool), mean(subS.lpd_pool)],
    meanLPDdiff = [mean(subC.lpd_diff), mean(subS.lpd_diff)],
)
CSV.write("output/baselines/subjectDevianceSummary.csv", summaryDF)
println(summaryDF)
println("Done subject deviance; outputs in output/baselines/")