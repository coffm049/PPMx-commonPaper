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
using MCMCDiagnosticTools
using CategoricalArrays

include("../code/loadNClean.jl")
include("../code/utilities.jl")
include("../code/analysis.jl")
include("../code/analysis2.jl")

#%% LOAD IN
dataDir = "/panfs/jay/groups/2/feczk001/shared/projects/FEZ_USERS/feczk001/UPPS_ABCD_FRF/code/jacob/"
# FRF labels
filepaths = [
  dataDir * "ADHDscores_list_ARMS1_merged.csv",
  dataDir * "ADHDscores_list_ARMS2_merged.csv"]
partitionLabel = "Li"
modelName = "List"

frfLabels = reduce(vcat, [CSV.read(f, DataFrame; select = ["subjectkey", "community"]) for f in filepaths])

fullDf = CSV.read("output/sampledDF.csv", DataFrame)
fullDf = leftjoin(fullDf, frfLabels, on =[:IID => :subjectkey])
A1train = innerjoin(fullDf, CSV.read("../data/a1TrFl.csv", DataFrame, header= ["IID"]), on = "IID")
A1test = innerjoin(fullDf, CSV.read("../data/a1TeFl.csv", DataFrame, header= ["IID"]), on = "IID")
A2train = innerjoin(fullDf, CSV.read("../data/a2TrFl.csv", DataFrame, header= ["IID"]), on = "IID")
A2test = innerjoin(fullDf, CSV.read("../data/a2TeFl.csv", DataFrame, header= ["IID"]), on = "IID")
A1train = A1train[completecases(A1train), :]
A2train = A2train[completecases(A2train), :]
A1test = A1test[completecases(A1test), :]
A2test = A2test[completecases(A2test), :]
A1Df = vcat(A1train, A2test)
A2Df = vcat(A2train, A2test)

modelVars= [:age, :female, :uPosUrg, :uLpers, :bbRR, :bbSum, :ADHDcomposite]
X1 = convert(Matrix{Float64}, Matrix(A1Df[:, modelVars]) .* 1.25)
X1 = hcat(ones(size(X1)[1], 1), X1)

X2 = convert(Matrix{Float64}, Matrix(A2Df[:, modelVars]) .* 1.25)
X2 = hcat(ones(size(X2)[1], 1), X2)

@load "output/subTotalStableCombined.jld2" simStable1 modelStable11 modelStable12
@load "output/subTotal2StableCombined.jld2" simStable2 modelStable21 modelStable22


#%% evaluate distribution
modelEval(:nTotal, "output/subTotalCombined.jld2", A2Df, "output/subTotal/subTotalm1")
modelEval(:nTotal, "output/subTotal2Combined.jld2", A1Df, "output/subTotal/subTotalm2")
modelEval(:nTotal, "output/subTotalStableCombined.jld2", A2Df, "output/subTotal/subTotalStablem1")
modelEval(:nTotal, "output/subTotal2StableCombined.jld2", A1Df, "output/subTotal/subTotalStablem2")

# Plots.histogram([s[:llik] for s in sim])
llk = kde([s[:llik] for s in sim1])
bestllk = llk.x[findmax(llk.density)[2]]
llkIndex = findmin([abs.(s[:llik] .- bestllk) for s in sim1])[2]
ym1a2, cm1a2 = postPred(X2, model12, [sim1[llkIndex]])
ym1a1, cm1a1 = postPred(X1, model12, [sim1[llkIndex]])

llk = kde([s[:llik] for s in sim2])
bestllk = llk.x[findmax(llk.density)[2]]
llkIndex = findmin([abs.(s[:llik] .- bestllk) for s in sim2])[2]
ym2a1, cm2a1 = postPred(X1, model21, [sim2[llkIndex]])
ym2a2, cm2a2 = postPred(X2, model21, [sim2[llkIndex]])

llk = kde([s[:llik] for s in simStable1])
bestllk = llk.x[findmax(llk.density)[2]]
llkIndex = findmin([abs.(s[:llik] .- bestllk) for s in simStable1])[2]
yms1a1, cms1a1 = postPred(X1, modelStable12, [simStable1[llkIndex]])
yms1a2, cms1a2 = postPred(X2, modelStable12, [simStable1[llkIndex]])

llk = kde([s[:llik] for s in simStable2])
bestllk = llk.x[findmax(llk.density)[2]]
llkIndex = findmin([abs.(s[:llik] .- bestllk) for s in simStable2])[2]
yms2a1, cms2a1 = postPred(X1, modelStable22, [simStable2[llkIndex]])
yms2a2, cms2a2 = postPred(X2, modelStable22, [simStable2[llkIndex]])

A1Df.m1c = cm1a1[1,:]
A1Df.ms1c = cms1a1[1,:]
A1Df.m2c = cm2a1[1,:]
A1Df.ms2c = cms2a1[1,:]

A2Df.m1c = cm1a2[1,:]
A2Df.ms1c = cms1a2[1,:]
A2Df.m2c = cm2a2[1,:]
A2Df.ms2c = cms2a2[1,:]

tabulate(A1Df, :m2c, :adhdLevel) |>
    tbl -> write_tex("output/subTotal/adhdcountsm2.tex", tbl)
tabulate(A1Df, :ms2c, :adhdLevel) |>
    tbl -> write_tex("output/subTotal/adhdcountsms2.tex", tbl)
tabulate(A2Df, :m1c, :adhdLevel) |>
    tbl -> write_tex("output/subTotal/adhdcountsm1.tex", tbl)
tabulate(A2Df, :ms1c, :adhdLevel) |>
    tbl -> write_tex("output/subTotal/adhdcountsms1.tex", tbl)


A1filt = A1Df |>
    tbl -> filter(
    row -> row.m1c ∈ [1, 3, 4,5,6] && row.ms1c ∈ [1,3,4,5,6,7] && 
    row.m2c ∈ [2,3,4,5,8,9] && row.ms2c ∈ [1,2,3,5,6],
        tbl)

A1frf = dropmissing(A1Df, :community)

A1Clustering = Matrix{Float64}(undef, 12, 4)
A1Clustering[1, :] .= Clustering.randindex(A1Df.m1c, A1Df.m2c)
A1Clustering[2, :] .= Clustering.randindex(A1filt.m1c, A1filt.m2c)
A1Clustering[3, :] .= Clustering.randindex(A1Df.ms1c, A1Df.ms2c)
A1Clustering[4, :] .= Clustering.randindex(A1filt.ms1c, A1filt.ms2c)
A1Clustering[5, :] .= Clustering.randindex(A1Df.m1c, A1Df.adhdLevel)
A1Clustering[6, :] .= Clustering.randindex(A1Df.m2c, A1Df.adhdLevel)
A1Clustering[7, :] .= Clustering.randindex(A1Df.ms1c, A1Df.adhdLevel)
A1Clustering[8, :] .= Clustering.randindex(A1Df.ms2c, A1Df.adhdLevel)
A1Clustering[9, :] .= Clustering.randindex(A1frf.m1c, A1frf.community)
A1Clustering[10, :] .= Clustering.randindex(A1frf.m2c, A1frf.community)
A1Clustering[11, :] .= Clustering.randindex(A1frf.ms1c, A1frf.community)
A1Clustering[12, :] .= Clustering.randindex(A1frf.ms2c, A1frf.community)
A1Clustering = DataFrame(A1Clustering, :auto)
rename!(A1Clustering, ["ri" ,"ari", "dis", "hub"])
A1Clustering.compared = ["m1-m2_full", "m1-m2_filt", "ms1-ms2_full", "ms1-ms2_filt", "m1-adhd", "m2-adhd", "ms1-adhd", "ms2-adhd", "m1-frf", "m2-frf", "ms1-frf", "ms2-frf"]
A1Clustering.ARMS .= 1


A2filt = A2Df |>
    tbl -> filter(
    row -> row.m1c ∈ [1, 4,5,6] && row.ms1c ∈ [4,5,6] && 
    row.m2c ∈ [9, 3] && row.ms2c ∈ [1,2],
        tbl)

A2frf = dropmissing(A2Df, :community)

A2Clustering = Matrix{Float64}(undef, 12, 4)
A2Clustering[1, :] .= Clustering.randindex(A2Df.m1c, A2Df.m2c)
A2Clustering[2, :] .= Clustering.randindex(A2filt.m1c, A2filt.m2c)
A2Clustering[3, :] .= Clustering.randindex(A2Df.ms1c, A2Df.ms2c)
A2Clustering[4, :] .= Clustering.randindex(A2filt.ms1c, A2filt.ms2c)
A2Clustering[5, :] .= Clustering.randindex(A2Df.m1c, A2Df.adhdLevel)
A2Clustering[6, :] .= Clustering.randindex(A2Df.m2c, A2Df.adhdLevel)
A2Clustering[7, :] .= Clustering.randindex(A2Df.ms1c, A2Df.adhdLevel)
A2Clustering[8, :] .= Clustering.randindex(A2Df.ms2c, A2Df.adhdLevel)
A2Clustering[9, :] .= Clustering.randindex(A2frf.m1c, A2frf.community)
A2Clustering[10, :] .= Clustering.randindex(A2frf.m2c, A2frf.community)
A2Clustering[11, :] .= Clustering.randindex(A2frf.ms1c, A2frf.community)
A2Clustering[12, :] .= Clustering.randindex(A2frf.ms2c, A2frf.community)
A2Clustering = DataFrame(A2Clustering, :auto)
rename!(A2Clustering, ["ri" ,"ari", "dis", "hub"])
A2Clustering.compared = ["m1-m2_full", "m1-m2_filt", "ms1-ms2_full", "ms1-ms2_filt", "m1-adhd", "m2-adhd", "ms1-adhd", "ms2-adhd", "m1-frf", "m2-frf", "ms1-frf", "ms2-frf"]
A2Clustering.ARMS .= 2

CSV.write("output/subTotal/riTable.csv", vcat(A1Clustering, A2Clustering))
