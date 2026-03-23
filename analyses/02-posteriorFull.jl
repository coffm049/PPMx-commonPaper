using Pkg
Pkg.activate("../../simulations/")
using Pipe
using JLD2
using KernelDensity
using DataFrames
using Plots
using StatsPlots
using GLM
using LaTeXStrings
using Turing
using Statistics
using TidierData
using Measures
using CategoricalArrays
using Clustering
using Latexify
using HypothesisTests

include("../code/loadNClean.jl")
#CSV.write("output/sampledDF.csv", fullDf)
fullDf, sds = loadNclean()
fullDf  = CSV.read("output/sampledDF.csv", DataFrame)
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

# combine(groupby(fullDf, :adhdLevel), [:nTotal, :nPicVocab] .=> (x -> mean(skipmissing(x))))

# fullDf = fullDf[fullDf.adhdLevel .∈ Ref([2,3,4]), :]
# fullDf.lTot= log.(0.001 .+ skipmissing(fullDf.nTotal .- minimum(skipmissing(fullDf.nTotal))))

#fullMod = lm(@formula(nPicVocab ~ age + female + uPosUrg + uLplanning + uLpers + bbRR + bbFS + bbSum + uNegUrg + ADHDcomposite), fullDf)
#selMod = lm(@formula(nTotal ~ age + female + uPosUrg + uLplanning + uLpers + bbRR + bbSum + ADHDcomposite), fullDf)
#
# 01sub-ppmxTotFulla.jl
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
sds[:female] = (1.0, 1.0)
sds[:int] = (1.0, 1.0)
modelVars= [:age, :female, :uPosUrg, :uLplanning, :uLpers, :uNegUrg, :bbRR, :bbFS, :bbSum]
Xtrain = convert(Matrix{Float64}, Matrix(train[:, modelVars]) .* 1.1)
Xtrain = hcat(ones(size(Xtrain)[1], 1), Xtrain)

Xtest = convert(Matrix{Float64}, Matrix(test[:, modelVars]) .* 1.1)
Xtest = hcat(ones(size(Xtest)[1], 1), Xtest)

y1 = convert(Vector{Float64}, train[:, :nTotal]) .* 6
y2 = convert(Vector{Float64}, test[:, :nTotal]) .* 6


#keykey = [
#  :Intercept,
#  :NIHTB_Total,
#  :UPPS_PosUrgency,
#  :NIHTB_List,
#  :NIHTB_Cryst,
#  :Female,
#  :NIHTB_Flanker,
#  :NIHTB_Pattern,
#  :NIHTB_Read,
#  :BISBAS_Reward,
#  :BIS,
#  :NIHTB_Fluid,
#  :UPPS_NegUrgency,
#  :NIHTB_Picture,
#  :BISBAS_Fun,
#  :UPPS_lackPers,
#  :BISBAS_Drive,
#  :ADHD_comp,
#  :NIHTB_PicVocab,
#  :Age,
#  :UPPS_lackPlan,
#  :FID,
#  :NIHTB_Card
#]
##%% LOAD IN

@load "output/openTotalFull1-shrunk.jld2" sim model
sim1= sim
@load "output/openTotalFull2-shrunk.jld2" sim
sim2= sim
sim = vcat(sim1, sim2)

modelVars= [:age, :female, :uPosUrg, :uLplanning, :uLpers, :uNegUrg, :bbRR, :bbFS, :bbSum]
for i in 1:length(modelVars)
    Plots.density([s[:prior_mean_beta][i] for s in sim1], title = uppercasefirst(string(modelVars[i])) * " coefficient posterior", label = "ARMS 1", fillopacity = 0.5)
    Plots.density!([s[:prior_mean_beta][i] for s in sim2], title = uppercasefirst(string(modelVars[i])) * " coefficient posterior", label = "ARMS 2", fillopacity = 0.5)
    # Plots.vline!([ests[i, 1]* 8 * sds[modelVars[i]][2]], label = "slr")
    # Plots.vline!([ests[i, 2]* 8 * sds[modelVars[i]][2]], label = "kmean")
    Plots.xlabel!(L"\hat \beta")
    Plots.savefig("output/openTotal/betafinal" * uppercasefirst(string(modelVars[i])) * "postMark.png")
end
betas = fill(NaN, (length(sim), length(sim1[1][:prior_mean_beta])))

nC = [maximum(s[:C]) for s in sim]

#80% are 5 cluster
#with remainder being 4 to 6
for (i, s) in enumerate(sim)
    if maximum(s[:C]) == 5
      betas[i, :] = s[:prior_mean_beta]
    end
end


betas = betas[.! any(isnan.(betas ), dims = 2)[:,1], :]
round.(mean(betas .> 0.0, dims = 1)[1,:], sigdigits = 2)
# 0.0     Intercept
# 0.11    age
# 0.82    female
# 0.044 * uPos
# 0.86    uLplanning
# 0.56    uLpers
# 0.98  * uNegUrg
# 0.63    bbRR
# 0.67    bbFS
# 0.52    bbSum
[quantile(b, [0.05, 0.95]) for b in eachcol(betas)]
M = DataFrame(round.(reduce(hcat, [quantile(b, [0.05, 0.95]) for b in eachcol(betas)])', digits = 2), :auto)

CSV.write("fullBetasNewFull.csv", M)

for i in 1:length(modelVars)
    Plots.density(betas[:,i], title = uppercasefirst(string(modelVars[i])) * " coefficient posterior", label = "Full")
    Plots.xlabel!(L"\hat \beta")
    Plots.savefig("output/openTotal/beta" * uppercasefirst(string(modelVars[i])) * "postMarkFull.png")
end


# conditional on Kn = mode(Kn)
cs = reduce(hcat, [s[:C] for s in sim])
cs = (maximum(cs, dims = 1) .== mode(maximum(cs, dims= 1)))
# 94 % had 9 clusters
ypred, cpred = postPred(Xtest, model, sim[cs[1,:]][1:100:end])
ypred = ypred'
cpred = cpred'
mods = [mode(r) for r in eachrow(cpred)]
pmods = mean(cpred .== mods, dims= 2)[:,1]


# identify the protyptes (all above 95% label consistency)
prototypes = []
pc = []
for c in 1:maximum(cpred)
    m = maximum(pmods[mods .== c])
    append!(pc, m)
    append!(prototypes, findfirst((pmods  .== m) .& (mods .== c)))
end

# relabe all occurences of this individual the same cluster
for (label, proto) in enumerate(prototypes)
    for col in eachcol(cpred)
        col[col .== col[proto]] .= label
    end
end

count_rows = [] 
# subset sizes
for j in 1:size(cpred, 2)
  counts = countmap(cpred[:, j])

  for (value, count) in counts
    push!(count_rows, (Column_Index = j, Value = value, count = count))
  end
end

count_rows = DataFrame(count_rows)

count_rows |>
  x -> groupby(x, :Value) |>
  x -> combine(x,
               :count => (y -> quantile(y, 0.05)) => :fifth,
               :count => (y -> quantile(y, 0.95)) => :ninetyfifth,
               )

plt = plot(count_rows.Column_Index, count_rows.count, group = count_rows.Value, 
     xlabel = "MCMC step",
     ylabel = "Subset size",
     )
Plots.savefig(plt, "output/openTotal/subsetSizeTrace.png")


# remove outlier subjects 2%
resids = ypred .- y2
resids = convert(Matrix{Union{Float64, Missing}}, resids)
resids[abs.(resids) .> 20] .= missing
# rmse
mse= sqrt.(mean((y1.- mean(y1)) .^2, dims = 1))
mseFull= (sqrt.(mean.(skipmissing.(eachrow(resids.^2))))/ mse) .^2
quantile(mseFull[.! isnan.(mseFull)], [0.05, 0.95])

# rmse Data 0.0.69 -  2.43
# rmse model 1.46
# pvalue
quantile(vec(mean(ypred .< y2, dims = 2)), [0.05, 0.95])
#  0.125
#  0.896

# filter to just 
nmodes = mean(cpred .== reshape([mode(c) for c in eachrow(cpred)], (1679, 1)), dims = 2) .> 0.5
test = vcat(A1test, A2test)
test = test[nmodes[:,1], :]
cpred = cpred[nmodes[:,1], :]


#  find mean covariate distributions
varPost = Array{Union{Missing, Float64}}(undef, size(cpred)[2], length(modelVars), length(pc))

for c in 1:5
  mask = cpred .== c
  for idx in 1:size(mask)[2]
    varPost[idx, 1:9, c] = mean(Matrix(test[mask[:, idx], modelVars]), dims = 1)[1,:]
  end
end
varPost[isnan.(varPost)] .= missing
# 2nd dimensino is the variables
centers= mapslices(x -> mean(skipmissing(x)), varPost, dims = 1)[1,:,:]
#  match cluasters on posterior mean

plots = Vector{Plots.Plot}(undef, 5)
varPost[isnan.(varPost) .| ismissing.(varPost)] .= 0
# violin!(plots, fill(1, length(varPost[:, 1,1])), varPost[:,1,1])
for clu in 1:5
    df1 = @pipe DataFrame(varPost[:, :, clu], :auto) |> 
      rename(_, modelVars) |>
      stack |>
      dropmissing |>
      # transform(groupby(_, :variable), :value => (x-> quantile(x, 0.125)) => :q5, ungroup= true) |>
      # transform(groupby(_, :variable), :value => (x-> quantile(x, 0.875)) => :q95, ungroup= true) |>
      # filter(row -> row.q5 < row.value < row.q95, _) |>
      transform(groupby(_, :variable), :value => (x-> mean(x)) => :m, ungroup = true) |>
      transform(_, :variable => categorical => :variable) # |>
      # combine(groupby(_, :variable), 
      #     :value => (x -> quantile(x, 0.05)) => :lower,
      #     :value => (x -> quantile(x, 0.95)) => :upper,
      # )
    df1.x = levelcode.(df1.variable)

    p = plot(; xticks = (1:length(levels(df1.variable)), levels(df1.variable)),
            legend=false)
    StatsPlots.boxplot!(p, df1.x, df1.value, fillcolor = :steelblue, title = "Subset" * string(clu), titlefont = font(10), outliers = false)

    plots[clu] = p
end
    #p=@df df1 groupedviolin(:variable, :value, side = :left, label = "ARMS1", outliers=false)
#   # @df df1 scatter!(:variable, :m, label = "ARMS1", side = :left)
    #@df df2 groupedviolin!(:variable, :value, side = :right, label = "ARMS2", outliers=  false)
#   # @df df2 scatter!(:variable, :m, label = "ARMS1", side = :right)


for sp in plots[1:3]
    xaxis!(sp, false, font = font(10))
    plot!(sp, legend = false, bottom_margin = -10*Plots.mm)
end
#plot!(plots[9], legend = false, legendfont = font(5))

p2 = plot(plots..., layout = (3,2), xrotation=45)
Plots.savefig(p2, "output/openTotal/groupCovarFull.png")


# [ ] find sampling distributions for the Associations...
betas = DataFrame(reduce(hcat, reduce(hcat, [map(x -> x[:beta] ,s[:lik_params]) for s in sim if maximum(s[:C]) == 5]))', :auto) |>
  tbl -> rename!(tbl, [:x2, :x3, :x4, :x5, :x6, :x7, :x8, :x9, :x10] .=> modelVars)
betas.Subset = repeat(1:5, Int(size(betas)[1]/ 5))
 
betas = stack(betas, Not([:Subset])) |>
    tbl -> filter(row -> !(row.variable in ["x1"]), tbl)

betas = @chain betas begin
  @group_by(Subset, variable)
  @filter(value > quantile(value, [0.05]))
  @filter(value < quantile(value, [0.95]))
  @ungroup
end

plots = Vector{Plots.Plot}(undef, 5)
for clu in 1:5
    df1 = @pipe betas |>
        subset(_, :Subset => x -> x .== clu) 
    #filter(row -> row.variable != "x10")
    
    p = plot(; xticks = (1:length(levels(df1.variable)), levels(df1.variable)),
             legend=false)
    StatsPlots.boxplot!(p, df1.variable, df1.value, fillcolor = :steelblue, title = "Subset" * string(clu), titlefont = font(10), outliers = false)
    plot!(p, title = "Subset" * string(clu), titlefont = font(10))
    plots[clu] = p
end

for sp in plots[4:5]
    plot!(sp, legend = false)
    xaxis!(sp, true, font = font(10))
end
for sp in plots[1:3]
  xaxis!(sp, false)
  plot!(sp, bar_width = 8, legend = false)
end
#plot!(plots[8], legend = true, legendfont = font(5))
p2 = plot(plots..., layout = (3,2), xrotation = 45, margin = -3mm)
Plots.savefig(p2, "output/openTotal/assocBoxFull.png")

dataDir = "/projects/standard/feczk001/shared/projects/FEZ_USERS/feczk001/UPPS_ABCD_FRF/code/jacob/"
# FRF labels
outcome = :nList
filepaths = [
  dataDir * "ADHDscores_list_ARMS1_merged.csv",
  dataDir * "ADHDscores_list_ARMS2_merged.csv"]
partitionLabel = "Li"
modelName = "List"

frfLabels = reduce(vcat, [CSV.read(f, DataFrame; select = ["subjectkey", "community"]) for f in filepaths])
rename!(frfLabels, ["IID", "frfLabel"])

train = leftjoin(train, frfLabels, on = "IID")
test = leftjoin(test, frfLabels, on = "IID")

preds = unique(cpred)
truths = unique(test.frfLabel)
xhips = fill(NaN, (size(cpred)[2]))
for i in 1:size(cpred)[2]
  contig = countmap(zip(cpred[:,i], test.frfLabel))
  values(contig) .> 4
  matrix = [get(contig, (p, t), 0) for p in preds, t in truths]
  mas = matrix .> 6 
  rowmas = sum.(eachrow(mas))
  colmas = sum.(eachcol(mas))
  matrix = matrix[rowmas .> 2, colmas .> 2]

  xhips[i] = pvalue(ChisqTest(matrix))
end

quantile(filter(!isnan, xhips), [0.05, 0.95])

# RI comparison
quantile([Clustering.randindex(s, test.ADHD1)[2] for s in eachcol(cpred)], [0.05, 0.95]) # 0.48 - 0.52
quantile([Clustering.randindex(s, test.ADHD2)[2] for s in eachcol(cpred)], [0.05, 0.95]) # 0.47 - 0.51
quantile([Clustering.randindex(s, test.ADHD3)[2] for s in eachcol(cpred)], [0.05, 0.95]) # 0.42 - 0.50
quantile([Clustering.randindex(s, test.ADHD4)[2] for s in eachcol(cpred)], [0.05, 0.95]) # 0.37 - 0.50
test.frfLabel = convert(Vector{Int64}, test.frfLabel)
quantile([Clustering.randindex(s, test.frfLabel)[2] for s in eachcol(cpred)], [0.05, 0.95]) # 0.52 - 0.67

gcounts = fill(NaN, (size(cpred)[2], 5)) 

for i in 1:size(cpred)[2]
  cs = collect(values(sort(delete!(countmap(cpred[:, i]), 0))))
  gcounts[i, 1:length(cs)] = cs
end

[quantile(filter(!isnan, c), [0.05,0.95]) for c in eachcol(gcounts)]
 # [57.35, 294.3]
 # [22.700000000000003, 89.29999999999998]
 # [7.3500000000000005, 27.0]
 # [5.0, 354.65]
 # [55.0, 281.6]
