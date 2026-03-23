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
using TexTables
using Turing
using Statistics
using TidierData
using Measures
using CategoricalArrays
using Clustering

include("../code/loadNClean.jl")
#CSV.write("output/sampledDF.csv", fullDf)
fullDf, sds = loadNclean()
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

# partition data
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
sds[:female] = (1.0, 1.0)
sds[:int] = (1.0, 1.0)
modelVars= [:age, :female, :uPosUrg, :uLplanning, :uLpers, :uNegUrg, :bbRR, :bbFS, :bbSum]
Xtrain = convert(Matrix{Float64}, Matrix(A1train[:, modelVars]) .* 1.1)
Xtrain = hcat(ones(size(Xtrain)[1], 1), Xtrain)

Xtrain2 = convert(Matrix{Float64}, Matrix(A2train[:, modelVars]) .* 1.1)
Xtrain2 = hcat(ones(size(Xtrain2)[1], 1), Xtrain2)

Xtest = convert(Matrix{Float64}, Matrix(A2Df[:, modelVars]) .* 1.1)
Xtest = hcat(ones(size(Xtest)[1], 1), Xtest)

Xtest2 = convert(Matrix{Float64}, Matrix(A1Df[:, modelVars]) .* 1.1)
Xtest2 = hcat(ones(size(Xtest2)[1], 1), Xtest2)

y2 = convert(Vector{Float64}, A1Df[:, :nTotal]) .* 6
y1 = convert(Vector{Float64}, A2Df[:, :nTotal]) .* 6


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

@load "output/openTotalFullDatamcmc1-shrunk.jld2" sim model
sim1= sim
@load "output/openTotalFullDatamcmc2-shrunk.jld2" sim
sim2= sim


modelVars= [:age, :female, :uPosUrg, :uLplanning, :uLpers, :uNegUrg, :bbRR, :bbFS, :bbSum]
for i in 1:length(modelVars)
    Plots.density([s[:prior_mean_beta][i] for s in sim1], title = uppercasefirst(string(modelVars[i])) * " coefficient posterior", label = "ARMS 1", fillopacity = 0.5)
    Plots.density!([s[:prior_mean_beta][i] for s in sim2], title = uppercasefirst(string(modelVars[i])) * " coefficient posterior", label = "ARMS 2", fillopacity = 0.5)
    # Plots.vline!([ests[i, 1]* 8 * sds[modelVars[i]][2]], label = "slr")
    # Plots.vline!([ests[i, 2]* 8 * sds[modelVars[i]][2]], label = "kmean")
    Plots.xlabel!(L"\hat \beta")
    Plots.savefig("output/openTotal/betafinal" * uppercasefirst(string(modelVars[i])) * "postGood.png")
end

betas = Matrix{Float64}(undef, length(sim1), length(sim1[1][:prior_mean_beta]))
betas2 = Matrix{Float64}(undef, length(sim2), length(sim2[1][:prior_mean_beta]))

nC = [maximum(s[:C]) for s in sim1] # 99% 8 cluster
nC = [maximum(s[:C]) for s in sim2] # 99% 8 


for (i, s) in enumerate(sim1)
    if maximum(s[:C]) == 8
        betas[i, :] = mean(reduce(hcat, [betas[:beta] for betas in s[:lik_params]]), dims = 2)
    end
end
for (i, s) in enumerate(sim2)
    if maximum(s[:C]) ==  8
        betas2[i, :] = mean(reduce(hcat, [betas[:beta] for betas in s[:lik_params]]), dims = 2)
    end
end
# betas = betas[.!any(abs.(betas) , dims=2)[:,1], :]
# betas2 = betas2[.!any(abs.(betas2), dims=2)[:,1], :]
mean(betas .> 0.0, dims = 1)[1,:]
mean(betas2 .> 0.0, dims = 1)[1,:]

mapslices(x -> quantile(x[.!isnan.(x)], [0.05, 0.5, 0.95]), betas, dims =1)'
mapslices(x -> quantile(x[.!isnan.(x)], [0.05, 0.5, 0.95]), betas2, dims =1)'

# ARM1, ARM2
#int*,*
#age*,
#female,
#uPosUrg*,*
#uLplanning*,
#uLpers*,*
#uNegUrg,
#bbRR
#bbFS
#bbSum*,*
#
# 5uLplanning was signif in ARMS 1 and 2 and this was the same for conditional as
# well as unconditional inference

for i in 1:length(modelVars)
    vec1 = betas[.! isnan.(betas[:, i]), i]
    vec2 = betas2[.! isnan.(betas2[:, i]), i]
    Plots.density(vec1, title = uppercasefirst(string(modelVars[i])) * " coefficient posterior", label = "ARMS 1", fillopacity = 0.5)
    Plots.density!(vec2, title = uppercasefirst(string(modelVars[i])) * " coefficient posterior", label = "ARMS 2", fillopacity = 0.5)
    #Plots.vline!([ests[i, 1]* 8 * sds[modelVars[i]][2]], label = "slr")
    #Plots.vline!([ests[i, 2]* 8 * sds[modelVars[i]][2]], label = "kmean")
    Plots.xlabel!(L"\hat \beta")
    Plots.savefig("output/openTotal/beta" * uppercasefirst(string(modelVars[i])) * "postGood.png")
end




# conditional on Kn = mode(Kn)
cs = reduce(hcat, [s[:C] for s in sim1])
cs = (maximum(cs, dims = 1) .== mode(maximum(cs, dims= 1)))
ypred, cpred = postPred(Xtest, model, sim1[cs[1,:]][1:100:end])
ypred = ypred'
cpred = cpred'
ypred12, cpred12 = postPred(Xtest2, model, sim1[cs[1,:]][1:100:end])
cpred12 = cpred12'
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


cs2 = reduce(hcat, [s[:C] for s in sim2])
# conditional on Kn = mode(Kn) = 6
cs2 = cs2[:, (maximum(cs2, dims = 1) .== mode(maximum(cs2, dims= 1)))[1,:]]
ypred2, cpred2 = postPred(Xtest2, model, sim2[cs2[1,:]][1:100:end])
ypred2 = ypred2'
cpred = cpred'
ypred22, cpred22 = postPred(Xtest, model, sim2[cs[1,:]][1:100:end])
cpred22 = cpred22'
mods = [mode(r) for r in eachrow(cs2)]
pmods = mean(cs2 .== mods, dims= 2)[:,1]


# [ ] randindex removing of only first 3 labels
# cpred[cpred .∈ Ref([4,5,6])] .= 0
# cpred2[cpred2 .∈ Ref([4,5,6])] .= 0
# cpred12[cpred12 .∈ Ref([4,5,6])] .= 0
# cpred22[cpred22 .∈ Ref([4,5,6])] .= 0
# cpred22 = cpred22'
# cpred12 = cpred12'


ris = [randindex(cpred12'[c, :], cpred2[c,:]) for c in 1:size(cpred12')[1]]
ris2 = [randindex(cpred22'[c, :], cpred[c,:]) for c in 1:size(cpred22')[1]]

# remove outlier subjects 2%
Plots.histogram(vec(ypred))
xlims!(-1e4, 1e4)

Plots.histogram(mean(abs.(ypred) .> 1e4, dims = 2) )
filt1 = (0.35 .< mean(ypred .< y1, dims = 2) .< 0.65)[:,1]
filt2 = (0.35 .< mean(ypred2 .< y2, dims = 2) .< 0.65)[:,1]
resids1 = ypred[filt1, :] .- y1[filt1]
resids2 = ypred2[filt2, :] .- y2[filt2]
filt12 = mean(abs.(resids1) .> 30, dims = 2) .> 0.1
filt22 = mean(abs.(resids2) .> 30, dims = 2) .> 0.1

filtresids1 = resids1[.!filt12[:, 1], :]
filtresids2 = resids2[.!filt22[:, 1], :]
filtresids1[filtresids1 .< -25] .= -25.0
filtresids1[filtresids1 .> 25] .= 25.0
filtresids2[filtresids2 .< -25] .= -25.0
filtresids2[filtresids2 .> 25] .= 25.0

# rmse
mse1= mean((filtresids1 ./1.42) .^2, dims = 1)
mse2= mean((filtresids2 ./1.42)  .^2, dims = 1)

# Plots.histogram(vec(mse1), label = "ARMS1 CV SSE", title = "Sum of Squares")
# Plots.histogram!(vec(mse2), label=  "ARMS2 CV SSE")
# Plots.vline!([var(y1), var(y2)], label = "Var(Y)")
# Plots.savefig("output/openTotal/sumofSquares.png")


# pvalue
quantile(vec(mean(ypred[filt1,:] .< y1[filt1,:], dims = 2)), [0, 1])
quantile(vec(mean(ypred2[filt2,:] .< y2[filt2,:], dims = 2)), [0, 1])



# identify the protyptes (all above 86% label consistency)
prototypes2 = []
pc2 = []
for c in 1:maximum(cs2)
    m = maximum(pmods[mods .== c])
    append!(pc2, m)
    append!(prototypes2, findfirst((pmods  .== m) .& (mods .== c)))
end

# relabe all occurences of this individual the same cluster
for (label, proto) in enumerate(prototypes2)
    for row in eachrow(cpred2)
        row[row .== row[proto]] .= label
    end
end

# [ ] filter to only high consistency subjects
nmodes = mean(cpred .== reshape([mode(c) for c in eachcol(cpred)], (1, 5370)), dims = 1) .> 0.5
nmodes2 = mean(cpred2 .== reshape([mode(c) for c in eachcol(cpred2)], (1, 5291)), dims = 1) .> 0.5
A2Dffil = A2Df[nmodes[1,:], :]
A1Dffil = A1Df[nmodes2[1,:], :]


#  find mean covariate distributions
varPost = Array{Union{Missing, Float64}}(undef, size(cpred)[1], length(modelVars), length(pc))
for (idx, c) in enumerate(1:length(pc))
  varPost[:, :, idx] = reduce(vcat, [mean(Matrix(A2Dffil[row .== c, modelVars]), dims=  1) for row in eachrow(cpred[:, nmodes[1,:]])])
end
varPost[isnan.(varPost)] .= missing

varPost2 = Array{Union{Missing, Float64}}(undef, size(cpred2)[1], length(modelVars), length(pc2))
for (idx, c) in enumerate(1:length(pc2))
    varPost2[:, :, idx] = reduce(vcat, [mean(Matrix(A1Dffil[row .== c, modelVars]), dims=  1) for row in eachrow(cpred2[:, nmodes2[1,:]])])
end
varPost2[isnan.(varPost2)] .= missing



centers= mapslices(x -> mean(skipmissing(x)), varPost, dims = 1)[1,:,:]
#  match cluasters on posterior mean
centers2= mapslices(x -> mean(skipmissing(x)), varPost2, dims = 1)[1,:,:]


dists = Matrix{Float64}(undef, size(centers)[2], size(centers2)[2])
for (idx, col) in enumerate(eachcol(centers))
    for (idx2, col2) in enumerate(eachcol(centers2))
        dists[idx, idx2] = sum(abs.(col .- col2))
    end
end
#dists[:, 2] .= 3000

matches = Matrix{Int64}(undef, 8,2)
matchDist = Vector{Float64}(undef, 8)
for i in 1:8
    coords = argmin(dists)
    matchDist[i] = dists[coords.I[1], coords.I[2]]
    dists[coords.I[1], :] .= 3000
    dists[:, coords.I[2]] .= 3000
    matches[i, :] = [coords.I[1], coords.I[2]]
end

machDF = DataFrame(matches, :auto)
rename!(machDF, [:ARMS1, :ARMS2])
machDF.dist = matchDist
# CSV.write("matches.csv", machDF)

machDF = CSV.read("matches.csv", DataFrame)





plots = Vector{Plots.Plot}(undef, 8)
#violin!(p, fill(1, length(varPost[:, 1,1])), varPost[:,1,1])
for clu in 1:8
    clu1 = matches[clu, 1]
    df1 = @pipe DataFrame(varPost[:, :, clu1], :auto) |> 
      rename(_, modelVars) |>
      stack |>
      dropmissing |>
      transform(groupby(_, :variable), :value => (x-> quantile(x, 0.05)) => :q5, ungroup= true) |>
      transform(groupby(_, :variable), :value => (x-> quantile(x, 0.95)) => :q95, ungroup= true) |>
      filter(row -> row.q5 < row.value < row.q95, _) |>
      transform(groupby(_, :variable), :value => (x-> mean(x)) => :m, ungroup = true) |>
      transform(_, :variable => categorical => :variable) |>
      combine(groupby(_, :variable), 
          :value => (x -> quantile(x, 0.05)) => :lower,
          :value => (x -> quantile(x, 0.95)) => :upper,
      )
    df1.ARMS .= "ARMS1"
    df1.x = levelcode.(df1.variable)

    clu2 = matches[clu, 2]
    df2 = @pipe DataFrame(varPost2[:, :, clu2], :auto) |> 
      rename(_, modelVars) |>
      stack |>
      dropmissing |>
      transform(groupby(_, :variable), :value => (x-> quantile(x, 0.05)) => :q5, ungroup= true) |>
      transform(groupby(_, :variable), :value => (x-> quantile(x, 0.95)) => :q95, ungroup= true) |>
      filter(row -> row.q5 < row.value < row.q95, _) |>
      transform(groupby(_, :variable), :value => (x-> mean(x)) => :m, ungroup = true) |>
      transform(_, :variable => categorical => :variable) |>
      combine(groupby(_, :variable), 
          :value => (x -> quantile(x, 0.05)) => :lower,
          :value => (x -> quantile(x, 0.95)) => :upper,
      )
    df2.x = levelcode.(df2.variable)
    df2.ARMS .= "ARMS2"
    # df = vcat(df1, df2)

    width = 0.4
    p = plot(; xlim = (0.5 - width, length(levels(df1.variable)) + 0.5 + width),
         ylim = (minimum(df1.lower)-0.5 - width, maximum(df1.upper) + 0.5 + width),
            xticks = (1:length(levels(df1.variable)), levels(df1.variable)),
            legend=false
            )
    plot!(p, [NaN], [NaN], label = "ARMS1", fillcolor = :steelblue)
    plot!(p, [NaN], [NaN], label = "ARMS2", fillcolor = :red, legend = true, legendfont = font(10))
    for row in eachrow(df1)
         xleft = row.x - width /2
         xright = row.x + width /2
         x_rect = [xleft, xright, xright, xleft]
         y_rect = [row.lower, row.lower, row.upper, row.upper]
         plot!(p, Shape(x_rect .- (width/2), y_rect), fillcolor = :steelblue, label = nothing)
    end
    for row in eachrow(df2)
         xleft = row.x - width /2
         xright = row.x + width /2
         x_rect = [xleft, xright, xright, xleft]
         y_rect = [row.lower, row.lower, row.upper, row.upper]
         plot!(p, Shape(x_rect .+ (width/2), y_rect), fillcolor = :red, label= nothing)
    end

    plot!(p, title = "Subset" * string(clu), titlefont = font(10))
    plots[clu] = p
end
    #p=@df df1 groupedviolin(:variable, :value, side = :left, label = "ARMS1", outliers=false)
#   # @df df1 scatter!(:variable, :m, label = "ARMS1", side = :left)
    #@df df2 groupedviolin!(:variable, :value, side = :right, label = "ARMS2", outliers=  false)
#   # @df df2 scatter!(:variable, :m, label = "ARMS1", side = :right)


for sp in plots[1:6]
    xaxis!(sp, false, font = font(10))
    plot!(sp, legend = false, bottom_margin = -10*Plots.mm)
end
plot!(plots[7], legend = true, legendfont = font(5))
plot!(plots[8], legend = false)

p2 = plot(plots..., layout = (4,2), xrotation=45)
Plots.savefig(p2, "output/openTotal/groupCovarInterval.png")


# [ ] find sampling distributions for the Associations...
betas = reduce(hcat, reduce(hcat, [map(x -> x[:beta] ,s[:lik_params]) for s in sim1 if maximum(s[:C]) == 8]))'
betas2 = reduce(hcat, reduce(hcat, [map(x -> x[:beta] ,s[:lik_params]) for s in sim2 if maximum(s[:C]) == 8]))'

betas= DataFrame(betas, :auto) 
rename!(betas, [:Intercept, :age, :female, :uPosUrg, :uLplanning, :uLpers, :uNegUrg, :bbRR, :bbFS, :bbSum])
betas.Subset = repeat(1:8, Int(size(betas)[1]/ 8))
 
betas2= DataFrame(betas2, :auto) 
rename!(betas2, [:Intercept, :age, :female, :uPosUrg, :uLplanning, :uLpers, :uNegUrg, :bbRR, :bbFS, :bbSum])
betas2.Subset = repeat(1:8, Int(size(betas2)[1]/ 8))

betas = stack(betas, Not([:Subset])) |>
    tbl -> filter(row -> !(row.variable in ["Intercept", "female", "age"]), tbl)
betas2 = stack(betas2, Not([:Subset])) |>
    tbl -> filter(row -> !(row.variable in ["Intercept", "female", "age"]), tbl)

betas = @chain betas begin
  @group_by(Subset, variable)
  @filter(value > quantile(value, [0.05]))
  @filter(value < quantile(value, [0.95]))
  @ungroup
end
betas2 = @chain betas2 begin
  @group_by(Subset, variable)
  @filter(value > quantile(value, [0.05]))
  @filter(value < quantile(value, [0.95]))
  @ungroup
end

# @pipe betas |>
#     subset(_, :Subset => x -> x .== 1) |>
#     transform(groupby(_, [:variable, :Subset]), :value => (x-> quantile(x, 0.10)) => :q5) |>
#     transform(groupby(_, [:variable, :Subset]), :value => (x-> quantile(x, 0.9)) => :q95) |>
#     filter(row -> row.q5 < row.value < row.q95, _)
    

plots = Vector{Plots.Plot}(undef, 8)
for clu in 1:8
    df1 = @pipe betas |>
        subset(_, :Subset => x -> x .== matches[clu, 1]) |>
        transform(groupby(_, [:variable, :Subset]), :value => (x-> quantile(x, 0.1)) => :q5) |>
        transform(groupby(_, [:variable, :Subset]), :value => (x-> quantile(x, 0.9)) => :q95) |>
        filter(row -> row.q5 < row.value < row.q95, _) |>
        transform(_, :variable => categorical => :variable) |>
        combine(groupby(_, :variable), 
            :value => (x -> quantile(x, 0.1)) => :lower,
            :value => (x -> quantile(x, 0.9)) => :upper,
        )
    df1.x = levelcode.(df1.variable)
    df1.ARMS .= "ARMS1"
    df2 =@pipe betas2 |>
        subset(_, :Subset => x -> x .== matches[clu, 2]) |>
        transform(groupby(_, [:variable, :Subset]), :value => (x-> quantile(x, 0.1)) => :q5) |>
        transform(groupby(_, [:variable, :Subset]), :value => (x-> quantile(x, 0.9)) => :q95) |>
        filter(row -> row.q5 < row.value < row.q95, _) |>
        transform(_, :variable => categorical => :variable) |>
        combine(groupby(_, :variable), 
            :value => (x -> quantile(x, 0.1)) => :lower,
            :value => (x -> quantile(x, 0.9)) => :upper,
        )
    df2.x = levelcode.(df2.variable)
    df2.ARMS .= "ARMS2"
    
    width = 0.4
    p = plot(; xlim = (0.5 - width, length(levels(df1.variable)) + 0.5 + width),
         ylim = (minimum(df1.lower)-0.5 - width, maximum(df1.upper) + 0.5 + width),
            xticks = (1:length(levels(df1.variable)), levels(df1.variable)),
            legend=false
            )
    plot!(p, [NaN], [NaN], label = "ARMS1", fillcolor = :steelblue)
    plot!(p, [NaN], [NaN], label = "ARMS2", fillcolor = :red, legend = true, legendfont = font(10))
    for row in eachrow(df1)
         xleft = row.x - width /2
         xright = row.x + width /2
         x_rect = [xleft, xright, xright, xleft]
         y_rect = [row.lower, row.lower, row.upper, row.upper]
         plot!(p, Shape(x_rect .- (width/2), y_rect), fillcolor = :steelblue, label = nothing)
    end
    for row in eachrow(df2)
         xleft = row.x - width /2
         xright = row.x + width /2
         x_rect = [xleft, xright, xright, xleft]
         y_rect = [row.lower, row.lower, row.upper, row.upper]
         plot!(p, Shape(x_rect .+ (width/2), y_rect), fillcolor = :red, label= nothing)
    end

    plot!(p, title = "Subset" * string(clu), titlefont = font(10))
    plots[clu] = p
end

for sp in plots[7:8]
    plot!(sp, bottom_margin = 2*Plots.mm, legend = false)
    xaxis!(sp, true, font = font(10))
end
for sp in plots[1:6]
  xaxis!(sp, false)
  plot!(sp, bar_width = 8, bottom_margin = -15*Plots.mm, legend = false)
end
plot!(plots[7], legend = true, legendfont = font(5))
p2 = plot(plots..., layout = (4,2), xrotation = 45)
Plots.savefig(p2, "output/openTotal/assocBox.png")


ypred, cpred = postPred(Xtrain, model, sim1[cs[1,:]][1:100:end])
ypred, cpred2 = postPred(Xtrain2, model, sim1[cs[1,:]][1:100:end])


A2train.m1c = mapslices(x -> mode(x), cpred2, dims =1)[1,:]
A1train.m1c = mapslices(x -> mode(x), cpred, dims =1)[1,:]
tabulate(A2train[1:1497, :], :m1c, :adhdLevel) |>
    tbl -> write_tex("output/openTotal/adhdcountsm1.tex", tbl)
tabulate(A1train[1:1498, :], :m1c, :adhdLevel) |>
    tbl -> write_tex("output/openTotal/adhdcountsm2.tex", tbl)

