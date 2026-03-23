include("../code/utilities.jl")


fullDf, standardization_params = loadNclean()

numCols = names(fullDF, x -> !(eltype(fullDF[:,x]) <: AbstractString))
numCols = numCols[2:(end -5)]

# explore data
cors = pairwise(cor, eachcol(fullDF[:, numCols]), skipmissing=:pairwise)
heatmap(cors)
histogram(vec(cors[abs.(cors) .<0.99]))

highCor = findall((cors .>0.75) .&& (cors .<0.99))
# a lot of the NIH toolbox outcomes are highly correlated
[(numCols[hc[1]], numCols[hc[2]]) for hc in highCor]

numCols = names(fullDF, x -> !(eltype(fullDF[:,x]) <: AbstractString))
UnicodePlots.histogram(collect(skipmissing(fullDF[:, numCols[20]])))
# uNegUrg - bimodal with peaks at +- 1
# uPosUrg - bimodal with peaks at +- 1
# bbRR - bimodal -0.25, 1.25 (1.25 seems to be censored too, happens at boundary)
# bbDrive - bimodal -1.25, 00.25 (-1.25 seems to be censored too, happens at boundary)
# Rest are symetric - all are normal looking or slightly skewed normal
