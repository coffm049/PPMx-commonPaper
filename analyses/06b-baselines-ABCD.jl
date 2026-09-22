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
using Distributions
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
using DPMM

include("../code/utilities.jl")
include("../code/loadNClean.jl")
include("../code/analysis.jl")
include("../code/analysis2.jl")
include("../code/ppmxBaselines.jl")
include("../code/salsoUtils.jl")

# ============================================================================
# 06b-baselines-ABCD.jl
# Sensitivity analysis: same as 06 but with female removed from covariates
# and no outcome/covariate post-standardisation scaling (*6 / *1.1).
# Outputs are written to output/baselines-nofemale/.
# ============================================================================

commonFiles = [
    "output/openTotalFullDatamcmc1-v2.jld2",
    "output/openTotalFullDatamcmc2-v2.jld2",
]
stdFile = "output/stdPPmxTot.jld2"
dataDir = "/projects/standard/feczk001/shared/projects/FEZ_USERS/feczk001/UPPS_ABCD_FRF/code/jacob/"
frfFiles = [
    dataDir * "ADHDscores_list_ARMS1_merged.csv",
    dataDir * "ADHDscores_list_ARMS2_merged.csv",
]
outcome = :nTotal

# ---- load + clean via loadNclean (self-contained, avoids stale sampledDF.csv) ----
fullDf, sds = loadNclean()
@info "loadNclean: $(nrow(fullDf)) subjects, $(ncol(fullDf)) columns, IID type: $(eltype(fullDf.IID))"

frfLabels = reduce(vcat, [CSV.read(f, DataFrame; select = ["subjectkey", "community"]) for f in frfFiles])
rename!(frfLabels, :subjectkey => :IID)
@info "frfLabels: $(nrow(frfLabels)) rows, IID type: $(eltype(frfLabels.IID))"
fullDf = leftjoin(fullDf, frfLabels, on = :IID)
@info "After FRF join: $(nrow(fullDf)) rows, community non-missing: $(sum(!ismissing, fullDf.community))"


A1train = innerjoin(fullDf, CSV.read("../data/a1TrFl.csv", DataFrame, header= ["IID"]), on = "IID")
A1test = innerjoin(fullDf, CSV.read("../data/a1TeFl.csv", DataFrame, header= ["IID"]), on = "IID")
A2train = innerjoin(fullDf, CSV.read("../data/a2TrFl.csv", DataFrame, header= ["IID"]), on = "IID")
A2test = innerjoin(fullDf, CSV.read("../data/a2TeFl.csv", DataFrame, header= ["IID"]), on = "IID")
@info "A1train: $(nrow(A1train)) rows, A2train: $(nrow(A2train)) rows"

# ---- NO FEMALE in covariate set ----
modelVars = [:age, :uPosUrg, :uLplanning, :uLpers, :uNegUrg, :bbRR, :bbFS, :bbSum]
predVars = [:age, :uPosUrg, :uLplanning, :uLpers]

# Check completecases only on core features (not community, which may be missing for non-FRF subjects)
featureCols = [modelVars..., :nTotal]
train = vcat(A1train[completecases(A1train[:, featureCols]), :], A2train[completecases(A2train[:, featureCols]), :])
test = vcat(A1test[completecases(A1test[:, featureCols]), :], A2test[completecases(A2test[:, featureCols]), :])
@info "train: $(nrow(train)) rows, test: $(nrow(test)) rows"
# FRF-matched subsets (ARI truth labels)
train_frf = train[.!ismissing.(train.community), :]
test_frf = test[.!ismissing.(test.community), :]
@info "FRF-matched: train_frf=$(nrow(train_frf)) rows, test_frf=$(nrow(test_frf)) rows"

# Truth labels: all FRF communities are in test set
teComm = findall(!ismissing, test_frf.community)
commTe = Int.(vec(test_frf[teComm, :community]))
trComm = Int[]
commTr = Int[]

# ---- NO post-standardisation scaling (*1.1 on covariates, *6 on outcome) ----
Xtrain = convert(Matrix{Float64}, Matrix(train[:, modelVars]))
Xtrain = hcat(ones(size(Xtrain, 1), 1), Xtrain)
Xtest = convert(Matrix{Float64}, Matrix(test_frf[:, modelVars]))
Xtest = hcat(ones(size(Xtest, 1), 1), Xtest)
ytrain = convert(Vector{Float64}, train[:, outcome])
ytest = convert(Vector{Float64}, test_frf[:, outcome])

# Safe ARI helper (returns missing for empty inputs)
safe_ari(a, b) = (length(a) == 0 || length(b) == 0) ? missing : Clustering.randindex(a, b)[1]

# ============================================================================
# 1. k-means baseline (on covariate columns)
# ============================================================================
kclust = argmin([kmeans(Xtrain[:, 2:end]', i).totalcost for i in 2:20])
km = kmeans(Xtrain[:, 2:end]', kclust)
kmTe = assign_to_centroids(Xtest[:, 2:end]', km.centers)
ariK_tr = missing  # no FRF training subjects
ariK_te = safe_ari(kmTe[teComm], commTe)

# per-cluster interaction regression, OOS RMSE
train.kclust = string.(km.assignments)
test_frf.kclust = string.(kmTe)
formK = term(outcome) ~ reduce(+, [term(Symbol(v)) for v in predVars]) * term(:kclust)
ck = Dict(:kclust => EffectsCoding())
kmLm = lm(formK, train; contrasts=ck)
rmseK = sqrt(mean(((predict(kmLm, test_frf)) .- test_frf[!, outcome]) .^ 2))

# ============================================================================
# 2. DP Gaussian mixture baseline (module DPMM)
# ============================================================================
dpm = dpm_regression_compare(copy(train), copy(test_frf), modelVars, predVars, outcome, :community;
                             alpha=1.0, iters=500, scale=1.1)
dpmTr = dpm.trainLabels
dpmTe = dpm.testLabels

# ============================================================================
# 3. PPMx-common  (mixDPM=true)
# ============================================================================
# v2.0: saved chains contain a few contaminant entries (bare lik_param/baseline
# dicts); keep only full per-iteration state dicts. Std chain (mixDPM=false)
# has no :prior_mean_beta by design.
is_full_common(s) = s isa AbstractDict && haskey(s, :C) && haskey(s, :prior_mean_beta) &&
    haskey(s, :lik_params) && s[:lik_params] isa AbstractVector && !isempty(s[:lik_params]) &&
    all(lp -> lp isa AbstractDict && haskey(lp, :mu) && haskey(lp, :sig) && haskey(lp, :beta), s[:lik_params])
is_full_std(s) = s isa AbstractDict && haskey(s, :C) &&
    haskey(s, :lik_params) && s[:lik_params] isa AbstractVector && !isempty(s[:lik_params]) &&
    all(lp -> lp isa AbstractDict && haskey(lp, :mu) && haskey(lp, :sig) && haskey(lp, :beta), s[:lik_params])
simC = Dict{Symbol,Any}[]
for f in commonFiles
    @load f sim model
    append!(simC, sim)
end
@load commonFiles[end] sim model
modelC = model
nC0 = length(simC); simC = filter(is_full_common, simC)
@info "Common chains: dropped $(nC0 - length(simC)) contaminants, kept $(length(simC))"
# training ARI from chain allocations (mode-number-of-clusters iterations)
cC_tr = [maximum(s[:C]) for s in simC]
ncC = mode(cC_tr)
ariC_tr = safe_ari([s[:C] for s in simC if maximum(s[:C]) == ncC][end][trComm], commTr)

yC, cC = postPred(Xtest, modelC, simC[1:100:end])
meanC = vec(mean(yC, dims = 1))
rmseC = sqrt(mean((meanC .- ytest) .^ 2))
ariC_te = Clustering.randindex(vec(mode.(eachcol(cC)))[teComm], commTe)[1]

# ============================================================================
# 4. Standard PPMx  (mixDPM=false)
# ============================================================================
@load stdFile sim model
simS = sim; modelS = model
nS0 = length(simS); simS = filter(is_full_std, simS)
@info "Std chain: dropped $(nS0 - length(simS)) contaminants, kept $(length(simS))"
cS_tr = [maximum(s[:C]) for s in simS]
ncS = mode(cS_tr)
ariS_tr = safe_ari([s[:C] for s in simS if maximum(s[:C]) == ncS][end][trComm], commTr)

yS, cS = postPred(Xtest, modelS, simS[1:100:end])
meanS = vec(mean(yS, dims = 1))
rmseS = sqrt(mean((meanS .- ytest) .^ 2))
ariS_te = Clustering.randindex(vec(mode.(eachcol(cS)))[teComm], commTe)[1]

# ============================================================================
# 5. SALSO point-estimate ARIs (Binder + VI) vs FRF communities
#    Posterior cluster matrices built from the chain allocations (training)
#    and postPred draws (test); only subjects with observed FRF communities.
#    k-means / DP-GMM have no posterior over partitions -> missing.
# ============================================================================
salsoCmat_te = cC[:, teComm]
salsoSmat_te = cS[:, teComm]

salsoC_binder_te = salso_ari(salsoCmat_te, commTe; loss=:binder).ari
salsoC_vi_te     = salso_ari(salsoCmat_te, commTe; loss=:VI).ari
salsoS_binder_te = salso_ari(salsoSmat_te, commTe; loss=:binder).ari
salsoS_vi_te     = salso_ari(salsoSmat_te, commTe; loss=:VI).ari
salsoC_binder_tr = missing
salsoC_vi_tr     = missing
salsoS_binder_tr = missing
salsoS_vi_tr     = missing

# ============================================================================
# 6. Out-of-sample Log Predictive Score (LPS) -- proper scoring rule (Rev 1 C4)
#    Replaces the Bayesian predictive p-value as the model-comparison criterion.
# ============================================================================
function _logsumexp(X; dims)
    maxvals = maximum(X; dims=dims)
    lse = log.(sum(exp.(X .- maxvals); dims=dims)) .+ maxvals
    return dropdims(lse; dims=dims)
end

function logscore(lDens)
    return mean(_logsumexp(lDens; dims=1)[1, :] .- log(size(lDens, 1)))
end

# PPMx-common and standard PPMx: posterior predictive log-density of held-out y
lpsC = logscore(postPredLogdens(Xtest, ytest, modelC, simC[1:100:end]))
lpsS = logscore(postPredLogdens(Xtest, ytest, modelS, simS[1:100:end]))
# k-means: per-cluster interaction LM predictive (Normal, residual SD)
kmPred = predict(kmLm, test_frf)
kmResid = test_frf[!, outcome] .- kmPred
kmSD = std(kmResid)
lpsK = mean(logpdf.(Ref(Normal(0.0, kmSD)), kmResid))
# DP-GMM baseline (lpsOOS returned by dpm_regression_compare)
lpsDpm = dpm.lpsOOS

# ============================================================================
# 7. SALSO point-estimate vs the primary (modal) partition (Rev 2 C3, Rev 3 C3)
#    The modal partition is retained as the primary point estimate; SALSO is
#    evaluated against it (Binder + VI loss) to confirm partition stability.
# ============================================================================
modalC_te = vec(mode.(eachcol(cC)))[teComm]
modalS_te = vec(mode.(eachcol(cS)))[teComm]

function ari_modal_salso(Cmat_samples, modalVec)
    part = salso_partition(Cmat_samples; loss = :binder)
    part === nothing && return (binder = missing, vi = missing)
    viPart = salso_partition(Cmat_samples; loss = :VI)
    return (
        binder = safe_ari(modalVec, part),
        vi     = safe_ari(modalVec, viPart),
    )
end

cSalso_te = ari_modal_salso(salsoCmat_te, modalC_te)
sSalso_te = ari_modal_salso(salsoSmat_te, modalS_te)
cSalso_tr = (binder = missing, vi = missing)
sSalso_tr = (binder = missing, vi = missing)

# ============================================================================
# comparison table
# ============================================================================
comparison = DataFrame(
    model = ["PPMx-common", "PPMx (standard)", "k-means", "DP-GMM"],
    trainARI = [ariC_tr, ariS_tr, ariK_tr, safe_ari(dpmTr[trComm], commTr)],
    testARI = [ariC_te, ariS_te, ariK_te, safe_ari(dpmTe[teComm], commTe)],
    testRMSE = [rmseC, rmseS, rmseK, dpm.rmseoos],
    testLPS = [lpsC, lpsS, lpsK, lpsDpm],
    nclusters = [ncC, ncS, kclust, dpm.nclusts],
    trainARISalsoBinder = [salsoC_binder_tr, salsoS_binder_tr, missing, missing],
    trainARISalsoVI     = [salsoC_vi_tr, salsoS_vi_tr, missing, missing],
    testARISalsoBinder  = [salsoC_binder_te, salsoS_binder_te, missing, missing],
    testARISalsoVI      = [salsoC_vi_te, salsoS_vi_te, missing, missing],
    trainARImodalSalsoBinder = [cSalso_tr.binder, sSalso_tr.binder, missing, missing],
    trainARImodalSalsoVI     = [cSalso_tr.vi, sSalso_tr.vi, missing, missing],
    testARImodalSalsoBinder  = [cSalso_te.binder, sSalso_te.binder, missing, missing],
    testARImodalSalsoVI      = [cSalso_te.vi, sSalso_te.vi, missing, missing],
)
mkpath("output/baselines-nofemale")
CSV.write("output/baselines-nofemale/frftotalComparison.csv", comparison)
println(comparison)

# plots: ARI vs RMSE
scatter(comparison.testARI, comparison.testRMSE,
        group = comparison.model, legend = :bottomleft, xlabel = "test ARI (FRF)",
        ylabel = "test RMSE", title = "FRF-total (no female): baselines vs PPMx")
Plots.savefig("output/baselines-nofemale/frftotalComparison.png")

println("Done baselines comparison (no female); outputs in output/baselines-nofemale/")
