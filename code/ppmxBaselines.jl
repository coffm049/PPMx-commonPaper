#
# ppmxBaselines.jl
#
# Complementary baselines / models for the PPMx-common paper on ABCD data:
#   1. "standard PPMx"   - PPMx with no common-effect model (mixDPM=false).
#      Unlike the common-effects model, `mcmc!` does NOT record
#      `:prior_mean_beta` / `:alpha`, so downstream summaries must use the
#      per-cluster regression coefficients stored in `s[:lik_params][k][:beta]`.
#   2. DP Gaussian mixture clustering baseline (module `DPMM` from DPMM.jl)
#      - cluster the covariate space, then fit a per-cluster interaction
#        regression (mirrors the existing k-means baseline).
#
# These helpers are shared by the ABCD scripts (05-ppmxStd-ABCD.jl and
# 06-baselines-ABCD.jl). They assume Pkg has been activated on the
# simulations environment and that DPMM has been dev-installed from the
# simulations/vendor/DPMM path (see README in that folder).

using StatsModels
using GLM
using Clustering
using JLD2
using Statistics
using Distributions

"""
    DPMM.ChainRecorder

Accumulates the label vector at each Gibbs iteration. Pass as the `scene`
keyword to `DPMM.fit` to retain the chain.
"""
mutable struct ChainRecorder
    chains::Vector{Vector{Int}}
end

function DPMM.record!(rec::ChainRecorder, labels, t)
    push!(rec.chains, copy(labels))
    return nothing
end

"""
    fit_standardPPMx(y, X, groupings; nburn=10000, nmc=6000, outfile::Union{String,Nothing}=nothing,
                     priors...)

Fit the "standard PPMx" model (mixDPM=false) following the same prior
setup as the PPMx-common ABCD scripts. Returns `(sim, model)`.
If `outfile` is given, also `@save`s it as `-std.jld2`.
"""
function fit_standardPPMx(y, X, groupings; nburn=10000, nmc=6000,
                          outfile::Union{String,Nothing}=nothing, kwargs...)
    model = Model_PPMx(y, X, groupings, similarity_type=:NN,
                       sampling_model=:Reg, init_lik_rand=true)

    prec = 0.01; alph = 1.0; bet = 1.0
    dims = size(X)[2]
    model.prior.base = Prior_base(
        repeat([0.0], dims),
        repeat([prec], dims),
        repeat([alph], dims),
        repeat([bet], dims),
    )
    model.prior.massParams = [1, 1]
    model.state.baseline.tau0 = 1e6

    mcmc!(model, nburn; mixDPM=false)
    sim = mcmc!(model, nmc; mixDPM=false)

    if !isnothing(outfile)
        @save outfile sim model
    end
    return sim, model
end

"""
    perClusterBetas(sim) -> Dict{Int, Vector{Vector{Float64}}}

Extract sampled per-cluster regression-coefficient vectors from each chain
iteration. Columns of each beta vector are the model covariates (posterior
mean from `:lik_params`). Standard PPMx does not record a global common beta.
"""
function perClusterBetas(sim)
    out = Dict{Int, Vector{Vector{Float64}}}()
    for (i, s) in enumerate(sim)
        nC = s[:C]
        out[i] = [s[:lik_params][k][:beta] for k in 1:nC]
    end
    return out
end

"""
    summarize_standardPPMx(sim, coefNum, ci=0.9) -> (median_beta, [qlo, qhi])

Posterior median and central interval for regression coefficient `coefNum`
across the most-popular number of clusters (mode of `s[:C]`). Standard PPMx
summarizes per-cluster betas only (no common-effect beta).
"""
function summarize_standardPPMx(sim, coefNum, ci=0.9)
    nc = mode([maximum(s[:C]) for s in sim])
    betas = [s[:lik_params][k][:beta][coefNum] for s in sim
             if maximum(s[:C]) == nc for k in 1:maximum(s[:C])]
    return median(betas), quantile(betas, [(1 - ci) / 2, (1 + ci) / 2])
end

"""
    standardPPMx_PP(Xtest, model, sim) -> (yPred, cPred)

Posterior predictive draws for a standard-PPMx chain (no prior_mean_beta).
Delegates to the model's `postPred`.
"""
function standardPPMx_PP(Xtest, model, sim)
    return postPred(Xtest, model, sim)
end

"""
    fit_DPMclustering(X; alpha=1.0, iters=200) -> (labels, centroids)

Cluster the covariate columns of `X` (columns = points) with a DP Gaussian
mixture using the DPMM.jl collapsed-Gibbs sampler. Returns integer labels
(1 per point) and the per-cluster centroids (dim x k).
"""
function fit_DPMclustering(X; alpha=1.0, iters=200)
    labels = Vector{Int}(DPMM.fit(X; algorithm=DPMM.CollapsedAlgorithm,
                                  α=alpha, T=iters))
    ks = unique(labels)
    centroids = hcat([vec(mean(X[:, labels .== k], dims=2)) for k in ks]...)
    return labels, centroids
end

"""
    fit_DPMclustering_chain(X; alpha=1.0, iters=500, burnin=250) -> (labels, chains, ks)

Like `fit_DPMclustering` but retains the full Gibbs chain of label vectors.
`chains` is a vector of `Vector{Int}` (one per post-burn-in iteration), and
`ks` is the sorted vector of unique cluster labels in the final state.
"""
function fit_DPMclustering_chain(X; alpha=1.0, iters=500, burnin=250)
    rec = DPMM.ChainRecorder(Vector{Vector{Int}}())
    labels = Vector{Int}(DPMM.fit(X; algorithm=DPMM.CollapsedAlgorithm,
                                  α=alpha, T=iters, scene=rec))
    chains = rec.chains
    # discard burn-in
    post = chains[(burnin+1):end]
    ks = unique(labels)
    return labels, post, ks
end

"""
    dpm_posterior_stats(chains)

Returns a NamedTuple with:
  - nclusts: posterior distribution of the number of clusters
  - sizes: posterior distribution of sorted cluster sizes
"""
function dpm_posterior_stats(chains)
    nclusts = [length(unique(c)) for c in chains]
    sizes = [sort([count(==(k), c) for k in unique(c)]) for c in chains]
    return (nclusts=nclusts, sizes=sizes)
end

"""
    assign_to_centroids(X, centroids) -> Vector{Int}

Assign each column of `X` (a point) to the index of the nearest column of
`centroids`. Used to project held-out (test) observations onto clusters
learned in training.
"""
function assign_to_centroids(X::AbstractMatrix, centroids::AbstractMatrix)
    return [argmin([norm(X[:, i] .- centroids[:, k])
                    for k in 1:size(centroids, 2)]) for i in 1:size(X, 2)]
end

"""
    dpm_regression_compare(trainDf, testDf, clustVars, predVars, outcome,
                           dmnLabels; alpha=1.0, iters=500, scale=1.1,
                           return_chain=false)

Cluster the training covariate matrix (columns = points, scaled by `scale`)
with a DP-GMM, project test rows onto the
same clusters via `assign_to_centroids`, fit a per-cluster interaction
regression of `outcome` on `predVars` crossed with `kclust`, and return a
NamedTuple of
  (ari, arioos, rmseoos, nclusts, trainLabels, testLabels, clustlm, lpsOOS).
If `return_chain=true`, also returns the post-burn-in Gibbs chain of label
vectors (`chain`) and posterior statistics (`poststats`).
ARI is computed against the truth partition `dmnLabels` on rows where that
column is observed; `clustlm`/`rmseoos` are `nothing`/`NaN` when the DPM
collapses to a single cluster.
"""
function dpm_regression_compare(trainDf, testDf, clustVars, predVars, outcome,
                                dmnLabels; alpha=1.0, iters=500, scale=1.1,
                                return_chain=false)
    Xtr = convert(Matrix{Float64}, Matrix(trainDf[:, clustVars])) .* scale
    Xtr = Xtr'   # dims x N; columns = points for DPMM (no intercept column)

    if return_chain
        labels, chains, ks = fit_DPMclustering_chain(Xtr; alpha=alpha, iters=iters)
    else
        labels, centroids = fit_DPMclustering(Xtr; alpha=alpha, iters=iters)
    end
    # relabel train compactly (1..K, same order as centroid columns) so train labels
    # and OOS nearest-centroid labels use the same scheme
    trainDf.kclust = string.(indexin(labels, unique(labels)))

    Xte = convert(Matrix{Float64}, Matrix(testDf[:, clustVars])) .* scale
    Xte = Xte'
    if return_chain
        # project test points onto final-state centroids
        centroids = hcat([vec(mean(Xtr[:, labels .== k], dims=2)) for k in ks]...)
    end
    testLabels = assign_to_centroids(Xte, centroids)
    testDf.kclust = string.(testLabels)

    # per-cluster interaction regression: (predVars) * kclust
    crossVars = intersect(predVars, clustVars)
    linearTerms = reduce(+, [term(Symbol(v)) for v in crossVars])
    form = term(outcome) ~ linearTerms * term(:kclust)
    contrasts = Dict(:kclust => EffectsCoding())
    randok = all(x -> x >= 2, [count(==(k), labels) for k in unique(labels)])
    clustlm = randok ? lm(form, trainDf; contrasts=contrasts) : nothing
    rmseoos = randok ? sqrt(mean(((predict(clustlm, testDf)) .- testDf[!, outcome]) .^ 2)) : NaN

    # ARI against the truth partition, only on rows where the truth is observed
    oktr = findall(!ismissing, trainDf[!, dmnLabels])
    okte = findall(!ismissing, testDf[!, dmnLabels])
    ari    = length(oktr) > 0 ? Clustering.randindex(labels[oktr], Int.(vec(trainDf[oktr, dmnLabels])))[1] : missing
    arioos = length(okte) > 0 ? Clustering.randindex(testLabels[okte], Int.(vec(testDf[okte, dmnLabels])))[1] : missing

    # OOS log predictive score (proper scoring rule): Normal predictive with
    # residual SD from the per-cluster interaction LM.
    lpsOOS = NaN
    if randok
        predDpm = predict(clustlm, testDf)
        residDpm = testDf[!, outcome] .- predDpm
        sdDpm = std(residDpm)
        lpsOOS = mean(logpdf.(Ref(Normal(0.0, sdDpm)), residDpm))
    end

    if return_chain
        poststats = dpm_posterior_stats(chains)
        return (ari=ari, arioos=arioos, rmseoos=rmseoos,
                nclusts=length(unique(labels)),
                trainLabels=labels, testLabels=testLabels, clustlm=clustlm,
                lpsOOS=lpsOOS, chain=chains, poststats=poststats)
    else
        return (ari=ari, arioos=arioos, rmseoos=rmseoos,
                nclusts=length(unique(labels)),
                trainLabels=labels, testLabels=testLabels, clustlm=clustlm,
                lpsOOS=lpsOOS)
    end
end