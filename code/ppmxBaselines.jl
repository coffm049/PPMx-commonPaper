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
using Random

"""
    ChainRecorder

Accumulates the label vector at each Gibbs iteration. Pass as the `scene`
keyword to `DPMM.fit` to retain the chain. This type lives in `Main` (this
file is `include`d, not a module); only `record!` is added to `DPMM`.
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
    fit_DPMclustering_chain(X; alpha=1.0, iters=4000, burnin=nothing, seed=20240601)
        -> (labels, chains, ks, burnin_used)

Like `fit_DPMclustering` but retains the full Gibbs chain of label vectors.
`chains` is a vector of `Vector{Int}` (one per retained iteration), and
`ks` is the sorted vector of unique cluster labels in the final state.

Burn-in. `burnin=nothing` (the default) selects it from the chain rather than
hard-coding a fraction: the number of clusters `K` is the quantity the DP-GMM
posterior is summarized over, so we find the first iteration after which `K`
has stopped drifting, and discard everything before it. Specifically we split
the chain into blocks, take the modal `K` of the second half as the reference
(the chain end is assumed equilibrated), and keep the first block whose modal
`K` matches that reference and whose every later block also matches. This is
deliberately conservative: if `K` never settles, the whole chain is discarded
and we fall back to half the chain, with `burnin_source` reporting which rule
fired so the caller can see it was not a clean convergence.

The final state is always appended to the retained draws so the summary
partition is itself represented among them.
"""
function fit_DPMclustering_chain(X; alpha=1.0, iters=4000, burnin=nothing, seed=20240601)
    # Seeds the global RNG, which is what DPMM.jl draws from unless it manages a
    # private one. This makes the run reproducible *if* it uses the global RNG;
    # if it does not, the chain still varies run to run and only the reported
    # seed documents the intent.
    Random.seed!(seed)
    # NB: this file is `include`d into Main, so the struct is `Main.ChainRecorder`,
    # not a member of the `DPMM` module. Only `record!` is added to DPMM.
    rec = ChainRecorder(Vector{Vector{Int}}())
    labels = Vector{Int}(DPMM.fit(X; algorithm=DPMM.CollapsedAlgorithm,
                                  α=alpha, T=iters, scene=rec))
    chains = rec.chains
    if isempty(chains)
        error("DPMM.fit did not call record! on the ChainRecorder, so no Gibbs " *
              "chain was retained. The `scene` keyword or the record! signature " *
              "expected by the installed DPMM version has changed.")
    end

    if burnin === nothing
        burnin, burnin_source = select_dpm_burnin(chains)
    else
        burnin_source = "user-specified"
    end
    burnin = clamp(burnin, 0, length(chains) - 1)

    post = chains[(burnin+1):end]
    # keep the final state so the summary partition is among the retained draws
    post[end] == labels || push!(post, copy(labels))
    ks = unique(labels)
    return labels, post, ks, burnin, burnin_source
end

"""
    select_dpm_burnin(chains; nblocks=20, tol=0) -> (burnin, source)

Data-driven burn-in for the DP-GMM chain, based on the trace of the number of
clusters `K`. Returns the number of leading iterations to discard and a short
string describing which rule fired.

Rationale: `K` is what the DP-GMM posterior is summarized over and what the
real-data table reports, so a chain whose `K` is still drifting has not
settled. We compare each block's modal `K` against the modal `K` of the second
half of the chain and keep the first block from which every remaining block
agrees. `tol` allows that many clusters of slack, for chains that hover
between two values. Falls back to half the chain when no block qualifies,
which is the honest default for a chain that never settles.
"""
function select_dpm_burnin(chains; nblocks::Int=20, tol::Int=0)
    n = length(chains)
    n < 4 && return (0, "chain too short ($(n) draws); no burn-in discarded")

    # blocks must be at least 2 draws wide, else a block can be empty
    nblocks = min(nblocks, max(2, div(n, 2)))

    Ks = [length(unique(c)) for c in chains]
    edges = round.(Int, range(1, n + 1; length=nblocks + 1))
    blockmodal = [mode(Ks[edges[b]:(edges[b+1] - 1)]) for b in 1:nblocks]

    # reference: modal K over the second half of the chain
    half = div(n, 2) + 1
    ref = mode(Ks[half:end])

    for b in 1:nblocks
        if all(abs.(blockmodal[b:end] .- ref) .<= tol)
            burnin = edges[b] - 1
            return (burnin, "K stable from block $b (blockmodal K=$(blockmodal[b:end]), ref=$ref)")
        end
    end
    return (half - 1, "K never stabilised (blockmodal K=$(blockmodal), ref=$ref); discarded first half")
end

"""
    dpm_posterior_stats(chains)

Returns a NamedTuple with:
  - nclusts: posterior distribution of the number of clusters
  - sizes: posterior distribution of sorted cluster sizes (one vector per draw,
    so the length varies with the number of clusters)
"""
function dpm_posterior_stats(chains)
    nclusts = [length(unique(c)) for c in chains]
    sizes = [sort([count(==(k), c) for k in unique(c)]) for c in chains]
    return (nclusts=nclusts, sizes=sizes)
end

"""
    dpm_posterior_summary(poststats) -> NamedTuple of scalars

Reduce the per-draw posterior draws to scalars that are safe to log and to
write to a CSV. `sizes` holds one variable-length vector per draw, so `mode`
is not meaningful on it; we report the modal number of clusters, the modal
cluster-size vector (by frequency), and the median largest/smallest cluster
size instead.
"""
function dpm_posterior_summary(poststats)
    nclusts = poststats.nclusts
    sizes = poststats.sizes
    modalK = isempty(nclusts) ? missing : mode(nclusts)
    # modal size vector: most frequent exact vector across draws
    modalSizes = missing
    if !isempty(sizes)
        counts = Dict{Vector{Int},Int}()
        for s in sizes
            counts[s] = get(counts, s, 0) + 1
        end
        modalSizes = reduce((a, b) -> counts[a] >= counts[b] ? a : b,
                            collect(keys(counts)))
    end
    largest = isempty(sizes) ? missing : median(last.(sizes))
    smallest = isempty(sizes) ? missing : median(first.(sizes))
    return (nclusts_mode=modalK,
            nclusts_median=isempty(nclusts) ? missing : median(nclusts),
            nclusts_min=isempty(nclusts) ? missing : minimum(nclusts),
            nclusts_max=isempty(nclusts) ? missing : maximum(nclusts),
            n_draws=length(nclusts),
            modal_sizes=modalSizes,
            median_largest_cluster=largest,
            median_smallest_cluster=smallest)
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
                                dmnLabels; alpha=1.0, iters=4000, scale=1.1,
                                return_chain=false, burnin=nothing, seed=20240601)
    Xtr = convert(Matrix{Float64}, Matrix(trainDf[:, clustVars])) .* scale
    Xtr = Xtr'   # dims x N; columns = points for DPMM (no intercept column)

    burnin_used = missing
    burnin_source = "no chain retained"
    if return_chain
        labels, chains, ks, burnin_used, burnin_source =
            fit_DPMclustering_chain(Xtr; alpha=alpha, iters=iters, burnin=burnin, seed=seed)
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
        postsummary = dpm_posterior_summary(poststats)
        return (ari=ari, arioos=arioos, rmseoos=rmseoos,
                nclusts=length(unique(labels)),
                trainLabels=labels, testLabels=testLabels, clustlm=clustlm,
                lpsOOS=lpsOOS, chain=chains, poststats=poststats,
                postsummary=postsummary, burnin=burnin_used,
                burnin_source=burnin_source, iters=iters, seed=seed)
    else
        return (ari=ari, arioos=arioos, rmseoos=rmseoos,
                nclusts=length(unique(labels)),
                trainLabels=labels, testLabels=testLabels, clustlm=clustlm,
                lpsOOS=lpsOOS)
    end
end