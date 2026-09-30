# salsoUtils.jl
# Decision-theoretic point estimates of the posterior partition via the R
# `salso` package (Dahl, Johnson, & Müller, 2022; https://CRAN.R-project.org/package=salso).
# Supports Binder loss and variation of information (VI), which reviewers requested.
#
# Relies on RCall + the R `salso` package. If either is unavailable the helpers
# return `nothing` and callers should treat results as missing.

using RCall

function salso_available()
    try
        RCall.reval("suppressPackageStartupMessages(library(salso))")
        return true
    catch
        @warn "RCall or the R 'salso' package is unavailable; SALSO metrics will be missing."
        return false
    end
end

"""
    salso_partition(C_mat, loss::Symbol=:VI; nRuns=16)

Given an `S × N` matrix of posterior cluster allocations (`S` MCMC draws, `N`
subjects), find the point-estimate partition minimizing posterior expected loss
under Binder loss (`loss=:binder`) or variation of information (`loss=:VI`).
Returns an `N`-vector of cluster labels, or `nothing` if R/salso is unavailable.
"""
function salso_partition(C_mat::AbstractMatrix{<:Integer}; loss::Symbol=:VI, nRuns::Int=16)
    salso_available() || return nothing

    # RCall expects a 0-based? No - integer matrix; salso treats equal labels as same cluster.
    CmatR = Int.(C_mat)

    if loss == :binder
        RCall.@rput CmatR
        RCall.@rput nRuns
        RCall.reval("part <- salso(CmatR, loss=binder(), nRuns=nRuns)")
    elseif loss == :VI
        RCall.@rput CmatR
        RCall.@rput nRuns
        RCall.reval("part <- salso(CmatR, loss=VI(), nRuns=nRuns)")
    else
        error("loss must be :binder or :VI")
    end

    part = RCall.rcopy(RCall.reval("part"))
    return Vector{Int}(vec(part))
end

"""
    salso_ari(C_mat, truth; loss=:VI, nRuns=16)

Partition point estimate (Binder or VI loss) followed by the adjusted Rand
index against `truth`. Returns a NamedTuple `(ari, nclusters)` or
`(ari=missing, nclusters=missing)` if R/salso unavailable.
"""
function salso_ari(C_mat::AbstractMatrix{<:Integer}, truth::AbstractVector{<:Integer};
                   loss::Symbol=:VI, nRuns::Int=16)
    part = salso_partition(C_mat; loss=loss, nRuns=nRuns)
    if part === nothing
        return (ari=missing, nclusters=missing)
    end
    ari = Clustering.randindex(part, collect(Int, truth))[1]
    return (ari=ari, nclusters=length(unique(part)))
end

"""
    binder_distance(a::Vector{Int}, b::Vector{Int})

Fraction of subject pairs on which two partitions disagree, i.e.

    d_B(a, b) = (1 / n^2) * sum_{i,j} 1{a_i != b_j and a_i != b_j and
                                           a_j != b_i and a_j != b_i}

This is the normalized Binder loss on a common scale in [0, 1]: 0 when the
two partitions are identical, 1 when they share no pair in common.
"""
function binder_distance(a::Vector{Int}, b::Vector{Int})
    n = length(a)
    n == length(b) || throw(DimensionMismatch("partitions must have equal length"))
    S = 0
    for i in 1:n, j in (i + 1):n
        if a[i] != a[j] && b[i] != b[j]
            S += 1
        end
    end
    return 2.0 * S / (n * n)
end

"""
    expected_binder_loss(C_mat, part::Vector{Int})

Mean Binder loss of the candidate `part` taken over the posterior sample of
partitions `C_mat` (draws in rows, subjects in columns). This is the
objective that SALSO minimizes when it selects a point estimate, so it is
the quantity to report as the "Binder loss" of the fitted partition.

Only defined for methods with a posterior over partitions. For a single
deterministic partition (k-means, DP-GMM) there is no such expectation, and
callers should record `missing`.
"""
function expected_binder_loss(C_mat::AbstractMatrix{<:Integer}, part::Vector{Int})
    n = size(C_mat, 2)
    size(C_mat, 2) == n || throw(DimensionMismatch("sample and partition disagree on n"))
    total = 0.0
    for s in axes(C_mat, 1)
        total += binder_distance(Vector{Int}(C_mat[s, :]), part)
    end
    return total / size(C_mat, 1)
end

"""
    salso_with_loss(C_mat; nRuns=16)

Returns `(part=..., loss=...)` where `part` is the Binder point estimate and
`loss` is the minimized expected Binder loss, or `(part=nothing,
loss=missing)` if R/salso is unavailable. Unlike `salso_ari` this also returns
the loss value, which the paper reports alongside the ARI of the point
estimate.
"""
function salso_with_loss(C_mat::AbstractMatrix{<:Integer}; nRuns::Int=16)
    part = salso_partition(C_mat; loss=:binder, nRuns=nRuns)
    part === nothing && return (part=nothing, loss=missing)
    return (part=part, loss=expected_binder_loss(C_mat, part))
end