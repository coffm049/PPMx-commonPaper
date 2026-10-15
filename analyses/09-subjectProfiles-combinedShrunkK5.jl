using Pkg
# v2.0: absolute shared env (relative ../../simulations/ resolved to ~/papers/simulations, an empty project).
Pkg.activate(expanduser("~/software/ProductPartitionModels.jl/simulations"))
# Distances is a transitive dep of Clustering but not in Project.toml; add if missing.
try
    using Distances
catch
    Pkg.add("Distances")
    using Distances
end
using JLD2, CSV, DataFrames, Plots, StatsPlots, Statistics, StatsBase, TidierData, Measures
using CategoricalArrays
# 300 dpi for the manuscript figures: dpi is a plot attribute in this
# Plots build, read by the GR backend at save time (backends/gr.jl:2092).
Plots.default(dpi=300)
include("../code/loadNClean.jl")
include("../code/salsoUtils.jl")

# --- Load combined shrunk K=5 (paper's primary) ---
fullDf, sds = loadNclean()
fullDf = CSV.read("output/sampledDF.csv", DataFrame)
transform!(fullDf, [:ADHD1, :ADHD2, :ADHD3, :ADHD4] => ByRow((a1,a2,a3,a4)-> a4==1 ? 4 : a3==1 ? 3 : a2==1 ? 2 : a1==1 ? 1 : 0) => :adhdLevel)
A1train = innerjoin(fullDf, CSV.read("../data/a1TrFl.csv", DataFrame, header=["IID"]), on="IID")
A1test  = innerjoin(fullDf, CSV.read("../data/a1TeFl.csv", DataFrame, header=["IID"]), on="IID")
A2train = innerjoin(fullDf, CSV.read("../data/a2TrFl.csv", DataFrame, header=["IID"]), on="IID")
A2test  = innerjoin(fullDf, CSV.read("../data/a2TeFl.csv", DataFrame, header=["IID"]), on="IID")
A1train = A1train[completecases(A1train),:]; A2train = A2train[completecases(A2train),:];
A1test  = A1test[completecases(A1test),:];  A2test = A2test[completecases(A2test),:];
train = vcat(A1train, A2train); test = vcat(A1test, A2test)
sds[:female]=(1.0,1.0); sds[:int]=(1.0,1.0)
modelVars = [:age,:female,:uPosUrg,:uLplanning,:uLpers,:uNegUrg,:bbRR,:bbFS,:bbSum]
Xtrain = convert(Matrix{Float64}, Matrix(train[:, modelVars]) .* 1.1); Xtrain = hcat(ones(size(Xtrain,1),1), Xtrain)
Xtest  = convert(Matrix{Float64}, Matrix(test[:, modelVars])  .* 1.1); Xtest  = hcat(ones(size(Xtest,1),1), Xtest)

# Covariates for which per-covariate deviation profiles are produced.
# `female` is EXCLUDED, for two reasons.
#
# 1. Methodological: D_i = beta_{C_i} - beta* for a *binary* covariate has no
#    interpretable kernel density or heatmap. The manuscript correspondingly
#    reports only "the eight continuous covariates" as separately rendered.
#    Sex is instead summarised as a proportion per subset (see
#    paper/figures/sexProportionBySubset.R -> tab:sexBySubset).
#
# 2. Data integrity: in the v2.0 refit chains the `female` slice of D -- and
#    therefore of Dbar -- is uninitialised garbage (the constant 2.0256e97 for
#    every subject), while the matching column of betaBar is clean. Same
#    signature as the `community` column dropped from anchors below. Emitting
#    it puts a corrupt column into a git-tracked artifact.
contVars = [v for v in modelVars if v != :female]
contCols = [findfirst(==(v), modelVars) + 1 for v in contVars]  # +1: Dbar col 1 is the intercept

# v2.0 estimator: read the v2.0 refit chains (see 01sub-ppmxTotFulla-shrunk.jl).
@load "output/openTotalFull2-v2.jld2" sim model
sim1=sim
@load "output/openTotalFull3-v2.jld2" sim
sim2=sim
simAll = vcat(sim1, sim2)
# v2.0: saved chains contain a few contaminant entries (bare lik_param/baseline
# dicts); keep only full per-iteration state dicts.
is_full_draw(s) = s isa AbstractDict && haskey(s, :C) && haskey(s, :prior_mean_beta) &&
    haskey(s, :lik_params) && s[:lik_params] isa AbstractVector && !isempty(s[:lik_params]) &&
    all(lp -> lp isa AbstractDict && haskey(lp, :mu) && haskey(lp, :sig) && haskey(lp, :beta), s[:lik_params])
n_raw = length(simAll)
simAll = filter(is_full_draw, simAll)
@info "Loaded $n_raw draws (dropped $(n_raw - length(simAll)) contaminants), model.n=$(model.n), p=$(model.p)"
@info "Available keys in first sim: $(collect(keys(simAll[1])))"

# --- Find the cluster assignments key ---
# v2.0 chains with mixDPM=true store :C in each sim dict.
# If :C is missing, try alternative names.
cluster_key = :C
if !(:C in keys(simAll[1]))
    alt_keys = filter(k -> lowercase(string(k)) in ["c", "clusters", "assignments", "partition", "label"], collect(keys(simAll[1])))
    if !isempty(alt_keys)
        cluster_key = alt_keys[1]
        @info "Using alternative cluster key: $cluster_key"
    else
        error("No cluster assignments key found in sim objects. Available keys: $(collect(keys(simAll[1])))")
    end
end

# --- Filter to modal K=5 ---
nc = [maximum(s[cluster_key]) for s in simAll]
modeK = mode(nc)
@info "K distribution" countmap(nc)
sim = filter(s -> maximum(s[cluster_key])==modeK, simAll)
@assert modeK==5 "Expected K=5, got $modeK"
@info "Filtered to K=$modeK: $(length(sim)) draws"

S = length(sim); n = model.n; p = model.p
@assert p == length(modelVars)+1

# --- SALSO (Binder) + prototypes ---
Cmat = reduce(hcat, [s[cluster_key] for s in sim])' # S x n
@info "Computing SALSO Binder (nRuns=100)..."
cSALSO = salso_partition(Cmat; loss=:binder, nRuns=100)
if cSALSO === nothing
    @warn "SALSO unavailable, falling back to posterior median partition"
    cSALSO = [mode(r) for r in eachrow(Cmat)] # fallback not ideal but keeps script runnable
end
# --- Align the raw chain to the SALSO labelling before scoring prototypes ---
# The chain stores arbitrary integer labels that may permute across draws, while
# SALSO returns its own labelling. Scoring agreement with
#     mean(Cmat[:,i] .== cSALSO[i])
# therefore compares two different labellings and is meaningless -- it returned
# a consistency of exactly 0.0 for one of the five subsets. Instead, for each
# draw we find the one-to-one map from that draw's labels onto the SALSO labels
# that maximises subject co-occurrence, and score agreement in the aligned
# space. This makes the score a genuine measure of how often a subject sits in
# the SALSO cluster it was assigned to.
function align_to_salso(Cmat::AbstractMatrix{<:Integer}, cSALSO::AbstractVector{<:Integer})
    S, n = size(Cmat)
    K = maximum(cSALSO)
    aligned = zeros(Int, S, n)
    n_overflow = 0   # draws whose chain has more distinct labels than SALSO clusters
    for s in 1:S
        labels = sort(unique(Cmat[s, :]))
        length(labels) > K && (n_overflow += 1)
        # M[c, j] = #{i : cSALSO[i] == c and Cmat[s, i] == labels[j]}
        M = zeros(Int, K, length(labels))
        for i in 1:n
            M[cSALSO[i], searchsortedfirst(labels, Cmat[s, i])] += 1
        end
        # greedy max-weight one-to-one assignment
        used_c = falses(K); used_j = falses(length(labels))
        lab2salso = Dict{Int,Int}()
        for _ in 1:min(K, length(labels))
            bc, bj, bv = 0, 0, -1
            for c in 1:K, j in eachindex(labels)
                if !used_c[c] && !used_j[j] && M[c, j] > bv
                    (bc, bj, bv) = (c, j, M[c, j])
                end
            end
            bv <= 0 && break
            used_c[bc] = true; used_j[bj] = true
            lab2salso[labels[bj]] = bc
        end
        # Any label left over (only possible when a draw has >K distinct labels,
        # e.g. labels 0..5 with max==5) would otherwise map to 0 and make the
        # affected subject's consistency exactly 0 in every draw. Give it the
        # SALSO cluster it co-occurs with most, so it is never silently wrong.
        for l in labels
            haskey(lab2salso, l) && continue
            cnt = [sum((Cmat[s, i] == l) & (cSALSO[i] == c) for i in 1:n) for c in 1:K]
            lab2salso[l] = argmax(cnt)
        end
        for i in 1:n
            aligned[s, i] = lab2salso[Cmat[s, i]]
        end
    end
    n_overflow > 0 && @warn "$n_overflow/$S draws have more distinct chain labels than SALSO clusters (K=$K); leftover labels mapped to their best-matching cluster."
    return aligned
end

Caligned = align_to_salso(Cmat, cSALSO)
pmods = [mean(Caligned[:, i] .== cSALSO[i]) for i in 1:n]
q = quantile(pmods, [0.0, 0.01, 0.05, 0.5, 0.95, 1.0])
@info "Prototype consistency (chain aligned to SALSO)" q
@info "subjects with consistency < 0.01: $(sum(pmods .< 0.01)) / $n"
for c in 1:maximum(cSALSO)
    idx = findall(cSALSO .== c)
    @info "  subset $c: n=$(length(idx)) median consistency=$(median(pmods[idx]))"
end

prototypes = Int[]; pc = []
for c in 1:maximum(cSALSO)
    idx = findall(cSALSO .== c)
    # most stable subject within SALSO cluster
    best = argmax(pmods[idx])
    push!(prototypes, idx[best])
    push!(pc, pmods[idx[best]])
end
@info "Prototypes (SALSO Binder, K=$modeK)" prototypes pc

# --- D_i = beta_{C_i} - beta* per iteration (label-switch-free) ---
# D : n x p x S
D = Array{Float64}(undef, n, p, S)
beta_i = Array{Float64}(undef, n, p, S)
for s in 1:S
    C = sim[s][cluster_key]
    bStar = sim[s][:prior_mean_beta] # p-vector, includes intercept
    lik = sim[s][:lik_params]
    for i in 1:n
        k = C[i]
        b_ik = lik[k][:beta] # p-vector
        D[i,:,s] = b_ik .- bStar
        beta_i[i,:,s] = b_ik
    end
end
Dbar = dropdims(mean(D, dims=3), dims=3) # n x p
# also keep beta posterior mean per subject
betaBar = dropdims(mean(beta_i, dims=3), dims=3)

# --- Guard: never write an uninitialised/divergent slice to a tracked artifact ---
# A constant extreme value across every subject (e.g. 2.03e97) is the signature
# of an unwritten slice, not a posterior mean. Fail loudly instead of shipping it.
function assert_finite_cols(M, cols, names, label)
    for (c, nm) in zip(cols, names)
        v = M[:, c]
        bad = count(!isfinite, v)
        bad == 0 || error("$label column '$nm' has $bad non-finite value(s); refusing to write.")
        spread = maximum(v) - minimum(v)
        spread == 0 && @warn "$label column '$nm' is constant ($(v[1])) across all $(length(v)) subjects."
    end
end
assert_finite_cols(Dbar, contCols, string.(contVars), "Dbar")
assert_finite_cols(betaBar, [findfirst(==(v), modelVars) + 1 for v in modelVars], string.(modelVars), "betaBar")

# --- Ordering: salso -> prototypes -> Dbar hierarchical ---
# Primary: SALSO label
# Secondary: within each SALSO cluster, order by hierarchical clustering of Dbar (Euclidean on 9 covariate deviations, excluding intercept)
using Distances, Clustering
ord = Int[]
for c in 1:maximum(cSALSO)
    idx = findall(cSALSO .== c)
    if length(idx) == 1
        append!(ord, idx)
    else
        # distance on Dbar[:,2:end] (drop intercept)
        sub = Dbar[idx, 2:end]' # p-1 x m
        # if prototypes includes one of idx, put prototype first
        proto = intersect(prototypes, idx)
        # hierarchical within cluster
        if length(idx) > 2
            dm = pairwise(Euclidean(), sub, dims=2)
            hc = hclust(dm, linkage=:average)
            perm = hc.order
            ordered = idx[perm]
            # move prototype to front if present
            if !isempty(proto)
                ordered = vcat(proto[1], setdiff(ordered, proto[1]))
            end
            append!(ord, ordered)
        else
            append!(ord, idx)
        end
    end
end
@assert length(ord)==n
@info "Ordering: SALSO primary, prototype first within cluster, Dbar hierarchical"

# --- Save subject-level posteriors ---
mkpath("output/openTotal/subjectProfilesK5-v2")
CSV.write("output/openTotal/subjectProfilesK5-v2/Dbar.csv", DataFrame(Dbar[:, [1; contCols]], Symbol.(["Intercept"; string.(contVars)])) |> x-> hcat(DataFrame(IID=train.IID[ord], salso=cSALSO[ord], prototype= [i in prototypes for i in ord]), x[ord,:]))
CSV.write("output/openTotal/subjectProfilesK5-v2/betaBar.csv", DataFrame(betaBar, Symbol.(["Intercept"; string.(modelVars)])) |> x-> hcat(DataFrame(IID=train.IID, salso=cSALSO), x))

# --- Heatmap all continuous covariates together (excluding intercept) ordered as above ---
# Dbar is in z-units (since X standardized *1.1, Y*6); keep as is for heatmap, also optionally back-transform
covNames = string.(contVars)
heat = Dbar[ord, contCols]' # (p-1) x n
# symmetric color limits at 2nd/98th pct for visibility
clim = Tuple(quantile(vec(heat), [0.02, 0.98]))
hAll = plot(heat, seriestype=:heatmap, c=:RdBu, clim=clim, yflip=true, yticks=(1:length(covNames), covNames),
    xticks=false, colorbar_title="D = β_i - β* (z)",
    title="Subject profiles Dbar (Binder SALSO order, K=5, n=$n)")
# add SALSO cluster boundaries as vertical lines
boundaries = cumsum([count(cSALSO[ord].==c) for c in 1:maximum(cSALSO)])
for b in boundaries[1:end-1]; vline!(hAll, [b+0.5], lw=1.2, lc=:black, label=""); end
# prototype ticks
for p in prototypes; idx = findfirst(ord .== p); scatter!(hAll, [idx], [0.5], ms=6, mc=:gold, shape=:star5, label=""); end
savefig(hAll, "output/openTotal/subjectProfilesK5-v2/heatmap_Dbar_all9.png")
@info "Saved heatmap all9"

# --- Plot each continuous covariate separately so you can combine as needed ---
for (j, var) in enumerate(contVars)
    col = contCols[j] # Dbar column (1 is intercept)
    vals = Dbar[ord, col]
    # per-covariate heatmap (1 x n) as single row
    hm = plot(reshape(vals, 1, n), seriestype=:heatmap, c=:RdBu, clim=Tuple(quantile(vals,[0.02,0.98])), yflip=true,
        yticks=(1, [string(var)]), xticks=false, colorbar_title="D",
        title="$var : Dbar (Binder order)")
    for b in boundaries[1:end-1]; vline!(hm, [b+0.5], lw=1, lc=:black, label=""); end
    savefig(hm, "output/openTotal/subjectProfilesK5-v2/heatmap_$(var).png")
    # per-covariate violin/box per SALSO cluster (distribution across subjects)
    df = DataFrame(D = Dbar[:,col], salso = string.(cSALSO), IID=train.IID)
    p = @df df violin(string.(:salso), :D, fillcolor=:steelblue, alpha=0.6, legend=false, title="$var : Dbar per SALSO cluster")
    @df df boxplot!(string.(:salso), :D, fillcolor=:white, alpha=0.0, legend=false)
    hline!(p, [0], ls=:dash, lc=:grey40, label="")
    ylabel!(p, "D = β_i - β*")
    savefig(p, "output/openTotal/subjectProfilesK5-v2/violin_$(var).png")
    # per-covariate ridge of anchor prototypes vs rest
    # density of D_i posterior for anchor vs all
    anchorIdx = prototypes
    # posterior draws for anchors: D[anchor, col, :]
    # plot densities separately
    plt = plot(title="$var : anchor posterior vs population", xlabel="D", ylabel="density", legend=:topright)
    for (k, anc) in enumerate(anchorIdx)
        density!(plt, D[anc, col, :], label="anchor C$(k) (IID $(train.IID[anc]))", lw=2)
    end
    # population density
    density!(plt, vec(D[:, col, :]), label="all subjects pooled", lw=1.5, ls=:dash, lc=:black, alpha=0.6)
    vline!(plt, [0], lc=:grey40, ls=:dot, label="")
    savefig(plt, "output/openTotal/subjectProfilesK5-v2/density_anchor_$(var).png")
end
@info "Saved per-covariate heatmaps/violins/densities for $(length(contVars)) continuous covariates"

# --- Anchor tables ---
# `community` dropped: causes garbage values when written to CSV (train.community
# contains pointer/uninitialized data in some environments).
# `female` dropped: binary covariate, no interpretable deviation profile, and its
# D slice is uninitialised garbage in these chains (see contVars comment above).
anchorDF = DataFrame(IID=train.IID[prototypes], salso=1:modeK, consistency=pc,
    adhdLevel=train.adhdLevel[prototypes])
for (j,var) in enumerate(contVars); anchorDF[!, var] = Dbar[prototypes, contCols[j]]; end
CSV.write("output/openTotal/subjectProfilesK5-v2/anchors.csv", anchorDF)
@info "Saved anchors.csv with Dbar per anchor"

# --- Per-subject credible intervals (for later forest) ---
# Continuous covariates only, for the same reason as anchors.csv above.
summaryRows = []
for i in 1:n
    for (j,var) in enumerate(contVars)
        col = contCols[j]
        draws = D[i,col,:]
        qs = quantile(draws, [0.05,0.5,0.95])
        push!(summaryRows, (IID=train.IID[i], covariate=string(var), salso=cSALSO[i], q05=qs[1], q50=qs[2], q95=qs[3], mean=mean(draws)))
    end
end
CSV.write("output/openTotal/subjectProfilesK5-v2/subjectPostSummary.csv", DataFrame(summaryRows))
@info "Saved subjectPostSummary.csv"

# --- Posterior predictive checks using ProductPartitionModels.postPred ---
# Compute predictive draws for the held-out test set, grouped by SALSO cluster
yPred, cAlloc, meanPred = postPred(Xtest, model, simAll[1:100:end])
# Assign each test subject to their posterior modal SALSO cluster
test_salso = Vector{Int}(undef, size(yPred, 2))
for j in 1:size(yPred, 2)
    test_salso[j] = mode(cAlloc[:, j])
end
postPredDF = DataFrame(
    IID = test.IID,
    salso = test_salso,
    nTotal_pred = vec(mean(meanPred, dims=1))
)
CSV.write("output/openTotal/subjectProfilesK5-v2/postPred.csv", postPredDF)
@info "Saved postPred.csv with posterior predictive nTotal per test subject"

@info "Done. Outputs in output/openTotal/subjectProfilesK5-v2/ : heatmap_Dbar_all9.png, heatmap_*.png ($(length(contVars))), violin_*.png ($(length(contVars))), density_anchor_*.png ($(length(contVars))), anchors.csv, Dbar.csv, betaBar.csv, subjectPostSummary.csv, postPred.csv. Continuous covariates only; \`female\` excluded (see contVars comment)."
