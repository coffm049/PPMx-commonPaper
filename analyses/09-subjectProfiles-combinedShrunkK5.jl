using Pkg
Pkg.activate("../../simulations/")
using JLD2, CSV, DataFrames, Plots, StatsPlots, Clustering, Statistics, StatsBase, TidierData, Measures
using CategoricalArrays
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

# v2.0 estimator: read the v2.0 refit chains (see 01sub-ppmxTotFulla-shrunk.jl).
@load "output/openTotalFull2-v2.jld2" sim model
sim1=sim
@load "output/openTotalFull3-v2.jld2" sim
sim2=sim
simAll = vcat(sim1, sim2)
@info "Loaded $(length(simAll)) draws, model.n=$(model.n), p=$(model.p)"

# --- Filter to modal K=5 ---
nc = [maximum(s[:C]) for s in simAll]
modeK = mode(nc)
@info "K distribution" countmap(nc)
sim = filter(s -> maximum(s[:C])==modeK, simAll)
@assert modeK==5 "Expected K=5, got $modeK"
@info "Filtered to K=$modeK: $(length(sim)) draws"

S = length(sim); n = model.n; p = model.p
@assert p == length(modelVars)+1

# --- SALSO (Binder) + prototypes ---
Cmat = reduce(hcat, [s[:C] for s in sim])' # S x n
@info "Computing SALSO Binder (nRuns=100)..."
cSALSO = salso_partition(Cmat; loss=:binder, nRuns=100)
if cSALSO === nothing
    @warn "SALSO unavailable, falling back to posterior median partition"
    cSALSO = [mode(r) for r in eachrow(Cmat)] # fallback not ideal but keeps script runnable
end
# prototypes as in 02-posteriorFull: max consistency >95%
# Use Cmat directly for prototypes (in-sample subjects = train)
mods = [mode(r) for r in eachrow(Cmat)]
# postPred for test not needed for train prototypes; use Cmat rows for consistency
# For train prototypes, use Cmat
pmods = vec(mean(Cmat .== mods, dims=2)[:,1]) # placeholder, will recompute per subject correct way:
# proper per-subject consistency vs modal
pmods = [mean(Cmat[:,i] .== cSALSO[i]) for i in 1:n]
prototypes = Int[]; pc=[]
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
    C = sim[s][:C]
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
CSV.write("output/openTotal/subjectProfilesK5-v2/Dbar.csv", DataFrame(Dbar, Symbol.(["Intercept"; string.(modelVars)])) |> x-> hcat(DataFrame(IID=train.IID[ord], salso=cSALSO[ord], prototype= [i in prototypes for i in ord]), x[ord,:]))
CSV.write("output/openTotal/subjectProfilesK5-v2/betaBar.csv", DataFrame(betaBar, Symbol.(["Intercept"; string.(modelVars)])) |> x-> hcat(DataFrame(IID=train.IID, salso=cSALSO), x))

# --- Heatmap all 9 together (excluding intercept) ordered as above ---
# Dbar is in z-units (since X standardized *1.1, Y*6); keep as is for heatmap, also optionally back-transform
covNames = string.(modelVars)
heat = Dbar[ord, 2:end]' # (p-1) x n
# symmetric color limits at 2nd/98th pct for visibility
clim = quantile(vec(heat), [0.02, 0.98])
hAll = heatmap(heat, c=:RdBu, clim=clim, yflip=true, yticks=(1:length(covNames), covNames),
    xticks=false, colorbar_title="D = β_i - β* (z)",
    title="Subject profiles Dbar (Binder SALSO order, K=5, n=$n)")
# add SALSO cluster boundaries as vertical lines
boundaries = cumsum([count(cSALSO[ord].==c) for c in 1:maximum(cSALSO)])
for b in boundaries[1:end-1]; vline!(hAll, [b+0.5], lw=1.2, lc=:black, label=""); end
# prototype ticks
for p in prototypes; idx = findfirst(ord .== p); scatter!(hAll, [idx], [0.5], ms=6, mc=:gold, shape=:star5, label=""); end
savefig(hAll, "output/openTotal/subjectProfilesK5-v2/heatmap_Dbar_all9.png")
@info "Saved heatmap all9"

# --- Plot all 9 separately so you can combine as needed ---
for (j, var) in enumerate(modelVars)
    col = j+1 # Dbar column (1 is intercept)
    vals = Dbar[ord, col]
    # per-covariate heatmap (1 x n) as single row
    hm = heatmap(reshape(vals, 1, n), c=:RdBu, clim=quantile(vals,[0.02,0.98]), yflip=true,
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
@info "Saved 9 separate heatmaps/violins/densities"

# --- Anchor tables ---
comm = hasproperty(train, :community) ? coalesce.(train.community[prototypes], missing) : fill(missing, length(prototypes))
anchorDF = DataFrame(IID=train.IID[prototypes], salso=1:modeK, consistency=pc,
    adhdLevel=train.adhdLevel[prototypes], community=comm)
for (j,var) in enumerate(modelVars); anchorDF[!, var] = Dbar[prototypes, j+1]; end
CSV.write("output/openTotal/subjectProfilesK5-v2/anchors.csv", anchorDF)
@info "Saved anchors.csv with Dbar per anchor"

# --- Per-subject credible intervals (for later forest) ---
# Save quantiles per subject per covariate (5,50,95) to allow quick plotting without holding full S
qDF = DataFrame(IID=repeat(train.IID, inner=length(modelVars)), 
    covariate=repeat(string.(modelVars), outer=n),
    salso=repeat(cSALSO, inner=length(modelVars)))
# This would be large if we store draws; instead store summary stats
summaryRows = []
for i in 1:n
    for (j,var) in enumerate(modelVars)
        col = j+1
        draws = D[i,col,:]
        qs = quantile(draws, [0.05,0.5,0.95])
        push!(summaryRows, (IID=train.IID[i], covariate=string(var), salso=cSALSO[i], q05=qs[1], q50=qs[2], q95=qs[3], mean=mean(draws)))
    end
end
CSV.write("output/openTotal/subjectProfilesK5-v2/subjectPostSummary.csv", DataFrame(summaryRows))
@info "Saved subjectPostSummary.csv"

@info "Done. Outputs in output/openTotal/subjectProfilesK5-v2/ : heatmap_Dbar_all9.png, heatmap_*.png (9), violin_*.png (9), density_anchor_*.png (9), anchors.csv, Dbar.csv, subjectPostSummary.csv"
