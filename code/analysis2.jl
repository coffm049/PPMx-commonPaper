

function modelEval(outcome::Symbol, datafile, testDF, prefix)
   
    fullDf = CSV.read("output/sampledDF.csv", DataFrame)
    dataDir = "/panfs/jay/groups/2/feczk001/shared/projects/FEZ_USERS/feczk001/UPPS_ABCD_FRF/code/jacob/"
    filepaths = [dataDir * "ADHDscores_flanker_ARMS1_merged.csv",
        dataDir * "ADHDscores_flanker_ARMS2_merged.csv"]
    frfLabels = reduce(vcat, [CSV.read(f, DataFrame; select = ["subjectkey", "community"]) for f in filepaths])
    rename!(frfLabels, ["IID", "frfLabel"])
    
    sim, mod, mod2 = jldopen(datafile, "r") do file
        allkeys = collect(keys(file))
        obj1 = file[allkeys[1]]
        obj2 = file[allkeys[2]]
        obj3 = file[allkeys[3]]
        return obj1, obj2, obj3
    end
    

    #%% partition data
    A1train = innerjoin(fullDf, CSV.read("../data/a1TrFl.csv", DataFrame, header= ["IID"]), on = "IID")
    A1test = innerjoin(fullDf, CSV.read("../data/a1TeFl.csv", DataFrame, header= ["IID"]), on = "IID")
    A2train = innerjoin(fullDf, CSV.read("../data/a2TrFl.csv", DataFrame, header= ["IID"]), on = "IID")
    A2test = innerjoin(fullDf, CSV.read("../data/a2TeFl.csv", DataFrame, header= ["IID"]), on = "IID")
    A2Df = vcat(A2train, A2test)
    A1train = A1train[completecases(A1train), :]
    A1test = A1test[completecases(A1test), :]
    A2Df = A2Df[completecases(A2Df), :]

    A1train = leftjoin(A1train, frfLabels, on = :IID)
    A1test = leftjoin(A1test, frfLabels, on = :IID)
    A2train = leftjoin(A2train, frfLabels, on = :IID)
    A2test = leftjoin(A2test, frfLabels, on = :IID)
    A2Df = leftjoin(A2Df, frfLabels, on = :IID)

    # sim = sim[3000:end]
    # Common effects
    betaEsts = []
    for i in 1:length(sim[1][:prior_mean_beta])
        append!(betaEsts, checkCommon(sim, i, prefix))
        writedlm(prefix * "Betas.txt", betaEsts)
    end

    #A1train[A1train.adhdLevel .!= 0, :]
    modelVars= [:age, :female, :uPosUrg, :uLplanning, :uLpers, :uNegUrg, :bbRR, :bbFS, :bbSum]
    X = convert(Matrix{Float64}, Matrix(A1train[:, modelVars]) .* 1.5)
    X = hcat(ones(size(X)[1]), X)
    y = convert(Vector{Float64}, A1train[:, :nTotal]) .* 10
    kclusts = argmin([kmeans(X', i).totalcost for i in 1:10])
    kmodel = kmeans(X', kclusts)
    A1train.kclust = string.(kmodel.assignments)
    
    # clustering
    Kn = [maximum(s[:C]) for s in sim]
    Plots.histogram(Kn, title = "# clusters", label = "PPMx-common")
    #Plots.vline!([kclusts], label = "kMeans")
    Plots.savefig(string(prefix) * "NumberofClusters.png")
    
    yPred, cPred = postPred(X, mod, sim[1:100:end])

    Plots.histogram(
        vec(reduce(vcat, [collect(values(countmap(s[:C]))) for s in sim])),
        title = "Cluster Sizes", label = "PPMx-common")
    Plots.vline!(collect(values(countmap(kmodel.assignments))), label = "kMeans")
    Plots.savefig(string(prefix) * "SizeofClusters.png")
    
    
    # Group characteristics (from best rind iteration)
    # bestRind = argmax([r[2] for r in rindMixvec])
    llk = kde([s[:llik] for s in sim])
    bestllk = llk.x[findmax(llk.density)[2]]
    llkIndex = findmin([abs.(s[:llik] .- bestllk) for s in sim])[2]
    
    # [ ] Should I take differences between groups for all partitions?
    # or just for the best?
    # doing best since label switching
    yPredBest, cPredBest = postPred(X, mod, [sim[llkIndex]])
    A1train.CbestRind = cPredBest[1,:]
    A1train.ybestRind = yPredBest[1,:]
    A1train.kclust = string.(assign_clusters(X', kmodel.centers))
    A1train = A1train[A1train.CbestRind .!= 0, :]
    A1train.CbestRind = string.(A1train.CbestRind)
    
    X = Matrix(A1test[:, modelVars]) .* 1.5
    X = convert(Matrix{Float64}, X)
    X = hcat(ones(size(X)[1]), X)
    A1test.kclust = string.(assign_clusters(X', kmodel.centers))
    yTestBest, cTestBest = postPred(X, mod, [sim[llkIndex]])
    A1test.CbestRind = cTestBest[1,:]
    A1test.ybestRind = yTestBest[1,:]
    A1test = A1test[A1test.CbestRind .!= 0, :]
    A1test.CbestRind = string.(A1test.CbestRind)
    
    X = Matrix(A2Df[:, modelVars]) .* 1.5
    X = convert(Matrix{Float64}, X)
    X = hcat(ones(size(X)[1]), X)
    yTestBest, cTestBest = postPred(X, mod, [sim[llkIndex]])
    A2Df.CbestRind = cTestBest[1,:]
    A2Df.ybestRind = yTestBest[1,:]
    A2Df.kclust = string.(assign_clusters(X', kmodel.centers))
    A2Df = A2Df[A2Df.CbestRind .!== 0, :]
    A2Df.CbestRind = string.(A2Df.CbestRind)
    

    #%% Train models
    # ppmx
    contrasts = Dict(:CbestRind => EffectsCoding())  # deviation from mean
    ppmlm = lm(@formula(nTotal ~ (age + female + uPosUrg + uLplanning + uLpers + uNegUrg+ bbRR + bbFS + bbSum) * CbestRind), A1train; contrasts=contrasts)
    assocTab = DataFrame(coeftable(ppmlm))
    assocTab[1, :Name] =  "CbestRind: 1"
    assocTab[2:10, :Name] =  assocTab[2:10, :Name] .* ": 1"
    transform!(assocTab, :Name => ByRow(s -> split(s, ": ")) => [:variable, :group])
    assocTab= assocTab[assocTab.variable .!= "CbestRind", :]
    assocTab = assocTab[[!occursin("female", v) for v in assocTab.variable], :]
    assocTab = assocTab[[!occursin("age", v) for v in assocTab.variable], :]
    rename!(assocTab, Symbol("Coef.") => :Coef)
    assocTab[:, :variable] = replace.(assocTab[:,:variable], " & CbestRind" => "")
    # kmeans
    
    X = hcat(ones(size(A1test)[1]), Matrix(A1test[:, modelVars]))' .* 1.5
    A1test.kclust = string.(assign_clusters(X, kmodel.centers))
    X = hcat(ones(size(A2Df)[1]), Matrix(A2Df[:, modelVars]))' .* 1.5
    A2Df.kclust = string.(assign_clusters(X .* 1.5, kmodel.centers))
    contrasts = Dict(:kclust => EffectsCoding())  # deviation from mean
    # [:age, :female, :uPosUrg, :uLpers, :bbRR, :bbSum, :ADHDcomposite]
    clustlm = lm(@formula(nTotal ~ (age + female + uPosUrg + uLpers + bbRR + bbSum + ADHDcomposite) * kclust), A1train; contrasts=contrasts)
    slr = lm(@formula(nFlank ~ age + female + uPosUrg + uLpers + bbRR + bbSum + ADHDcomposite), A1train; contrasts=contrasts)
    write_tex(string(prefix) * "ppmFit.tex", regtable(ppmlm))
    
    @df assocTab scatter(:variable, :Coef, group = :group, xrotation=90, title = "Modeled associations", legend = :outerright)
    #Plots.savefig(string(prefix) * "GroupAssocDifferences.png")
    
    regtable(slr, clustlm, ppmlm) |>
        (tbl -> write_tex(string(prefix) * "linMods.tex" ,tbl))
    
    # group difference summaries for all data partitions
    # partitionSummaries(A1train, ppmlm, clustlm, slr, prefix * "train", outcome)
    # partitionSummaries(A1test, ppmlm, clustlm, slr, prefix * "test", outcome)
    filDf = dropmissing(fullDf, modelVars)
    X = convert(Matrix{Float64}, Matrix(filDf[:, modelVars]) .* 1.25)
    X = hcat(ones(size(X)[1]), X)
    yTestBest, cTestBest = postPred(X, mod, [sim[llkIndex]])
    filDf.CbestRind = cTestBest[1,:]
    filDf.kclust = assign_clusters(X', kmodel.centers)
    
    for (ARMS, ddf) in enumerate([A1Df, A2Df])

        df = filDf |> tbl -> filter(row -> row.IID ∈ ddf.IID, tbl)
        # tabulate(filDf, :ADHD1, :CbestRind) |> 
        #     (tbl -> write_tex(string(prefix) * "bestRindContingiency.tex", tbl))
        #%% Group differences
        # df.CbestRind = string.(df.CbestRind)
        contrasts = Dict(:CbestRind => EffectsCoding())  # deviation from mean
        bbRRDiff = lm(@formula(bbRR ~ CbestRind), df; contrasts = contrasts)
        bbRRF = ftest(bbRRDiff.model)
        bbDriveDiff = lm(@formula(bbDrive ~ CbestRind), df; contrasts = contrasts)
        bbDriveF = ftest(bbDriveDiff.model)
        bbFSDiff = lm(@formula(bbFS ~ CbestRind), df; contrasts = contrasts)
        bbFSF = ftest(bbFSDiff.model)
        bbSumDiff = lm(@formula(bbSum ~ CbestRind), df; contrasts = contrasts)
        bbSumF = ftest(bbSumDiff.model)
        uNegUrgDiff = lm(@formula(uNegUrg ~ CbestRind), df; contrasts = contrasts)
        uNegUrgF = ftest(uNegUrgDiff.model)
        uPosUrgDiff = lm(@formula(uPosUrg ~ CbestRind), df; contrasts = contrasts)
        uPosUrgF = ftest(uPosUrgDiff.model)
        uLplanningDiff = lm(@formula(uLplanning ~ CbestRind), df; contrasts = contrasts)
        uLplanningF = ftest(uLplanningDiff.model)
        uLpersDiff = lm(@formula(uLpers ~ CbestRind), df; contrasts = contrasts)
        uLpersF = ftest(uLpersDiff.model)
        nPicVocabDiff = lm(@formula(nPicVocab ~ CbestRind), df; contrasts = contrasts)
        nPicVocabF = ftest(nPicVocabDiff.model)
        nFlankDiff = lm(@formula(nFlank ~ CbestRind), df; contrasts = contrasts)
        nFlankF = ftest(nFlankDiff.model)
        nListDiff = lm(@formula(nList ~ CbestRind), df; contrasts = contrasts)
        nListF = ftest(nListDiff.model)
        # nCbestRindardDiff = lm(@formula(nCbestRindard ~ CbestRind), df; contrasts = contrasts)
        # nCbestRinardF = ftest(nCbestRinardDiff.model)
        nPatternDiff = lm(@formula(nPattern ~ CbestRind), df; contrasts = contrasts)
        nPatternF = ftest(nPatternDiff.model)
        nPicDiff = lm(@formula(nPic ~ CbestRind), df; contrasts = contrasts)
        nPicF = ftest(nPicDiff.model)
        nReadDiff = lm(@formula(nRead ~ CbestRind), df; contrasts = contrasts)
        nReadF = ftest(nReadDiff.model)
        nFluidDiff = lm(@formula(nFluid ~ CbestRind), df; contrasts = contrasts)
        nFluidF = ftest(nFluidDiff.model)
        #nCbestRindrystDiff = lm(@formula(nCbestRindryst ~ CbestRind), df; contrasts = contrasts)
        #bbRRF = ftest(bbRRDiff.model)
        nTotalDiff = lm(@formula(nTotal ~ CbestRind), df; contrasts = contrasts)
        nTotalF = ftest(nTotalDiff.model)
        ADHDcompositeDiff = lm(@formula(ADHDcomposite ~ CbestRind), df; contrasts = contrasts)
        ADHDcompositeF = ftest(ADHDcompositeDiff.model)
        
        modelVars= [:age, :female, :uPosUrg, :uLplanning, :uLpers, :uNegUrg, :bbRR, :bbFS, :bbSum]
        regtable(
          "Modeled" => (nPicVocabDiff, uPosUrgDiff, uLpersDiff, bbRRDiff, bbSumDiff, ADHDcompositeDiff, uLplanningDiff, bbFSDiff, uNegUrgDiff),
          "Unmodeled" => (nFlankDiff, nFlankDiff, nListDiff,
            nPatternDiff, nPicDiff, nReadDiff, nFluidDiff, nTotalDiff, bbDriveDiff)
          ) |>
          (tbl -> write_tex(string(prefix) * "A" *  string(ARMS) * "GroupCovarDifferences.tex", tbl))
        
        tall = @chain df begin
          stack([
           "bbRR",
           "bbDrive",
           "bbFS",
           "bbSum",
           "uNegUrg",
           "uPosUrg",
           "uLplanning",
           "uLpers",
           "nPicVocab",
           "nFlank",
           "nList",
           "nCard",
           "nPattern",
           "nPic",
           "nRead",
           "nFluid",
           "nCryst",
           "nTotal"])
          transform(:variable => ByRow(x -> occursin(r"^u", x)) => :Modeled)
        end
        plots = [@df d groupedboxplot(:variable, :value, group=:CbestRind, facet = :Modeled, legend = :outerright, xrotation=90, outliers = false) for d in groupby(tall, :Modeled)]
        Plots.title!(plots[1], "UModeled differences")
        Plots.title!(plots[2], "Modeled differences")
        plot(plots..., layout = (2,:))
        # Plots.savefig(string(prefix) * "A" *  string(ARMS) * "GroupCovarDifferences.png")
        
        # [x] RMSE
        # make it R2
        # resid1 = df[:, outcome] .- yPred'
        mixR2 = adjRr2(ppmlm, df, outcome)
        slrR2 = adjRr2(slr, df, outcome)
        clustR2 = adjRr2(clustlm, df, outcome)
        # df = dropmissing(df, [:community, :CbestRind])
        # df.frfLabel = convert.(Int, df.frfLabel)
        # df.CbestRind = parse.(Int, df.CbestRind)
        
        # global ri
        # try
        #     ri = Clustering.randindex(df.frfLabel, df.CbestRind)
        # catch
        #     ri = (1e6, 1e6, 1e6, 1e6)
        #     @warn "missing labels, couldn't compute ri"
        # end
        
        writedlm(prefix * "A" *  string(ARMS)* "R2Vec.txt", [mixR2, slrR2, clustR2])

    end

    return 0
end

