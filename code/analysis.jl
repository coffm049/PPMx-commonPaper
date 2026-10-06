function adjRr2(mod, df, outcome)
    df.CbestRind = string.(df.CbestRind)
    df = df[df.CbestRind .!= "0", :]
    df.kclust = string.(df.kclust)
    
    Xnew = apply_schema(df, mod)
    Xnew = modelmatrix(mod.mf; data= Xnew)
    ssRes = sum((predict(mod, Xnew) .- df[:,outcome]).^2)
    ssTot = sum(df[:,outcome] .^ 2)
    R2 = 1 .- (ssRes ./ ssTot)
    den = dof_residual(mod)
    

    # return 1- (((1-R2) * (nrow(df)-1)) / den)
    return R2
end


function checkCommon(sim, coefNum, prefix)
    commonBeta = [s[:prior_mean_beta][coefNum] for s in sim]
    CI = minimumSet(commonBeta; ci=0.9)
    println("CI:" * string(CI))
    # p = lineplot(commonBeta)
    # hline!(p, CI)
    # hline!(p, 0)
    # histogram(commonBeta)
    pl = plot(commonBeta, label = "Beta" * string(coefNum-1))
    Plots.hline!(CI, label = "90% CI")
    ph = Plots.histogram(commonBeta, label = "Beta" * string(coefNum- 1))
    Plots.vline!(CI, label = "90% CI")
    meanBeta = mean(commonBeta)
    print("Mean: " * string(meanBeta))
    # display(p)
    plot(pl, ph)
    Plots.savefig(prefix  * "Beta" * string(coefNum - 1) * ".png")

    return [meanBeta, CI]
end

# summarizes RMSE, clustering, and gruop differences for each partition
function partitionSummaries(df, ppmlm, clustlm, slr, prefix, outcome)
    # [ ] Generate X and y

    # tabulate(df, :ADHD1, :CbestRind) |> 
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
    # ADHDcompositeDiff = lm(@formula(ADHDcomposite ~ CbestRind), df; contrasts = contrasts)
    # ADHDcompositeF = ftest(ADHDcompositeDiff.model)
    
    regtable(
      "Modeled" => (uPosUrgDiff, uLplanningDiff, uLpersDiff),
      "Unmodeled" => (nPicVocabDiff, nFlankDiff, nFlankDiff, nListDiff,
        nPatternDiff, nPicDiff, nReadDiff, nFluidDiff, nTotalDiff, bbRRDiff, bbDriveDiff, bbFSDiff, bbSumDiff,uNegUrgDiff)
      ) |>
      (tbl -> write_tex(string(prefix) * "GroupCovarDifferences.tex", tbl))
    
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
    # Plots.savefig(string(prefix) * "GroupCovarDifferences.png")
    
    # [x] RMSE
    # make it R2
    # resid1 = df[:, outcome] .- yPred'
    mixR2 = adjR2(ppmlm, df, outcome)
    slrR2 = adjR2(slr, df, outcome)
    clustR2 = adjR2(clustlm, df, outcome)
    df = dropmissing(df, [:frfLabel, :CbestRind])
    df.frfLabel = convert.(Int, df.frfLabel)
    df.CbestRind = parse.(Int, df.CbestRind)
    
    global ri
    try
        ri = Clustering.randindex(df.frfLabel, df.CbestRind)
    catch
        ri = (1e6, 1e6, 1e6, 1e6)
        @warn "missing labels, couldn't compute ri"
    end
    
    writedlm(prefix * "R2Vec.txt", [mixR2, slrR2, clustR2, ri...])
    return [mixR2, slrR2, clustR2, ri...]
end


# [ ] update everywhere that states nFlank to be the outcome symbol
function modelEval(outcome::Symbol)
   
    fullDf, standardization_params = loadNclean()
    prefix = "output/" * string(outcome) * "/" * string(outcome)
    
    dataDir = "/panfs/jay/groups/2/feczk001/shared/projects/FEZ_USERS/feczk001/UPPS_ABCD_FRF/code/jacob/"
    # FRF labels
    if outcome == :nFlank 
        filepaths = [dataDir * "ADHDscores_flanker_ARMS1_merged.csv",
        dataDir * "ADHDscores_flanker_ARMS2_merged.csv"]
        partitionLabel = "Fl"
        modelName = "Flanker"
    elseif outcome == :nList
        filepaths = [
          dataDir * "ADHDscores_list_ARMS1_merged.csv",
          dataDir * "ADHDscores_list_ARMS2_merged.csv"]
        partitionLabel = "Li"
        modelName = "List"
    elseif outcome == :subList
        filepaths = [
          dataDir * "ADHDscores_list_ARMS1_merged.csv",
          dataDir * "ADHDscores_list_ARMS2_merged.csv"]
        partitionLabel = "Li"
        modelName = "subList"
        outcome = :nList
    elseif outcome == :subList
        filepaths = [
          dataDir * "ADHDscores_list_ARMS1_merged.csv",
          dataDir * "ADHDscores_list_ARMS2_merged.csv"]
        partitionLabel = "Li"
        modelName = "subList"
        outcome = :nList
    elseif outcome == :nFluid
        filepaths = [
          dataDir * "ADHDscores_fluid_ARMS1_merged.csv",
          dataDir * "ADHDscoFlres_fluid_ARMS2_merged.csv"]
        partitionLabel = "Flu"
        modelName = "Fluid"
    end 

    frfLabels = reduce(vcat, [CSV.read(f, DataFrame; select = ["subjectkey", "community"]) for f in filepaths])
    rename!(frfLabels, ["IID", "frfLabel"])
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
    
    if modelName == "subList"
        fullDf = fullDf[fullDf.adhdLevel .!= 0, :]
    end
        

    #%% partition data
    A2train = innerjoin(fullDf, CSV.read("../data/a1Tr" * partitionLabel*  ".csv", DataFrame, header= ["IID"]), on = "IID")
    A2test = innerjoin(fullDf, CSV.read("../data/a1Te" * partitionLabel * ".csv", DataFrame, header= ["IID"]), on = "IID")
    A1train = innerjoin(fullDf, CSV.read("../data/a2Tr" * partitionLabel *".csv", DataFrame, header= ["IID"]), on = "IID")
    A1test = innerjoin(fullDf, CSV.read("../data/a2Te" * partitionLabel * ".csv", DataFrame, header= ["IID"]), on = "IID")
    A2Df = vcat(A2train, A2test)
    A1train = A1train[completecases(A1train), :]
    A1train = leftjoin(A1train, frfLabels, on = :IID)
    A1test = A1test[completecases(A1test), :]
    A1test = leftjoin(A1test, frfLabels, on = :IID)
    A2Df = A2Df[completecases(A2Df), :]
    A2Df = leftjoin(A2Df, frfLabels, on = :IID)
    
    
    @load "output/" * modelName * "a1.0b1.0m1.0_1.0.jld2" sim2
    prefix = prefix * "2"
    sim = sim2
    sim = sim[3000:end]
    # Common effects
    betaEsts = []
    for i in 1:length(sim[1][:prior_mean_beta])
        append!(betaEsts, checkCommon(sim, i, prefix))
        writedlm(prefix * "Betas.txt", betaEsts)
    end
    A1train[A1train.adhdLevel .!= 0, :]
    X = Matrix(A1train[:, [:age, :female, :uPosUrg, :uLplanning, :uLpers]]) .* 1.75
    X = convert(Matrix{Float64}, X)
    y = convert(Vector{Float64}, A1train[:, outcome])
    X = hcat(ones(size(X)[1]), X)
    kclusts = argmin([kmeans(X', i).totalcost for i in 1:10])
    kmodel = kmeans(X', kclusts)
    A1train.kclust = string.(kmodel.assignments)
    model = Model_PPMx(y, X, 1, similarity_type=:NN, sampling_model=:Reg, init_lik_rand=true)
    # set priors for base measure sampling
    prec = 0.01
    alph=1.0
    bet=1.0
    dims = size(X)[2]
    model.prior.base = Prior_base(
        repeat([0.0], dims + 1),
        repeat([prec], dims + 1), #1.0
        repeat([alph], dims + 1), # 1.0
        repeat([bet], dims + 1) # 1.0
    )
    model.prior.massParams = [1.0, 1.0] # 1e-3 for  common 10, inter 5 
    model.state.baseline.tau0 = 1e6
    
    # clustering
    Kn = [maximum(s[:C]) for s in sim]
    Plots.histogram(Kn, title = "# clusters", label = "PPMx-common")
    #Plots.vline!([kclusts], label = "kMeans")
    Plots.savefig(string(prefix) * "NumberofClusters.png")
    
    yPred, cPred = postPred(X, model, sim)

    rindMixvec = [Clustering.randindex(s, A1train.ADHD1) for s in eachrow(cPred)]
    rindkMean = Clustering.randindex(kmodel.assignments, A1train.ADHD1)
    Plots.histogram([r[2] for r in rindMixvec], title = "Rand Index", label = "PPMx-common")
    xlims!(rindkMean[2] - 0.05, maximum([r[2] for r in rindMixvec]))
    Plots.vline!([rindkMean[2]], label = "kMeans")
    Plots.savefig(string(prefix) * "randIndex.png")
    
    clustCounts= [collect(values(countmap(s[:C]))) for s in sim]
    # histogram(vec(reduce(hcat, clustCounts)))
    Plots.histogram(vec(reduce(vcat, clustCounts)), title = "Cluster Sizes", label = "PPMx-common")
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
    yPredBest, cPredBest = postPred(X, model, [sim[llkIndex]])
    A1train.CbestRind = cPredBest[1,:]
    A1train.ybestRind = yPredBest[1,:]
    A1train.kclust = string.(assign_clusters(X', kmodel.centers))
    A1train = A1train[A1train.CbestRind .!= 0, :]
    A1train.CbestRind = categorical(A1train.CbestRind)
    
    X = Matrix(A1test[:, [:age, :female, :uPosUrg, :uLplanning, :uLpers]]) .* 1.75
    X = convert(Matrix{Float64}, X)
    X = hcat(ones(size(X)[1]), X)
    A1test.kclust = string.(assign_clusters(X', kmodel.centers))
    yTestBest, cTestBest = postPred(X, model, [sim[llkIndex]])
    A1test.CbestRind = cTestBest[1,:]
    A1test.ybestRind = yTestBest[1,:]
    A1test = A1test[A1test.CbestRind .!= 0, :]
    A1test.CbestRind = categorical(A1test.CbestRind)
    
    X = Matrix(A2Df[:, [:age, :female, :uPosUrg, :uLplanning, :uLpers]]) .* 1.75
    X = convert(Matrix{Float64}, X)
    X = hcat(ones(size(X)[1]), X)
    yTestBest, cTestBest = postPred(X, model, [sim[llkIndex]])
    A2Df.CbestRind = cTestBest[1,:]
    A2Df.ybestRind = yTestBest[1,:]
    A2Df.kclust = string.(assign_clusters(X', kmodel.centers))
    A2Df = A2Df[A2Df.CbestRind .!== 0, :]
    A2Df.CbestRind = categorical(A2Df.CbestRind)
    

    #%% Train models
    # ppmx
    contrasts = Dict(:CbestRind => EffectsCoding())  # deviation from mean
    ppmForm = @formula(outcome ~ (age + female + uPosUrg + uLplanning + uLpers) * CbestRind)
    ppmForm = term(outcome) ~ term.(ppmForm.rhs)
    ppmlm = lm(ppmForm, A1train; contrasts=contrasts)
    assocTab = DataFrame(coeftable(ppmlm))[2:end, :]
    assocTab[1:5, :Name] =  assocTab[1:5, :Name] .* ": 1"
    transform!(assocTab, :Name => ByRow(s -> split(s, ": ")) => [:variable, :group])
    assocTab= assocTab[assocTab.variable .!= "CbestRind", :]
    assocTab = assocTab[[!occursin("female", v) for v in assocTab.variable], :]
    assocTab = assocTab[[!occursin("age", v) for v in assocTab.variable], :]
    rename!(assocTab, Symbol("Coef.") => :Coef)
    assocTab[:, :variable] = replace.(assocTab[:,:variable], " & CbestRind" => "")
    # kmeans
    
    X = hcat(ones(size(A1test)[1]), Matrix(A1test[:, [:age, :female, :uPosUrg, :uLplanning, :uLpers]]))' .* 1.75
    A1test.kclust = string.(assign_clusters(X, kmodel.centers))
    X = hcat(ones(size(A2Df)[1]), Matrix(A2Df[:, [:age, :female, :uPosUrg, :uLplanning, :uLpers]]))' .* 1.75
    A2Df.kclust = string.(assign_clusters(X .* 1.75, kmodel.centers))
    contrasts = Dict(:kclust => EffectsCoding())  # deviation from mean
    clustform = @formula(nFlank ~ (age + female + uPosUrg + uLplanning + uLpers) * kclust)
    clustform = term(outcome) ~ term.(clustform.rhs)
    clustlm = lm(clustform, A1train; contrasts=contrasts)
    slrform = @formula(nFlank ~ age + female + uPosUrg + uLplanning + uLpers)
    slrform = term(outcome) ~ term.(slrform.rhs)
    slr = lm(slrform, A1train; contrasts=contrasts)
    write_tex(string(prefix) * "ppmFit.tex", regtable(ppmlm))
    
    @df assocTab scatter(:variable, :Coef, group = :group, xrotation=90, title = "Modeled associations", legend = :outerright)
    #Plots.savefig(string(prefix) * "GroupAssocDifferences.png")
    
    # regtable(slr, clustlm, ppmlm) |>
    #     (tbl -> write_tex(string(prefix) * "linMods.tex" ,tbl))
    
    # group difference summaries for all data partitions
    partitionSummaries(A1train, ppmlm, clustlm, slr, prefix * "train", outcome)
    partitionSummaries(A1test, ppmlm, clustlm, slr, prefix * "test", outcome)
    partitionSummaries(A2Df, ppmlm, clustlm, slr, prefix * "A2", outcome)

    return 0
end
