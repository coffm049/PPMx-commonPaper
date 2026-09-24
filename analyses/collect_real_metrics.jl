using CSV, DataFrames, Statistics, StatsBase, DataFramesMeta

"""
Collect real-world model metrics from analysis outputs and produce
paper-ready summary tables.
"""
function collect_real_metrics(analyses_dir="."; output_prefix="paper_real_")
    # Baseline comparisons
    base_files = [
        "output/baselines/frftotalComparison.csv",
        "output/baselines-female/frftotalComparison.csv",
        "output/baselines-male/frftotalComparison.csv"
    ]
    
    cv_files = [
        "output/baselines/combined5foldCV.csv",
        "output/baselines-female/combined5foldCV.csv",
        "output/baselines-male/combined5foldCV.csv"
    ]

    mfit_files = [
        "output/sexStratifiedMfit/mfit_comparison.csv",
        "output/sexStratifiedMfit/mfit_deltas.csv"
    ]

    all_exist = all(isfile, vcat(base_files, cv_files, mfit_files))
    if !all_exist
        missing_files = filter(f -> !isfile(f), vcat(base_files, cv_files, mfit_files))
        @warn "Some expected files missing: $missing_files"
    end

    # 1. Combined baseline comparison (main paper table)
    if isfile(base_files[1])
        base_combined = CSV.read(base_files[1], DataFrame)
        # Select key metrics for paper
        paper_cols = [:model, :testRMSE, :testRMSE_l, :testRMSE_u, :testRMSE_trim,
                      :testLPS, :testLPS_l, :testLPS_u,
                      :testARI, :testARI_l, :testARI_u, :testARI_trim,
                      :testARISalsoBinder, :testARISalsoVI,
                      :testARImodalSalsoBinder, :testARImodalSalsoVI,
                      :nclusters]
        existing = [c for c in paper_cols if c in names(base_combined)]
        base_paper = select(base_combined, existing...)
        CSV.write("$(output_prefix)baselines_combined.csv", base_paper)
        @info "Wrote $(output_prefix)baselines_combined.csv"
    end

    # 2. Sex-stratified baseline comparisons
    for (label, path) in [("female", base_files[2]), ("male", base_files[3])]
        if isfile(path)
            df = CSV.read(path, DataFrame)
            paper_cols = [:model, :testRMSE, :testRMSE_l, :testRMSE_u, :testRMSE_trim,
                          :testLPS, :testLPS_l, :testLPS_u,
                          :testARI, :testARI_l, :testARI_u, :testARI_trim,
                          :testARISalsoBinder, :testARISalsoVI,
                          :testARImodalSalsoBinder, :testARImodalSalsoVI,
                          :nclusters]
            existing = [c for c in paper_cols if c in names(df)]
            df_paper = select(df, existing...)
            CSV.write("$(output_prefix)baselines_$(label).csv", df_paper)
            @info "Wrote $(output_prefix)baselines_$(label).csv"
        end
    end

    # 3. CV RMSE summary
    cv_dfs = []
    for (label, path) in [("combined", cv_files[1]), ("female", cv_files[2]), ("male", cv_files[3])]
        if isfile(path)
            df = CSV.read(path, DataFrame)
            df[!, :sample] .= label
            push!(cv_dfs, df)
        end
    end
    if !isempty(cv_dfs)
        cv_all = vcat(cv_dfs...)
        cv_summary = @chain cv_all begin
            groupby(:sample)
            combine(:rmse => (mean ∘ skipmissing) => :mean_rmse,
                    :rmse => (std ∘ skipmissing) => :sd_rmse,
                    :rmse => (x -> quantile(skipmissing(x), 0.025)) => :rmse_l,
                    :rmse => (x -> quantile(skipmissing(x), 0.975)) => :rmse_u,
                    :fold => length => :n_folds)
        end
        CSV.write("$(output_prefix)cv_summary.csv", cv_summary)
        @info "Wrote $(output_prefix)cv_summary.csv"
        # Also save fold-level
        CSV.write("$(output_prefix)cv_folds.csv", cv_all)
    end

    # 4. Model fit comparison (combined vs female vs male)
    if isfile(mfit_files[1])
        mfit = CSV.read(mfit_files[1], DataFrame)
        CSV.write("$(output_prefix)mfit_comparison.csv", mfit)
        @info "Wrote $(output_prefix)mfit_comparison.csv"
    end
    if isfile(mfit_files[2])
        deltas = CSV.read(mfit_files[2], DataFrame)
        CSV.write("$(output_prefix)mfit_deltas.csv", deltas)
        @info "Wrote $(output_prefix)mfit_deltas.csv"
    end

    # 5. Master combined table (all metrics wide)
    if isfile(base_files[1]) && isfile(cv_files[1])
        base_combined = CSV.read(base_files[1], DataFrame)
        cv_combined = CSV.read(cv_files[1], DataFrame)
        cv_mean = mean(skipmissing(cv_combined.rmse))
        base_combined[!, :cvRMSE] .= cv_mean
        CSV.write("$(output_prefix)master.csv", base_combined)
        @info "Wrote $(output_prefix)master.csv"
    end
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    collect_real_metrics()
end