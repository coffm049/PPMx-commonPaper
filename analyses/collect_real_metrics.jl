using CSV, DataFrames, Statistics, StatsBase, DataFramesMeta

"""
Collect real-world model metrics from analysis outputs and produce
paper-ready summary tables.
"""
function collect_real_metrics(analyses_dir="."; output_prefix="paper_real_")
    # Baseline comparisons (combined sample only; sex enters the model as a
    # covariate in the single fit, so no sex-stratified fits are collected)
    base_files = [
        "output_v3-dl-centered/baselines/frftotalComparison.csv"
    ]

    if !isfile(base_files[1])
        @warn "Missing expected baseline comparison: $(base_files[1])"
    end

    if isfile(base_files[1])
        base_combined = CSV.read(base_files[1], DataFrame)
        paper_cols = [:model, :testRMSE, :testRMSE_l, :testRMSE_u,
                      :testLPS, :testLPS_l, :testLPS_u,
                      :testARI, :testARI_l, :testARI_u,
                      :testARISalsoBinder, :testARISalsoVI,
                      :testARImodalSalsoBinder, :testARImodalSalsoVI,
                      :testBinderLoss,
                      :nclusters]
        existing = [c for c in paper_cols if c in names(base_combined)]
        base_paper = select(base_combined, existing...)
        CSV.write("$(output_prefix)baselines_combined.csv", base_paper)
        @info "Wrote $(output_prefix)baselines_combined.csv"

        CSV.write("$(output_prefix)master.csv", base_paper)
        @info "Wrote $(output_prefix)master.csv"
    end
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    collect_real_metrics()
end