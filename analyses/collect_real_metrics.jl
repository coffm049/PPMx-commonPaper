using CSV, DataFrames, Statistics, StatsBase, DataFramesMeta

"""
Collect real-world model metrics from analysis outputs and produce
paper-ready summary tables.
"""
function collect_real_metrics(analyses_dir="."; output_prefix="paper_real_")
    base_file = joinpath(analyses_dir, "output_v3-dl-centered/baselines/frftotalComparison.csv")
    isfile(base_file) || error("Missing required baseline comparison: $base_file")

    base_combined = CSV.read(base_file, DataFrame)
    nrow(base_combined) > 0 || error("Baseline comparison is empty: $base_file")

    paper_cols = [:model, :testRMSE, :testRMSE_l, :testRMSE_u,
                  :testLPS, :testLPS_l, :testLPS_u,
                  :testARI, :testARI_l, :testARI_u,
                  :testARISalsoBinder, :testARISalsoVI,
                  :testARImodalSalsoBinder, :testARImodalSalsoVI,
                  :testBinderLoss,
                  :nclusters]
    required = [:model, :testRMSE, :testLPS, :testARI, :nclusters]
    missing_required = setdiff(required, propertynames(base_combined))
    isempty(missing_required) || error("Baseline comparison missing required columns: $(missing_required)")

    existing = [c for c in paper_cols if c in propertynames(base_combined)]
    isempty(existing) && error("No expected paper columns found in $base_file; check column names/types.")

    base_paper = select(base_combined, existing...)
    nrow(base_paper) > 0 || error("Selected paper table has no rows: $base_file")

    CSV.write(joinpath(analyses_dir, "$(output_prefix)baselines_combined.csv"), base_paper)
    @info "Wrote $(output_prefix)baselines_combined.csv ($(nrow(base_paper)) rows)"

    CSV.write(joinpath(analyses_dir, "$(output_prefix)master.csv"), base_paper)
    @info "Wrote $(output_prefix)master.csv ($(nrow(base_paper)) rows)"
    base_paper
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    collect_real_metrics()
end