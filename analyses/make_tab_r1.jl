using CSV, DataFrames, Printf

function fmt_ci(v, l, u; digits=3)
    (ismissing(v) || (v isa Number && isnan(v))) && return "\\textemdash"
    if ismissing(l) || ismissing(u) || (l isa Number && isnan(l)) || (u isa Number && isnan(u))
        return @sprintf("%.*f", digits, v)
    end
    @sprintf("%.*f (%.*f--%.*f)", digits, v, digits, l, digits, u)
end

function fmt(v; digits=3)
    (ismissing(v) || (v isa Number && isnan(v))) && return "\\textemdash"
    @sprintf("%.*f", digits, v)
end

function make_tab_r1(comparison_file; output_file="tab_R1.tex", n_test=6237)
    df = CSV.read(comparison_file, DataFrame)
    sort!(df, :model)
    io = IOBuffer()
    println(io, raw"\begin{table}[!htp]")
    println(io, raw"\centering")
    println(io, "\\caption{Real-data model comparison on the ABCD dataset, combined sample (test \$n=$(n_test)\$ FRF-matched subjects). Three methods: PPMx-common, \$k\$-means, and DP-GMM. Bayesian models report the central 90\\% posterior interval (5\\%/95\\% quantiles across draws); \$k\$-means is a deterministic point estimate.}")
    println(io, raw"\label{tab:R1}")
    println(io, raw"\resizebox{\textwidth}{!}{%")
    println(io, raw"\begin{tabular}{lcccccc}")
    println(io, raw"\toprule")
    println(io, raw"Model & RMSE & LPS & ARI (vs FRF) & ARI (SALSO Binder) & Binder Loss & $n_{\mathrm{clusters}}$ \\")
    println(io, raw"\midrule")
    for row in eachrow(df)
        rmse = fmt_ci(row.testRMSE, row.testRMSE_l, row.testRMSE_u)
        lps = fmt_ci(row.testLPS, row.testLPS_l, row.testLPS_u)
        ari = fmt_ci(row.testARI, row.testARI_l, row.testARI_u)
        ari_salso = fmt(get(row, :testARISalsoBinder, missing))
        binder = fmt_ci(row.testBinderLoss, get(row, :testBinderLoss_l, missing), get(row, :testBinderLoss_u, missing))
        println(io, "$(row.model) & $rmse & $lps & $ari & $ari_salso & $binder & $(Int(row.nclusters)) \\\\")
    end
    println(io, raw"\bottomrule")
    println(io, raw"\end{tabular}}")
    println(io, raw"\caption*{RMSE and Binder loss: lower is better; LPS and ARI: higher is better. Bayesian intervals propagate partition uncertainty; $k$-means has no posterior. RMSE is reported as-is from the baseline comparison output.}")
    println(io, raw"\end{table}")
    write(output_file, String(take!(io)))
    @info "Wrote $output_file"
end

comparison_file = get(ARGS, 1, "output_v3-dl-centered/baselines/frftotalComparison.csv")
output_file = get(ARGS, 2, "tab_R1.tex")
n_test = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 6237
make_tab_r1(comparison_file; output_file=output_file, n_test=n_test)
