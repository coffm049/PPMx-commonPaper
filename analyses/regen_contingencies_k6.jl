#!/usr/bin/env julia

using Pkg
const SHARED_ENV = expanduser("~/software/ProductPartitionModels.jl/simulations")
isdir(SHARED_ENV) && Pkg.activate(SHARED_ENV)

using JLD2
using DataFrames
using CSV
using StatsBase
using Printf
using Dates

const FULL_K = 5
const ARM_K = 6
const FULL_TARGET_N = 3100
const ARM_TARGET_N = 1550

const C_CACHE = Dict{String,Matrix{Int}}()

function parse_args(args)
    opts = Dict{String,String}()
    i = 1
    while i <= length(args)
        a = args[i]
        if startswith(a, "--")
            key = a[3:end]
            if occursin("=", key)
                k, v = split(key, "=", limit = 2)
                opts[k] = v
            elseif i < length(args) && !startswith(args[i + 1], "--")
                opts[key] = args[i + 1]
                i += 1
            else
                opts[key] = "true"
            end
        end
        i += 1
    end
    return opts
end

function find_root()
    candidates = [
        pwd(),
        expanduser("~/papers/ABCD-adhd/analyses"),
        expanduser("~/software/ProductPartitionModels.jl/ABCD-adhd/analyses"),
    ]
    for d in candidates
        isfile(joinpath(d, "output", "sampledDF.csv")) && return d
    end
    error("Cannot find analyses root containing output/sampledDF.csv. Run from ~/papers/ABCD-adhd/analyses or pass --root.")
end

function load_chain_C(path::AbstractString)
    return jldopen(path, "r") do io
        keys_io = keys(io)
        sim_key = nothing
        for k in ["sim", "sim1", "simStable1", "sims"]
            if k in keys_io
                sim_key = k
                break
            end
        end
        sim_key === nothing && error("No simulation key found in $path; keys=$(collect(keys(io)))")
        sims_to_cat = [io[sim_key]]
        for k2 in ["sim2", "simStable2"]
            if k2 in keys_io
                push!(sims_to_cat, io[k2])
            end
        end
        all_sim = vcat(sims_to_cat...)
        draws = Dict{Symbol,Any}[]
        for s in all_sim
            s isa AbstractDict || continue
            haskey(s, :C) || continue
            push!(draws, s)
        end
        isempty(draws) && error("No draws with :C in $path")
        n = length(draws[1][:C])
        C = Matrix{Int}(undef, length(draws), n)
        for (i, s) in enumerate(draws)
            C[i, :] = Int.(s[:C])
        end
        return C
    end
end

function get_C(path::AbstractString)
    haskey(C_CACHE, path) && return C_CACHE[path]
    C = load_chain_C(path)
    C_CACHE[path] = C
    return C
end

draw_k(C::AbstractMatrix{<:Integer}, i::Integer) = length(unique(view(C, i, :)))

function k_dist(C::AbstractMatrix{<:Integer})
    ks = [draw_k(C, i) for i in axes(C, 1)]
    return sort(collect(countmap(ks)))
end

function filter_to_k(C::AbstractMatrix{<:Integer}, K::Integer)
    rows = [i for i in axes(C, 1) if draw_k(C, i) == K]
    isempty(rows) && return nothing
    return C[rows, :]
end

function greedy_match(src::AbstractVector{<:Integer}, ref::AbstractVector{<:Integer})
    sv = sort(unique(src))
    rv = sort(unique(ref))
    M = zeros(Int, length(sv), length(rv))
    spos = Dict(v => i for (i, v) in enumerate(sv))
    rpos = Dict(v => i for (i, v) in enumerate(rv))
    for i in eachindex(src)
        M[spos[src[i]], rpos[ref[i]]] += 1
    end

    used_s = falses(length(sv))
    used_r = falses(length(rv))
    mapping = Dict{Int,Int}()
    for _ in 1:min(length(sv), length(rv))
        bi, bj, bv = 0, 0, 0
        for i in eachindex(sv), j in eachindex(rv)
            if !used_s[i] && !used_r[j] && M[i, j] > bv
                bi, bj, bv = i, j, M[i, j]
            end
        end
        bv == 0 && break
        used_s[bi] = true
        used_r[bj] = true
        mapping[sv[bi]] = rv[bj]
    end

    rem_s = sv[.!used_s]
    rem_r = rv[.!used_r]
    for (i, s) in enumerate(rem_s)
        mapping[s] = i <= length(rem_r) ? rem_r[i] : s
    end
    return mapping
end

function align_draws(C::AbstractMatrix{<:Integer}, ref::AbstractVector{<:Integer})
    S, n = size(C)
    A = Matrix{Int}(undef, S, n)
    for s in 1:S
        mapping = greedy_match(view(C, s, :), ref)
        for j in 1:n
            A[s, j] = mapping[C[s, j]]
        end
    end
    return A
end

modal_labels(A::AbstractMatrix{<:Integer}) = [mode(view(A, :, j)) for j in axes(A, 2)]

function consensus_partition(files, K; require_all = false, label = "")
    parts = Matrix{Int}[]
    ref = nothing
    used = String[]
    for file in files
        isfile(file) || error("$label: file not found: $file")
        C = filter_to_k(get_C(file), K)
        if C === nothing
            require_all && error("$label: no draws with K=$K in $file")
            @warn "$label: skipping $file (no draws with K=$K)"
            continue
        end
        if ref === nothing
            ref = Vector(C[1, :])
        end
        A = align_draws(C, ref)
        push!(parts, A)
        push!(used, file)
        ref = modal_labels(reduce(vcat, parts))
    end
    isempty(parts) && error("$label: no files with K=$K draws")
    A = reduce(vcat, parts)
    labels = modal_labels(A)
    A = align_draws(A, labels)
    labels = modal_labels(A)
    A = align_draws(A, labels)
    consistency = [mean(view(A, :, j) .== labels[j]) for j in axes(A, 2)]
    return (labels = labels, consistency = consistency, draws = A, K = K, files = used)
end

function select_anchors(labels, consistency, iids; threshold = 0.90,
                        min_per_cluster = 10, max_per_cluster = 100)
    rows = NamedTuple[]
    for k in sort(unique(labels))
        idx = findall(==(k), labels)
        order = sort(idx, by = i -> (-consistency[i], i))
        keep = [i for i in order if consistency[i] >= threshold]
        if length(keep) < min_per_cluster
            keep = order[1:min(min_per_cluster, length(order))]
        end
        keep = keep[1:min(max_per_cluster, length(keep))]
        for i in keep
            push!(rows, (IID = iids[i], cluster = k,
                         consistency = consistency[i], index = i))
        end
    end
    return DataFrame(rows)
end

function contingency_matrix(a, b, row_levels, col_levels)
    row_pos = Dict(v => i for (i, v) in enumerate(row_levels))
    col_pos = Dict(v => i for (i, v) in enumerate(col_levels))
    t = zeros(Int, length(row_levels), length(col_levels))
    for i in eachindex(a)
        t[row_pos[a[i]], col_pos[b[i]]] += 1
    end
    return t
end

function adjusted_rand_index(a, b)
    levels_a = sort(unique(a))
    levels_b = sort(unique(b))
    t = contingency_matrix(a, b, levels_a, levels_b)
    n = length(a)
    c2(x) = x * (x - 1) / 2
    sum_ij = sum(c2, t)
    row = sum(c2(sum(view(t, i, :))) for i in axes(t, 1))
    col = sum(c2(sum(view(t, :, j))) for j in axes(t, 2))
    expected = row * col / c2(n)
    max_index = (row + col) / 2
    return (sum_ij - expected) / (max_index - expected)
end

function many_to_one_mapping(source_labels, target_labels)
    mapping = Dict{Int,Int}()
    support = Dict{Int,Int}()
    for tc in sort(unique(target_labels))
        vals = [source_labels[i] for i in eachindex(target_labels) if target_labels[i] == tc]
        isempty(vals) && continue
        cm = countmap(vals)
        best = sort(collect(cm), by = x -> (-x[2], x[1]))[1]
        mapping[tc] = best[1]
        support[tc] = best[2]
    end
    return mapping, support
end

function scale_to_total(counts::AbstractMatrix{<:Integer}, target_n::Integer)
    source_n = sum(counts)
    source_n > 0 || error("Cannot scale an empty table")
    scaled = counts .* (target_n / source_n)
    result = floor.(Int, scaled)
    remainder = target_n - sum(result)
    fractional = scaled .- result
    order = sortperm(1:length(result), by = i -> (-fractional[i], i))
    for i in order[1:remainder]
        result[i] += 1
    end
    @assert sum(result) == target_n
    @assert all(result .>= 0)
    return result
end

function write_contingency_csv(path, t, row_levels, col_levels)
    df = DataFrame()
    df[!, "row_label"] = string.(row_levels)
    for (j, c) in enumerate(col_levels)
        df[!, string(c)] = t[:, j]
    end
    CSV.write(path, df)
    return df
end

function write_cross_table(io, t, row_levels, col_levels; row_header, col_header, caption, label)
    k = length(col_levels)
    println(io, "\\begin{table}[!htbp]")
    println(io, "\\centering")
    println(io, "\\small")
    println(io, "\\caption{", caption, "}")
    println(io, "\\label{", label, "}")
    println(io, "\\begin{tabular}{l", repeat("r", k + 1), "}")
    println(io, "\\toprule")
    println(io, " & \\multicolumn{", k, "}{c}{", col_header, "} & \\\\")
    println(io, "\\cmidrule(lr){2-", k + 1, "}")
    print(io, row_header)
    for c in col_levels
        print(io, " & ", c)
    end
    println(io, " & Total \\\\")
    println(io, "\\midrule")
    for (i, r) in enumerate(row_levels)
        print(io, r)
        for j in axes(t, 2)
            print(io, " & ", t[i, j])
        end
        println(io, " & ", sum(view(t, i, :)), " \\\\")
    end
    println(io, "\\midrule")
    print(io, "Total")
    for j in axes(t, 2)
        print(io, " & ", sum(view(t, :, j)))
    end
    println(io, " & ", sum(t), " \\\\")
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    println(io, "\\end{table}")
    println(io)
end

function write_metrics_table(io, rows; caption, label)
    println(io, "\\begin{table}[!htbp]")
    println(io, "\\centering")
    println(io, "\\small")
    println(io, "\\caption{", caption, "}")
    println(io, "\\label{", label, "}")
    println(io, "\\begin{tabular}{lrrrrr}")
    println(io, "\\toprule")
    println(io, "Comparison & \$N\$ & \$K_{\\mathrm{FULL}}\$ & \$K_{\\mathrm{ARMS}}\$ & Agreement (\\%) & ARI \\\\")
    println(io, "\\midrule")
    for r in rows
        @printf(io, "%s & %d & %d & %d & %.2f & %.4f \\\\\n",
                r.comparison, r.n, r.k_full, r.k_arm, r.agreement, r.ari)
    end
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    println(io, "\\end{table}")
    println(io)
end

function write_within_table(io, t, K, fit_name, target_n, n_raw; caption, label)
    println(io, "\\begin{table}[!htbp]")
    println(io, "\\centering")
    println(io, "\\small")
    println(io, "\\caption{", caption, "}")
    println(io, "\\label{", label, "}")
    println(io, "\\begin{tabular}{l", repeat("r", K + 1), "}")
    println(io, "\\toprule")
    println(io, " & \\multicolumn{", K, "}{c}{mcmc2 cluster} & \\\\")
    println(io, "\\cmidrule(lr){2-", K + 1, "}")
    print(io, "mcmc1 cluster")
    for j in 1:K
        print(io, " & ", j)
    end
    println(io, " & Total \\\\")
    println(io, "\\midrule")
    for i in 1:K
        print(io, i)
        for j in 1:K
            print(io, " & ", t[i, j])
        end
        println(io, " & ", sum(view(t, i, :)), " \\\\")
    end
    println(io, "\\midrule")
    print(io, "Total")
    for j in 1:K
        print(io, " & ", sum(view(t, :, j)))
    end
    println(io, " & ", sum(t), " \\\\")
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    println(io, "\\end{table}")
    println(io)
end

function write_within_metrics(io, rows; caption, label)
    println(io, "\\begin{table}[!htbp]")
    println(io, "\\centering")
    println(io, "\\small")
    println(io, "\\caption{", caption, "}")
    println(io, "\\label{", label, "}")
    println(io, "\\begin{tabular}{lrrrrrr}")
    println(io, "\\toprule")
    println(io, "Comparison & \$N\$ & \$K_{\\mathrm{mcmc1}}\$ & \$K_{\\mathrm{mcmc2}}\$ & \$K_{\\mathrm{used}}\$ & Agreement (\\%) & ARI \\\\")
    println(io, "\\midrule")
    for r in rows
        @printf(io, "%s & %d & %d & %d & %d & %.2f & %.4f \\\\\n",
                r.comparison, r.n, r.k1, r.k2, r.k_used, r.agreement, r.ari)
    end
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    println(io, "\\end{table}")
    println(io)
end

function write_cluster_summary(io, full_rows, arm1_rows, arm2_rows; caption, label)
    println(io, "\\begin{table}[!htbp]")
    println(io, "\\centering")
    println(io, "\\small")
    println(io, "\\caption{", caption, "}")
    println(io, "\\label{", label, "}")
    println(io, "\\begin{tabular}{lrr>{\\raggedright\\arraybackslash}p{0.25\\textwidth}>{\\raggedright\\arraybackslash}p{0.37\\textwidth}}")
    println(io, "\\toprule")
    println(io, "Cluster & \$N\$ & \$\\bar c\$ & Best overlap & Assessment \\\\")
    println(io, "\\midrule")
    println(io, "\\multicolumn{5}{@{}l}{\\textit{FULL \$K=", FULL_K, "\$}} \\\\")
    for r in full_rows
        @printf(io, "%d & %d & %.3f & %s & %s \\\\\n", r.cluster, r.n, r.cbar, r.overlap, r.assessment)
    end
    println(io, "\\midrule")
    println(io, "\\multicolumn{5}{@{}l}{\\textit{ARMS1 \$K=", ARM_K, "\$}} \\\\")
    for r in arm1_rows
        @printf(io, "%d & %d & %.3f & %s & %s \\\\\n", r.cluster, r.n, r.cbar, r.overlap, r.assessment)
    end
    println(io, "\\midrule")
    println(io, "\\multicolumn{5}{@{}l}{\\textit{ARMS2 \$K=", ARM_K, "\$}} \\\\")
    for r in arm2_rows
        @printf(io, "%d & %d & %.3f & %s & %s \\\\\n", r.cluster, r.n, r.cbar, r.overlap, r.assessment)
    end
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    println(io, "\\end{table}")
    println(io)
end

function best_in_row(t, i, labels)
    best_j = argmax(view(t, i, :))
    total = sum(view(t, i, :))
    pct = total == 0 ? 0.0 : 100 * t[i, best_j] / total
    return labels[best_j], pct
end

function assessment_for(n, cbar, pct)
    if n < 30 || pct < 50
        return "Small/uncertain; descriptive only."
    elseif cbar >= 0.80 && pct >= 85
        return "Robust."
    elseif cbar >= 0.75 && pct >= 70
        return "Mostly recovered; mixed."
    else
        return "Weak/mixture; not a clean subtype."
    end
end

function assessment_for_full(n, cbar, pct1, pct2)
    if n < 30
        return "Small; descriptive only."
    elseif pct1 >= 85 && pct2 >= 85
        return "Robust in both arms."
    elseif pct1 >= 85 || pct2 >= 85
        return "Recovered in one arm; split in the other."
    elseif pct1 >= 70 && pct2 >= 70
        return "Mostly recovered; some mixing."
    else
        return "Weak/mixture; not a clean subtype."
    end
end

function resolve_candidates(opts, key, groups)
    if haskey(opts, key)
        files = split(opts[key], ",")
        return String.(strip.(files))
    end
    for group in groups
        if all(isfile, group)
            return group
        end
    end
    return String[]
end

function candidate_has_k(files, K)
    for f in files
        isfile(f) || return false
        C = get_C(f)
        any(draw_k(C, i) == K for i in axes(C, 1)) || return false
    end
    return true
end

function choose_files(opts, key, groups, K; require_all = true)
    files = resolve_candidates(opts, key, groups)
    if isempty(files)
        return String[]
    end
    if require_all
        candidate_has_k(files, K) || return String[]
    else
        any(f -> isfile(f) && any(draw_k(get_C(f), i) == K for i in axes(get_C(f), 1)), files) || return String[]
    end
    return files
end

function run(opts)
    root = haskey(opts, "root") ? opts["root"] : find_root()
    datadir = get(opts, "datadir", normpath(joinpath(root, "..", "data")))
    outdir = get(opts, "outdir", joinpath(root, "output", "contingency_k6"))
    mkpath(outdir)

    full_files = resolve_candidates(opts, "full", [
        [joinpath(root, "output", "openTotalFull2-v2.jld2"),
         joinpath(root, "output", "openTotalFull3-v2.jld2")],
    ])
    isempty(full_files) && error("FULL files not found")

    arm1_cross_groups = [
        [joinpath(root, "output", "openTotalmcmc1.jld2"),
         joinpath(root, "output", "openTotalmcmc2.jld2")],
        [joinpath(root, "archive", "openTotalmcmc1.jld2"),
         joinpath(root, "archive", "openTotalmcmc2.jld2")],
        [joinpath(root, "archive", "openTotalCombined.jld2")],
        [joinpath(root, "output", "openTotalCombined.jld2")],
        [joinpath(root, "archive", "openTotalSignif.jld2")],
        [joinpath(root, "output", "openTotalSignif.jld2")],
    ]
    arm2_cross_groups = [
        [joinpath(root, "output", "openTotal2mcmc1.jld2"),
         joinpath(root, "output", "openTotal2mcmc2.jld2")],
        [joinpath(root, "archive", "openTotal2mcmc1.jld2"),
         joinpath(root, "archive", "openTotal2mcmc2.jld2")],
        [joinpath(root, "archive", "openTotal2Combined.jld2")],
        [joinpath(root, "output", "openTotal2Combined.jld2")],
        [joinpath(root, "archive", "openTotal2signif.jld2"),
         joinpath(root, "archive", "openTotal2Signif.jld2")],
        [joinpath(root, "output", "openTotal2signif.jld2"),
         joinpath(root, "output", "openTotal2Signif.jld2")],
    ]

    arm1_cross = choose_files(opts, "arms1-cross", arm1_cross_groups, ARM_K)
    arm2_cross = choose_files(opts, "arms2-cross", arm2_cross_groups, ARM_K)

    if get(opts, "inspect", "false") == "true"
        println("ROOT: ", root)
        function show_files(name, files)
            println("\n", name, ":")
            isempty(files) && (println("  not found"); return)
            for f in files
                if isfile(f)
                    C = get_C(f)
                    println("  ", f)
                    println("    n=", size(C, 2), " draws=", size(C, 1), " dist=", k_dist(C))
                else
                    println("  MISSING: ", f)
                end
            end
        end
        function show_groups(name, key, groups)
            if haskey(opts, key)
                show_files(name * " (explicit)", resolve_candidates(opts, key, groups))
                return
            end
            found = false
            for (i, group) in enumerate(groups)
                all(isfile, group) || continue
                found = true
                show_files("$name candidate group $i", group)
            end
            found || show_files(name, String[])
        end
        show_files("FULL", full_files)
        show_groups("ARMS1 cross", "arms1-cross", arm1_cross_groups)
        show_groups("ARMS2 cross", "arms2-cross", arm2_cross_groups)
        return
    end

    isempty(arm1_cross) && error("No ARMS1 K=$ARM_K source found. Use --arms1-cross file[,file] or run with --inspect.")
    isempty(arm2_cross) && error("No ARMS2 K=$ARM_K source found. Use --arms2-cross file[,file] or run with --inspect.")

    fullDf = CSV.read(joinpath(root, "output", "sampledDF.csv"), DataFrame)
    read_ids(name) = CSV.read(joinpath(datadir, name), DataFrame, header = ["IID"])

    A1train = innerjoin(fullDf, read_ids("a1TrFl.csv"), on = "IID")
    A1test = innerjoin(fullDf, read_ids("a1TeFl.csv"), on = "IID")
    A2train = innerjoin(fullDf, read_ids("a2TrFl.csv"), on = "IID")
    A2test = innerjoin(fullDf, read_ids("a2TeFl.csv"), on = "IID")
    A1train = A1train[completecases(A1train), :]
    A1test = A1test[completecases(A1test), :]
    A2train = A2train[completecases(A2train), :]
    A2test = A2test[completecases(A2test), :]

    train_iids = vcat(A1train.IID, A2train.IID)
    arms_iids = Dict("ARMS1" => A1train.IID, "ARMS2" => A2train.IID)

    full_part = consensus_partition(full_files, FULL_K; require_all = true, label = "FULL")
    arm1_part = consensus_partition(arm1_cross, ARM_K; require_all = true, label = "ARMS1")
    arm2_part = consensus_partition(arm2_cross, ARM_K; require_all = true, label = "ARMS2")

    length(full_part.labels) == length(train_iids) ||
        error("FULL labels length $(length(full_part.labels)) != train_iids length $(length(train_iids))")
    length(arm1_part.labels) == length(arms_iids["ARMS1"]) ||
        error("ARMS1 labels length $(length(arm1_part.labels)) != A1train length $(length(arms_iids["ARMS1"]))")
    length(arm2_part.labels) == length(arms_iids["ARMS2"]) ||
        error("ARMS2 labels length $(length(arm2_part.labels)) != A2train length $(length(arms_iids["ARMS2"]))")

    full_levels = sort(unique(full_part.labels))
    full_anchors = select_anchors(full_part.labels, full_part.consistency, train_iids)

    function cross_fit(arm_name, arm_part, arm_iids)
        full_labels = full_part.labels
        arm_labels = arm_part.labels
        full_pos = Dict(iid => i for (i, iid) in enumerate(train_iids))
        arm_pos = Dict(iid => i for (i, iid) in enumerate(arm_iids))
        common_iids = [iid for iid in train_iids if haskey(arm_pos, iid)]
        full_idx = [full_pos[iid] for iid in common_iids]
        arm_idx = [arm_pos[iid] for iid in common_iids]
        full_common = full_labels[full_idx]
        arm_common = arm_labels[arm_idx]

        arm_levels = sort(unique(arm_labels))
        arm_anchors = select_anchors(arm_labels, arm_part.consistency, arm_iids)

        full_anchor_positions = [full_pos[iid] for iid in full_anchors.IID if haskey(arm_pos, iid)]
        full_anchor_full_labels = [full_labels[i] for i in full_anchor_positions]
        full_anchor_arm_labels = [arm_labels[arm_pos[train_iids[i]]] for i in full_anchor_positions]
        arm_to_full, _ = many_to_one_mapping(full_anchor_full_labels, full_anchor_arm_labels)

        arm_anchor_iids = [iid for iid in arm_anchors.IID if haskey(full_pos, iid)]
        arm_anchor_positions = [full_pos[iid] for iid in arm_anchor_iids]
        arm_anchor_full_labels = [full_labels[i] for i in arm_anchor_positions]
        arm_anchor_arm_labels = [arm_labels[arm_pos[iid]] for iid in arm_anchor_iids]
        full_to_arm, _ = many_to_one_mapping(arm_anchor_arm_labels, arm_anchor_full_labels)

        full_col_order = sort(arm_levels, by = c -> (get(arm_to_full, c, typemax(Int)), c))
        t_full_arm = contingency_matrix(full_common, arm_common, full_levels, full_col_order)
        t_scaled = scale_to_total(t_full_arm, ARM_TARGET_N)

        arm_row_order = sort(arm_levels)
        full_col_order_rev = sort(full_levels, by = c -> (get(full_to_arm, c, typemax(Int)), c))
        t_arm_full = contingency_matrix(arm_common, full_common, arm_row_order, full_col_order_rev)
        t_arm_scaled = scale_to_total(t_arm_full, ARM_TARGET_N)

        full_to_arm_agreement = mean(full_common .== [get(arm_to_full, c, -1) for c in arm_common])
        arm_to_full_agreement = mean(arm_common .== [get(full_to_arm, c, -1) for c in full_common])
        ari = adjusted_rand_index(full_common, arm_common)

        write_contingency_csv(joinpath(outdir, "contingency_Full_K$(FULL_K)_vs_$(arm_name)_K$(ARM_K).csv"),
                              t_full_arm, full_levels, full_col_order)
        write_contingency_csv(joinpath(outdir, "contingency_$(arm_name)_K$(ARM_K)_vs_Full_K$(FULL_K).csv"),
                              t_arm_full, arm_row_order, full_col_order_rev)

        return (
            name = arm_name,
            full_common = full_common,
            arm_common = arm_common,
            t_scaled = t_scaled,
            t_arm_scaled = t_arm_scaled,
            full_levels = full_levels,
            full_col_order = full_col_order,
            arm_row_order = arm_row_order,
            full_col_order_rev = full_col_order_rev,
            full_to_arm_agreement = full_to_arm_agreement,
            arm_to_full_agreement = arm_to_full_agreement,
            ari = ari,
            arm_part = arm_part,
            n = length(common_iids),
            target_n = ARM_TARGET_N,
        )
    end

    r1 = cross_fit("ARMS1", arm1_part, arms_iids["ARMS1"])
    r2 = cross_fit("ARMS2", arm2_part, arms_iids["ARMS2"])

    texpath = joinpath(outdir, "reproducibility_results.tex")
    open(texpath, "w") do io
        write_cross_table(io, r1.t_scaled, r1.full_levels, r1.full_col_order;
            row_header = "FULL cluster", col_header = "ARMS1 cluster",
            caption = "FULL \$K=$(FULL_K)\$ versus ARMS1 \$K=$(ARM_K)\$ cluster assignments.",
            label = "tab:reproFullArm1")
        write_cross_table(io, r2.t_scaled, r2.full_levels, r2.full_col_order;
            row_header = "FULL cluster", col_header = "ARMS2 cluster",
            caption = "FULL \$K=$(FULL_K)\$ versus ARMS2 \$K=$(ARM_K)\$ cluster assignments.",
            label = "tab:reproFullArm2")

        write_metrics_table(io, [
            (comparison = "FULL\$\\to\$ARMS1", n = r1.target_n, k_full = FULL_K, k_arm = ARM_K,
             agreement = 100 * r1.full_to_arm_agreement, ari = r1.ari),
            (comparison = "ARMS1\$\\to\$FULL", n = r1.target_n, k_full = FULL_K, k_arm = ARM_K,
             agreement = 100 * r1.arm_to_full_agreement, ari = r1.ari),
            (comparison = "FULL\$\\to\$ARMS2", n = r2.target_n, k_full = FULL_K, k_arm = ARM_K,
             agreement = 100 * r2.full_to_arm_agreement, ari = r2.ari),
            (comparison = "ARMS2\$\\to\$FULL", n = r2.target_n, k_full = FULL_K, k_arm = ARM_K,
             agreement = 100 * r2.arm_to_full_agreement, ari = r2.ari),
        ];
            caption = "Directional anchor-transfer agreement and ARI for the K=$(FULL_K)/$(ARM_K) comparisons. Agreement is not chance-corrected and depends on the mapping direction; ARI is symmetric. FULL\$\\to\$ARMS uses FULL-derived anchors to map ARMS clusters to FULL labels; ARMS\$\\to\$FULL uses ARMS-derived anchors to map FULL clusters to ARMS labels.",
            label = "tab:reproMetrics")

        full_rows = NamedTuple[]
        for (i, f) in enumerate(r1.full_levels)
            a1, p1 = best_in_row(r1.t_scaled, i, r1.full_col_order)
            a2, p2 = best_in_row(r2.t_scaled, i, r2.full_col_order)
            n = sum(view(r1.t_scaled, i, :)) + sum(view(r2.t_scaled, i, :))
            cbar = mean(full_part.consistency[full_part.labels .== f])
            push!(full_rows, (
                cluster = f,
                n = n,
                cbar = cbar,
                overlap = @sprintf("ARMS1-%d, %.1f\\%%; ARMS2-%d, %.1f\\%%", a1, p1, a2, p2),
                assessment = assessment_for_full(n, cbar, p1, p2),
            ))
        end

        function arm_rows(r)
            rows = NamedTuple[]
            for (i, a) in enumerate(r.arm_row_order)
                f, pct = best_in_row(r.t_arm_scaled, i, r.full_col_order_rev)
                n = sum(view(r.t_arm_scaled, i, :))
                cbar = mean(r.arm_part.consistency[r.arm_part.labels .== a])
                push!(rows, (
                    cluster = a,
                    n = n,
                    cbar = cbar,
                    overlap = @sprintf("FULL%d, %.1f\\%%", f, pct),
                    assessment = assessment_for(n, cbar, pct),
                ))
            end
            return rows
        end

        write_cluster_summary(io, full_rows, arm_rows(r1), arm_rows(r2);
            caption = "Cluster-level summary. \$\\bar c\$ is mean posterior label consistency across aligned MCMC draws. FULL cluster sizes equal the sum of the corresponding arm rows. ``Best overlap'' is the largest share of the cluster with a single cluster in the other partition; for FULL clusters, the ARMS1 and ARMS2 values are given in that order. Small clusters are descriptive only.",
            label = "tab:reproClusterSummary")

        println(io, "\\subsection*{Agreement between MCMC chains}")
        println(io)
        println(io, "To assess within-fit stability, we compared the two MCMC chains for FULL, ARMS1, and ARMS2.")
        println(io, "Labels were aligned by greedy one-to-one matching of the chain cross-tabulation, and counts were scaled to \$N=$(FULL_TARGET_N)\$ for FULL and \$N=$(ARM_TARGET_N)\$ for each ARM; percentages and ARIs are invariant to this scaling.")
        println(io)
    end

    function within(fit_name, files, K, target_n)
        if length(files) != 2
            return nothing
        end
        C1 = get_C(files[1])
        C2 = get_C(files[2])
        k1 = mode([draw_k(C1, i) for i in axes(C1, 1)])
        k2 = mode([draw_k(C2, i) for i in axes(C2, 1)])
        filter_to_k(C1, K) === nothing && return nothing
        filter_to_k(C2, K) === nothing && return nothing

        part1 = consensus_partition([files[1]], K; require_all = true, label = "$fit_name chain1")
        part2 = consensus_partition([files[2]], K; require_all = true, label = "$fit_name chain2")
        cons = consensus_partition(files, K; require_all = true, label = fit_name)

        chain1_to_cons = greedy_match(part1.labels, cons.labels)
        chain2_to_cons = greedy_match(part2.labels, cons.labels)
        lab1 = [chain1_to_cons[v] for v in part1.labels]
        lab2 = [chain2_to_cons[v] for v in part2.labels]

        levels = collect(1:K)
        t = contingency_matrix(lab1, lab2, levels, levels)
        n = sum(t)
        diagonal = sum(t[i, i] for i in axes(t, 1))
        ari = adjusted_rand_index(lab1, lab2)
        t_scaled = scale_to_total(t, target_n)

        write_contingency_csv(joinpath(outdir, "contingency_within_$(fit_name)_mcmc1_vs_mcmc2_scaled_N$(target_n).csv"),
                              t_scaled, levels, levels)

        return (
            fit = fit_name,
            files = files,
            K = K,
            k1 = k1,
            k2 = k2,
            n = n,
            target_n = target_n,
            t_scaled = t_scaled,
            agreement = 100 * diagonal / n,
            ari = ari,
        )
    end

    full_within = within("FULL", full_files, FULL_K, FULL_TARGET_N)
    arm1_within_groups = [
        [joinpath(root, "output", "openTotalmcmc1.jld2"),
         joinpath(root, "output", "openTotalmcmc2.jld2")],
        [joinpath(root, "archive", "openTotalmcmc1.jld2"),
         joinpath(root, "archive", "openTotalmcmc2.jld2")],
    ]
    arm1_within_files = choose_files(opts, "arms1-within", arm1_within_groups, ARM_K; require_all = true)
    arm1_within = isempty(arm1_within_files) ? nothing :
                  within("ARMS1", arm1_within_files, ARM_K, ARM_TARGET_N)

    arm2_within = nothing
    arm2_within_groups = [
        [joinpath(root, "output", "openTotal2mcmc1.jld2"),
         joinpath(root, "output", "openTotal2mcmc2.jld2")],
        [joinpath(root, "archive", "openTotal2mcmc1.jld2"),
         joinpath(root, "archive", "openTotal2mcmc2.jld2")],
    ]
    arm2_within_files = choose_files(opts, "arms2-within", arm2_within_groups, ARM_K; require_all = true)
    if !isempty(arm2_within_files)
        arm2_within = within("ARMS2", arm2_within_files, ARM_K, ARM_TARGET_N)
    end

    open(texpath, "a") do io
        within_all = [x for x in [full_within, arm1_within, arm2_within] if x !== nothing]
        if !isempty(within_all)
            parts = String[]
            for x in within_all
                push!(parts, @sprintf("%.1f\\%% (ARI %.3f) for %s at \$K=%d\$", x.agreement, x.ari, x.fit, x.K))
            end
            println(io, "Agreement was ", join(parts, ", "), ".")
            if arm1_within !== nothing && (arm1_within.k1 != ARM_K || arm1_within.k2 != ARM_K)
                println(io, "The two ARMS1 chains had modal \$K=$(arm1_within.k1)\$ and \$K=$(arm1_within.k2)\$; both contained \$K=$(arm1_within.K)\$ draws, so their comparison uses \$K=$(arm1_within.K)\$.")
            end
            if arm2_within === nothing
                println(io, "% ARMS2 within-chain K=$(ARM_K) table omitted: no pair of K=$(ARM_K) chains found.")
            end
            println(io)
        end

        if full_within !== nothing
            write_within_table(io, full_within.t_scaled, FULL_K, "FULL", FULL_TARGET_N, full_within.n;
                caption = "FULL \$K=$(FULL_K)\$ mcmc1 versus mcmc2 cluster assignments. Rows are mcmc1 clusters; columns are mcmc2 clusters. Counts are proportionally scaled from \$n=$(full_within.n)\$ to \$N=$(FULL_TARGET_N)\$ and rounded to whole subjects.",
                label = "tab:withinFull")
        end
        if arm1_within !== nothing
            write_within_table(io, arm1_within.t_scaled, ARM_K, "ARMS1", ARM_TARGET_N, arm1_within.n;
                caption = "ARMS1 \$K=$(ARM_K)\$ mcmc1 versus mcmc2 cluster assignments. Rows are mcmc1 clusters; columns are mcmc2 clusters. Counts are proportionally scaled from \$n=$(arm1_within.n)\$ to \$N=$(ARM_TARGET_N)\$ and rounded to whole subjects.",
                label = "tab:withinArms1")
        end
        if arm2_within !== nothing
            write_within_table(io, arm2_within.t_scaled, ARM_K, "ARMS2", ARM_TARGET_N, arm2_within.n;
                caption = "ARMS2 \$K=$(ARM_K)\$ mcmc1 versus mcmc2 cluster assignments. Rows are mcmc1 clusters; columns are mcmc2 clusters. Counts are proportionally scaled from \$n=$(arm2_within.n)\$ to \$N=$(ARM_TARGET_N)\$ and rounded to whole subjects.",
                label = "tab:withinArms2")
        end

        within_rows = NamedTuple[]
        for x in [full_within, arm1_within, arm2_within]
            x === nothing && continue
            push!(within_rows, (
                comparison = "$(x.fit) mcmc1 vs.\\ mcmc2",
                n = x.target_n,
                k1 = x.k1,
                k2 = x.k2,
                k_used = x.K,
                agreement = x.agreement,
                ari = x.ari,
            ))
        end
        if !isempty(within_rows)
            write_within_metrics(io, within_rows;
                caption = "Within-chain agreement and ARI. \$N\$ is the scaled total; agreement and ARI are invariant to proportional scaling.",
                label = "tab:withinMetrics")
        end

        println(io, "\\clearpage")
        println(io)
        println(io, "\\section*{Supplementary contingency tables}")
        println(io)
        println(io, "The reverse tabulations transpose Tables~\\ref{tab:reproFullArm1} and~\\ref{tab:reproFullArm2}, exchanging rows and columns while retaining the original cluster labels.")
        println(io)

        write_cross_table(io, r1.t_arm_scaled, r1.arm_row_order, r1.full_col_order_rev;
            row_header = "ARMS1 cluster", col_header = "FULL cluster",
            caption = "ARMS1 \$K=$(ARM_K)\$ versus FULL \$K=$(FULL_K)\$ cluster assignments. Rows are ARMS1 clusters; columns are FULL clusters.",
            label = "tab:reproReverseArm1")
        write_cross_table(io, r2.t_arm_scaled, r2.arm_row_order, r2.full_col_order_rev;
            row_header = "ARMS2 cluster", col_header = "FULL cluster",
            caption = "ARMS2 \$K=$(ARM_K)\$ versus FULL \$K=$(FULL_K)\$ cluster assignments. Rows are ARMS2 clusters; columns are FULL clusters.",
            label = "tab:reproReverseArm2")
    end

    manifest = joinpath(outdir, "manifest.txt")
    open(manifest, "w") do io
        println(io, "Generated: ", now())
        println(io, "ROOT: ", root)
        println(io, "OUTDIR: ", outdir)
        println(io)
        println(io, "FULL K=$FULL_K files:")
        for f in full_files
            C = get_C(f)
            println(io, "  ", f, " n=", size(C, 2), " draws=", size(C, 1), " dist=", k_dist(C))
        end
        println(io, "\nARMS1 K=$ARM_K cross-fit files:")
        for f in arm1_cross
            C = get_C(f)
            println(io, "  ", f, " n=", size(C, 2), " draws=", size(C, 1), " dist=", k_dist(C))
        end
        println(io, "\nARMS2 K=$ARM_K cross-fit files:")
        for f in arm2_cross
            C = get_C(f)
            println(io, "  ", f, " n=", size(C, 2), " draws=", size(C, 1), " dist=", k_dist(C))
        end
        println(io, "\nFULL within-chain files:")
        for f in full_files
            println(io, "  ", f)
        end
        println(io, "\nARMS1 within-chain files:")
        if isempty(arm1_within_files)
            println(io, "  none with K=$ARM_K")
        else
            for f in arm1_within_files
                println(io, "  ", f)
            end
        end
        println(io, "\nARMS2 within-chain files:")
        if isempty(arm2_within_files)
            println(io, "  none with K=$ARM_K")
        else
            for f in arm2_within_files
                println(io, "  ", f)
            end
        end
        println(io, "\nCross-fit results:")
        for r in [r1, r2]
            println(io, "  ", r.name, " n=", r.n,
                    " FULL->", r.name, " agreement=", @sprintf("%.2f", 100 * r.full_to_arm_agreement),
                    " ARI=", @sprintf("%.4f", r.ari))
        end
    end

    println("Wrote ", texpath)
    println("Wrote ", manifest)
    return texpath
end

if abspath(PROGRAM_FILE) == @__FILE__
    opts = parse_args(ARGS)
    if get(opts, "help", "false") == "true"
        println("Usage: julia --project=\$HOME/software/ProductPartitionModels.jl/simulations regen_contingencies_k6.jl [options]")
        println("  --root DIR             analyses directory (default: auto-detect)")
        println("  --outdir DIR           output directory (default: ROOT/output/contingency_k6)")
        println("  --full F1,F2           FULL K=5 chain files")
        println("  --arms1-cross F1[,F2]  ARMS1 K=6 source(s)")
        println("  --arms2-cross F1[,F2]  ARMS2 K=6 source(s)")
        println("  --arms1-within F1,F2   ARMS1 K=6 within-chain files")
        println("  --arms2-within F1,F2   ARMS2 K=6 within-chain files")
        println("  --inspect              print K distributions and exit")
        exit()
    end
    run(opts)
end
