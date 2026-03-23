
function assign_clusters(X, centroids)
    return [argmin([norm(X[:, i] - centroids[:, j]) for j in 1:size(centroids, 2)]) for i in 1:size(X, 2)]
end

# note rmse will hvae to be inverted
function findEquilibrated(; rmse::Vector{Float64}, rind::Vector{Float64}, tol::Float64=0.01)
    rmseEq = abs.(rmse .- minimum(rmse)) .<= (tol .* minimum(rmse))
    rindEq = abs.(rind .- maximum(rind)) .<= (tol .* maximum(rind))

    # 'and' v1 and v2 bit vectors together elementwise
    eq = rmseEq .& rindEq
    eq = findfirst(x -> x == 1, eq)
    if isnothing(eq)
        eq = 1
    end
    return eq
end

function clean_dataframe(df::DataFrame)
    # Step 1: Replace NaN and nothing with missing
    for name in names(df)
        df[!, name] = replace(df[!, name]) do val
            (val === nothing || (val isa AbstractFloat && isnan(val))) ? missing : val
        end
    end

    # Step 2: Filter for complete cases
    df_clean = df[completecases(df), :]

    # Step 3: Convert column types to remove Union{Missing, T}
    for name in names(df_clean)
        col = df_clean[!, name]
        if eltype(col) <: Union{Missing, Any} && !any(ismissing, col)
            T = nonmissingtype(eltype(col))
            df_clean[!, name] = convert(Vector{T}, col)
        end
    end

    return df_clean
end

function minimumSet(variable;  ci = [0.9])
    cb1dens = kde(variable)
    dx = cb1dens.x[2] - cb1dens.x[1]
    idx = sortperm(cb1dens.density, rev=true)
    ecdf = cumsum(cb1dens.density[idx]) * dx
    minSet = cb1dens.x[idx[1:findfirst(ecdf .> 0.9)]]
    minx = minimum(minSet)
    maxx = maximum(minSet)
    return [minx, maxx]
end
