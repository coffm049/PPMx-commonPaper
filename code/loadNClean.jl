using Pkg
Pkg.activate("../../simulations/")
using StatsBase
using Statistics
# using StatsPlots
# using Distributions
# using Random
# using Plots
using DataFrames
using StatsModels
# using LinearAlgebra
using ProductPartitionModels
using Clustering
using JLD2
using CSV
using GLM
using TidierData
# using HypothesisTests
# using LaTeXStrings
using UnicodePlots
using MAT
include("utilities.jl")

function loadNclean()

    # ADHD labels
    #labels = CSV.read("/panfs/jay/groups/2/feczk001/shared/projects/FEZ_USERS/feczk001/UPPS_ABCD_FRF/data/ABCD_ADHD_prevalence.csv", DataFrame, 
    labels = CSV.read("/projects/standard/feczk001/shared/projects/FEZ_USERS/feczk001/UPPS_ABCD_FRF/data/ABCD_ADHD_prevalence.csv", DataFrame, 
                             types = Dict(
                             1=>String,
                             2=>String, 
                             3=>String,
                             4=>String,
                             5=>String,
    #                         6=>Float64
                             ))
    rename!(labels, Dict("subjectkey" => "IID"))
    labels = filter(:ADHDcomposite => x -> !isnan(x), labels)
    
    fullDF = CSV.read("../data/ADHDCovars.csv", DataFrame; normalizenames = true)
    #nn = names(fullDF)
    rename!(fullDF, Dict(
        "bisbas_ss_bas_rr" => "bbRR",
        "bisbas_ss_bas_drive" => "bbDrive",
        "bisbas_ss_bas_fs" => "bbFS",
        "bisbas_ss_bis_sum" => "bbSum",
        "upps_ss_negative_urgency" => "uNegUrg",
        "upps_ss_positive_urgency" => "uPosUrg",
        "upps_ss_lack_of_planning" => "uLplanning",
        "upps_ss_lack_of_perseverance" => "uLpers",
        # note all nihtbx items are age corrected
        "nihtbx_picvocab_agecorrected" => "nPicVocab",
        "nihtbx_flanker_agecorrected" => "nFlank",
        "nihtbx_list_agecorrected" => "nList",
        "nihtbx_cardsort_agecorrected" => "nCard",
        "nihtbx_pattern_agecorrected" => "nPattern",
        "nihtbx_picture_agecorrected" => "nPic",
        "nihtbx_reading_agecorrected" => "nRead",
        "nihtbx_fluidcomp_agecorrected" => "nFluid",
        "nihtbx_cryst_agecorrected" => "nCryst",
        "nihtbx_totalcomp_agecorrected" => "nTotal"
    ))
    select!(fullDF, [
        "FID",
        "IID",
        "abcd_site",
        "age",
        "female",
        "household_income",
        "high_educ",
        "race_ethnicity",
        "rel_relationship",
        "bbRR",
        "bbDrive",
        "bbFS",
        "bbSum",
        "uNegUrg",
        "uPosUrg",
        "uLplanning",
        "uLpers",
        # note all nihtbx items are age corrected
        "nPicVocab",
        "nFlank",
        "nList",
        "nCard",
        "nPattern",
        "nPic",
        "nRead",
        "nFluid",
        "nCryst",
        "nTotal"
    ])
    
    
    # CLEAN
    # convert strings to numerics
    for name in names(fullDF)
        if !(name ∈ ["key", "female", "FID", "IID", "abcd_site", "household_income", "high_educ", "race_ethnicity", "rel_relationship", "female"]) &&eltype(fullDF[!, name]) <: AbstractString
            println(name)
            fullDF[!, name] = map(x -> tryparse(Float64, x) === nothing ? missing : parse(Float64, x), fullDF[!, name])
        end
    end
    
    fullDF = innerjoin(fullDF, labels, on = "IID")
    
    # Standardize
    standardization_params = Dict()
    
    for name in names(fullDF)
        if !(eltype(fullDF[:,name]) <:AbstractString)
            col = fullDF[!, name]
            μ = mean(skipmissing(col))
            σ = std(skipmissing(col))
            fullDF[!, name] = (col .- μ) ./ σ
            standardization_params[Symbol(name)] = (μ, σ)
        end
    end
    
    fullDF.female = map(x -> x == "no" ? 0 : x == "yes" ? 1 : missing, fullDF.female)
    fullDF.ADHD1 = map(x -> x == "Ctrl" ? 0 : x == "ADHD" ? 1 : missing, fullDF.ADHD1)
    fullDF.ADHD2 = map(x -> x == "Ctrl" ? 0 : x == "ADHD" ? 1 : missing, fullDF.ADHD2)
    fullDF.ADHD3 = map(x -> x == "Ctrl" ? 0 : x == "ADHD" ? 1 : missing, fullDF.ADHD3)
    fullDF.ADHD4 = map(x -> x == "Ctrl" ? 0 : x == "ADHD" ? 1 : missing, fullDF.ADHD4)

    # some subjects have complete missingness 
    # this filter removew 33 subjects out of all 11854
    rowMissing = map(row -> count(ismissing, row), eachrow(fullDF)) .<= 2
    filtereDf = fullDF[rowMissing, :]
    # fullDF = fullDF[completecases(fullDF), :]
    # colMissing = map(col -> count(ismissing, col), eachcol(filtereDf))
    # rowMissing = map(row -> count(ismissing, row), eachrow(filtereDf))
    return fullDF, standardization_params
end
