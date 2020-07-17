using Pkg;
Pkg.activate(@__DIR__)
Pkg.instantiate()

using MLJ
using PrettyPrinting
import DataFrames: DataFrame, select!, Not, describe
import Statistics
using Dates
using UrlDownload
using CSV


df = DataFrame(urldownload("https://raw.githubusercontent.com/tlienart/DataScienceTutorialsData.jl/master/data/kc_housing.csv", true))
describe(df)

df.is_renovated = df.yr_renovated .== 0

select!(df, Not([:id, :date, :yr_renovated]))
CSV.write(joinpath(@__DIR__, "..", "house.csv"), df)
