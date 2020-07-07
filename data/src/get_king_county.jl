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
CSV.write("house.csv", df)
