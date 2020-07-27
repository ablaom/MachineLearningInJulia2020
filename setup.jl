# Setup:

isbinder() = "jovyan" in split(pwd(), "/")

const REPO = "https://github.com/ablaom/MachineLearningInJulia2020"
using Pkg
uuid = Pkg.TOML.parsefile(joinpath(DIR,"Project.toml"))["uuid"]
uuid == "4764ce03-6504-4302-ab9a-b32cdba420f3" ||
    error("It appears this file is not in the same directory as other "*
          "files it needs (and in particular, the intended "*
          "Project.toml file). "*
          "A complete tutorial can be obtained by cloning the"*
          "MachineLearningInJulia repository from $REPO. ")
if !isbinder()
    Pkg.activate(DIR)
    Pkg.instantiate()
else
    @info "Skipping package instantiation as binder notebook. "
end
using CategoricalArrays
import MLJLinearModels
import DataFrames
import CSV
import DecisionTree
import MLJFlux
import Plots
import MLJLinearModels
import MultivariateStats
using MLJ
