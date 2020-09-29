# Setup:

isbinder() = "jovyan" in split(pwd(), "/")

const REPO = "https://github.com/ablaom/MachineLearningInJulia2020"
using Pkg

if !isbinder()
    Pkg.activate(DIR)
    Pkg.instantiate()
    using CategoricalArrays
    import MLJLinearModels
    import DataFrames
    import CSV
    import DecisionTree
    using MLJ
    import MLJLinearModels
    import MultivariateStats
    import MLJFlux
    import Plots
else
    @info "Skipping package instantiation as binder notebook. "
end
@info "Done loading"
