# Setup:

const REPO = "https://github.com/ablaom/MachineLearningInJulia2020"

using Pkg
DIR = @__DIR__
last(splitpath(DIR)) == "MachineLearningInJulia2020" ||
    error("This script is not intended to be run outside "*
          "of the MachineLearningInJulia2020 root directory. "*
          "\n You can clone that repository from $REPO .")
Pkg.activate(DIR)
Pkg.instantiate()
using CategoricalArrays
import MLJLinearModels
import DataFrames
import CSV
import DecisionTree
import MLJFlux
#import Plots
import MLJLinearModels
import MultivariateStats
using MLJ
