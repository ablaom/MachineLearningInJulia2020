# # Machine Learning in Julia, JuliaCon2020

# A workshop introducing the machine learning toolbox
# [MLJ](https://alan-turing-institute.github.io/MLJ.jl/stable/)

include(joinpath(@__DIR__, "setup.jl"))

# ## Part 1: Data representation

# ### Scientific types

# To help you focus on the intended *purpose* or *interpretation* of
# data, MLJ models specify data requirements using *scientific types*,
# instead of machine types. An example of a scientific type is
# `OrderedFactor`. The other basic "scalar" scientific types are
# illustrated below:

# ![](assets/scitypes.png)

# A scientific type is an ordinary Julia type (so it can be used for
# method dispatch, for example) but it usually has no instances. The
# `scitype` function is used to articulate MLJ's convention about how
# different machine types will be interpreted by MLJ models:

using MLJ
scitype(3.141)

#-

time = [2.3, 4.5, 4.2, 1.8, 7.1]
scitype(time)

# To fix data which MLJ is interpreting incorrectly, we use the
# `coerce` method:

height = [185, 153, 163, 114, 180]
scitype(height)

#-

height = coerce(height, Continuous)

# Here's an example of data we would want interpreted as `OrderedFactor`:

exam_mark = ["rotten", "great", "bla",  missing, "great"]
scitype(exam_mark)

#-

exam_mark = coerce(exam_mark, OrderedFactor)

#-

levels(exam_mark)

# Use `levels!` to put the classes in the right order:

levels!(exam_mark, ["rotten", "bla", "great"])
exam_mark[1] < exam_mark[2]

# Subsampling preserves levels:

levels(exam_mark[1:2])

# Note that there is no separate scientific type for binary
# data. Binary data is `OrderedFactor{2}` if it has an intrinsic
# "true" class (eg, "pass"/"fail") and `Multiclass{2}` otherwise (eg,
# "male"/"female").


# ### Two-dimensional data

# Whenever it makes sense, MLJ Models generally expect two-dimensional
# data to be *tabular*. All the tabular formats implementing the
# [Tables.jl API](https://juliadata.github.io/Tables.jl/stable/) (see
# this
# [list](https://github.com/JuliaData/Tables.jl/blob/master/INTEGRATIONS.md))
# have a scientific type of `Table` and can be used with such models.

# The simplest example of a table is a the julia native *column
# table*, which is just a named tuple of equal-length vectors:


column_table = (h=height, e=exam_mark, t=time)

#-

scitype(column_table)

#-

# Notice the `Table{K}` type parameter `K` encodes the scientific
# types of the columns. To see the individual types of columns, we use
# the `schema` method:

schema(column_table)

# Here are four other examples of tables:

row_table = [(a=1, b=3.4),
             (a=2, b=4.5),
             (a=3, b=5.6)]
schema(row_table)

#-

using DataFrames
df = DataFrame(column_table)

#-

schema(df)

#-

using CSV
file = CSV.File(joinpath(DIR, "data", "horse.csv"));
schema(file) # (triggers a file read)


# Most MLJ models do not accept matrix in lieu of a table, but you can
# wrap a matrix as a table:

matrix_table = MLJ.table(rand(2,3))
schema(matrix_table)

# Under the hood many algorithms convert tabular data to matrices. If
# your table is a wrapped matrix like the above, then the compiler
# will generally collapse the conversions to a no-op.


# ### Fixing scientific types in tabular data

# To show how we can correct the scientific types of data in tables,
# we introduce a cleaned up version of the UCI Horse Colic Data Set
# (the cleaning workflow is described
# [here](https://alan-turing-institute.github.io/DataScienceTutorials.jl/end-to-end/horse/#dealing_with_missing_values))

using CSV
file = CSV.File(joinpath(DIR, "data", "horse.csv"));
horse = CSV.DataFrame!(file); # convert to data frame without copying columns
first(horse, 4)

#-

# From [the UCI
# docs](http://archive.ics.uci.edu/ml/datasets/Horse+Colic) we can
# surmise how each variable ought to be interpreted (a step in our
# workflow that cannot reliably be left to the computer):

# variable                    | scientific type (interpretation)
# ----------------------------|-----------------------------------
# `:surgery`                  | Multiclass
# `:age`                      | Multiclass
# `:rectal_temperature`       | Continuous
# `:pulse`                    | Continuous
# `:respiratory_rate`         | Continuous
# `:temperature_extremities`  | OrderedFactor
# `:mucous_membranes`         | Multiclass
# `:capillary_refill_time`    | Multiclass
# `:pain`                     | OrderedFactor
# `:peristalsis`              | OrderedFactor
# `:abdominal_distension`     | OrderedFactor
# `:packed_cell_volume`       | Continuous
# `:total_protein`            | Continuous
# `:outcome`                  | Multiclass
# `:surgical_lesion`          | OrderedFactor
# `:cp_data`                  | Multiclass

# Let's see how MLJ will actually interpret the data, as it is
# currently encoded:

schema(horse)

# As a first correction step, we can get MLJ to "guess" the
# appropriate fix, using the `autotype` method:

autotype(horse)

#-

# Okay, this is not perfect, but a step in the right direction, which
# we implement like this:

coerce!(horse, autotype(horse));
schema(horse)

# All remaining `Count` data should be `Continuous`:

coerce!(horse, Count => Continuous);
schema(horse)

# We'll correct the remaining truant entries manually:

coerce!(horse,
        :surgery               => Multiclass,
        :age                   => Multiclass,
        :mucous_membranes      => Multiclass,
        :capillary_refill_time => Multiclass,
        :outcome               => Multiclass,
        :cp_data               => Multiclass);
schema(horse)


# #### Resources for Part 1
#
# - From the MLJ manual:
#    - [A preview of data type specification in
#   MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/getting_started/#A-preview-of-data-type-specification-in-MLJ-1)
#    - [Data containers and scientific types](https://alan-turing-institute.github.io/MLJ.jl/dev/getting_started/#Data-containers-and-scientific-types-1)
#    - [Working with Categorical Data](https://alan-turing-institute.github.io/MLJ.jl/dev/working_with_categorical_data/)
# - [Summary](https://alan-turing-institute.github.io/MLJScientificTypes.jl/dev/#Summary-of-the-MLJ-convention-1) of the MLJ convention for representing scientific types 
# - [MLJScientificTypes.jl](https://alan-turing-institute.github.io/MLJScientificTypes.jl/dev/)
# - From Data Science Tutorials:
#     - [Data interpretation: Scientific Types](https://alan-turing-institute.github.io/DataScienceTutorials.jl/data/scitype/)
#     - [Horse colic data](https://alan-turing-institute.github.io/DataScienceTutorials.jl/end-to-end/horse/)
# - [UCI Horse Colic Data Set](http://archive.ics.uci.edu/ml/datasets/Horse+Colic)

# #### Exercises

# ##### Ex 1 

# Try to guessing the how each cell below will evaluate before doing so:

scitype(42)

#-

scitype(["who", "why", "what", "when"])

#-

t = (3.141, 42, "how")
scitype(t)

#-

A = rand(2, 3)
scitype(A)

#-

using SparseArrays
Asparse = sparse(A)
scitype(Asparse)

#-

using CategoricalArrays
C1 = categorical(A)
scitype(C1)

#-

C2 = categorical(A, ordered=true)
scitype(C2)

#-

v = [1, 2, missing, 4]
scitype(v)

#- 

scitype(v[1:2])

# Can you guess at the general behaviour of
# `scitype` with respect to tuples, abstract arrays and missing
# values? The answers are
# [here](https://github.com/alan-turing-institute/ScientificTypes.jl#2-the-scitype-and-scitype-methods)
# (ignore "Property 1").


# ##### Ex 2

# Coerce the following vector to make MLJ recognize it as an ordered
# factor (with the factors in appropriate order):

quality = ["good", "poor", "poor", "excellent", missing, "good", "excellent"]

#-


# ##### Ex 3 (fixing scitypes in a table)

# Fix the scitypes for the [House Prices in King County data
# set](https://mlr3gallery.mlr-org.com/posts/2020-01-30-house-prices-in-king-county/):

file = CSV.File(joinpath(DIR, "data", "house.csv"));
house = CSV.DataFrame!(file); # convert to data frame without copying columns
first(house, 4)

# ---

# ## Part 2: Selecting, training and evaluating models

# The "Hello World!" of machine learning is to classify Fisher's
# famous iris data set. This time, we'll grab the data from
# [OpenML](https://www.openml.org):

iris = OpenML.load(61); # a row table
iris = DataFrame(iris);
first(iris, 4)


# **Our goal** is to build and evaluate models for predicting the
# `:class` variable, given the four remaining measurement variables.


# ### Step 1. Inspect and fix scientific types

schema(iris)

#-

coerce!(iris, :class => Multiclass);
schema(iris)


# ### Step 2. Split data into input and target parts

# Here's how we split the data into target and input features, which
# is needed for MLJ supervised models. We randomize the data at the
# same time:

y, X, = unpack(iris, ==(:class), col->true; rng=123);
scitype(y)

# Do `?unpack` to learn more. 


# ### On searching for a model

# Here's how to see *all* models (not immediately useful):

kitchen_sink = models()

# Each entry contains metadata for a model whose defining code is not yet loaded:

meta = kitchen_sink[3]

#-

targetscitype = meta.target_scitype

#-

scitype(y) <: targetscitype

# So this model won't do. Let's  find all pure julia classifiers:

filter(meta) = AbstractVector{Finite} <: meta.target_scitype &&
        meta.is_pure_julia

models(filter)


# Find all models with "Classifier" in `name` (or `docstring`):

models("Classifier")


# Find all (supervised) models that match my data!

models(matching(X, y))



# ### Step 3. Select and instantiate a model

model = @load NeuralNetworkClassifier

#-

info(model)

# In MLJ a *model* is just a struct containing hyperparameters, and
# that's all. A model does not store *learned* parameters. Models are
# mutable:

model.epochs = 12

# And all models have a key-word constructor that works once `@load`
# has been performed:

NeuralNetworkClassifier(epochs=12) == model


# ### On fitting and predicting

# In MLJ a model and training/evaluation data are typically bound
# together in a machine:

mach = machine(model, X, y)

# A machine stores *learned* parameters, among other things. Let's fit
# our machine on some subset of the data and make a prediction on a
# 30% holdout set. We start by dividing all row indices into `train` and
# `test` subsets:

train, test = partition(eachindex(y), 0.7)

#-

fit!(mach, rows=train, verbosity=2)

# Machines remember the last set of hyperparameters used during fit,
# which, in the case of iterative models, allows them to restart
# computations where they left off, when the iteration parameter is
# increased:

model.epochs = model.epochs + 4
fit!(mach, rows=train, verbosity=2)

# By default, we can also increase `:learning_rate` without a cold restart:

model.epochs = model.epochs + 4
model.optimiser.eta = 10*model.optimiser.eta
fit!(mach, rows=train, verbosity=2)

# However, change the regularization parameter and training will
# restart from scratch:

model.lambda = 0.001
fit!(mach, rows=train, verbosity=2)

# Let's train silently for a total of 50 epochs, and look at a prediction:

model.epochs = 50
fit!(mach, rows=train)
yhat = predict(mach, X[test,:]);
yhat[1]

# What's going on here?

info(model).prediction_type

# **Important**:
# - In MLJ, a model that can predict probabilities (and not just point values) will do so by default. (These models have supertype `Proababilistic`, while point-estimate predictors have supertype `Deterministic`.)
# - For most probabilistic predictors, the predicted object is a `Distributions.Distribution` object, supporting the `Distributions.jl` [API](https://juliastats.org/Distributions.jl/latest/extends/#Create-a-Distribution-1) for such objects. In particular, the methods `rand`,  `pdf`, `mode`, `median` and `mean` will apply, where appropriate.

# So, to obtain the probability of "Iris-virginica" in the first test
# prediction, we do

pdf(yhat[1], "Iris-virginica")

# To get the most likely observation, we do

mode(yhat[1])

# These can be broadcast over multiple predictions in the usual way:

broadcast(pdf, yhat[1:4], "Iris-versicolor")

#-

mode.(yhat[1:4])

# Or, alternatively, you can use the `predict_mode` operation instead
# of `predict`:

predict_mode(mach, X[test,:])[1:4]

# For a more conventional matrix of probabilities you can do this:

L = levels(y)
pdf(yhat, L)[1:4, :]

# However, in a typical MLJ workflow, this is not as useful as you
# might imagine. In particular, all probablistic performance measures
# in MLJ expect distribution objects in their first slot:

cross_entropy(yhat, y[test]) |> mean

# To apply a deterministic measure, we first need to obtain point-estimates:

misclassification_rate(mode.(yhat), y[test])


# ### Step 4. Evaluate the model performance

# Naturally, MLJ provides boilerplate code for carrying out a model
# evaluation with a lot less fuss. Let's repeat the performance
# evaluation above and add an extra measure, `brier_score`:

evaluate!(mach, resampling=Holdout(fraction_train=0.7),
                measures=[cross_entropy, brier_score])

# Or applying cross-validation instead:

evaluate!(mach, resampling=CV(nfolds=6),
                measures=[cross_entropy, brier_score])

# Or, Monte-Carlo cross-validation (cross-validation repeated
# randomizied folds)

evaluate!(mach, resampling=CV(nfolds=6, rng=123),
                repeats=3,
                measures=[cross_entropy, brier_score])





using Literate #src
Literate.markdown(@__FILE__, @__DIR__) #src
Literate.notebook(@__FILE__, @__DIR__) #src
