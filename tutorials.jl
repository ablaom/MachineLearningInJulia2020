# # Machine Learning in Julia, JuliaCon2020

# A workshop introducing the machine learning toolbox
# [MLJ](https://alan-turing-institute.github.io/MLJ.jl/stable/)
# ### Environment instantiation

# The following loads a Julia environment and assumes you have the
# following files in the same directory as this file:

# - Project.toml
# - Manifest.toml
# - setup.jl

include(joinpath(@__DIR__, "setup.jl"))

# If this is the notebook version of the tutorial, then it is
# recommended that you clear all cell outputs before attempting the
# tutorial.

# ## Contents

# ### Basic

# - [Part 1 - Data Representation](#part-1-data-representation)
# - [Part 2 - Selecting, Training and Evaluating Models](#part-2-selecting-training-and-evaluating-models)
# - [Part 3 - Transformers and Pipelines](#part-3-transformers-and-pipelines)

# ### Advanced

# - [Part 4 - Tuning Hyper-parameters](#part-4-tuning-hyperparameters)


# <a id='part-1-data-representation'></a>
# ## Part 1 - Data Representation

# > **Goals:**
# > 1. Learn how MLJ specifies it's data requirements using "scientific" types
# > 2. Understand the options for representing tabular data
# > 3. Learn how to inspect and fix the representation of data to meet MLJ requirements


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

# Here's an example of data we would want interpreted as
# `OrderedFactor` but isn't:

exam_mark = ["rotten", "great", "bla",  missing, "great"]
scitype(exam_mark)

#-

exam_mark = coerce(exam_mark, OrderedFactor)

#-

levels(exam_mark)

# Use `levels!` to put the classes in the right order:

levels!(exam_mark, ["rotten", "bla", "great"])
exam_mark[1] < exam_mark[2]

# When subsampling, no levels are not lost:

levels(exam_mark[1:2])

# **Note on binary data.** There is no separate scientific type for binary
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
# types of the columns. (This is useful when comparing table scitypes
# with `<:`). To inspect the individual column scitypes, we use the
# `schema` method instead:

schema(column_table)

# Here are four other examples of tables:

row_table = [(a=1, b=3.4),
             (a=2, b=4.5),
             (a=3, b=5.6)]
schema(row_table)

#-

import DataFrames
df = DataFrames.DataFrame(column_table)

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


# **Manipulating tabular data.** In this workshop we assume
# familiarity with some kind of tabular data container (although it is
# possible, in principle, to carry out the exercises without this.)
# For a quick start introduction to `DataFrames`, see [this
# tutorial](https://alan-turing-institute.github.io/DataScienceTutorials.jl/data/dataframe/)

# ### Fixing scientific types in tabular data

# To show how we can correct the scientific types of data in tables,
# we introduce a cleaned up version of the UCI Horse Colic Data Set
# (the cleaning workflow is described
# [here](https://alan-turing-institute.github.io/DataScienceTutorials.jl/end-to-end/horse/#dealing_with_missing_values))

using CSV
file = CSV.File(joinpath(DIR, "data", "horse.csv"));
horse = DataFrames.DataFrame(file); # convert to data frame without copying columns
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


# ### Resources for Part 1
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


# ### Exercises for Part 1


# #### Exercise 1

# Try to guess how each code snippet below will evaluate:

scitype(42)

#-

questions = ["who", "why", "what", "when"]
scitype(questions)

#-

elscitype(questions)

#-

t = (3.141, 42, "how")
scitype(t)

#-

A = rand(2, 3)

# -

scitype(A)

#-

elscitype(A)

#-

using SparseArrays
Asparse = sparse(A)

#-

scitype(Asparse)

#-

using CategoricalArrays
C1 = categorical(A)

#-

scitype(C1)

#-

elscitype(C1)

#-

C2 = categorical(A, ordered=true)
scitype(C2)

#-

v = [1, 2, missing, 4]
scitype(v)

#-

elscitype(v)

#-

scitype(v[1:2])

# Can you guess at the general behaviour of
# `scitype` with respect to tuples, abstract arrays and missing
# values? The answers are
# [here](https://github.com/alan-turing-institute/ScientificTypes.jl#2-the-scitype-and-scitype-methods)
# (ignore "Property 1").


# #### Exercise 2

# Coerce the following vector to make MLJ recognize it as a vector of
# ordered factors (with an appropriate ordering):

quality = ["good", "poor", "poor", "excellent", missing, "good", "excellent"]

#-


# #### Exercise 3 (fixing scitypes in a table)

# Fix the scitypes for the [House Prices in King
# County](https://mlr3gallery.mlr-org.com/posts/2020-01-30-house-prices-in-king-county/)
# dataset:

file = CSV.File(joinpath(DIR, "data", "house.csv"));
house = DataFrames.DataFrame(file); # convert to data frame without copying columns
first(house, 4)

# (Two features in the original data set have been deemed uninformative
# and dropped, namely `:id` and `:date`. The original feature
# `:yr_renovated` has been replaced by the `Bool` feature `is_renovated`.)

# <a id='part-2-selecting-training-and-evaluating-models'></a>
# ## Part 2 - Selecting, Training and Evaluating Models

# > **Goals:**
# > 1. Search MLJ's database of model metadata to identify model candidates for a supervised learning task.
# > 2. Evaluate the performance of a model on a holdout set using basic `fit!`/`predict` workflow.
# > 3. Inspect the outcomes of training and save these to a file.
# > 3. Evaluate performance using other resampling strategies, such as cross-validation, in one line, using `evaluate!`
# > 4. Plot a "learning curve", to inspect performance as a function of some model hyper-parameter, such as an iteration parameter

# The "Hello World!" of machine learning is to classify Fisher's
# famous iris data set. This time, we'll grab the data from
# [OpenML](https://www.openml.org):

iris = OpenML.load(61); # a row table
iris = DataFrames.DataFrame(iris);
first(iris, 4)


# **Main goal.** To build and evaluate models for predicting the
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

y, X = unpack(iris, ==(:class), name->true; rng=123);
scitype(y)

# Do `?unpack` to learn more:

@doc unpack

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

filt(meta) = AbstractVector{Finite} <: meta.target_scitype &&
        meta.is_pure_julia
models(filt)


# Find all models with "Classifier" in `name` (or `docstring`):

models("Classifier")


# Find all (supervised) models that match my data!

models(matching(X, y))



# ### Step 3. Select and instantiate a model

model = @load NeuralNetworkClassifier

#-

info(model)

# In MLJ a *model* is just a struct containing hyper-parameters, and
# that's all. A model does not store *learned* parameters. Models are
# mutable:

model.epochs = 12

# And all models have a key-word constructor that works once `@load`
# has been performed:

NeuralNetworkClassifier(epochs=12) == model


# ### On fitting, predicting, and inspecting models

# In MLJ a model and training/validation data are typically bound
# together in a machine:

mach = machine(model, X, y)

# A machine stores *learned* parameters, among other things. We'll
# train this machine on 70% of the data and evaluate on a 30% holdout
# set. Let's start by dividing all row indices into `train` and `test`
# subsets:

train, test = partition(eachindex(y), 0.7)

#-

fit!(mach, rows=train, verbosity=2)

# After training, one can inspect the learned parameters:

fitted_params(mach)

#-

# Everything else the user might be interested in is accessed from the
# training *report*:

report(mach)

# You save a machine like this:

MLJ.save("neural_net.jlso", mach)

# And retrieve it like this:

mach2 = machine("neural_net.jlso")
predict(mach2, X)[1:3]

# If you want to fit a retrieved model, you will need to bind some data to it:

mach3 = machine("neural_net.jlso", X, y)
fit!(mach3)

# Machines remember the last set of hyper-parameters used during fit,
# which, in the case of iterative models, allows for a warm restart of
# computations in the case that only the iteration parameter is
# increased:

model.epochs = model.epochs + 4
fit!(mach, rows=train, verbosity=2)

# By default (for this particular model) we can also increase
# `:learning_rate` without triggering a cold restart:

model.epochs = model.epochs + 4
model.optimiser.eta = 10*model.optimiser.eta
fit!(mach, rows=train, verbosity=2)

# However, change any other parameter and training will restart from
# scratch:

model.lambda = 0.001
fit!(mach, rows=train, verbosity=2)

# Let's train silently for a total of 50 epochs, and look at a prediction:

model.epochs = 50
fit!(mach, rows=train)
yhat = predict(mach, X[test,:]); # or predict(mach, rows=test)
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

predict_mode(mach, X[test,:])[1:4] # or predict_mode(mach, rows=test)[1:4]

# For a more conventional matrix of probabilities you can do this:

L = levels(y)
pdf(yhat, L)[1:4, :]

# However, in a typical MLJ workflow, this is not as useful as you
# might imagine. In particular, all probablistic performance measures
# in MLJ expect distribution objects in their first slot:

cross_entropy(yhat, y[test]) |> mean

# To apply a deterministic measure, we first need to obtain point-estimates:

misclassification_rate(mode.(yhat), y[test])

# We note in passing that there is also a search tool for measures
# analogous to `models`:

measures(matching(y))


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

e = evaluate!(mach, resampling=CV(nfolds=6, rng=123),
                repeats=3,
              measures=[cross_entropy, brier_score])

# One can access the following properties of the output `e` of an
# evaluation: `measure`, `measurement`, `per_fold` (measurement for
# each fold) and `per_observation` (measurement per observation, if
# reported).

# We finally note that you can restrict the rows of observations from
# which train and test folds are drawn, by specifying `rows=...`. For
# example, imagining the last 30% of target observations are `missing`
# you might have a workflow like this:

train, test = partition(eachindex(y), 0.7)
mach = machine(model, X, y)
evaluate!(mach, resampling=CV(nfolds=6),
          measures=[cross_entropy, brier_score],
          rows=train)     # cv estimate, resampling from `train`
fit!(mach, rows=train)    # re-train using all of `train` observations
predict(mach, rows=test); # and predict missing targets


# ### On learning curves

# Since our model is an iterative one, we might want to inspect the
# out-of-sample performance as a function of the iteration
# parameter. For this we can use the `learning_curve` function (which,
# incidentally can be applied to any model hyper-parameter). This
# starts by defining a one-dimensional range object for the parameter
# (more on this when we discuss tuning in Part 4):

r = range(model, :epochs, lower=1, upper=50, scale=:log)
curve = learning_curve(mach,
                       range=r,
                       resampling=Holdout(fraction_train=0.7), # (default)
                       measure=cross_entropy)

using Plots
pyplot(size=(490,300))
plt=plot(curve.parameter_values, curve.measurements)
xlabel!(plt, "epochs")
ylabel!(plt, "cross entropy on holdout set")
plt

# We will return to learning curves when we look at tuning in Part 4.


# ### Resources for Part 2

# - From the MLJ manual:
#     - [Getting Started](https://alan-turing-institute.github.io/MLJ.jl/dev/getting_started/)
#     - [Model Search](https://alan-turing-institute.github.io/MLJ.jl/dev/model_search/)
#     - [Evaluating Performance](https://alan-turing-institute.github.io/MLJ.jl/dev/evaluating_model_performance/) (using `evaluate!`)
#     - [Learning Curves](https://alan-turing-institute.github.io/MLJ.jl/dev/learning_curves/)
#     - [Performance Measures](https://alan-turing-institute.github.io/MLJ.jl/dev/performance_measures/) (loss functions, scores, etc)
# - From Data Science Tutorials:
#     - [Choosing and evaluating a model](https://alan-turing-institute.github.io/DataScienceTutorials.jl/getting-started/choosing-a-model/)
#     - [Fit, predict, transform](https://alan-turing-institute.github.io/DataScienceTutorials.jl/getting-started/fit-and-predict/)


# ### Exercises for Part 2


# #### Exercise 4

# (a) Identify all supervised MLJ models that can be applied (without
# type coercion or one-hot encoding) to a supervised learning problem
# with input features `X4` and target `y4` defined below:

import Distributions
poisson = Distributions.Poisson

age = 18 .+ 60*rand(10);
salary = coerce(rand([:small, :big, :huge], 10), OrderedFactor);
levels!(salary, [:small, :big, :huge]);
X4 = DataFrames.DataFrame(age=age, salary=salary)

n_devices(salary) = salary > :small ? rand(poisson(1.3)) : rand(poisson(2.9))
y4 = [n_devices(row.salary) for row in eachrow(X4)]

# (b) What models can be applied if you coerce the salary to a
# `Continuous` scitype?


# #### Exercise 5 (unpack)

# After evaluating the following ...

data = (a = [1, 2, 3, 4],
     b = rand(4),
     c = rand(4),
     d = coerce(["male", "female", "female", "male"], OrderedFactor));
pretty(data)

using Tables
y, X, w = unpack(data, ==(:a),
                 name -> elscitype(Tables.getcolumn(data, name)) == Continuous,
                 name -> true);

# ...attempt to guess the evaluations of the following:

y

#-

pretty(X)

#-

w


# #### Exercise 6 (first steps in modelling Horse Colic)

# (a) Suppose we want to use predict the `:outcome` variable in the
# Horse Colic study introduced in Part 1, based on the remaining
# variables that are `Continuous` (one-hot encoding categorical
# variables is discussed later in Part 3) *while ignoring the others*.
# Extract from the `horse` data set (defined in Part 1) appropriate
# input features `X` and target variable `y`. (Do not, however,
# randomize the observations.)

# (b) Create a 70:30 `train`/`test` split of the data and train a
# `LogisticClassifier` model, from the `MLJLinearModels` package, on
# the `train` rows. Use `lambda=100` and default values for the
# other hyper-parameters. (Although one would normally standardize
# (whiten) the continuous features for this model, do not do so here.)
# After training:

# - (i) Recalling that a logistic classifier (aka logistic regressor) is
#   a linear-based model learning a *vector* of coefficients for each
#   feature (one coefficient for each target class), use the
#   `fitted_params` method to find this vector of coefficients in the
#   case of the `:pulse` feature. (To convert a vector of pairs `v =
#   [x1 => y1, x2 => y2, ...]` into a dictionary, do `Dict(v)`.)

# - (ii) Evaluate the `cross_entropy` performance on the `test`
#   observations.

# - &star;(iii) In how many `test` observations does the predicted
#   probablility of the observed class exceed 50%?

# - (iv) Find the `misclassification_rate` in the `test`
#   set. (*Hint.* As this measure is deterministic, you will either
#   need to broadcast `mode` or use `predict_mode` instead of
#   `predict`.)

# (c) Instead use a `RandomForestClassifier` model from the
#     `DecisionTree` package and:
#
# - (i) Generate an appropriate learning curve to convince yourself
#   that out-of-sample estimates of the `cross_entropy` loss do not
#   substatially improve for `n_trees > 50`. Use default values for
#   all other hyper-parameters, and feel free to use all available
#   data to generate the curve.

# - (ii) Fix `n_trees=90` and use `evaluate!` to obtain a 9-fold
#   cross-validation estimate of the `cross_entropy`, restricting
#   sub-sampling to the `train` observations.

# - (iii) Now use *all* available data but set
#   `resampling=Holdout(fraction_train=0.7)` to obtain a score you can
#   compare with the `KNNClassifier` in part (b)(iii). Which model is
#   better?

# <a id='part-2-transformers-and-pipelines'></a>
# ## Part 3 - Transformers and Pipelines

# ### Transformers

# Unsupervised models, which receive no target `y` during training,
# always have a `transform` operation. They sometimes also support an
# `inverse_transform` operation, with obvious meaning, and sometimes
# support a `predict` operation (see the clustering example discussed
# [here](https://alan-turing-institute.github.io/MLJ.jl/dev/transformers/#Transformers-that-also-predict-1)).
# Otherwise, they are handled much like supervised models.

# Here's a simple standardization example:

x = rand(100);
@show mean(x) std(x);

#-

model = UnivariateStandardizer() # a built-in model
mach = machine(model, x)
fit!(mach)
x̂ = transform(mach, x);
@show mean(x̂) std(x̂);

# This particular model has an `inverse_transform`:

inverse_transform(mach, x̂) ≈ x


# ### Re-encoding the King County House data as continuous

# For further illustrations of tranformers, let's re-encode *all* of the
# King County House input features (see [Ex
# 3](#ex-3-fixing-scitypes-in-a-table)) into a set of `Continuous`
# features. We do this with the `ContinousEncoder` model, which, by
# default, will:

# - one-hot encode all `Multiclass` features
# - coerce all `OrderedFactor` features to `Continuous` ones
# - coerce all `Count` features to `Continuous` ones (there aren't any)
# - drop any remaining non-Continuous features (none of these either)

# First, we reload the data and fix the scitypes (Exercise 3):

file = CSV.File(joinpath(DIR, "data", "house.csv"));
house = DataFrames.DataFrame(file)
coerce!(house, autotype(file))
coerce!(house, Count => Continuous, :zipcode => Multiclass);
schema(house)

#-

y, X = unpack(house, ==(:price), name -> true, rng=123);

# Instantiate the unsupervised model (transformer):

encoder = ContinuousEncoder() # a built-in model; no need to @load it

# Bind the model to the data and fit!

mach = machine(encoder, X) |> fit!;

# Transform and inspect the result:

Xcont = transform(mach, X);
schema(Xcont)


# ### More transformers

# Here's how to list all of MLJ's unsupervised models:

models(m->!m.is_supervised)

# Some commonly used ones are built-in (do not require `@load`ing):

# model type                  | does what?
# ----------------------------|----------------------------------------------
# ContinuousEncoder | transform input to a table of `Continuous` features (see above)
# FeatureSelector | retain or dump selected features
# FillImputer | impute missing values
# OneHotEncoder | one-hot encoder `Multiclass` (and optionally `OrderedFactor`) features
# Standardizer | standardize (whiten) the `Continuous` features in a table
# UnivariateBoxCoxTransformer | apply a learned Box-Cox transformation to a vector
# UnivariateDiscretizer | discretize a `Continuous` vector, and hence render its elscityp `OrderedFactor`
# UnivariateStandardizer| standardize (whiten) a `Continuous` vector

# In addition to "dynamic" transformers (ones that learn something
# from the data and must be `fit!`) users can wrap ordinary functions
# as transformers, and such *static* transformers can depend on
# parameters, like the dynamic ones. See
# [here](https://alan-turing-institute.github.io/MLJ.jl/dev/transformers/#Static-transformers-1)
# for how to define your own static transformers.


# ### Pipelines

length(schema(Xcont).names)

# Let's suppose that additionally we'd like to reduce the dimension of
# our data.  A model that will do this is `PCA` from
# `MultivariateStats`:

reducer = @load PCA

# Now, rather simply repeating the workflow above, applying the new
# transformation to `Xcont`, we can combine both the encoding and the
# dimension-reducing models into a single model, known as a
# *pipeline*. While MLJ offers a powerful interface for composing
# models in a variety of ways, we'll stick to these simplest class of
# composite models for now. The easiest way to construct them is using
# the `@pipeline` macro:

pipe = @pipeline encoder reducer

# Notice that `pipe` is an *instance* of an automatically generated
# type (called `Pipeline<some digits>`).

# The new model behaves like any other transformer:

mach = machine(pipe, X) |> fit!;
Xsmall = transform(mach, X)
schema(Xsmall)

# Want to combine this pre-processing with ridge regression?

rgs = @load RidgeRegressor pkg=MLJLinearModels
pipe2 = @pipeline encoder reducer rgs

# Now our pipeline is a supervised model, instead of a transformer,
# whose performance we can evaluate:

mach = machine(pipe2, X, y) |> fit!
evaluate!(mach, measure=mae, resampling=Holdout()) # CV(nfolds=6) is default


# ### Training of composite models is "smart"

# Now notice what happens if we train on all the data, then change a
# regressor hyper-parameter and retrain:

fit!(mach)

#-

pipe2.ridge_regressor.lambda = 0.1
fit!(mach)

# Second time only the ridge regressor is retrained!

# Mutate a hyper-parameter of the `PCA` model and every model except
# the `ContinuousEncoder` (which comes before it will be retrained):

pipe2.pca.pratio = 0.9999
fit!(mach)


# ### Inspecting composite models

# The dot syntax used above to change the values of *nested*
# hyper-parameters is also useful when inspecting the learned
# parameters and report generated when training a composite model:

fitted_params(mach).ridge_regressor

#-

report(mach).pca


# ### Incorporating target transformations

# Next, suppose that instead of using the raw `:price` as the
# training target, we want to use the log-price (a common practice in
# dealing with house price data). However, suppose that we still want
# to report final *predictions* on the original linear scale (and use
# these for evaluation purposes). Then we supply appropriate functions
# to key-word arguments `target` and `inverse`.

# First we'll overload `log` and `exp` for broadcasting:
Base.log(v::AbstractArray) = log.(v)
Base.exp(v::AbstractArray) = exp.(v)

# Now for the new pipeline:

pipe3 = @pipeline encoder reducer rgs target=log inverse=exp
mach = machine(pipe3, X, y)
evaluate!(mach, measure=mae)

# MLJ will also allow you to insert *learned* target
# transformations. For example, we might want to apply
# `UnivariateStandardizer()` to the target, to standarize it, or
# `UnivariateBoxCoxTransformer()` to make it look Gaussian. Then
# instead of specifying a *function* for `target`, we specify a
# unsupervised *model* (or model type). One does not specify `inverse`
# because only models implementing `inverse_transform` are
# allowed.

# Let's see which of these two options results in a better outcome:

box = UnivariateBoxCoxTransformer(n=20)
stand = UnivariateStandardizer()

pipe4 = @pipeline encoder reducer rgs target=box
mach = machine(pipe4, X, y)
evaluate!(mach, measure=mae)

#-

pipe4.target = stand
evaluate!(mach, measure=mae)


# ### Resources for Part 3

# - From the MLJ manual:
#     - [Transformers and other unsupervised models](https://alan-turing-institute.github.io/MLJ.jl/dev/transformers/)
#     - [Linear pipelines](https://alan-turing-institute.github.io/MLJ.jl/dev/composing_models/#Linear-pipelines-1)
# - From Data Science Tutorials:
#     - [Composing models](https://alan-turing-institute.github.io/DataScienceTutorials.jl/getting-started/composing-models/)


# ### Exercises for Part 3

# #### Exercise 7

# Consider again the Horse Colic classification problem considered in
# Exercise 6, but with all features, `Finite` and `Infinite`:

y, X = unpack(horse, ==(:outcome), name -> true);
schema(X)

# (a) Define a pipeline that:
# - uses `Standardizer` to ensure that features that are already
#   continuous are centred at zero and have unit variance
# - re-encodes the full set of features as `Continuous`, using
#   `ContinuousEncoder`
# - uses the `KMeans` clustering model from `Clustering.jl`
#   to reduce the dimension of the feature space to `k=10`.
# - trains a `EvoTreeClassifier` (a gradient tree boosting
#   algorithm in `EvoTrees.jl`) on the reduced data, using
#   `nrounds=50` and default values for the other
#    hyper-parameters

# (b) Evaluate the pipeline on all data, using 6-fold cross-validation
# and `cross_entropy` loss.

# &star;(c) Plot a learning curve which examines the effect on this loss
# as the tree booster parameter `max_depth` varies from 2 to 10.


# ## Part 4 - Tuning Hyper-parameters

r = range(pipe3, :(ridge_regressor.lambda), lower = 1e-6, upper=10, scale=:log)

# If you're curious, you can see what `lambda` values this range will
# generate for a given resolution:

iterator(r, 10)



# ## Solutions to exercises


# #### Exercise 2 solution

quality = coerce(quality, OrderedFactor);
levels!(quality, ["poor", "good", "excellent"]);
elscitype(quality)


# #### Exercise 3 solution

# First pass:

coerce!(house, autotype(house));
schema(house)

#-

# All the "sqft" fields refer to "square feet" so are
# really `Continuous`. We'll regard `:yr_built` (the other `Count`
# variable above) as `Continuous` as well. So:

coerce!(house, Count => Continuous);

# And `:zipcode` should not be ordered:

coerce!(house, :zipcode => Multiclass);
schema(house)

# `:bathrooms` looks like it has a lot of levels, but on further
# inspection we see why, and `OrderedFactor` remains appropriate:

import StatsBase.countmap
countmap(house.bathrooms)


# #### Exercise 4 solution

# 4(a)

# There are *no* models that apply immediately:

models(matching(X4, y4))

# 4(b)

y4 = coerce(y4, Continuous);
models(matching(X4, y4))


# #### Exercise 6 solution

# 6(a)

y, X = unpack(horse,
              ==(:outcome),
              name -> elscitype(Tables.getcolumn(horse, name)) == Continuous);

# 6(b)(i)

model = @load LogisticClassifier pkg=MLJLinearModels;
model.lambda = 100
mach = machine(model, X, y)
fit!(mach, rows=train)
fitted_params(mach)

#-

coefs_given_feature = Dict(fitted_params(mach).coefs)
coefs_given_feature[:pulse]

#6(b)(ii)

yhat = predict(mach, rows=test); # or predict(mach, X[test,:])
err = cross_entropy(yhat, y[test]) |> mean

# 6(b)(iii)

# The predicted probabilities of the actual observations in the test
# are given by

p = broadcast(pdf, yhat, y[test]);

# The number of times this probability exceeds 50% is:
n50 = filter(x -> x > 0.5, p) |> length

# Or, as a proportion:

n50/length(test)

# 6(b)(iv)

misclassification_rate(mode.(yhat), y[test])

# 6(c)(i)

model = @load RandomForestClassifier pkg=DecisionTree
mach = machine(model, X, y)
evaluate!(mach, resampling=CV(nfolds=6), measure=cross_entropy)

r = range(model, :n_trees, lower=10, upper=70, scale=:log)

# Since random forests are inherently randomized, we generate multiple
# curves:

plt = plot()
for i in 1:4
    curve = learning_curve(mach,
                           range=r,
                           resampling=Holdout(),
                           measure=cross_entropy)
    plt=plot!(curve.parameter_values, curve.measurements)
end
xlabel!(plt, "n_trees")
ylabel!(plt, "cross entropy")
plt

# 6(c)(ii)

evaluate!(mach, resampling=CV(nfolds=9),
                measure=cross_entropy,
                rows=train).measurement[1]

model.n_trees = 90

# 6(c)(iii)

err_forest = evaluate!(mach, resampling=Holdout(),
                       measure=cross_entropy).measurement[1]

# #### Exercise 7

# (a)

@load KMeans pkg=Clustering
@load EvoTreeClassifier
pipe = @pipeline(Standardizer,
                 ContinuousEncoder,
                 KMeans(k=10),
                 EvoTreeClassifier(nrounds=50))

# (b)

mach = machine(pipe, X, y)
evaluate!(mach, resampling=CV(nfolds=6), measure=cross_entropy)

# (c)

r = range(pipe, :(evo_tree_classifier.max_depth), lower=1, upper=10)

curve = learning_curve(mach,
                       range=r,
                       resampling=CV(nfolds=6),
                       measure=cross_entropy)

plt = plot(curve.parameter_values, curve.measurements)
xlabel!(plt, "max_depth")
ylabel!(plt, "CV estimate of cross entropy")
plt

# Here's a second curve using a different random seed for the booster:

pipe.evo_tree_classifier.seed = 123
curve = learning_curve(mach,
                       range=r,
                       resampling=CV(nfolds=6),
                       measure=cross_entropy)
plot!(curve.parameter_values, curve.measurements)

using Literate #src
Literate.markdown(@__FILE__, @__DIR__, execute=true) #src
Literate.notebook(@__FILE__, @__DIR__, execute=false) #src
