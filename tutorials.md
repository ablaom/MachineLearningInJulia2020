```@meta
EditURL = "<unknown>/tutorials.jl"
```

# Machine Learning in Julia, JuliaCon2020

A workshop introducing the machine learning toolbox
[MLJ](https://alan-turing-institute.github.io/MLJ.jl/stable/)

```@example tutorials
include(joinpath(@__DIR__, "setup.jl"))
```

## Part 1: Data representation

> **Goals:**
> 1. Learn how MLJ specifies it's data requirements using "scientific" types
> 2. Understand the options for representing tabular data
> 3. Learn how to inspect and fix the representation of data to meet MLJ requirements

### Scientific types

To help you focus on the intended *purpose* or *interpretation* of
data, MLJ models specify data requirements using *scientific types*,
instead of machine types. An example of a scientific type is
`OrderedFactor`. The other basic "scalar" scientific types are
illustrated below:

![](assets/scitypes.png)

A scientific type is an ordinary Julia type (so it can be used for
method dispatch, for example) but it usually has no instances. The
`scitype` function is used to articulate MLJ's convention about how
different machine types will be interpreted by MLJ models:

```@example tutorials
using MLJ
scitype(3.141)
```

```@example tutorials
time = [2.3, 4.5, 4.2, 1.8, 7.1]
scitype(time)
```

To fix data which MLJ is interpreting incorrectly, we use the
`coerce` method:

```@example tutorials
height = [185, 153, 163, 114, 180]
scitype(height)
```

```@example tutorials
height = coerce(height, Continuous)
```

Here's an example of data we would want interpreted as
`OrderedFactor` but isn't:

```@example tutorials
exam_mark = ["rotten", "great", "bla",  missing, "great"]
scitype(exam_mark)
```

```@example tutorials
exam_mark = coerce(exam_mark, OrderedFactor)
```

```@example tutorials
levels(exam_mark)
```

Use `levels!` to put the classes in the right order:

```@example tutorials
levels!(exam_mark, ["rotten", "bla", "great"])
exam_mark[1] < exam_mark[2]
```

Subsampling preserves levels:

```@example tutorials
levels(exam_mark[1:2])
```

**Note on binary data.** There is no separate scientific type for binary
data. Binary data is `OrderedFactor{2}` if it has an intrinsic
"true" class (eg, "pass"/"fail") and `Multiclass{2}` otherwise (eg,
"male"/"female").

### Two-dimensional data

Whenever it makes sense, MLJ Models generally expect two-dimensional
data to be *tabular*. All the tabular formats implementing the
[Tables.jl API](https://juliadata.github.io/Tables.jl/stable/) (see
this
[list](https://github.com/JuliaData/Tables.jl/blob/master/INTEGRATIONS.md))
have a scientific type of `Table` and can be used with such models.

The simplest example of a table is a the julia native *column
table*, which is just a named tuple of equal-length vectors:

```@example tutorials
column_table = (h=height, e=exam_mark, t=time)
```

```@example tutorials
scitype(column_table)
```

Notice the `Table{K}` type parameter `K` encodes the scientific
types of the columns. To see the individual types of columns, we use
the `schema` method instead:

```@example tutorials
schema(column_table)
```

Here are four other examples of tables:

```@example tutorials
row_table = [(a=1, b=3.4),
             (a=2, b=4.5),
             (a=3, b=5.6)]
schema(row_table)
```

```@example tutorials
using DataFrames
df = DataFrames.DataFrame(column_table)
```

```@example tutorials
schema(df)
```

```@example tutorials
using CSV
file = CSV.File(joinpath(DIR, "data", "horse.csv"));
schema(file) # (triggers a file read)
```

Most MLJ models do not accept matrix in lieu of a table, but you can
wrap a matrix as a table:

```@example tutorials
matrix_table = MLJ.table(rand(2,3))
schema(matrix_table)
```

Under the hood many algorithms convert tabular data to matrices. If
your table is a wrapped matrix like the above, then the compiler
will generally collapse the conversions to a no-op.

### Fixing scientific types in tabular data

To show how we can correct the scientific types of data in tables,
we introduce a cleaned up version of the UCI Horse Colic Data Set
(the cleaning workflow is described
[here](https://alan-turing-institute.github.io/DataScienceTutorials.jl/end-to-end/horse/#dealing_with_missing_values))

```@example tutorials
using CSV
file = CSV.File(joinpath(DIR, "data", "horse.csv"));
horse = CSV.DataFrame!(file); # convert to data frame without copying columns
first(horse, 4)
```

From [the UCI
docs](http://archive.ics.uci.edu/ml/datasets/Horse+Colic) we can
surmise how each variable ought to be interpreted (a step in our
workflow that cannot reliably be left to the computer):

variable                    | scientific type (interpretation)
----------------------------|-----------------------------------
`:surgery`                  | Multiclass
`:age`                      | Multiclass
`:rectal_temperature`       | Continuous
`:pulse`                    | Continuous
`:respiratory_rate`         | Continuous
`:temperature_extremities`  | OrderedFactor
`:mucous_membranes`         | Multiclass
`:capillary_refill_time`    | Multiclass
`:pain`                     | OrderedFactor
`:peristalsis`              | OrderedFactor
`:abdominal_distension`     | OrderedFactor
`:packed_cell_volume`       | Continuous
`:total_protein`            | Continuous
`:outcome`                  | Multiclass
`:surgical_lesion`          | OrderedFactor
`:cp_data`                  | Multiclass

Let's see how MLJ will actually interpret the data, as it is
currently encoded:

```@example tutorials
schema(horse)
```

As a first correction step, we can get MLJ to "guess" the
appropriate fix, using the `autotype` method:

```@example tutorials
autotype(horse)
```

Okay, this is not perfect, but a step in the right direction, which
we implement like this:

```@example tutorials
coerce!(horse, autotype(horse));
schema(horse)
```

All remaining `Count` data should be `Continuous`:

```@example tutorials
coerce!(horse, Count => Continuous);
schema(horse)
```

We'll correct the remaining truant entries manually:

```@example tutorials
coerce!(horse,
        :surgery               => Multiclass,
        :age                   => Multiclass,
        :mucous_membranes      => Multiclass,
        :capillary_refill_time => Multiclass,
        :outcome               => Multiclass,
        :cp_data               => Multiclass);
schema(horse)
```

### Resources for Part 1

- From the MLJ manual:
   - [A preview of data type specification in
  MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/getting_started/#A-preview-of-data-type-specification-in-MLJ-1)
   - [Data containers and scientific types](https://alan-turing-institute.github.io/MLJ.jl/dev/getting_started/#Data-containers-and-scientific-types-1)
   - [Working with Categorical Data](https://alan-turing-institute.github.io/MLJ.jl/dev/working_with_categorical_data/)
- [Summary](https://alan-turing-institute.github.io/MLJScientificTypes.jl/dev/#Summary-of-the-MLJ-convention-1) of the MLJ convention for representing scientific types
- [MLJScientificTypes.jl](https://alan-turing-institute.github.io/MLJScientificTypes.jl/dev/)
- From Data Science Tutorials:
    - [Data interpretation: Scientific Types](https://alan-turing-institute.github.io/DataScienceTutorials.jl/data/scitype/)
    - [Horse colic data](https://alan-turing-institute.github.io/DataScienceTutorials.jl/end-to-end/horse/)
- [UCI Horse Colic Data Set](http://archive.ics.uci.edu/ml/datasets/Horse+Colic)

### Exercises for Part 1

#### Ex 1

Try to guess how each code snippet below will evaluate:

```@example tutorials
scitype(42)
```

```@example tutorials
questions = ["who", "why", "what", "when"]
scitype(questions)
```

```@example tutorials
elscitype(questions)
```

```@example tutorials
t = (3.141, 42, "how")
scitype(t)
```

```@example tutorials
A = rand(2, 3)
scitype(A)
```

```@example tutorials
elscitype(A)
```

```@example tutorials
using SparseArrays
Asparse = sparse(A)
scitype(Asparse)
```

```@example tutorials
using CategoricalArrays
C1 = categorical(A)
scitype(C1)
```

```@example tutorials
elscitype(C1)
```

```@example tutorials
C2 = categorical(A, ordered=true)
scitype(C2)
```

```@example tutorials
v = [1, 2, missing, 4]
scitype(v)
```

```@example tutorials
elscitype(v)
```

```@example tutorials
scitype(v[1:2])
```

Can you guess at the general behaviour of
`scitype` with respect to tuples, abstract arrays and missing
values? The answers are
[here](https://github.com/alan-turing-institute/ScientificTypes.jl#2-the-scitype-and-scitype-methods)
(ignore "Property 1").

#### Ex 2

Coerce the following vector to make MLJ recognize it as an ordered
factor (with the factors in appropriate order):

```@example tutorials
quality = ["good", "poor", "poor", "excellent", missing, "good", "excellent"]
```

#### Ex 3 (fixing scitypes in a table)

Fix the scitypes for the [House Prices in King County
data](https://mlr3gallery.mlr-org.com/posts/2020-01-30-house-prices-in-king-county/)
dataset:

```@example tutorials
file = CSV.File(joinpath(DIR, "data", "house.csv"));
house = CSV.DataFrame!(file); # convert to data frame without copying columns
first(house, 4)
```

## Part 2: Selecting, training and evaluating models

The "Hello World!" of machine learning is to classify Fisher's
famous iris data set. This time, we'll grab the data from
[OpenML](https://www.openml.org):

```@example tutorials
iris = OpenML.load(61); # a row table
iris = DataFrames.DataFrame(iris);
first(iris, 4)
```

**Goal.** To build and evaluate models for predicting the
`:class` variable, given the four remaining measurement variables.

### Step 1. Inspect and fix scientific types

```@example tutorials
schema(iris)
```

```@example tutorials
coerce!(iris, :class => Multiclass);
schema(iris)
```

### Step 2. Split data into input and target parts

Here's how we split the data into target and input features, which
is needed for MLJ supervised models. We randomize the data at the
same time:

```@example tutorials
y, X = unpack(iris, ==(:class), name->true; rng=123);
scitype(y)
```

Do `?unpack` to learn more:

```@example tutorials
@doc unpack
```

### On searching for a model

Here's how to see *all* models (not immediately useful):

```@example tutorials
kitchen_sink = models()
```

Each entry contains metadata for a model whose defining code is not yet loaded:

```@example tutorials
meta = kitchen_sink[3]
```

```@example tutorials
targetscitype = meta.target_scitype
```

```@example tutorials
scitype(y) <: targetscitype
```

So this model won't do. Let's  find all pure julia classifiers:

```@example tutorials
filt(meta) = AbstractVector{Finite} <: meta.target_scitype &&
        meta.is_pure_julia
models(filt)
```

Find all models with "Classifier" in `name` (or `docstring`):

```@example tutorials
models("Classifier")
```

Find all (supervised) models that match my data!

```@example tutorials
models(matching(X, y))
```

### Step 3. Select and instantiate a model

```@example tutorials
model = @load NeuralNetworkClassifier
```

```@example tutorials
info(model)
```

In MLJ a *model* is just a struct containing hyperparameters, and
that's all. A model does not store *learned* parameters. Models are
mutable:

```@example tutorials
model.epochs = 12
```

And all models have a key-word constructor that works once `@load`
has been performed:

```@example tutorials
NeuralNetworkClassifier(epochs=12) == model
```

### On fitting and predicting

In MLJ a model and training/evaluation data are typically bound
together in a machine:

```@example tutorials
mach = machine(model, X, y)
```

A machine stores *learned* parameters, among other things. We'll
train this machine on 70% of the data and evaluate on a 30% holdout
set. Let's start by dividing all row indices into `train` and `test`
subsets:

```@example tutorials
train, test = partition(eachindex(y), 0.7)
```

```@example tutorials
fit!(mach, rows=train, verbosity=2)
```

Machines remember the last set of hyperparameters used during fit,
which, in the case of iterative models, allows them to restart
computations where they left off, when the iteration parameter is
increased:

```@example tutorials
model.epochs = model.epochs + 4
fit!(mach, rows=train, verbosity=2)
```

By default, we can also increase `:learning_rate` without a cold restart:

```@example tutorials
model.epochs = model.epochs + 4
model.optimiser.eta = 10*model.optimiser.eta
fit!(mach, rows=train, verbosity=2)
```

However, change the regularization parameter and training will
restart from scratch:

```@example tutorials
model.lambda = 0.001
fit!(mach, rows=train, verbosity=2)
```

Let's train silently for a total of 50 epochs, and look at a prediction:

```@example tutorials
model.epochs = 50
fit!(mach, rows=train)
yhat = predict(mach, X[test,:]); # or predict(mach, rows=test)
yhat[1]
```

What's going on here?

```@example tutorials
info(model).prediction_type
```

**Important**:
- In MLJ, a model that can predict probabilities (and not just point values) will do so by default. (These models have supertype `Proababilistic`, while point-estimate predictors have supertype `Deterministic`.)
- For most probabilistic predictors, the predicted object is a `Distributions.Distribution` object, supporting the `Distributions.jl` [API](https://juliastats.org/Distributions.jl/latest/extends/#Create-a-Distribution-1) for such objects. In particular, the methods `rand`,  `pdf`, `mode`, `median` and `mean` will apply, where appropriate.

So, to obtain the probability of "Iris-virginica" in the first test
prediction, we do

```@example tutorials
pdf(yhat[1], "Iris-virginica")
```

To get the most likely observation, we do

```@example tutorials
mode(yhat[1])
```

These can be broadcast over multiple predictions in the usual way:

```@example tutorials
broadcast(pdf, yhat[1:4], "Iris-versicolor")
```

```@example tutorials
mode.(yhat[1:4])
```

Or, alternatively, you can use the `predict_mode` operation instead
of `predict`:

```@example tutorials
predict_mode(mach, X[test,:])[1:4] # or predict_mode(mach, rows=test)[1:4]
```

For a more conventional matrix of probabilities you can do this:

```@example tutorials
L = levels(y)
pdf(yhat, L)[1:4, :]
```

However, in a typical MLJ workflow, this is not as useful as you
might imagine. In particular, all probablistic performance measures
in MLJ expect distribution objects in their first slot:

```@example tutorials
cross_entropy(yhat, y[test]) |> mean
```

To apply a deterministic measure, we first need to obtain point-estimates:

```@example tutorials
misclassification_rate(mode.(yhat), y[test])
```

### Step 4. Evaluate the model performance

Naturally, MLJ provides boilerplate code for carrying out a model
evaluation with a lot less fuss. Let's repeat the performance
evaluation above and add an extra measure, `brier_score`:

```@example tutorials
evaluate!(mach, resampling=Holdout(fraction_train=0.8),
          measures=[cross_entropy, brier_score])
```

Or applying cross-validation instead:

```@example tutorials
evaluate!(mach, resampling=CV(nfolds=6),
          measures=[cross_entropy, brier_score])
```

Or, Monte-Carlo cross-validation (cross-validation repeated
randomizied folds)

```@example tutorials
e = evaluate!(mach, resampling=CV(nfolds=6, rng=123),
                repeats=3,
              measures=[cross_entropy, brier_score])
```

One can access the following properties of the output `e` of an
evaluation: `measure`, `measurement`, `per_fold` (measurement for
each fold) and `per_observation` (measurement per observation, if
reported).

We finally note that you can restrict the rows of observations from
which train and test folds are drawn, by specifying `rows=...`. For
example, imagining the last 30% of target observations are `missing`
you might have a workflow like this:

```@example tutorials
train, test = partition(eachindex(y), 0.7)
mach = machine(model, X, y)
evaluate!(mach, resampling=CV(nfolds=6),
          measures=[cross_entropy, brier_score],
          rows=train)     # cv estimate, resampling from `train`
fit!(mach, rows=train)    # re-train using all of `train` observations
predict(mach, rows=test); # and predict missing targets
nothing #hide
```

### On learning curves

Since our model is an iterative one, we might want to inspect the
out-of-sample performance as a function of the iteration
parameter. For this we can use the `learning_curve` function (which,
incidentally can be applied to any model hyper-parameter). This
starts by defining a one-dimensional range object for the parameter
(more on this when we discuss tuning in Part 4):

```@example tutorials
r = range(model, :epochs, lower=1, upper=60, scale=:log)
curve = learning_curve(mach,
                       range=r,
                       resampling=Holdout(fraction_train=0.7), # (default)
                       measure=cross_entropy)

using Plots
pyplot()
plt=plot(curve.parameter_values, curve.measurements)
xlabel!(plt, "epochs")
ylabel!(plt, "cross entropy on holdout set")
savefig("iris_learning_curve.png")
```

### Exercises for Part 2

#### Ex 4

(a) Identify all supervised MLJ models that can be applied (without
type coercion or one-hot encoding) to a supervised learning problem
with input features `X4` and target `y4` defined below:

```@example tutorials
import Distributions
poisson = Distributions.Poisson

age = 18 .+ 60*rand(10);
salary = coerce(rand([:small, :big, :huge], 10), OrderedFactor);
levels!(salary, [:small, :big, :huge]);
X4 = DataFrames.DataFrame(age=age, salary=salary)

n_devices(salary) = salary > :small ? rand(poisson(1.3)) : rand(poisson(2.9))
y4 = [n_devices(row.salary) for row in eachrow(X4)]
```

(b) What models can be applied if you coerce the salary to a
`Continuous` scitype?

#### Ex 5 (unpack)

After evaluating the following ...

```@example tutorials
data = (a = [1, 2, 3, 4],
     b = rand(4),
     c = rand(4),
     d = coerce(["male", "female", "female", "male"], OrderedFactor));
pretty(data)

using Tables
y, X, w = unpack(data, ==(:a),
                 name -> elscitype(Tables.getcolumn(data, name)) == Continuous,
                 name -> true)
```

...attempt to guess the evaluations of the following:

```@example tutorials
y
```

```@example tutorials
pretty(X)
```

```@example tutorials
w
```

#### Ex 6 (horse data)

(a) Suppose we want to use predict the `:outcome` variable in the
Horse Colic study introduced in Part 1, based on the remaining
variables that are `Continuous` (one-hot encoding categorical
variables is discussed later in Part 3) *while ignoring the others*.
Extract from the `horse` data set (defined in Part 1) appropriate
input features `X` and target variable `y`? (Do not, however,
randomize the obserations.)

(b) Create a 70:30 `train`/`test` split of the data and train a
`KNNClassifier` model on the `train` data, using `K = 20` and
default values for the other hyper-parameters. (Although one would
normally standardize (whiten) the continuous features for this
model, do not do so here.) After training:

- (i) Evaluate the `cross_entropy` performance on the `test`
  observations.

- &star;(ii) In how many `test` observations does the predicted
  probablility of the observed class exceed 50%?

- &star;(iii) Find the `misclassification_rate` in the `test`
  set. (*Hint.* As this measure is deterministic, you will either
  need to broadcast `mode` or use `predict_mode` instead of
  `predict`.)

(c) Instead use a `RandomForestClassifier` model from the
    `DecisionTree` package and:

- (i) Generate an appropriate learning curve to
  convince yourself that out-of-sample estimates of the
  `cross_entropy` loss do not substatially improve for `n_trees >
  50`. Use default values for all other hyper-parameters, and feel
  free to use all available data to generate the curve.

- (ii) Fix `n_trees=90` and use `evaluate!` to obtain a 9-fold
  cross-validation estimate of the `cross_entropy`, restricting
  sub-sampling to the `train` observations.

- (iii) Now use *all* available data but set
  `resampling=Holdout(fraction_train=0.7)` to obtain a score you can
  compare with the `KNNClassifier` in part (b)(iii). Which model is
  better?

## Solutions to exercises

#### Ex 2 solution

```@example tutorials
quality = coerce(quality, OrderedFactor);
levels!(quality, ["poor", "good", "excellent"]);
elscitype(quality)
```

#### Ex 3 solution

TODO

#### Ex 4 solution

4(a)

There are *no* models that apply immediately:

```@example tutorials
models(matching(X4, y4))
```

4(b)

```@example tutorials
y4 = coerce(y4, Continuous);
models(matching(X4, y4))
```

#### Ex 6 solution

6(a)

```@example tutorials
y, X = unpack(horse,
              ==(:outcome),
              name -> elscitype(Tables.getcolumn(horse, name)) == Continuous);
nothing #hide
```

6(b)(i)

```@example tutorials
model = @load KNNClassifier
model.K = 20
mach = machine(model, X, y)
fit!(mach, rows=train)
yhat = predict(mach, rows=test) # or predict(mach, X[test,:]);
err = cross_entropy(yhat, y[test]) |> mean
```

6(b)(ii)

The predicted probabilities of the actual observations in the test
are given by

```@example tutorials
p = broadcast(pdf, yhat, y[test]);
nothing #hide
```

The number of times this probability exceeds 50% is:

```@example tutorials
n50 = filter(x -> x > 0.5, p) |> length
```

Or, as a proportion:

```@example tutorials
n50/length(test)
```

6(c)(iii)

```@example tutorials
misclassification_rate(mode.(yhat), y[test])
```

6(c)(i)

```@example tutorials
model = @load RandomForestClassifier pkg=DecisionTree
mach = machine(model, X, y)
evaluate!(mach, resampling=CV(nfolds=6), measure=cross_entropy)

r = range(model, :n_trees, lower=10, upper=70, scale=:log)
```

Since random forests are inherently randomized, we generate multiple
curves:

```@example tutorials
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
```

6(c)(ii)

```@example tutorials
evaluate!(mach, resampling=CV(nfolds=9),
                measure=cross_entropy,
                rows=train).measurement[1]

model.n_trees = 90
```

6(c)(iii)

```@example tutorials
err_forest = evaluate!(mach, resampling=Holdout(),
                       measure=cross_entropy).measurement[1]
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

