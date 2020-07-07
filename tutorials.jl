# # Machine Learning in Julia, JuliaCon2020

# A workshop introducing the machine learning toolbox
# [MLJ](https://alan-turing-institute.github.io/MLJ.jl/stable/)


# Setup:

using Pkg
DIR = @__DIR__
@assert last(splitpath(DIR)) == "MachineLearningInJulia2020"
Pkg.activate(DIR)
Pkg.instantiate()
using CategoricalArrays #src

# ## Part 1: Data representation

# ### Scientific types

# #### Resources
#
# - From the MLJ manual: [A preview of data type specification in
#   MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/getting_started/#A-preview-of-data-type-specification-in-MLJ-1), [Data containers and scientific types](https://alan-turing-institute.github.io/MLJ.jl/dev/getting_started/#Data-containers-and-scientific-types-1), [Working with Categorical Data](https://alan-turing-institute.github.io/MLJ.jl/dev/working_with_categorical_data/)

# - [MLJScientificTypes.jl](https://alan-turing-institute.github.io/MLJScientificTypes.jl/dev/)

# - From Data Science Tutorials: [Data interpretation: Scientific
#   Types](https://alan-turing-institute.github.io/DataScienceTutorials.jl/data/scitype/),

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

exam_mark = ["rotten", "great", "bla", "great", "bla"]
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

# #### Resources

# - [UCI Horse Colic Data Set](http://archive.ics.uci.edu/ml/datasets/Horse+Colic)

# - From Julia Data Science Tutorials: [Horse colic data](https://alan-turing-institute.github.io/DataScienceTutorials.jl/end-to-end/horse/)


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


# ### Exercises

# #### Ex 1 (scitype of tuples and arrays)

# Evaluate the following cells:

t = (3.141, 42, "how")
scitype(t)

#-

A = rand(2, 3)

using SparseArrays
Asparse = sparse(A)

#-

using CategoricalArrays
Acategorical = categorical(A)

scitype(A)

#-

scitype(Asparse)

#-

scitype(Acategorical)

#-

v = [1, 2, missing, 4]
scitype(v)


# Try to guess the following before evaluation:

scitype(v[1:2])


# Can you guess at the general behaviour of `scitype` with respect to
# tuples, abstract arrays and missing values? The answers are
# [here](https://github.com/alan-turing-institute/ScientificTypes.jl#2-the-scitype-and-scitype-methods)
# (ignore "Property 1").


# #### Ex 2 (fixing scitypes in a table)

# Fix the scitypes for the [House Prices in King County data
# set](https://mlr3gallery.mlr-org.com/posts/2020-01-30-house-prices-in-king-county/):

file = CSV.File(joinpath(DIR, "data", "homes.csv"));
homes = CSV.DataFrame!(file); # convert to data frame without copying columns
first(homes, 4)


# ## Part 2: Selecting, training and evaluating models




using Literate #src
Literate.markdown(@__FILE__, @__DIR__) #src
Literate.notebook(@__FILE__, @__DIR__) #src
