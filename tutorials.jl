# # Machine Learning in Julia, JuliaCon2020

# A workshop introducing the machine learning toolbox
# [MLJ](https://alan-turing-institute.github.io/MLJ.jl/stable/)


# Setup:

using Pkg
DIR = @__DIR__
@assert last(splitpath(DIR)) == "MachineLearningInJulia2020"
Pkg.activate(DIR)
Pkg.instantiate()


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

# To have a `Finite` element scitype, an array must be a
# [`CategoricalArray`](https://juliadata.github.io/CategoricalArrays.jl/stable/). For
# `OrderedFactor`, use `levels!` to put the classes in the right order:

exam_mark = ["rotten", "great", "bla", "great", "bla"]
scitype(exam_mark)

#-

exam_mark = coerce(exam_mark, OrderedFactor)
levels(exam_mark)

#-

levels!(exam_mark, ["rotten", "bla", "great"])
exam_mark[1] < exam_mark[2]

# Subsampling preserves levels:

levels(exam_mark[1:2])


# ### Two-dimensional data

# Whenever it makes sense, MLJ Models generally expect two-dimensional
# data to be *tabular*. All the tabular formats listed
# [here](https://github.com/JuliaData/Tables.jl/blob/master/INTEGRATIONS.md)
# have a scientific type of `Table` and can be used with such
# models.

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




using Literate #src
Literate.markdown(@__FILE__, @__DIR__) #src
