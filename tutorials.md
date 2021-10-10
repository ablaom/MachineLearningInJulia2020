```@meta
EditURL = "<unknown>/tutorials.jl"
```

# Machine Learning in Julia, JuliaCon2020

A workshop introducing the machine learning toolbox
[MLJ](https://alan-turing-institute.github.io/MLJ.jl/stable/)

### Set-up

The following instantiates a package environment and pre-loads some
packages, to avoid delays later on.

The package environment has been created using **Julia 1.6** and may not
instantiate properly for other Julia versions.

````julia
VERSION
````

````
v"1.6.3"
````

````julia
DIR = @__DIR__
include(joinpath(DIR, "setup.jl"))
````

````
  Activating environment at `~/Google Drive/Julia/MLJ/MachineLearningInJulia2020/Project.toml`
[ Info: Done loading

````

## General resources

- [List of methods introduced in this tutorial](methods.md)
- [MLJ Cheatsheet](https://alan-turing-institute.github.io/MLJ.jl/dev/mlj_cheatsheet/)
- [Common MLJ Workflows](https://alan-turing-institute.github.io/MLJ.jl/dev/common_mlj_workflows/)
- [MLJ manual](https://alan-turing-institute.github.io/MLJ.jl/dev/)
- [Data Science Tutorials in Julia](https://juliaai.github.io/DataScienceTutorials.jl/)

## Contents

### Basic

- [Part 1 - Data Representation](#part-1-data-representation)
- [Part 2 - Selecting, Training and Evaluating Models](#part-2-selecting-training-and-evaluating-models)
- [Part 3 - Transformers and Pipelines](#part-3-transformers-and-pipelines)
- [Part 4 - Tuning Hyper-parameters](#part-4-tuning-hyper-parameters)
- [Part 5 - Advanced model composition](#part-5-advanced-model-composition)
- [Solutions to Exercises](#solutions-to-exercises)

<a id='part-1-data-representation'></a>

## Part 1 - Data Representation

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

````julia
using MLJ
scitype(3.141)
````

````
Continuous
````

````julia
time = [2.3, 4.5, 4.2, 1.8, 7.1]
scitype(time)
````

````
AbstractVector{Continuous} (alias for AbstractArray{Continuous, 1})
````

To fix data which MLJ is interpreting incorrectly, we use the
`coerce` method:

````julia
height = [185, 153, 163, 114, 180]
scitype(height)
````

````
AbstractVector{Count} (alias for AbstractArray{Count, 1})
````

````julia
height = coerce(height, Continuous)
````

````
5-element Vector{Float64}:
 185.0
 153.0
 163.0
 114.0
 180.0
````

Here's an example of data we would want interpreted as
`OrderedFactor` but isn't:

````julia
exam_mark = ["rotten", "great", "bla",  missing, "great"]
scitype(exam_mark)
````

````
AbstractVector{Union{Missing, Textual}} (alias for AbstractArray{Union{Missing, Textual}, 1})
````

````julia
exam_mark = coerce(exam_mark, OrderedFactor)
````

````
5-element CategoricalArray{Union{Missing, String},1,UInt32}:
 "rotten"
 "great"
 "bla"
 missing
 "great"
````

````julia
levels(exam_mark)
````

````
3-element Vector{String}:
 "bla"
 "great"
 "rotten"
````

Use `levels!` to put the classes in the right order:

````julia
levels!(exam_mark, ["rotten", "bla", "great"])
exam_mark[1] < exam_mark[2]
````

````
true
````

When sub-sampling, no levels are lost:

````julia
levels(exam_mark[1:2])
````

````
3-element Vector{String}:
 "rotten"
 "bla"
 "great"
````

**Note on binary data.** There is no separate scientific type for
binary data. Binary data is `OrderedFactor{2}` or
`Multiclass{2}`. If a binary measure like `truepositive` is a
applied to `OrderedFactor{2}` then the "positive" class is assumed
to appear *second* in the ordering. If such a measure is applied to
`Multiclass{2}` data, a warning is issued. A single `OrderedFactor`
can be coerced to a single `Continuous` variable, for models that
require this, while a `Multiclass` variable can only be one-hot
encoded.

### Two-dimensional data

Whenever it makes sense, MLJ Models generally expect two-dimensional
data to be *tabular*. All the tabular formats implementing the
[Tables.jl API](https://juliadata.github.io/Tables.jl/stable/) (see
this
[list](https://github.com/JuliaData/Tables.jl/blob/master/INTEGRATIONS.md))
have a scientific type of `Table` and can be used with such models.

Probably the simplest example of a table is the julia native *column
table*, which is just a named tuple of equal-length vectors:

````julia
column_table = (h=height, e=exam_mark, t=time)
````

````
(h = [185.0, 153.0, 163.0, 114.0, 180.0],
 e = Union{Missing, CategoricalValue{String, UInt32}}["rotten", "great", "bla", missing, "great"],
 t = [2.3, 4.5, 4.2, 1.8, 7.1],)
````

````julia
scitype(column_table)
````

````
Table{Union{AbstractVector{Union{Missing, OrderedFactor{3}}}, AbstractVector{Continuous}}}
````

Notice the `Table{K}` type parameter `K` encodes the scientific
types of the columns. (This is useful when comparing table scitypes
with `<:`). To inspect the individual column scitypes, we use the
`schema` method instead:

````julia
schema(column_table)
````

````
┌─────────┬──────────────────────────────────────────────────┬──────────────────────────────────┐
│ _.names │ _.types                                          │ _.scitypes                       │
├─────────┼──────────────────────────────────────────────────┼──────────────────────────────────┤
│ h       │ Float64                                          │ Continuous                       │
│ e       │ Union{Missing, CategoricalValue{String, UInt32}} │ Union{Missing, OrderedFactor{3}} │
│ t       │ Float64                                          │ Continuous                       │
└─────────┴──────────────────────────────────────────────────┴──────────────────────────────────┘
_.nrows = 5

````

Here are five other examples of tables:

````julia
dict_table = Dict(:h => height, :e => exam_mark, :t => time)
schema(dict_table)
````

````
┌─────────┬──────────────────────────────────────────────────┬──────────────────────────────────┐
│ _.names │ _.types                                          │ _.scitypes                       │
├─────────┼──────────────────────────────────────────────────┼──────────────────────────────────┤
│ e       │ Union{Missing, CategoricalValue{String, UInt32}} │ Union{Missing, OrderedFactor{3}} │
│ h       │ Float64                                          │ Continuous                       │
│ t       │ Float64                                          │ Continuous                       │
└─────────┴──────────────────────────────────────────────────┴──────────────────────────────────┘
_.nrows = 5

````

(To control column order here, instead use `LittleDict` from
OrderedCollections.jl.)

````julia
row_table = [(a=1, b=3.4),
             (a=2, b=4.5),
             (a=3, b=5.6)]
schema(row_table)
````

````
┌─────────┬─────────┬────────────┐
│ _.names │ _.types │ _.scitypes │
├─────────┼─────────┼────────────┤
│ a       │ Int64   │ Count      │
│ b       │ Float64 │ Continuous │
└─────────┴─────────┴────────────┘
_.nrows = 3

````

````julia
import DataFrames
df = DataFrames.DataFrame(column_table)
````

```@raw html
<div class="data-frame"><p>5 rows × 3 columns</p><table class="data-frame"><thead><tr><th></th><th>h</th><th>e</th><th>t</th></tr><tr><th></th><th title="Float64">Float64</th><th title="Union{Missing, CategoricalValue{String, UInt32}}">Cat…?</th><th title="Float64">Float64</th></tr></thead><tbody><tr><th>1</th><td>185.0</td><td>rotten</td><td>2.3</td></tr><tr><th>2</th><td>153.0</td><td>great</td><td>4.5</td></tr><tr><th>3</th><td>163.0</td><td>bla</td><td>4.2</td></tr><tr><th>4</th><td>114.0</td><td><em>missing</em></td><td>1.8</td></tr><tr><th>5</th><td>180.0</td><td>great</td><td>7.1</td></tr></tbody></table></div>
```

````julia
schema(df) == schema(column_table)
````

````
true
````

````julia
using CSV
file = CSV.File(joinpath(DIR, "data", "horse.csv"));
schema(file) # (triggers a file read)
````

````
┌─────────────────────────┬─────────┬────────────┐
│ _.names                 │ _.types │ _.scitypes │
├─────────────────────────┼─────────┼────────────┤
│ surgery                 │ Int64   │ Count      │
│ age                     │ Int64   │ Count      │
│ rectal_temperature      │ Float64 │ Continuous │
│ pulse                   │ Int64   │ Count      │
│ respiratory_rate        │ Int64   │ Count      │
│ temperature_extremities │ Int64   │ Count      │
│ mucous_membranes        │ Int64   │ Count      │
│ capillary_refill_time   │ Int64   │ Count      │
│ pain                    │ Int64   │ Count      │
│ peristalsis             │ Int64   │ Count      │
│ abdominal_distension    │ Int64   │ Count      │
│ packed_cell_volume      │ Float64 │ Continuous │
│ total_protein           │ Float64 │ Continuous │
│ outcome                 │ Int64   │ Count      │
│ surgical_lesion         │ Int64   │ Count      │
│ cp_data                 │ Int64   │ Count      │
└─────────────────────────┴─────────┴────────────┘
_.nrows = 366

````

Most MLJ models do not accept matrix in lieu of a table, but you can
wrap a matrix as a table:

````julia
matrix_table = MLJ.table(rand(2,3))
schema(matrix_table)
````

````
┌─────────┬─────────┬────────────┐
│ _.names │ _.types │ _.scitypes │
├─────────┼─────────┼────────────┤
│ x1      │ Float64 │ Continuous │
│ x2      │ Float64 │ Continuous │
│ x3      │ Float64 │ Continuous │
└─────────┴─────────┴────────────┘
_.nrows = 2

````

The matrix is *not* copied, only wrapped. Some models may perform
better if one wraps the adjoint of the transpose - see
[here](https://alan-turing-institute.github.io/MLJ.jl/dev/getting_started/#Observations-correspond-to-rows,-not-columns).

**Manipulating tabular data.** In this workshop we assume
familiarity with some kind of tabular data container (although it is
possible, in principle, to carry out the exercises without this.)
For a quick start introduction to `DataFrames`, see [this
tutorial](https://juliaai.github.io/DataScienceTutorials.jl/data/dataframe/)

### Fixing scientific types in tabular data

To show how we can correct the scientific types of data in tables,
we introduce a cleaned up version of the UCI Horse Colic Data Set
(the cleaning work-flow is described
[here](https://juliaai.github.io/DataScienceTutorials.jl/end-to-end/horse/#dealing_with_missing_values))

````julia
using CSV
file = CSV.File(joinpath(DIR, "data", "horse.csv"));
horse = DataFrames.DataFrame(file); # convert to data frame without copying columns
first(horse, 4)
````

```@raw html
<div class="data-frame"><p>4 rows × 16 columns</p><table class="data-frame"><thead><tr><th></th><th>surgery</th><th>age</th><th>rectal_temperature</th><th>pulse</th><th>respiratory_rate</th><th>temperature_extremities</th><th>mucous_membranes</th><th>capillary_refill_time</th><th>pain</th><th>peristalsis</th><th>abdominal_distension</th><th>packed_cell_volume</th><th>total_protein</th><th>outcome</th><th>surgical_lesion</th><th>cp_data</th></tr><tr><th></th><th title="Int64">Int64</th><th title="Int64">Int64</th><th title="Float64">Float64</th><th title="Int64">Int64</th><th title="Int64">Int64</th><th title="Int64">Int64</th><th title="Int64">Int64</th><th title="Int64">Int64</th><th title="Int64">Int64</th><th title="Int64">Int64</th><th title="Int64">Int64</th><th title="Float64">Float64</th><th title="Float64">Float64</th><th title="Int64">Int64</th><th title="Int64">Int64</th><th title="Int64">Int64</th></tr></thead><tbody><tr><th>1</th><td>2</td><td>1</td><td>38.5</td><td>66</td><td>66</td><td>3</td><td>1</td><td>2</td><td>5</td><td>4</td><td>4</td><td>45.0</td><td>8.4</td><td>2</td><td>2</td><td>2</td></tr><tr><th>2</th><td>1</td><td>1</td><td>39.2</td><td>88</td><td>88</td><td>3</td><td>4</td><td>1</td><td>3</td><td>4</td><td>2</td><td>50.0</td><td>85.0</td><td>3</td><td>2</td><td>2</td></tr><tr><th>3</th><td>2</td><td>1</td><td>38.3</td><td>40</td><td>40</td><td>1</td><td>3</td><td>1</td><td>3</td><td>3</td><td>1</td><td>33.0</td><td>6.7</td><td>1</td><td>2</td><td>1</td></tr><tr><th>4</th><td>1</td><td>9</td><td>39.1</td><td>164</td><td>164</td><td>4</td><td>6</td><td>2</td><td>2</td><td>4</td><td>4</td><td>48.0</td><td>7.2</td><td>2</td><td>1</td><td>1</td></tr></tbody></table></div>
```

From [the UCI
docs](http://archive.ics.uci.edu/ml/datasets/Horse+Colic) we can
surmise how each variable ought to be interpreted (a step in our
work-flow that cannot reliably be left to the computer):

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

````julia
schema(horse)
````

````
┌─────────────────────────┬─────────┬────────────┐
│ _.names                 │ _.types │ _.scitypes │
├─────────────────────────┼─────────┼────────────┤
│ surgery                 │ Int64   │ Count      │
│ age                     │ Int64   │ Count      │
│ rectal_temperature      │ Float64 │ Continuous │
│ pulse                   │ Int64   │ Count      │
│ respiratory_rate        │ Int64   │ Count      │
│ temperature_extremities │ Int64   │ Count      │
│ mucous_membranes        │ Int64   │ Count      │
│ capillary_refill_time   │ Int64   │ Count      │
│ pain                    │ Int64   │ Count      │
│ peristalsis             │ Int64   │ Count      │
│ abdominal_distension    │ Int64   │ Count      │
│ packed_cell_volume      │ Float64 │ Continuous │
│ total_protein           │ Float64 │ Continuous │
│ outcome                 │ Int64   │ Count      │
│ surgical_lesion         │ Int64   │ Count      │
│ cp_data                 │ Int64   │ Count      │
└─────────────────────────┴─────────┴────────────┘
_.nrows = 366

````

As a first correction step, we can get MLJ to "guess" the
appropriate fix, using the `autotype` method:

````julia
autotype(horse)
````

````
Dict{Symbol, Type} with 11 entries:
  :abdominal_distension => OrderedFactor
  :pain => OrderedFactor
  :surgery => OrderedFactor
  :mucous_membranes => OrderedFactor
  :surgical_lesion => OrderedFactor
  :outcome => OrderedFactor
  :capillary_refill_time => OrderedFactor
  :age => OrderedFactor
  :temperature_extremities => OrderedFactor
  :peristalsis => OrderedFactor
  :cp_data => OrderedFactor
````

Okay, this is not perfect, but a step in the right direction, which
we implement like this:

````julia
coerce!(horse, autotype(horse));
schema(horse)
````

````
┌─────────────────────────┬─────────────────────────────────┬──────────────────┐
│ _.names                 │ _.types                         │ _.scitypes       │
├─────────────────────────┼─────────────────────────────────┼──────────────────┤
│ surgery                 │ CategoricalValue{Int64, UInt32} │ OrderedFactor{2} │
│ age                     │ CategoricalValue{Int64, UInt32} │ OrderedFactor{2} │
│ rectal_temperature      │ Float64                         │ Continuous       │
│ pulse                   │ Int64                           │ Count            │
│ respiratory_rate        │ Int64                           │ Count            │
│ temperature_extremities │ CategoricalValue{Int64, UInt32} │ OrderedFactor{4} │
│ mucous_membranes        │ CategoricalValue{Int64, UInt32} │ OrderedFactor{6} │
│ capillary_refill_time   │ CategoricalValue{Int64, UInt32} │ OrderedFactor{3} │
│ pain                    │ CategoricalValue{Int64, UInt32} │ OrderedFactor{5} │
│ peristalsis             │ CategoricalValue{Int64, UInt32} │ OrderedFactor{4} │
│ abdominal_distension    │ CategoricalValue{Int64, UInt32} │ OrderedFactor{4} │
│ packed_cell_volume      │ Float64                         │ Continuous       │
│ total_protein           │ Float64                         │ Continuous       │
│ outcome                 │ CategoricalValue{Int64, UInt32} │ OrderedFactor{3} │
│ surgical_lesion         │ CategoricalValue{Int64, UInt32} │ OrderedFactor{2} │
│ cp_data                 │ CategoricalValue{Int64, UInt32} │ OrderedFactor{2} │
└─────────────────────────┴─────────────────────────────────┴──────────────────┘
_.nrows = 366

````

All remaining `Count` data should be `Continuous`:

````julia
coerce!(horse, Count => Continuous);
schema(horse)
````

````
┌─────────────────────────┬─────────────────────────────────┬──────────────────┐
│ _.names                 │ _.types                         │ _.scitypes       │
├─────────────────────────┼─────────────────────────────────┼──────────────────┤
│ surgery                 │ CategoricalValue{Int64, UInt32} │ OrderedFactor{2} │
│ age                     │ CategoricalValue{Int64, UInt32} │ OrderedFactor{2} │
│ rectal_temperature      │ Float64                         │ Continuous       │
│ pulse                   │ Float64                         │ Continuous       │
│ respiratory_rate        │ Float64                         │ Continuous       │
│ temperature_extremities │ CategoricalValue{Int64, UInt32} │ OrderedFactor{4} │
│ mucous_membranes        │ CategoricalValue{Int64, UInt32} │ OrderedFactor{6} │
│ capillary_refill_time   │ CategoricalValue{Int64, UInt32} │ OrderedFactor{3} │
│ pain                    │ CategoricalValue{Int64, UInt32} │ OrderedFactor{5} │
│ peristalsis             │ CategoricalValue{Int64, UInt32} │ OrderedFactor{4} │
│ abdominal_distension    │ CategoricalValue{Int64, UInt32} │ OrderedFactor{4} │
│ packed_cell_volume      │ Float64                         │ Continuous       │
│ total_protein           │ Float64                         │ Continuous       │
│ outcome                 │ CategoricalValue{Int64, UInt32} │ OrderedFactor{3} │
│ surgical_lesion         │ CategoricalValue{Int64, UInt32} │ OrderedFactor{2} │
│ cp_data                 │ CategoricalValue{Int64, UInt32} │ OrderedFactor{2} │
└─────────────────────────┴─────────────────────────────────┴──────────────────┘
_.nrows = 366

````

We'll correct the remaining truant entries manually:

````julia
coerce!(horse,
        :surgery               => Multiclass,
        :age                   => Multiclass,
        :mucous_membranes      => Multiclass,
        :capillary_refill_time => Multiclass,
        :outcome               => Multiclass,
        :cp_data               => Multiclass);
schema(horse)
````

````
┌─────────────────────────┬─────────────────────────────────┬──────────────────┐
│ _.names                 │ _.types                         │ _.scitypes       │
├─────────────────────────┼─────────────────────────────────┼──────────────────┤
│ surgery                 │ CategoricalValue{Int64, UInt32} │ Multiclass{2}    │
│ age                     │ CategoricalValue{Int64, UInt32} │ Multiclass{2}    │
│ rectal_temperature      │ Float64                         │ Continuous       │
│ pulse                   │ Float64                         │ Continuous       │
│ respiratory_rate        │ Float64                         │ Continuous       │
│ temperature_extremities │ CategoricalValue{Int64, UInt32} │ OrderedFactor{4} │
│ mucous_membranes        │ CategoricalValue{Int64, UInt32} │ Multiclass{6}    │
│ capillary_refill_time   │ CategoricalValue{Int64, UInt32} │ Multiclass{3}    │
│ pain                    │ CategoricalValue{Int64, UInt32} │ OrderedFactor{5} │
│ peristalsis             │ CategoricalValue{Int64, UInt32} │ OrderedFactor{4} │
│ abdominal_distension    │ CategoricalValue{Int64, UInt32} │ OrderedFactor{4} │
│ packed_cell_volume      │ Float64                         │ Continuous       │
│ total_protein           │ Float64                         │ Continuous       │
│ outcome                 │ CategoricalValue{Int64, UInt32} │ Multiclass{3}    │
│ surgical_lesion         │ CategoricalValue{Int64, UInt32} │ OrderedFactor{2} │
│ cp_data                 │ CategoricalValue{Int64, UInt32} │ Multiclass{2}    │
└─────────────────────────┴─────────────────────────────────┴──────────────────┘
_.nrows = 366

````

### Resources for Part 1

- From the MLJ manual:
   - [A preview of data type specification in
  MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/getting_started/#A-preview-of-data-type-specification-in-MLJ-1)
   - [Data containers and scientific types](https://alan-turing-institute.github.io/MLJ.jl/dev/getting_started/#Data-containers-and-scientific-types-1)
   - [Working with Categorical Data](https://alan-turing-institute.github.io/MLJ.jl/dev/working_with_categorical_data/)
- [Summary](https://juliaai.github.io/ScientificTypes.jl/dev/#Summary-of-the-MLJ-convention-1) of the MLJ convention for representing scientific types
- [ScientificTypes.jl](https://juliaai.github.io/ScientificTypes.jl/dev/)
- From Data Science Tutorials:
    - [Data interpretation: Scientific Types](https://juliaai.github.io/DataScienceTutorials.jl/data/scitype/)
    - [Horse colic data](https://juliaai.github.io/DataScienceTutorials.jl/end-to-end/horse/)
- [UCI Horse Colic Data Set](http://archive.ics.uci.edu/ml/datasets/Horse+Colic)

### Exercises for Part 1

#### Exercise 1

Try to guess how each code snippet below will evaluate:

````julia
scitype(42)
````

````
Count
````

````julia
questions = ["who", "why", "what", "when"]
scitype(questions)
````

````
AbstractVector{Textual} (alias for AbstractArray{Textual, 1})
````

````julia
elscitype(questions)
````

````
Textual
````

````julia
t = (3.141, 42, "how")
scitype(t)
````

````
Tuple{Continuous, Count, Textual}
````

````julia
A = rand(2, 3)
````

````
2×3 Matrix{Float64}:
 0.307453  0.598374  0.301268
 0.749687  0.288024  0.000220251
````

-

````julia
scitype(A)
````

````
AbstractMatrix{Continuous} (alias for AbstractArray{Continuous, 2})
````

````julia
elscitype(A)
````

````
Continuous
````

````julia
using SparseArrays
Asparse = sparse(A)
````

````
2×3 SparseMatrixCSC{Float64, Int64} with 6 stored entries:
 0.307453  0.598374  0.301268
 0.749687  0.288024  0.000220251
````

````julia
scitype(Asparse)
````

````
AbstractMatrix{Continuous} (alias for AbstractArray{Continuous, 2})
````

````julia
using CategoricalArrays
C1 = categorical(A)
````

````
2×3 CategoricalArray{Float64,2,UInt32}:
 0.307453  0.598374  0.301268
 0.749687  0.288024  0.000220251
````

````julia
scitype(C1)
````

````
AbstractMatrix{Multiclass{6}} (alias for AbstractArray{Multiclass{6}, 2})
````

````julia
elscitype(C1)
````

````
Multiclass{6}
````

````julia
C2 = categorical(A, ordered=true)
scitype(C2)
````

````
AbstractMatrix{OrderedFactor{6}} (alias for AbstractArray{OrderedFactor{6}, 2})
````

````julia
v = [1, 2, missing, 4]
scitype(v)
````

````
AbstractVector{Union{Missing, Count}} (alias for AbstractArray{Union{Missing, Count}, 1})
````

````julia
elscitype(v)
````

````
Union{Missing, Count}
````

````julia
scitype(v[1:2])
````

````
AbstractVector{Union{Missing, Count}} (alias for AbstractArray{Union{Missing, Count}, 1})
````

Can you guess at the general behavior of
`scitype` with respect to tuples, abstract arrays and missing
values? The answers are
[here](https://github.com/juliaai/ScientificTypesBase.jl#2-the-scitype-and-scitype-methods)
(ignore "Property 1").

#### Exercise 2

Coerce the following vector to make MLJ recognize it as a vector of
ordered factors (with an appropriate ordering):

````julia
quality = ["good", "poor", "poor", "excellent", missing, "good", "excellent"]
````

````
7-element Vector{Union{Missing, String}}:
 "good"
 "poor"
 "poor"
 "excellent"
 missing
 "good"
 "excellent"
````

#### Exercise 3 (fixing scitypes in a table)

Fix the scitypes for the [House Prices in King
County](https://mlr3gallery.mlr-org.com/posts/2020-01-30-house-prices-in-king-county/)
dataset:

````julia
file = CSV.File(joinpath(DIR, "data", "house.csv"));
house = DataFrames.DataFrame(file); # convert to data frame without copying columns
first(house, 4)
````

```@raw html
<div class="data-frame"><p>4 rows × 19 columns</p><table class="data-frame"><thead><tr><th></th><th>price</th><th>bedrooms</th><th>bathrooms</th><th>sqft_living</th><th>sqft_lot</th><th>floors</th><th>waterfront</th><th>view</th><th>condition</th><th>grade</th><th>sqft_above</th><th>sqft_basement</th><th>yr_built</th><th>zipcode</th><th>lat</th><th>long</th><th>sqft_living15</th><th>sqft_lot15</th><th>is_renovated</th></tr><tr><th></th><th title="Float64">Float64</th><th title="Int64">Int64</th><th title="Float64">Float64</th><th title="Int64">Int64</th><th title="Int64">Int64</th><th title="Float64">Float64</th><th title="Int64">Int64</th><th title="Int64">Int64</th><th title="Int64">Int64</th><th title="Int64">Int64</th><th title="Int64">Int64</th><th title="Int64">Int64</th><th title="Int64">Int64</th><th title="Int64">Int64</th><th title="Float64">Float64</th><th title="Float64">Float64</th><th title="Int64">Int64</th><th title="Int64">Int64</th><th title="Bool">Bool</th></tr></thead><tbody><tr><th>1</th><td>221900.0</td><td>3</td><td>1.0</td><td>1180</td><td>5650</td><td>1.0</td><td>0</td><td>0</td><td>3</td><td>7</td><td>1180</td><td>0</td><td>1955</td><td>98178</td><td>47.5112</td><td>-122.257</td><td>1340</td><td>5650</td><td>1</td></tr><tr><th>2</th><td>538000.0</td><td>3</td><td>2.25</td><td>2570</td><td>7242</td><td>2.0</td><td>0</td><td>0</td><td>3</td><td>7</td><td>2170</td><td>400</td><td>1951</td><td>98125</td><td>47.721</td><td>-122.319</td><td>1690</td><td>7639</td><td>0</td></tr><tr><th>3</th><td>180000.0</td><td>2</td><td>1.0</td><td>770</td><td>10000</td><td>1.0</td><td>0</td><td>0</td><td>3</td><td>6</td><td>770</td><td>0</td><td>1933</td><td>98028</td><td>47.7379</td><td>-122.233</td><td>2720</td><td>8062</td><td>1</td></tr><tr><th>4</th><td>604000.0</td><td>4</td><td>3.0</td><td>1960</td><td>5000</td><td>1.0</td><td>0</td><td>0</td><td>5</td><td>7</td><td>1050</td><td>910</td><td>1965</td><td>98136</td><td>47.5208</td><td>-122.393</td><td>1360</td><td>5000</td><td>1</td></tr></tbody></table></div>
```

(Two features in the original data set have been deemed uninformative
and dropped, namely `:id` and `:date`. The original feature
`:yr_renovated` has been replaced by the `Bool` feature `is_renovated`.)

<a id='part-2-selecting-training-and-evaluating-models'></a>

## Part 2 - Selecting, Training and Evaluating Models

> **Goals:**
> 1. Search MLJ's database of model metadata to identify model candidates for a supervised learning task.
> 2. Evaluate the performance of a model on a holdout set using basic `fit!`/`predict` work-flow.
> 3. Inspect the outcomes of training and save these to a file.
> 3. Evaluate performance using other resampling strategies, such as cross-validation, in one line, using `evaluate!`
> 4. Plot a "learning curve", to inspect performance as a function of some model hyper-parameter, such as an iteration parameter

The "Hello World!" of machine learning is to classify Fisher's
famous iris data set. This time, we'll grab the data from
[OpenML](https://www.openml.org):

````julia
OpenML.describe_dataset(61)
````

```@raw html
<div class="markdown"><p><strong>Author</strong>: R.A. Fisher   <strong>Source</strong>: <a href="https://archive.ics.uci.edu/ml/datasets/Iris">UCI</a> - 1936 - Donated by Michael Marshall   <strong>Please cite</strong>:   </p>
<p><strong>Iris Plants Database</strong>   This is perhaps the best known database to be found in the pattern recognition literature.  Fisher&#39;s paper is a classic in the field and is referenced frequently to this day.  &#40;See Duda &amp; Hart, for example.&#41;  The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant.  One class is     linearly separable from the other 2; the latter are NOT linearly separable from each other.</p>
<p>Predicted attribute: class of iris plant.   This is an exceedingly simple domain.  </p>
<h3>Attribute Information:</h3>
<pre><code>1. sepal length in cm
2. sepal width in cm
3. petal length in cm
4. petal width in cm
5. class: 
   -- Iris Setosa
   -- Iris Versicolour
   -- Iris Virginica</code></pre>
</div>
```

````julia
iris = OpenML.load(61); # a row table
iris = DataFrames.DataFrame(iris);
first(iris, 4)
````

```@raw html
<div class="data-frame"><p>4 rows × 5 columns</p><table class="data-frame"><thead><tr><th></th><th>sepallength</th><th>sepalwidth</th><th>petallength</th><th>petalwidth</th><th>class</th></tr><tr><th></th><th title="Union{Missing, Float64}">Float64?</th><th title="Union{Missing, Float64}">Float64?</th><th title="Union{Missing, Float64}">Float64?</th><th title="Union{Missing, Float64}">Float64?</th><th title="Union{Missing, CategoricalValue{String, UInt32}}">Cat…?</th></tr></thead><tbody><tr><th>1</th><td>5.1</td><td>3.5</td><td>1.4</td><td>0.2</td><td>Iris-setosa</td></tr><tr><th>2</th><td>4.9</td><td>3.0</td><td>1.4</td><td>0.2</td><td>Iris-setosa</td></tr><tr><th>3</th><td>4.7</td><td>3.2</td><td>1.3</td><td>0.2</td><td>Iris-setosa</td></tr><tr><th>4</th><td>4.6</td><td>3.1</td><td>1.5</td><td>0.2</td><td>Iris-setosa</td></tr></tbody></table></div>
```

**Main goal.** To build and evaluate models for predicting the
`:class` variable, given the four remaining measurement variables.

### Step 1. Inspect and fix scientific types

````julia
schema(iris)
````

````
┌─────────────┬──────────────────────────────────────────────────┬───────────────────────────────┐
│ _.names     │ _.types                                          │ _.scitypes                    │
├─────────────┼──────────────────────────────────────────────────┼───────────────────────────────┤
│ sepallength │ Union{Missing, Float64}                          │ Union{Missing, Continuous}    │
│ sepalwidth  │ Union{Missing, Float64}                          │ Union{Missing, Continuous}    │
│ petallength │ Union{Missing, Float64}                          │ Union{Missing, Continuous}    │
│ petalwidth  │ Union{Missing, Float64}                          │ Union{Missing, Continuous}    │
│ class       │ Union{Missing, CategoricalValue{String, UInt32}} │ Union{Missing, Multiclass{3}} │
└─────────────┴──────────────────────────────────────────────────┴───────────────────────────────┘
_.nrows = 150

````

Unfortunately, `Missing` is appearing in the element type, despite
the fact there are no missing values (see this
[issue](https://github.com/JuliaAI/OpenML.jl/issues/10)). To do this
we have to explicilty tighten the types:

FIX ME!!!!!!

````julia
iris = coerce(iris,
              Union{Missing,Continuous}=>Continuous,
              Union{Missing,Multiclass}=>Multiclass,
              tight=true)
schema(iris)
````

````
┌─────────────┬──────────────────────────────────┬───────────────┐
│ _.names     │ _.types                          │ _.scitypes    │
├─────────────┼──────────────────────────────────┼───────────────┤
│ sepallength │ Float64                          │ Continuous    │
│ sepalwidth  │ Float64                          │ Continuous    │
│ petallength │ Float64                          │ Continuous    │
│ petalwidth  │ Float64                          │ Continuous    │
│ class       │ CategoricalValue{String, UInt32} │ Multiclass{3} │
└─────────────┴──────────────────────────────────┴───────────────┘
_.nrows = 150

````

### Step 2. Split data into input and target parts

Here's how we split the data into target and input features, which
is needed for MLJ supervised models. We randomize the data at the
same time:

````julia
y, X = unpack(iris, ==(:class), name->true; rng=123);
scitype(y)
````

````
AbstractVector{Multiclass{3}} (alias for AbstractArray{Multiclass{3}, 1})
````

Here's one way to access the documentation (at the REPL, `?unpack`
also works):

````julia
@doc unpack
````

```@raw html
<div class="markdown"><pre><code>t1, t2, ...., tk &#61; unpack&#40;table, f1, f2, ... fk;
                         wrap_singles&#61;false,
                         shuffle&#61;false,
                         rng::Union&#123;AbstractRNG,Int,Nothing&#125;&#61;nothing&#41;</code></pre>
<p>Horizontally split any Tables.jl compatible <code>table</code> into smaller tables &#40;or vectors&#41; <code>t1, t2, ..., tk</code> by making column selections <strong>without replacement</strong> by successively applying the columnn name filters <code>f1</code>, <code>f2</code>, ..., <code>fk</code>. A <em>filter</em> is any object <code>f</code> such that <code>f&#40;name&#41;</code> is <code>true</code> or <code>false</code> for each column <code>name::Symbol</code> of <code>table</code>. For example, use the filter <code>_ -&gt; true</code> to pick up all remaining columns of the table.</p>
<p>Whenever a returned table contains a single column, it is converted to a vector unless <code>wrap_singles&#61;true</code>.</p>
<p>Scientific type conversions can be optionally specified &#40;note semicolon&#41;:</p>
<pre><code>unpack&#40;table, t...; col1&#61;&gt;scitype1, col2&#61;&gt;scitype2, ... &#41;</code></pre>
<p>If <code>shuffle&#61;true</code> then the rows of <code>table</code> are first shuffled, using the global RNG, unless <code>rng</code> is specified; if <code>rng</code> is an integer, it specifies the seed of an automatically generated Mersenne twister. If <code>rng</code> is specified then <code>shuffle&#61;true</code> is implicit.</p>
<h3>Example</h3>
<pre><code>julia&gt; table &#61; DataFrame&#40;x&#61;&#91;1,2&#93;, y&#61;&#91;&#39;a&#39;, &#39;b&#39;&#93;, z&#61;&#91;10.0, 20.0&#93;, w&#61;&#91;&quot;A&quot;, &quot;B&quot;&#93;&#41;
julia&gt; Z, XY &#61; unpack&#40;table, &#61;&#61;&#40;:z&#41;, &#33;&#61;&#40;:w&#41;;
               :x&#61;&gt;Continuous, :y&#61;&gt;Multiclass&#41;
julia&gt; XY
2×2 DataFrame
│ Row │ x       │ y            │
│     │ Float64 │ Categorical… │
├─────┼─────────┼──────────────┤
│ 1   │ 1.0     │ &#39;a&#39;          │
│ 2   │ 2.0     │ &#39;b&#39;          │

julia&gt; Z
2-element Array&#123;Float64,1&#125;:
 10.0
 20.0</code></pre>


</div>
```

### On searching for a model

Here's how to see *all* models (not immediately useful):

````julia
all_models = models()
````

````
183-element Vector{NamedTuple{(:name, :package_name, :is_supervised, :abstract_type, :deep_properties, :docstring, :fit_data_scitype, :hyperparameter_ranges, :hyperparameter_types, :hyperparameters, :implemented_methods, :inverse_transform_scitype, :is_pure_julia, :is_wrapper, :iteration_parameter, :load_path, :package_license, :package_url, :package_uuid, :predict_scitype, :prediction_type, :supports_class_weights, :supports_online, :supports_training_losses, :supports_weights, :transform_scitype, :input_scitype, :target_scitype, :output_scitype), T} where T<:Tuple}:
 (name = ABODDetector, package_name = OutlierDetectionNeighbors, ... )
 (name = ABODDetector, package_name = OutlierDetectionPython, ... )
 (name = AEDetector, package_name = OutlierDetectionNetworks, ... )
 (name = ARDRegressor, package_name = ScikitLearn, ... )
 (name = AdaBoostClassifier, package_name = ScikitLearn, ... )
 (name = AdaBoostRegressor, package_name = ScikitLearn, ... )
 (name = AdaBoostStumpClassifier, package_name = DecisionTree, ... )
 (name = AffinityPropagation, package_name = ScikitLearn, ... )
 (name = AgglomerativeClustering, package_name = ScikitLearn, ... )
 (name = BaggingClassifier, package_name = ScikitLearn, ... )
 (name = BaggingRegressor, package_name = ScikitLearn, ... )
 (name = BayesianLDA, package_name = MultivariateStats, ... )
 (name = BayesianLDA, package_name = ScikitLearn, ... )
 (name = BayesianQDA, package_name = ScikitLearn, ... )
 (name = BayesianRidgeRegressor, package_name = ScikitLearn, ... )
 (name = BayesianSubspaceLDA, package_name = MultivariateStats, ... )
 (name = BernoulliNBClassifier, package_name = ScikitLearn, ... )
 (name = Birch, package_name = ScikitLearn, ... )
 (name = CBLOFDetector, package_name = OutlierDetectionPython, ... )
 (name = COFDetector, package_name = OutlierDetectionNeighbors, ... )
 (name = COFDetector, package_name = OutlierDetectionPython, ... )
 (name = COPODDetector, package_name = OutlierDetectionPython, ... )
 (name = ComplementNBClassifier, package_name = ScikitLearn, ... )
 (name = ConstantClassifier, package_name = MLJModels, ... )
 (name = ConstantRegressor, package_name = MLJModels, ... )
 (name = ContinuousEncoder, package_name = MLJModels, ... )
 (name = DBSCAN, package_name = ScikitLearn, ... )
 (name = DNNDetector, package_name = OutlierDetectionNeighbors, ... )
 (name = DSADDetector, package_name = OutlierDetectionNetworks, ... )
 (name = DecisionTreeClassifier, package_name = BetaML, ... )
 (name = DecisionTreeClassifier, package_name = DecisionTree, ... )
 (name = DecisionTreeRegressor, package_name = BetaML, ... )
 (name = DecisionTreeRegressor, package_name = DecisionTree, ... )
 (name = DeterministicConstantClassifier, package_name = MLJModels, ... )
 (name = DeterministicConstantRegressor, package_name = MLJModels, ... )
 (name = DummyClassifier, package_name = ScikitLearn, ... )
 (name = DummyRegressor, package_name = ScikitLearn, ... )
 (name = ESADDetector, package_name = OutlierDetectionNetworks, ... )
 (name = ElasticNetCVRegressor, package_name = ScikitLearn, ... )
 (name = ElasticNetRegressor, package_name = MLJLinearModels, ... )
 (name = ElasticNetRegressor, package_name = ScikitLearn, ... )
 (name = EpsilonSVR, package_name = LIBSVM, ... )
 (name = EvoTreeClassifier, package_name = EvoTrees, ... )
 (name = EvoTreeCount, package_name = EvoTrees, ... )
 (name = EvoTreeGaussian, package_name = EvoTrees, ... )
 (name = EvoTreeRegressor, package_name = EvoTrees, ... )
 (name = ExtraTreesClassifier, package_name = ScikitLearn, ... )
 (name = ExtraTreesRegressor, package_name = ScikitLearn, ... )
 (name = FactorAnalysis, package_name = MultivariateStats, ... )
 (name = FeatureAgglomeration, package_name = ScikitLearn, ... )
 (name = FeatureSelector, package_name = MLJModels, ... )
 (name = FillImputer, package_name = MLJModels, ... )
 (name = GMMClusterer, package_name = BetaML, ... )
 (name = GaussianNBClassifier, package_name = NaiveBayes, ... )
 (name = GaussianNBClassifier, package_name = ScikitLearn, ... )
 (name = GaussianProcessClassifier, package_name = ScikitLearn, ... )
 (name = GaussianProcessRegressor, package_name = ScikitLearn, ... )
 (name = GradientBoostingClassifier, package_name = ScikitLearn, ... )
 (name = GradientBoostingRegressor, package_name = ScikitLearn, ... )
 (name = HBOSDetector, package_name = OutlierDetectionPython, ... )
 (name = HuberRegressor, package_name = MLJLinearModels, ... )
 (name = HuberRegressor, package_name = ScikitLearn, ... )
 (name = ICA, package_name = MultivariateStats, ... )
 (name = IForestDetector, package_name = OutlierDetectionPython, ... )
 (name = ImageClassifier, package_name = MLJFlux, ... )
 (name = KMeans, package_name = BetaML, ... )
 (name = KMeans, package_name = Clustering, ... )
 (name = KMeans, package_name = ParallelKMeans, ... )
 (name = KMeans, package_name = ScikitLearn, ... )
 (name = KMedoids, package_name = BetaML, ... )
 (name = KMedoids, package_name = Clustering, ... )
 (name = KNNClassifier, package_name = NearestNeighborModels, ... )
 (name = KNNDetector, package_name = OutlierDetectionNeighbors, ... )
 (name = KNNDetector, package_name = OutlierDetectionPython, ... )
 (name = KNNRegressor, package_name = NearestNeighborModels, ... )
 (name = KNeighborsClassifier, package_name = ScikitLearn, ... )
 (name = KNeighborsRegressor, package_name = ScikitLearn, ... )
 (name = KPLSRegressor, package_name = PartialLeastSquaresRegressor, ... )
 (name = KernelPCA, package_name = MultivariateStats, ... )
 (name = KernelPerceptronClassifier, package_name = BetaML, ... )
 (name = LADRegressor, package_name = MLJLinearModels, ... )
 (name = LDA, package_name = MultivariateStats, ... )
 (name = LGBMClassifier, package_name = LightGBM, ... )
 (name = LGBMRegressor, package_name = LightGBM, ... )
 (name = LMDDDetector, package_name = OutlierDetectionPython, ... )
 (name = LOCIDetector, package_name = OutlierDetectionPython, ... )
 (name = LODADetector, package_name = OutlierDetectionPython, ... )
 (name = LOFDetector, package_name = OutlierDetectionNeighbors, ... )
 (name = LOFDetector, package_name = OutlierDetectionPython, ... )
 (name = LarsCVRegressor, package_name = ScikitLearn, ... )
 (name = LarsRegressor, package_name = ScikitLearn, ... )
 (name = LassoCVRegressor, package_name = ScikitLearn, ... )
 (name = LassoLarsCVRegressor, package_name = ScikitLearn, ... )
 (name = LassoLarsICRegressor, package_name = ScikitLearn, ... )
 (name = LassoLarsRegressor, package_name = ScikitLearn, ... )
 (name = LassoRegressor, package_name = MLJLinearModels, ... )
 (name = LassoRegressor, package_name = ScikitLearn, ... )
 (name = LinearBinaryClassifier, package_name = GLM, ... )
 (name = LinearCountRegressor, package_name = GLM, ... )
 (name = LinearRegressor, package_name = GLM, ... )
 (name = LinearRegressor, package_name = MLJLinearModels, ... )
 (name = LinearRegressor, package_name = MultivariateStats, ... )
 (name = LinearRegressor, package_name = ScikitLearn, ... )
 (name = LinearSVC, package_name = LIBSVM, ... )
 (name = LogisticCVClassifier, package_name = ScikitLearn, ... )
 (name = LogisticClassifier, package_name = MLJLinearModels, ... )
 (name = LogisticClassifier, package_name = ScikitLearn, ... )
 (name = MCDDetector, package_name = OutlierDetectionPython, ... )
 (name = MeanShift, package_name = ScikitLearn, ... )
 (name = MiniBatchKMeans, package_name = ScikitLearn, ... )
 (name = MissingImputator, package_name = BetaML, ... )
 (name = MultiTaskElasticNetCVRegressor, package_name = ScikitLearn, ... )
 (name = MultiTaskElasticNetRegressor, package_name = ScikitLearn, ... )
 (name = MultiTaskLassoCVRegressor, package_name = ScikitLearn, ... )
 (name = MultiTaskLassoRegressor, package_name = ScikitLearn, ... )
 (name = MultinomialClassifier, package_name = MLJLinearModels, ... )
 (name = MultinomialNBClassifier, package_name = NaiveBayes, ... )
 (name = MultinomialNBClassifier, package_name = ScikitLearn, ... )
 (name = MultitargetKNNClassifier, package_name = NearestNeighborModels, ... )
 (name = MultitargetKNNRegressor, package_name = NearestNeighborModels, ... )
 (name = MultitargetLinearRegressor, package_name = MultivariateStats, ... )
 (name = MultitargetNeuralNetworkRegressor, package_name = MLJFlux, ... )
 (name = MultitargetRidgeRegressor, package_name = MultivariateStats, ... )
 (name = NeuralNetworkClassifier, package_name = MLJFlux, ... )
 (name = NeuralNetworkRegressor, package_name = MLJFlux, ... )
 (name = NuSVC, package_name = LIBSVM, ... )
 (name = NuSVR, package_name = LIBSVM, ... )
 (name = OCSVMDetector, package_name = OutlierDetectionPython, ... )
 (name = OPTICS, package_name = ScikitLearn, ... )
 (name = OneClassSVM, package_name = LIBSVM, ... )
 (name = OneHotEncoder, package_name = MLJModels, ... )
 (name = OrthogonalMatchingPursuitCVRegressor, package_name = ScikitLearn, ... )
 (name = OrthogonalMatchingPursuitRegressor, package_name = ScikitLearn, ... )
 (name = PCA, package_name = MultivariateStats, ... )
 (name = PCADetector, package_name = OutlierDetectionPython, ... )
 (name = PLSRegressor, package_name = PartialLeastSquaresRegressor, ... )
 (name = PPCA, package_name = MultivariateStats, ... )
 (name = PassiveAggressiveClassifier, package_name = ScikitLearn, ... )
 (name = PassiveAggressiveRegressor, package_name = ScikitLearn, ... )
 (name = PegasosClassifier, package_name = BetaML, ... )
 (name = PerceptronClassifier, package_name = BetaML, ... )
 (name = PerceptronClassifier, package_name = ScikitLearn, ... )
 (name = ProbabilisticSGDClassifier, package_name = ScikitLearn, ... )
 (name = QuantileRegressor, package_name = MLJLinearModels, ... )
 (name = RANSACRegressor, package_name = ScikitLearn, ... )
 (name = RODDetector, package_name = OutlierDetectionPython, ... )
 (name = RandomForestClassifier, package_name = BetaML, ... )
 (name = RandomForestClassifier, package_name = DecisionTree, ... )
 (name = RandomForestClassifier, package_name = ScikitLearn, ... )
 (name = RandomForestRegressor, package_name = BetaML, ... )
 (name = RandomForestRegressor, package_name = DecisionTree, ... )
 (name = RandomForestRegressor, package_name = ScikitLearn, ... )
 (name = RidgeCVClassifier, package_name = ScikitLearn, ... )
 (name = RidgeCVRegressor, package_name = ScikitLearn, ... )
 (name = RidgeClassifier, package_name = ScikitLearn, ... )
 (name = RidgeRegressor, package_name = MLJLinearModels, ... )
 (name = RidgeRegressor, package_name = MultivariateStats, ... )
 (name = RidgeRegressor, package_name = ScikitLearn, ... )
 (name = RobustRegressor, package_name = MLJLinearModels, ... )
 (name = SGDClassifier, package_name = ScikitLearn, ... )
 (name = SGDRegressor, package_name = ScikitLearn, ... )
 (name = SODDetector, package_name = OutlierDetectionPython, ... )
 (name = SOSDetector, package_name = OutlierDetectionPython, ... )
 (name = SVC, package_name = LIBSVM, ... )
 (name = SVMClassifier, package_name = ScikitLearn, ... )
 (name = SVMLinearClassifier, package_name = ScikitLearn, ... )
 (name = SVMLinearRegressor, package_name = ScikitLearn, ... )
 (name = SVMNuClassifier, package_name = ScikitLearn, ... )
 (name = SVMNuRegressor, package_name = ScikitLearn, ... )
 (name = SVMRegressor, package_name = ScikitLearn, ... )
 (name = SpectralClustering, package_name = ScikitLearn, ... )
 (name = Standardizer, package_name = MLJModels, ... )
 (name = SubspaceLDA, package_name = MultivariateStats, ... )
 (name = TSVDTransformer, package_name = TSVD, ... )
 (name = TheilSenRegressor, package_name = ScikitLearn, ... )
 (name = UnivariateBoxCoxTransformer, package_name = MLJModels, ... )
 (name = UnivariateDiscretizer, package_name = MLJModels, ... )
 (name = UnivariateFillImputer, package_name = MLJModels, ... )
 (name = UnivariateStandardizer, package_name = MLJModels, ... )
 (name = UnivariateTimeTypeToContinuous, package_name = MLJModels, ... )
 (name = XGBoostClassifier, package_name = XGBoost, ... )
 (name = XGBoostCount, package_name = XGBoost, ... )
 (name = XGBoostRegressor, package_name = XGBoost, ... )
````

Each entry contains metadata for a model whose defining code is not yet loaded:

````julia
meta = all_models[3]
````

````
[35mAEDetector from OutlierDetectionNetworks.jl.[39m
[35m[Documentation](https://github.com/OutlierDetectionJL/OutlierDetectionNetworks.jl).[39m
(name = "AEDetector",
 package_name = "OutlierDetectionNetworks",
 is_supervised = false,
 abstract_type = MLJModelInterface.UnsupervisedDetector,
 deep_properties = (),
 docstring = "AEDetector from OutlierDetectionNetworks.jl.\n[Documentation](https://github.com/OutlierDetectionJL/OutlierDetectionNetworks.jl).",
 fit_data_scitype = Tuple{Union{Table{_s52} where _s52<:(AbstractVector{_s51} where _s51<:Continuous), AbstractMatrix{_s689} where _s689<:Continuous}},
 hyperparameter_ranges = (nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing),
 hyperparameter_types = ("Flux.Chain", "Flux.Chain", "Integer", "Integer", "Bool", "Bool", "Any", "Function"),
 hyperparameters = (:encoder, :decoder, :batchsize, :epochs, :shuffle, :partial, :opt, :loss),
 implemented_methods = [:clean!, :fit, :transform],
 inverse_transform_scitype = Unknown,
 is_pure_julia = true,
 is_wrapper = false,
 iteration_parameter = nothing,
 load_path = "OutlierDetectionNetworks.AEDetector",
 package_license = "MIT",
 package_url = "https://github.com/OutlierDetectionJL/OutlierDetectionNetworks.jl",
 package_uuid = "c7f57e37-4fcb-4a0b-a36c-c2204bc839a7",
 predict_scitype = Unknown,
 prediction_type = :unknown,
 supports_class_weights = false,
 supports_online = false,
 supports_training_losses = false,
 supports_weights = false,
 transform_scitype = AbstractVector{_s689} where _s689<:Continuous,
 input_scitype = Union{Table{_s52} where _s52<:(AbstractVector{_s51} where _s51<:Continuous), AbstractMatrix{_s689} where _s689<:Continuous},
 target_scitype = AbstractVector{_s689} where _s689<:Union{Missing, OrderedFactor{2}},
 output_scitype = AbstractVector{_s689} where _s689<:Continuous,)
````

````julia
targetscitype = meta.target_scitype
````

````
AbstractVector{_s689} where _s689<:Union{Missing, OrderedFactor{2}} (alias for AbstractArray{_s689, 1} where _s689<:Union{Missing, OrderedFactor{2}})
````

````julia
scitype(y) <: targetscitype
````

````
false
````

So this model won't do. Let's  find all pure julia classifiers:

````julia
filter_julia_classifiers(meta) =
    AbstractVector{Finite} <: meta.target_scitype &&
    meta.is_pure_julia

models(filter_julia_classifiers)
````

````
21-element Vector{NamedTuple{(:name, :package_name, :is_supervised, :abstract_type, :deep_properties, :docstring, :fit_data_scitype, :hyperparameter_ranges, :hyperparameter_types, :hyperparameters, :implemented_methods, :inverse_transform_scitype, :is_pure_julia, :is_wrapper, :iteration_parameter, :load_path, :package_license, :package_url, :package_uuid, :predict_scitype, :prediction_type, :supports_class_weights, :supports_online, :supports_training_losses, :supports_weights, :transform_scitype, :input_scitype, :target_scitype, :output_scitype), T} where T<:Tuple}:
 (name = AdaBoostStumpClassifier, package_name = DecisionTree, ... )
 (name = BayesianLDA, package_name = MultivariateStats, ... )
 (name = BayesianSubspaceLDA, package_name = MultivariateStats, ... )
 (name = ConstantClassifier, package_name = MLJModels, ... )
 (name = DecisionTreeClassifier, package_name = BetaML, ... )
 (name = DecisionTreeClassifier, package_name = DecisionTree, ... )
 (name = DeterministicConstantClassifier, package_name = MLJModels, ... )
 (name = EvoTreeClassifier, package_name = EvoTrees, ... )
 (name = GaussianNBClassifier, package_name = NaiveBayes, ... )
 (name = KNNClassifier, package_name = NearestNeighborModels, ... )
 (name = KernelPerceptronClassifier, package_name = BetaML, ... )
 (name = LDA, package_name = MultivariateStats, ... )
 (name = LogisticClassifier, package_name = MLJLinearModels, ... )
 (name = MultinomialClassifier, package_name = MLJLinearModels, ... )
 (name = MultinomialNBClassifier, package_name = NaiveBayes, ... )
 (name = NeuralNetworkClassifier, package_name = MLJFlux, ... )
 (name = PegasosClassifier, package_name = BetaML, ... )
 (name = PerceptronClassifier, package_name = BetaML, ... )
 (name = RandomForestClassifier, package_name = BetaML, ... )
 (name = RandomForestClassifier, package_name = DecisionTree, ... )
 (name = SubspaceLDA, package_name = MultivariateStats, ... )
````

Find all models with "Classifier" in `name` (or `docstring`):

````julia
models("Classifier")
````

````
45-element Vector{NamedTuple{(:name, :package_name, :is_supervised, :abstract_type, :deep_properties, :docstring, :fit_data_scitype, :hyperparameter_ranges, :hyperparameter_types, :hyperparameters, :implemented_methods, :inverse_transform_scitype, :is_pure_julia, :is_wrapper, :iteration_parameter, :load_path, :package_license, :package_url, :package_uuid, :predict_scitype, :prediction_type, :supports_class_weights, :supports_online, :supports_training_losses, :supports_weights, :transform_scitype, :input_scitype, :target_scitype, :output_scitype), T} where T<:Tuple}:
 (name = AdaBoostClassifier, package_name = ScikitLearn, ... )
 (name = AdaBoostStumpClassifier, package_name = DecisionTree, ... )
 (name = BaggingClassifier, package_name = ScikitLearn, ... )
 (name = BernoulliNBClassifier, package_name = ScikitLearn, ... )
 (name = ComplementNBClassifier, package_name = ScikitLearn, ... )
 (name = ConstantClassifier, package_name = MLJModels, ... )
 (name = DecisionTreeClassifier, package_name = BetaML, ... )
 (name = DecisionTreeClassifier, package_name = DecisionTree, ... )
 (name = DeterministicConstantClassifier, package_name = MLJModels, ... )
 (name = DummyClassifier, package_name = ScikitLearn, ... )
 (name = EvoTreeClassifier, package_name = EvoTrees, ... )
 (name = ExtraTreesClassifier, package_name = ScikitLearn, ... )
 (name = GaussianNBClassifier, package_name = NaiveBayes, ... )
 (name = GaussianNBClassifier, package_name = ScikitLearn, ... )
 (name = GaussianProcessClassifier, package_name = ScikitLearn, ... )
 (name = GradientBoostingClassifier, package_name = ScikitLearn, ... )
 (name = ImageClassifier, package_name = MLJFlux, ... )
 (name = KNNClassifier, package_name = NearestNeighborModels, ... )
 (name = KNeighborsClassifier, package_name = ScikitLearn, ... )
 (name = KernelPerceptronClassifier, package_name = BetaML, ... )
 (name = LGBMClassifier, package_name = LightGBM, ... )
 (name = LinearBinaryClassifier, package_name = GLM, ... )
 (name = LogisticCVClassifier, package_name = ScikitLearn, ... )
 (name = LogisticClassifier, package_name = MLJLinearModels, ... )
 (name = LogisticClassifier, package_name = ScikitLearn, ... )
 (name = MultinomialClassifier, package_name = MLJLinearModels, ... )
 (name = MultinomialNBClassifier, package_name = NaiveBayes, ... )
 (name = MultinomialNBClassifier, package_name = ScikitLearn, ... )
 (name = MultitargetKNNClassifier, package_name = NearestNeighborModels, ... )
 (name = NeuralNetworkClassifier, package_name = MLJFlux, ... )
 (name = PassiveAggressiveClassifier, package_name = ScikitLearn, ... )
 (name = PegasosClassifier, package_name = BetaML, ... )
 (name = PerceptronClassifier, package_name = BetaML, ... )
 (name = PerceptronClassifier, package_name = ScikitLearn, ... )
 (name = ProbabilisticSGDClassifier, package_name = ScikitLearn, ... )
 (name = RandomForestClassifier, package_name = BetaML, ... )
 (name = RandomForestClassifier, package_name = DecisionTree, ... )
 (name = RandomForestClassifier, package_name = ScikitLearn, ... )
 (name = RidgeCVClassifier, package_name = ScikitLearn, ... )
 (name = RidgeClassifier, package_name = ScikitLearn, ... )
 (name = SGDClassifier, package_name = ScikitLearn, ... )
 (name = SVMClassifier, package_name = ScikitLearn, ... )
 (name = SVMLinearClassifier, package_name = ScikitLearn, ... )
 (name = SVMNuClassifier, package_name = ScikitLearn, ... )
 (name = XGBoostClassifier, package_name = XGBoost, ... )
````

Find all (supervised) models that match my data!

````julia
models(matching(X, y))
````

````
47-element Vector{NamedTuple{(:name, :package_name, :is_supervised, :abstract_type, :deep_properties, :docstring, :fit_data_scitype, :hyperparameter_ranges, :hyperparameter_types, :hyperparameters, :implemented_methods, :inverse_transform_scitype, :is_pure_julia, :is_wrapper, :iteration_parameter, :load_path, :package_license, :package_url, :package_uuid, :predict_scitype, :prediction_type, :supports_class_weights, :supports_online, :supports_training_losses, :supports_weights, :transform_scitype, :input_scitype, :target_scitype, :output_scitype), T} where T<:Tuple}:
 (name = AdaBoostClassifier, package_name = ScikitLearn, ... )
 (name = AdaBoostStumpClassifier, package_name = DecisionTree, ... )
 (name = BaggingClassifier, package_name = ScikitLearn, ... )
 (name = BayesianLDA, package_name = MultivariateStats, ... )
 (name = BayesianLDA, package_name = ScikitLearn, ... )
 (name = BayesianQDA, package_name = ScikitLearn, ... )
 (name = BayesianSubspaceLDA, package_name = MultivariateStats, ... )
 (name = ConstantClassifier, package_name = MLJModels, ... )
 (name = DecisionTreeClassifier, package_name = BetaML, ... )
 (name = DecisionTreeClassifier, package_name = DecisionTree, ... )
 (name = DeterministicConstantClassifier, package_name = MLJModels, ... )
 (name = DummyClassifier, package_name = ScikitLearn, ... )
 (name = EvoTreeClassifier, package_name = EvoTrees, ... )
 (name = ExtraTreesClassifier, package_name = ScikitLearn, ... )
 (name = GaussianNBClassifier, package_name = NaiveBayes, ... )
 (name = GaussianNBClassifier, package_name = ScikitLearn, ... )
 (name = GaussianProcessClassifier, package_name = ScikitLearn, ... )
 (name = GradientBoostingClassifier, package_name = ScikitLearn, ... )
 (name = KNNClassifier, package_name = NearestNeighborModels, ... )
 (name = KNeighborsClassifier, package_name = ScikitLearn, ... )
 (name = KernelPerceptronClassifier, package_name = BetaML, ... )
 (name = LDA, package_name = MultivariateStats, ... )
 (name = LGBMClassifier, package_name = LightGBM, ... )
 (name = LinearSVC, package_name = LIBSVM, ... )
 (name = LogisticCVClassifier, package_name = ScikitLearn, ... )
 (name = LogisticClassifier, package_name = MLJLinearModels, ... )
 (name = LogisticClassifier, package_name = ScikitLearn, ... )
 (name = MultinomialClassifier, package_name = MLJLinearModels, ... )
 (name = NeuralNetworkClassifier, package_name = MLJFlux, ... )
 (name = NuSVC, package_name = LIBSVM, ... )
 (name = PassiveAggressiveClassifier, package_name = ScikitLearn, ... )
 (name = PegasosClassifier, package_name = BetaML, ... )
 (name = PerceptronClassifier, package_name = BetaML, ... )
 (name = PerceptronClassifier, package_name = ScikitLearn, ... )
 (name = ProbabilisticSGDClassifier, package_name = ScikitLearn, ... )
 (name = RandomForestClassifier, package_name = BetaML, ... )
 (name = RandomForestClassifier, package_name = DecisionTree, ... )
 (name = RandomForestClassifier, package_name = ScikitLearn, ... )
 (name = RidgeCVClassifier, package_name = ScikitLearn, ... )
 (name = RidgeClassifier, package_name = ScikitLearn, ... )
 (name = SGDClassifier, package_name = ScikitLearn, ... )
 (name = SVC, package_name = LIBSVM, ... )
 (name = SVMClassifier, package_name = ScikitLearn, ... )
 (name = SVMLinearClassifier, package_name = ScikitLearn, ... )
 (name = SVMNuClassifier, package_name = ScikitLearn, ... )
 (name = SubspaceLDA, package_name = MultivariateStats, ... )
 (name = XGBoostClassifier, package_name = XGBoost, ... )
````

### Step 3. Select and instantiate a model

To load the code defining a new model type we use the `@load` macro:

````julia
NeuralNetworkClassifier = @load NeuralNetworkClassifier
````

````
MLJFlux.NeuralNetworkClassifier
````

Other ways to load model code are described
[here](https://alan-turing-institute.github.io/MLJ.jl/dev/loading_model_code/#Loading-Model-Code).

We'll instantiate this type with default values for the
hyperparameters:

````julia
model = NeuralNetworkClassifier()
````

````
NeuralNetworkClassifier(
    builder = Short(
            n_hidden = 0,
            dropout = 0.5,
            σ = NNlib.σ),
    finaliser = NNlib.softmax,
    optimiser = Flux.Optimise.ADAM(0.001, (0.9, 0.999), IdDict{Any, Any}()),
    loss = Flux.Losses.crossentropy,
    epochs = 10,
    batch_size = 1,
    lambda = 0.0,
    alpha = 0.0,
    rng = Random._GLOBAL_RNG(),
    optimiser_changes_trigger_retraining = false,
    acceleration = CPU1{Nothing}(nothing))
````

````julia
info(model)
````

````
[35mA neural network model for making probabilistic predictions of a `Multiclass` or `OrderedFactor` target, given a table of `Continuous` features. [39m
[35m→ based on [MLJFlux](https://github.com/alan-turing-institute/MLJFlux.jl).[39m
[35m→ do `@load NeuralNetworkClassifier pkg="MLJFlux"` to use the model.[39m
[35m→ do `?NeuralNetworkClassifier` for documentation.[39m
(name = "NeuralNetworkClassifier",
 package_name = "MLJFlux",
 is_supervised = true,
 abstract_type = Probabilistic,
 deep_properties = (:optimiser, :builder),
 docstring = "A neural network model for making probabilistic predictions of a `Multiclass` or `OrderedFactor` target, given a table of `Continuous` features. \n→ based on [MLJFlux](https://github.com/alan-turing-institute/MLJFlux.jl).\n→ do `@load NeuralNetworkClassifier pkg=\"MLJFlux\"` to use the model.\n→ do `?NeuralNetworkClassifier` for documentation.",
 fit_data_scitype = Tuple{Table{var"#s53"} where var"#s53"<:(AbstractVector{var"#s52"} where var"#s52"<:Continuous), AbstractVector{var"#s76"} where var"#s76"<:Finite},
 hyperparameter_ranges = (nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing),
 hyperparameter_types = ("MLJFlux.Short", "typeof(NNlib.softmax)", "Flux.Optimise.ADAM", "typeof(Flux.Losses.crossentropy)", "Int64", "Int64", "Float64", "Float64", "Union{Int64, AbstractRNG}", "Bool", "ComputationalResources.AbstractResource"),
 hyperparameters = (:builder, :finaliser, :optimiser, :loss, :epochs, :batch_size, :lambda, :alpha, :rng, :optimiser_changes_trigger_retraining, :acceleration),
 implemented_methods = Any[],
 inverse_transform_scitype = Unknown,
 is_pure_julia = true,
 is_wrapper = false,
 iteration_parameter = :epochs,
 load_path = "MLJFlux.NeuralNetworkClassifier",
 package_license = "MIT",
 package_url = "https://github.com/alan-turing-institute/MLJFlux.jl",
 package_uuid = "094fc8d1-fd35-5302-93ea-dabda2abf845",
 predict_scitype = AbstractVector{ScientificTypesBase.Density{_s25} where _s25<:Finite},
 prediction_type = :probabilistic,
 supports_class_weights = false,
 supports_online = false,
 supports_training_losses = true,
 supports_weights = false,
 transform_scitype = Unknown,
 input_scitype = Table{var"#s53"} where var"#s53"<:(AbstractVector{var"#s52"} where var"#s52"<:Continuous),
 target_scitype = AbstractVector{var"#s76"} where var"#s76"<:Finite,
 output_scitype = Unknown,)
````

In MLJ a *model* is just a struct containing hyper-parameters, and
that's all. A model does not store *learned* parameters. Models are
mutable:

````julia
model.epochs = 12
````

````
12
````

And all models have a key-word constructor that works once `@load`
has been performed:

````julia
NeuralNetworkClassifier(epochs=12) == model
````

````
true
````

### On fitting, predicting, and inspecting models

In MLJ a model and training/validation data are typically bound
together in a machine:

````julia
mach = machine(model, X, y)
````

````
Machine{NeuralNetworkClassifier{Short,…},…} trained 0 times; caches data
  args: 
    1:	Source @902 ⏎ `Table{AbstractVector{Continuous}}`
    2:	Source @305 ⏎ `AbstractVector{Multiclass{3}}`

````

A machine stores *learned* parameters, among other things. We'll
train this machine on 70% of the data and evaluate on a 30% holdout
set. Let's start by dividing all row indices into `train` and `test`
subsets:

````julia
train, test = partition(eachindex(y), 0.7)
````

````
([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105], [106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150])
````

Now we can `fit!`...

````julia
fit!(mach, rows=train, verbosity=2)
````

````
Machine{NeuralNetworkClassifier{Short,…},…} trained 1 time; caches data
  args: 
    1:	Source @902 ⏎ `Table{AbstractVector{Continuous}}`
    2:	Source @305 ⏎ `AbstractVector{Multiclass{3}}`

````

... and `predict`:

````julia
yhat = predict(mach, rows=test);  # or `predict(mach, Xnew)`
yhat[1:3]
````

````
3-element MLJBase.UnivariateFiniteVector{Multiclass{3}, String, UInt32, Float64}:
 UnivariateFinite{Multiclass{3}}(Iris-setosa=>0.308, Iris-versicolor=>0.358, Iris-virginica=>0.334)
 UnivariateFinite{Multiclass{3}}(Iris-setosa=>0.289, Iris-versicolor=>0.368, Iris-virginica=>0.343)
 UnivariateFinite{Multiclass{3}}(Iris-setosa=>0.412, Iris-versicolor=>0.302, Iris-virginica=>0.287)
````

We'll have more to say on the form of this prediction shortly.

After training, one can inspect the learned parameters:

````julia
fitted_params(mach)
````

````
(chain = Chain(Chain(Dense(4, 3, σ), Dropout(0.5), Dense(3, 3)), softmax),)
````

Everything else the user might be interested in is accessed from the
training *report*:

````julia
report(mach)
````

````
(training_losses = [1.2559176769988762, 1.3170392208064197, 1.2810116884925107, 1.2562834872760997, 1.1201985415208149, 1.101753341902836, 1.1226891488962834, 1.1124477674171902, 1.0381021980346619, 1.0523111166616987, 1.0648892577341496, 1.0271606808156204, 1.086799948938903],)
````

You save a machine like this:

````julia
MLJ.save("neural_net.jlso", mach)
````

And retrieve it like this:

````julia
mach2 = machine("neural_net.jlso")
yhat = predict(mach2, X);
yhat[1:3]
````

````
3-element MLJBase.UnivariateFiniteVector{Multiclass{3}, String, UInt32, Float64}:
 UnivariateFinite{Multiclass{3}}(Iris-setosa=>0.285, Iris-versicolor=>0.369, Iris-virginica=>0.346)
 UnivariateFinite{Multiclass{3}}(Iris-setosa=>0.306, Iris-versicolor=>0.359, Iris-virginica=>0.335)
 UnivariateFinite{Multiclass{3}}(Iris-setosa=>0.284, Iris-versicolor=>0.367, Iris-virginica=>0.349)
````

If you want to fit a retrieved model, you will need to bind some data to it:

````julia
mach3 = machine("neural_net.jlso", X, y)
fit!(mach3)
````

````
Machine{NeuralNetworkClassifier{Short,…},…} trained 2 times; caches data
  args: 
    1:	Source @848 ⏎ `Table{AbstractVector{Continuous}}`
    2:	Source @236 ⏎ `AbstractVector{Multiclass{3}}`

````

Machines remember the last set of hyper-parameters used during fit,
which, in the case of iterative models, allows for a warm restart of
computations in the case that only the iteration parameter is
increased:

````julia
model.epochs = model.epochs + 4
fit!(mach, rows=train, verbosity=2)
````

````
Machine{NeuralNetworkClassifier{Short,…},…} trained 2 times; caches data
  args: 
    1:	Source @902 ⏎ `Table{AbstractVector{Continuous}}`
    2:	Source @305 ⏎ `AbstractVector{Multiclass{3}}`

````

For this particular model we can also increase `:learning_rate`
without triggering a cold restart:

````julia
model.epochs = model.epochs + 4
model.optimiser.eta = 10*model.optimiser.eta
fit!(mach, rows=train, verbosity=2)
````

````
Machine{NeuralNetworkClassifier{Short,…},…} trained 3 times; caches data
  args: 
    1:	Source @902 ⏎ `Table{AbstractVector{Continuous}}`
    2:	Source @305 ⏎ `AbstractVector{Multiclass{3}}`

````

However, change any other parameter and training will restart from
scratch:

````julia
model.lambda = 0.001
fit!(mach, rows=train, verbosity=2)
````

````
Machine{NeuralNetworkClassifier{Short,…},…} trained 4 times; caches data
  args: 
    1:	Source @902 ⏎ `Table{AbstractVector{Continuous}}`
    2:	Source @305 ⏎ `AbstractVector{Multiclass{3}}`

````

Iterative models that implement warm-restart for training can be
controlled externally (eg, using an out-of-sample stopping
criterion). See
[here](https://alan-turing-institute.github.io/MLJ.jl/dev/controlling_iterative_models/)
for details.

Let's train silently for a total of 50 epochs, and look at a
prediction:

````julia
model.epochs = 50
fit!(mach, rows=train)
yhat = predict(mach, X[test,:]); # or predict(mach, rows=test)
yhat[1]
````

````
UnivariateFinite{Multiclass{3}}(Iris-setosa=>0.118, Iris-versicolor=>0.524, Iris-virginica=>0.358)
````

What's going on here?

````julia
info(model).prediction_type
````

````
:probabilistic
````

**Important**:
- In MLJ, a model that can predict probabilities (and not just point values) will do so by default.
- For most probabilistic predictors, the predicted object is a `Distributions.Distribution` object, supporting the `Distributions.jl` [API](https://juliastats.org/Distributions.jl/latest/extends/#Create-a-Distribution-1) for such objects. In particular, the methods `rand`,  `pdf`, `logpdf`, `mode`, `median` and `mean` will apply, where appropriate.

So, to obtain the probability of "Iris-virginica" in the first test
prediction, we do

````julia
pdf(yhat[1], "Iris-virginica")
````

````
0.35804834874791513
````

To get the most likely observation, we do

````julia
mode(yhat[1])
````

````
CategoricalValue{String, UInt32} "Iris-versicolor"
````

These can be broadcast over multiple predictions in the usual way:

````julia
broadcast(pdf, yhat[1:4], "Iris-versicolor")
````

````
4-element Vector{Float64}:
 0.5239484952144228
 0.4731109844693249
 0.056882068083702046
 0.3486877328180898
````

````julia
mode.(yhat[1:4])
````

````
4-element CategoricalArray{String,1,UInt32}:
 "Iris-versicolor"
 "Iris-versicolor"
 "Iris-setosa"
 "Iris-virginica"
````

Or, alternatively, you can use the `predict_mode` operation instead
of `predict`:

````julia
predict_mode(mach, X[test,:])[1:4] # or predict_mode(mach, rows=test)[1:4]
````

````
4-element CategoricalArray{String,1,UInt32}:
 "Iris-versicolor"
 "Iris-versicolor"
 "Iris-setosa"
 "Iris-virginica"
````

For a more conventional matrix of probabilities you can do this:

````julia
L = levels(y)
pdf(yhat, L)[1:4, :]
````

````
4×3 Matrix{Float64}:
 0.118003   0.523948   0.358048
 0.0597223  0.473111   0.467167
 0.941423   0.0568821  0.0016948
 0.0150815  0.348688   0.636231
````

However, in a typical MLJ work-flow, this is not as useful as you
might imagine. In particular, all probabilistic performance measures
in MLJ expect distribution objects in their first slot:

````julia
cross_entropy(yhat, y[test]) |> mean
````

````
0.38563146383967883
````

To apply a deterministic measure, we first need to obtain point-estimates:

````julia
misclassification_rate(mode.(yhat), y[test])
````

````
0.044444444444444446
````

We note in passing that there is also a search tool for measures
analogous to `models`:

````julia
measures()
````

````
61-element Vector{NamedTuple{(:name, :instances, :human_name, :target_scitype, :supports_weights, :supports_class_weights, :prediction_type, :orientation, :reports_each_observation, :aggregation, :is_feature_dependent, :docstring, :distribution_type), T} where T<:Tuple}:
 (name = BrierLoss, instances = [brier_loss], ...)
 (name = BrierScore, instances = [brier_score], ...)
 (name = LPLoss, instances = [l1, l2], ...)
 (name = LogCoshLoss, instances = [log_cosh, log_cosh_loss], ...)
 (name = LogLoss, instances = [log_loss, cross_entropy], ...)
 (name = LogScore, instances = [log_score], ...)
 (name = SphericalScore, instances = [spherical_score], ...)
 (name = Accuracy, instances = [accuracy], ...)
 (name = AreaUnderCurve, instances = [area_under_curve, auc], ...)
 (name = BalancedAccuracy, instances = [balanced_accuracy, bacc, bac], ...)
 (name = ConfusionMatrix, instances = [confusion_matrix, confmat], ...)
 (name = FScore, instances = [f1score], ...)
 (name = FalseDiscoveryRate, instances = [false_discovery_rate, falsediscovery_rate, fdr], ...)
 (name = FalseNegative, instances = [false_negative, falsenegative], ...)
 (name = FalseNegativeRate, instances = [false_negative_rate, falsenegative_rate, fnr, miss_rate], ...)
 (name = FalsePositive, instances = [false_positive, falsepositive], ...)
 (name = FalsePositiveRate, instances = [false_positive_rate, falsepositive_rate, fpr, fallout], ...)
 (name = MatthewsCorrelation, instances = [matthews_correlation, mcc], ...)
 (name = MeanAbsoluteError, instances = [mae, mav, mean_absolute_error, mean_absolute_value], ...)
 (name = MeanAbsoluteProportionalError, instances = [mape], ...)
 (name = MisclassificationRate, instances = [misclassification_rate, mcr], ...)
 (name = MulticlassFScore, instances = [macro_f1score, micro_f1score, multiclass_f1score], ...)
 (name = MulticlassFalseDiscoveryRate, instances = [multiclass_falsediscovery_rate, multiclass_fdr], ...)
 (name = MulticlassFalseNegative, instances = [multiclass_false_negative, multiclass_falsenegative], ...)
 (name = MulticlassFalseNegativeRate, instances = [multiclass_false_negative_rate, multiclass_fnr, multiclass_miss_rate, multiclass_falsenegative_rate], ...)
 (name = MulticlassFalsePositive, instances = [multiclass_false_positive, multiclass_falsepositive], ...)
 (name = MulticlassFalsePositiveRate, instances = [multiclass_false_positive_rate, multiclass_fpr, multiclass_fallout, multiclass_falsepositive_rate], ...)
 (name = MulticlassNegativePredictiveValue, instances = [multiclass_negative_predictive_value, multiclass_negativepredictive_value, multiclass_npv], ...)
 (name = MulticlassPrecision, instances = [multiclass_positive_predictive_value, multiclass_ppv, multiclass_positivepredictive_value, multiclass_recall], ...)
 (name = MulticlassTrueNegative, instances = [multiclass_true_negative, multiclass_truenegative], ...)
 (name = MulticlassTrueNegativeRate, instances = [multiclass_true_negative_rate, multiclass_tnr, multiclass_specificity, multiclass_selectivity, multiclass_truenegative_rate], ...)
 (name = MulticlassTruePositive, instances = [multiclass_true_positive, multiclass_truepositive], ...)
 (name = MulticlassTruePositiveRate, instances = [multiclass_true_positive_rate, multiclass_tpr, multiclass_sensitivity, multiclass_recall, multiclass_hit_rate, multiclass_truepositive_rate], ...)
 (name = NegativePredictiveValue, instances = [negative_predictive_value, negativepredictive_value, npv], ...)
 (name = Precision, instances = [positive_predictive_value, ppv, positivepredictive_value, precision], ...)
 (name = RootMeanSquaredError, instances = [rms, rmse, root_mean_squared_error], ...)
 (name = RootMeanSquaredLogError, instances = [rmsl, rmsle, root_mean_squared_log_error], ...)
 (name = RootMeanSquaredLogProportionalError, instances = [rmslp1], ...)
 (name = RootMeanSquaredProportionalError, instances = [rmsp], ...)
 (name = TrueNegative, instances = [true_negative, truenegative], ...)
 (name = TrueNegativeRate, instances = [true_negative_rate, truenegative_rate, tnr, specificity, selectivity], ...)
 (name = TruePositive, instances = [true_positive, truepositive], ...)
 (name = TruePositiveRate, instances = [true_positive_rate, truepositive_rate, tpr, sensitivity, recall, hit_rate], ...)
 (name = DWDMarginLoss, instances = [dwd_margin_loss], ...)
 (name = ExpLoss, instances = [exp_loss], ...)
 (name = L1HingeLoss, instances = [l1_hinge_loss], ...)
 (name = L2HingeLoss, instances = [l2_hinge_loss], ...)
 (name = L2MarginLoss, instances = [l2_margin_loss], ...)
 (name = LogitMarginLoss, instances = [logit_margin_loss], ...)
 (name = ModifiedHuberLoss, instances = [modified_huber_loss], ...)
 (name = PerceptronLoss, instances = [perceptron_loss], ...)
 (name = SigmoidLoss, instances = [sigmoid_loss], ...)
 (name = SmoothedL1HingeLoss, instances = [smoothed_l1_hinge_loss], ...)
 (name = ZeroOneLoss, instances = [zero_one_loss], ...)
 (name = HuberLoss, instances = [huber_loss], ...)
 (name = L1EpsilonInsLoss, instances = [l1_epsilon_ins_loss], ...)
 (name = L2EpsilonInsLoss, instances = [l2_epsilon_ins_loss], ...)
 (name = LPDistLoss, instances = [lp_dist_loss], ...)
 (name = LogitDistLoss, instances = [logit_dist_loss], ...)
 (name = PeriodicLoss, instances = [periodic_loss], ...)
 (name = QuantileLoss, instances = [quantile_loss], ...)
````

### Step 4. Evaluate the model performance

Naturally, MLJ provides boilerplate code for carrying out a model
evaluation with a lot less fuss. Let's repeat the performance
evaluation above and add an extra measure, `brier_score`:

````julia
evaluate!(mach, resampling=Holdout(fraction_train=0.7),
          measures=[cross_entropy, brier_score])
````

````
PerformanceEvaluation object with these fields:
  measure, measurement, operation, per_fold,
  per_observation, fitted_params_per_fold,
  report_per_fold, train_test_pairs
Extract:
┌────────────────────────────┬─────────────┬───────────┬──────────┐
│ measure                    │ measurement │ operation │ per_fold │
├────────────────────────────┼─────────────┼───────────┼──────────┤
│ LogLoss(tol = 2.22045e-16) │ 0.386       │ predict   │ [0.386]  │
│ BrierScore()               │ -0.217      │ predict   │ [-0.217] │
└────────────────────────────┴─────────────┴───────────┴──────────┘

````

Or applying cross-validation instead:

````julia
evaluate!(mach, resampling=CV(nfolds=6),
          measures=[cross_entropy, brier_score])
````

````
PerformanceEvaluation object with these fields:
  measure, measurement, operation, per_fold,
  per_observation, fitted_params_per_fold,
  report_per_fold, train_test_pairs
Extract:
┌────────────────────────────┬─────────────┬───────────┬──────────────────────────────────────────────────┐
│ measure                    │ measurement │ operation │ per_fold                                         │
├────────────────────────────┼─────────────┼───────────┼──────────────────────────────────────────────────┤
│ LogLoss(tol = 2.22045e-16) │ 0.402       │ predict   │ [0.355, 0.326, 0.516, 0.435, 0.444, 0.337]       │
│ BrierScore()               │ -0.229      │ predict   │ [-0.179, -0.185, -0.314, -0.259, -0.251, -0.183] │
└────────────────────────────┴─────────────┴───────────┴──────────────────────────────────────────────────┘

````

Or, Monte Carlo cross-validation (cross-validation repeated
randomized folds)

````julia
e = evaluate!(mach, resampling=CV(nfolds=6, rng=123),
              repeats=3,
              measures=[cross_entropy, brier_score])
````

````
PerformanceEvaluation object with these fields:
  measure, measurement, operation, per_fold,
  per_observation, fitted_params_per_fold,
  report_per_fold, train_test_pairs
Extract:
┌────────────────────────────┬─────────────┬───────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ measure                    │ measurement │ operation │ per_fold                                                                                                                                        │
├────────────────────────────┼─────────────┼───────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ LogLoss(tol = 2.22045e-16) │ 0.406       │ predict   │ [0.408, 0.358, 0.383, 0.476, 0.509, 0.428, 0.385, 0.406, 0.417, 0.36, 0.385, 0.361, 0.437, 0.368, 0.436, 0.388, 0.373, 0.432]                   │
│ BrierScore()               │ -0.227      │ predict   │ [-0.221, -0.186, -0.21, -0.267, -0.298, -0.248, -0.207, -0.248, -0.223, -0.196, -0.209, -0.191, -0.246, -0.206, -0.245, -0.222, -0.204, -0.263] │
└────────────────────────────┴─────────────┴───────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

````

One can access the following properties of the output `e` of an
evaluation: `measure`, `measurement`, `per_fold` (measurement for
each fold) and `per_observation` (measurement per observation, if
reported).

We finally note that you can restrict the rows of observations from
which train and test folds are drawn, by specifying `rows=...`. For
example, imagining the last 30% of target observations are `missing`
you might have a work-flow like this:

````julia
train, test = partition(eachindex(y), 0.7)
mach = machine(model, X, y)
evaluate!(mach, resampling=CV(nfolds=6),
          measures=[cross_entropy, brier_score],
          rows=train)     # cv estimate, resampling from `train`
fit!(mach, rows=train)    # re-train using all of `train` observations
predict(mach, rows=test); # and predict missing targets
nothing #hide
````

````
[ Info: Creating subsamples from a subset of all rows. 
Evaluating over 6 folds:  33%[========>                ]  ETA: 0:00:10[KEvaluating over 6 folds:  50%[============>            ]  ETA: 0:00:08[KEvaluating over 6 folds:  67%[================>        ]  ETA: 0:00:05[KEvaluating over 6 folds:  83%[====================>    ]  ETA: 0:00:03[KEvaluating over 6 folds: 100%[=========================] Time: 0:00:15[K
[ Info: Training Machine{NeuralNetworkClassifier{Short,…},…}.
Optimising neural net:  4%[>                        ]  ETA: 0:00:01[KOptimising neural net:  6%[=>                       ]  ETA: 0:00:02[KOptimising neural net:  8%[=>                       ]  ETA: 0:00:02[KOptimising neural net: 10%[==>                      ]  ETA: 0:00:02[KOptimising neural net: 12%[==>                      ]  ETA: 0:00:02[KOptimising neural net: 14%[===>                     ]  ETA: 0:00:02[KOptimising neural net: 16%[===>                     ]  ETA: 0:00:02[KOptimising neural net: 18%[====>                    ]  ETA: 0:00:02[KOptimising neural net: 20%[====>                    ]  ETA: 0:00:02[KOptimising neural net: 22%[=====>                   ]  ETA: 0:00:02[KOptimising neural net: 24%[=====>                   ]  ETA: 0:00:02[KOptimising neural net: 25%[======>                  ]  ETA: 0:00:02[KOptimising neural net: 27%[======>                  ]  ETA: 0:00:02[KOptimising neural net: 29%[=======>                 ]  ETA: 0:00:02[KOptimising neural net: 31%[=======>                 ]  ETA: 0:00:02[KOptimising neural net: 33%[========>                ]  ETA: 0:00:02[KOptimising neural net: 35%[========>                ]  ETA: 0:00:02[KOptimising neural net: 37%[=========>               ]  ETA: 0:00:02[KOptimising neural net: 39%[=========>               ]  ETA: 0:00:02[KOptimising neural net: 41%[==========>              ]  ETA: 0:00:02[KOptimising neural net: 43%[==========>              ]  ETA: 0:00:02[KOptimising neural net: 45%[===========>             ]  ETA: 0:00:02[KOptimising neural net: 47%[===========>             ]  ETA: 0:00:02[KOptimising neural net: 49%[============>            ]  ETA: 0:00:02[KOptimising neural net: 51%[============>            ]  ETA: 0:00:02[KOptimising neural net: 53%[=============>           ]  ETA: 0:00:01[KOptimising neural net: 55%[=============>           ]  ETA: 0:00:01[KOptimising neural net: 57%[==============>          ]  ETA: 0:00:01[KOptimising neural net: 59%[==============>          ]  ETA: 0:00:01[KOptimising neural net: 61%[===============>         ]  ETA: 0:00:01[KOptimising neural net: 63%[===============>         ]  ETA: 0:00:01[KOptimising neural net: 65%[================>        ]  ETA: 0:00:01[KOptimising neural net: 67%[================>        ]  ETA: 0:00:01[KOptimising neural net: 69%[=================>       ]  ETA: 0:00:01[KOptimising neural net: 71%[=================>       ]  ETA: 0:00:01[KOptimising neural net: 73%[==================>      ]  ETA: 0:00:01[KOptimising neural net: 75%[==================>      ]  ETA: 0:00:01[KOptimising neural net: 76%[===================>     ]  ETA: 0:00:01[KOptimising neural net: 78%[===================>     ]  ETA: 0:00:01[KOptimising neural net: 80%[====================>    ]  ETA: 0:00:01[KOptimising neural net: 82%[====================>    ]  ETA: 0:00:01[KOptimising neural net: 84%[=====================>   ]  ETA: 0:00:00[KOptimising neural net: 86%[=====================>   ]  ETA: 0:00:00[KOptimising neural net: 88%[======================>  ]  ETA: 0:00:00[KOptimising neural net: 90%[======================>  ]  ETA: 0:00:00[KOptimising neural net: 92%[=======================> ]  ETA: 0:00:00[KOptimising neural net: 94%[=======================> ]  ETA: 0:00:00[KOptimising neural net: 96%[========================>]  ETA: 0:00:00[KOptimising neural net: 98%[========================>]  ETA: 0:00:00[KOptimising neural net:100%[=========================] Time: 0:00:03[K

````

### On learning curves

Since our model is an iterative one, we might want to inspect the
out-of-sample performance as a function of the iteration
parameter. For this we can use the `learning_curve` function (which,
incidentally can be applied to any model hyper-parameter). This
starts by defining a one-dimensional range object for the parameter
(more on this when we discuss tuning in Part 4):

````julia
r = range(model, :epochs, lower=1, upper=50, scale=:log)
````

````
NumericRange(1 ≤ epochs ≤ 50; origin=25.5, unit=24.5) on log scale
````

````julia
curve = learning_curve(mach,
                       range=r,
                       resampling=Holdout(fraction_train=0.7), # (default)
                       measure=cross_entropy)

using Plots
plotly(size=(490,300))
plt=plot(curve.parameter_values, curve.measurements)
xlabel!(plt, "epochs")
ylabel!(plt, "cross entropy on holdout set")
plt
````

```@raw html
<script src="https://cdn.plot.ly/plotly-1.57.1.min.js"></script>    <div id="e5683faa-45fb-4948-ba54-ee096cf8e5b1" style="width:490px;height:300px;"></div>
    <script>
    
        var PLOT = document.getElementById('e5683faa-45fb-4948-ba54-ee096cf8e5b1');
    Plotly.plot(PLOT, [
    {
        "xaxis": "x",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y",
        "x": [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            13.0,
            15.0,
            17.0,
            19.0,
            22.0,
            25.0,
            29.0,
            33.0,
            38.0,
            44.0,
            50.0
        ],
        "showlegend": true,
        "mode": "lines",
        "name": "y1",
        "zmin": null,
        "legendgroup": "y1",
        "zmax": null,
        "line": {
            "color": "rgba(0, 154, 250, 1.000)",
            "shape": "linear",
            "dash": "solid",
            "width": 1
        },
        "y": [
            1.014482835586597,
            0.8906537188852359,
            0.80018892536832,
            0.7213789861219907,
            0.6560466894803221,
            0.6211108673666452,
            0.6027467415004643,
            0.5641339316414179,
            0.5695412837060354,
            0.5368794093704342,
            0.5321354713239868,
            0.506893812880953,
            0.4815506013025007,
            0.46761503880050437,
            0.43944016265777447,
            0.4371013523323308,
            0.4325800979592121,
            0.4102093175525475,
            0.3912220593381234,
            0.3862430222928144,
            0.37213387852132523,
            0.37284775983378593
        ],
        "type": "scatter"
    }
]
, {
    "showlegend": true,
    "xaxis": {
        "showticklabels": true,
        "gridwidth": 0.5,
        "tickvals": [
            0.0,
            10.0,
            20.0,
            30.0,
            40.0,
            50.0
        ],
        "range": [
            -0.47,
            51.47
        ],
        "domain": [
            0.09363561697644936,
            0.9919652900530291
        ],
        "mirror": false,
        "tickangle": 0,
        "showline": true,
        "ticktext": [
            "0",
            "10",
            "20",
            "30",
            "40",
            "50"
        ],
        "zeroline": false,
        "tickfont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "zerolinecolor": "rgba(0, 0, 0, 1.000)",
        "anchor": "y",
        "visible": true,
        "ticks": "inside",
        "tickmode": "array",
        "linecolor": "rgba(0, 0, 0, 1.000)",
        "showgrid": true,
        "title": "epochs",
        "gridcolor": "rgba(0, 0, 0, 0.100)",
        "titlefont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 15
        },
        "tickcolor": "rgb(0, 0, 0)",
        "type": "-"
    },
    "paper_bgcolor": "rgba(255, 255, 255, 1.000)",
    "annotations": [],
    "height": 300,
    "margin": {
        "l": 0,
        "b": 20,
        "r": 0,
        "t": 20
    },
    "plot_bgcolor": "rgba(255, 255, 255, 1.000)",
    "yaxis": {
        "showticklabels": true,
        "gridwidth": 0.5,
        "tickvals": [
            0.4,
            0.5,
            0.6000000000000001,
            0.7000000000000001,
            0.8,
            0.9,
            1.0
        ],
        "range": [
            0.3528634098093671,
            1.0337533042985552
        ],
        "domain": [
            0.10108632254301551,
            0.9868766404199475
        ],
        "mirror": false,
        "tickangle": 0,
        "showline": true,
        "ticktext": [
            "0.4",
            "0.5",
            "0.6",
            "0.7",
            "0.8",
            "0.9",
            "1.0"
        ],
        "zeroline": false,
        "tickfont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "zerolinecolor": "rgba(0, 0, 0, 1.000)",
        "anchor": "x",
        "visible": true,
        "ticks": "inside",
        "tickmode": "array",
        "linecolor": "rgba(0, 0, 0, 1.000)",
        "showgrid": true,
        "title": "cross entropy on holdout set",
        "gridcolor": "rgba(0, 0, 0, 0.100)",
        "titlefont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 15
        },
        "tickcolor": "rgb(0, 0, 0)",
        "type": "-"
    },
    "legend": {
        "yanchor": "auto",
        "xanchor": "auto",
        "bordercolor": "rgba(0, 0, 0, 1.000)",
        "bgcolor": "rgba(255, 255, 255, 1.000)",
        "borderwidth": 1,
        "tracegroupgap": 0,
        "y": 1.0,
        "font": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "title": {
            "font": {
                "color": "rgba(0, 0, 0, 1.000)",
                "family": "sans-serif",
                "size": 15
            },
            "text": ""
        },
        "traceorder": "normal",
        "x": 1.0
    },
    "width": 490
}
);

    
    </script>

```

We will return to learning curves when we look at tuning in Part 4.

### Resources for Part 2

- From the MLJ manual:
    - [Getting Started](https://alan-turing-institute.github.io/MLJ.jl/dev/getting_started/)
    - [Model Search](https://alan-turing-institute.github.io/MLJ.jl/dev/model_search/)
    - [Evaluating Performance](https://alan-turing-institute.github.io/MLJ.jl/dev/evaluating_model_performance/) (using `evaluate!`)
    - [Learning Curves](https://alan-turing-institute.github.io/MLJ.jl/dev/learning_curves/)
    - [Performance Measures](https://alan-turing-institute.github.io/MLJ.jl/dev/performance_measures/) (loss functions, scores, etc)
- From Data Science Tutorials:
    - [Choosing and evaluating a model](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/choosing-a-model/)
    - [Fit, predict, transform](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/fit-and-predict/)

### Exercises for Part 2

#### Exercise 4

(a) Identify all supervised MLJ models that can be applied (without
type coercion or one-hot encoding) to a supervised learning problem
with input features `X4` and target `y4` defined below:

````julia
import Distributions
poisson = Distributions.Poisson

age = 18 .+ 60*rand(10);
salary = coerce(rand(["small", "big", "huge"], 10), OrderedFactor);
levels!(salary, ["small", "big", "huge"]);
small = CategoricalValue("small", salary)
````

````
CategoricalValue{String, UInt32} "small" (1/3)
````

````julia
X4 = DataFrames.DataFrame(age=age, salary=salary)

n_devices(salary) = salary > small ? rand(poisson(1.3)) : rand(poisson(2.9))
y4 = [n_devices(row.salary) for row in eachrow(X4)]
````

````
10-element Vector{Int64}:
 1
 0
 2
 0
 3
 3
 1
 2
 0
 0
````

(b) What models can be applied if you coerce the salary to a
`Continuous` scitype?

#### Exercise 5 (unpack)

After evaluating the following ...

````julia
data = (a = [1, 2, 3, 4],
        b = rand(4),
        c = rand(4),
        d = coerce(["male", "female", "female", "male"], OrderedFactor));
pretty(data)
````

````
┌───────┬────────────┬────────────┬──────────────────────────────────┐
│ a     │ b          │ c          │ d                                │
│ Int64 │ Float64    │ Float64    │ CategoricalValue{String, UInt32} │
│ Count │ Continuous │ Continuous │ OrderedFactor{2}                 │
├───────┼────────────┼────────────┼──────────────────────────────────┤
│ 1     │ 0.402444   │ 0.603449   │ male                             │
│ 2     │ 0.0351777  │ 0.910565   │ female                           │
│ 3     │ 0.970689   │ 0.682197   │ female                           │
│ 4     │ 0.966209   │ 0.949514   │ male                             │
└───────┴────────────┴────────────┴──────────────────────────────────┘

````

````julia
using Tables
y, X, w = unpack(data,
                 ==(:a),
                 name -> elscitype(Tables.getcolumn(data, name)) == Continuous,
                 name -> true);
nothing #hide
````

...attempt to guess the evaluations of the following:

````julia
y
````

````
4-element Vector{Int64}:
 1
 2
 3
 4
````

````julia
pretty(X)
````

````
┌────────────┬────────────┐
│ b          │ c          │
│ Float64    │ Float64    │
│ Continuous │ Continuous │
├────────────┼────────────┤
│ 0.402444   │ 0.603449   │
│ 0.0351777  │ 0.910565   │
│ 0.970689   │ 0.682197   │
│ 0.966209   │ 0.949514   │
└────────────┴────────────┘

````

````julia
w
````

````
4-element CategoricalArray{String,1,UInt32}:
 "male"
 "female"
 "female"
 "male"
````

#### Exercise 6 (first steps in modeling Horse Colic)

(a) Suppose we want to use predict the `:outcome` variable in the
Horse Colic study introduced in Part 1, based on the remaining
variables that are `Continuous` (one-hot encoding categorical
variables is discussed later in Part 3) *while ignoring the others*.
Extract from the `horse` data set (defined in Part 1) appropriate
input features `X` and target variable `y`. (Do not, however,
randomize the observations.)

(b) Create a 70:30 `train`/`test` split of the data and train a
`LogisticClassifier` model, from the `MLJLinearModels` package, on
the `train` rows. Use `lambda=100` and default values for the
other hyper-parameters. (Although one would normally standardize
(whiten) the continuous features for this model, do not do so here.)
After training:

- (i) Recalling that a logistic classifier (aka logistic regressor) is
  a linear-based model learning a *vector* of coefficients for each
  feature (one coefficient for each target class), use the
  `fitted_params` method to find this vector of coefficients in the
  case of the `:pulse` feature. (You can convert a vector of pairs `v =
  [x1 => y1, x2 => y2, ...]` into a dictionary with `Dict(v)`.)

- (ii) Evaluate the `cross_entropy` performance on the `test`
  observations.

- &star;(iii) In how many `test` observations does the predicted
  probability of the observed class exceed 50%?

- (iv) Find the `misclassification_rate` in the `test`
  set. (*Hint.* As this measure is deterministic, you will either
  need to broadcast `mode` or use `predict_mode` instead of
  `predict`.)

(c) Instead use a `RandomForestClassifier` model from the
    `DecisionTree` package and:

- (i) Generate an appropriate learning curve to convince yourself
  that out-of-sample estimates of the `cross_entropy` loss do not
  substantially improve for `n_trees > 50`. Use default values for
  all other hyper-parameters, and feel free to use all available
  data to generate the curve.

- (ii) Fix `n_trees=90` and use `evaluate!` to obtain a 9-fold
  cross-validation estimate of the `cross_entropy`, restricting
  sub-sampling to the `train` observations.

- (iii) Now use *all* available data but set
  `resampling=Holdout(fraction_train=0.7)` to obtain a score you can
  compare with the `KNNClassifier` in part (b)(iii). Which model is
  better?

<a id='part-3-transformers-and-pipelines'></a>

## Part 3 - Transformers and Pipelines

### Transformers

Unsupervised models, which receive no target `y` during training,
always have a `transform` operation. They sometimes also support an
`inverse_transform` operation, with obvious meaning, and sometimes
support a `predict` operation (see the clustering example discussed
[here](https://alan-turing-institute.github.io/MLJ.jl/dev/transformers/#Transformers-that-also-predict-1)).
Otherwise, they are handled much like supervised models.

Here's a simple standardization example:

````julia
x = rand(100);
@show mean(x) std(x);
nothing #hide
````

````
mean(x) = 0.49113868534293464
std(x) = 0.2869983282501564

````

````julia
model = Standardizer() # a built-in model
mach = machine(model, x)
fit!(mach)
xhat = transform(mach, x);
@show mean(xhat) std(xhat);
nothing #hide
````

````
[ Info: Training Machine{Standardizer,…}.
mean(xhat) = -1.021405182655144e-16
std(xhat) = 0.9999999999999999

````

This particular model has an `inverse_transform`:

````julia
inverse_transform(mach, xhat) ≈ x
````

````
true
````

### Re-encoding the King County House data as continuous

For further illustrations of transformers, let's re-encode *all* of the
King County House input features (see [Ex
3](#ex-3-fixing-scitypes-in-a-table)) into a set of `Continuous`
features. We do this with the `ContinuousEncoder` model, which, by
default, will:

- one-hot encode all `Multiclass` features
- coerce all `OrderedFactor` features to `Continuous` ones
- coerce all `Count` features to `Continuous` ones (there aren't any)
- drop any remaining non-Continuous features (none of these either)

First, we reload the data and fix the scitypes (Exercise 3):

````julia
file = CSV.File(joinpath(DIR, "data", "house.csv"));
house = DataFrames.DataFrame(file);
coerce!(house, autotype(file));
coerce!(house, Count => Continuous, :zipcode => Multiclass);
schema(house)
````

````
┌───────────────┬───────────────────────────────────┬───────────────────┐
│ _.names       │ _.types                           │ _.scitypes        │
├───────────────┼───────────────────────────────────┼───────────────────┤
│ price         │ Float64                           │ Continuous        │
│ bedrooms      │ CategoricalValue{Int64, UInt32}   │ OrderedFactor{13} │
│ bathrooms     │ CategoricalValue{Float64, UInt32} │ OrderedFactor{30} │
│ sqft_living   │ Float64                           │ Continuous        │
│ sqft_lot      │ Float64                           │ Continuous        │
│ floors        │ CategoricalValue{Float64, UInt32} │ OrderedFactor{6}  │
│ waterfront    │ CategoricalValue{Int64, UInt32}   │ OrderedFactor{2}  │
│ view          │ CategoricalValue{Int64, UInt32}   │ OrderedFactor{5}  │
│ condition     │ CategoricalValue{Int64, UInt32}   │ OrderedFactor{5}  │
│ grade         │ CategoricalValue{Int64, UInt32}   │ OrderedFactor{12} │
│ sqft_above    │ Float64                           │ Continuous        │
│ sqft_basement │ Float64                           │ Continuous        │
│ yr_built      │ Float64                           │ Continuous        │
│ zipcode       │ CategoricalValue{Int64, UInt32}   │ Multiclass{70}    │
│ lat           │ Float64                           │ Continuous        │
│ long          │ Float64                           │ Continuous        │
│ sqft_living15 │ Float64                           │ Continuous        │
│ sqft_lot15    │ Float64                           │ Continuous        │
│ is_renovated  │ CategoricalValue{Bool, UInt32}    │ OrderedFactor{2}  │
└───────────────┴───────────────────────────────────┴───────────────────┘
_.nrows = 21613

````

````julia
y, X = unpack(house, ==(:price), name -> true, rng=123);
nothing #hide
````

Instantiate the unsupervised model (transformer):

````julia
encoder = ContinuousEncoder() # a built-in model; no need to @load it
````

````
ContinuousEncoder(
    drop_last = false,
    one_hot_ordered_factors = false)
````

Bind the model to the data and fit!

````julia
mach = machine(encoder, X) |> fit!;
nothing #hide
````

````
[ Info: Training Machine{ContinuousEncoder,…}.

````

Transform and inspect the result:

````julia
Xcont = transform(mach, X);
schema(Xcont)
````

````
┌────────────────┬─────────┬────────────┐
│ _.names        │ _.types │ _.scitypes │
├────────────────┼─────────┼────────────┤
│ bedrooms       │ Float64 │ Continuous │
│ bathrooms      │ Float64 │ Continuous │
│ sqft_living    │ Float64 │ Continuous │
│ sqft_lot       │ Float64 │ Continuous │
│ floors         │ Float64 │ Continuous │
│ waterfront     │ Float64 │ Continuous │
│ view           │ Float64 │ Continuous │
│ condition      │ Float64 │ Continuous │
│ grade          │ Float64 │ Continuous │
│ sqft_above     │ Float64 │ Continuous │
│ sqft_basement  │ Float64 │ Continuous │
│ yr_built       │ Float64 │ Continuous │
│ zipcode__98001 │ Float64 │ Continuous │
│ zipcode__98002 │ Float64 │ Continuous │
│ zipcode__98003 │ Float64 │ Continuous │
│ zipcode__98004 │ Float64 │ Continuous │
│ zipcode__98005 │ Float64 │ Continuous │
│ zipcode__98006 │ Float64 │ Continuous │
│ zipcode__98007 │ Float64 │ Continuous │
│ zipcode__98008 │ Float64 │ Continuous │
│ zipcode__98010 │ Float64 │ Continuous │
│ zipcode__98011 │ Float64 │ Continuous │
│ zipcode__98014 │ Float64 │ Continuous │
│ zipcode__98019 │ Float64 │ Continuous │
│ zipcode__98022 │ Float64 │ Continuous │
│ zipcode__98023 │ Float64 │ Continuous │
│ zipcode__98024 │ Float64 │ Continuous │
│ zipcode__98027 │ Float64 │ Continuous │
│ zipcode__98028 │ Float64 │ Continuous │
│ zipcode__98029 │ Float64 │ Continuous │
│ zipcode__98030 │ Float64 │ Continuous │
│ zipcode__98031 │ Float64 │ Continuous │
│ zipcode__98032 │ Float64 │ Continuous │
│ zipcode__98033 │ Float64 │ Continuous │
│ zipcode__98034 │ Float64 │ Continuous │
│ zipcode__98038 │ Float64 │ Continuous │
│ zipcode__98039 │ Float64 │ Continuous │
│ zipcode__98040 │ Float64 │ Continuous │
│ zipcode__98042 │ Float64 │ Continuous │
│ zipcode__98045 │ Float64 │ Continuous │
│ zipcode__98052 │ Float64 │ Continuous │
│ zipcode__98053 │ Float64 │ Continuous │
│ zipcode__98055 │ Float64 │ Continuous │
│ zipcode__98056 │ Float64 │ Continuous │
│ zipcode__98058 │ Float64 │ Continuous │
│ zipcode__98059 │ Float64 │ Continuous │
│ zipcode__98065 │ Float64 │ Continuous │
│ zipcode__98070 │ Float64 │ Continuous │
│ zipcode__98072 │ Float64 │ Continuous │
│ zipcode__98074 │ Float64 │ Continuous │
│ zipcode__98075 │ Float64 │ Continuous │
│ zipcode__98077 │ Float64 │ Continuous │
│ zipcode__98092 │ Float64 │ Continuous │
│ zipcode__98102 │ Float64 │ Continuous │
│ zipcode__98103 │ Float64 │ Continuous │
│ zipcode__98105 │ Float64 │ Continuous │
│ zipcode__98106 │ Float64 │ Continuous │
│ zipcode__98107 │ Float64 │ Continuous │
│ zipcode__98108 │ Float64 │ Continuous │
│ zipcode__98109 │ Float64 │ Continuous │
│ zipcode__98112 │ Float64 │ Continuous │
│ zipcode__98115 │ Float64 │ Continuous │
│ zipcode__98116 │ Float64 │ Continuous │
│ zipcode__98117 │ Float64 │ Continuous │
│ zipcode__98118 │ Float64 │ Continuous │
│ zipcode__98119 │ Float64 │ Continuous │
│ zipcode__98122 │ Float64 │ Continuous │
│ zipcode__98125 │ Float64 │ Continuous │
│ zipcode__98126 │ Float64 │ Continuous │
│ zipcode__98133 │ Float64 │ Continuous │
│ zipcode__98136 │ Float64 │ Continuous │
│ zipcode__98144 │ Float64 │ Continuous │
│ zipcode__98146 │ Float64 │ Continuous │
│ zipcode__98148 │ Float64 │ Continuous │
│ zipcode__98155 │ Float64 │ Continuous │
│ zipcode__98166 │ Float64 │ Continuous │
│ zipcode__98168 │ Float64 │ Continuous │
│ zipcode__98177 │ Float64 │ Continuous │
│ zipcode__98178 │ Float64 │ Continuous │
│ zipcode__98188 │ Float64 │ Continuous │
│ zipcode__98198 │ Float64 │ Continuous │
│ zipcode__98199 │ Float64 │ Continuous │
│ lat            │ Float64 │ Continuous │
│ long           │ Float64 │ Continuous │
│ sqft_living15  │ Float64 │ Continuous │
│ sqft_lot15     │ Float64 │ Continuous │
│ is_renovated   │ Float64 │ Continuous │
└────────────────┴─────────┴────────────┘
_.nrows = 21613

````

### More transformers

Here's how to list all of MLJ's unsupervised models:

````julia
models(m->!m.is_supervised)
````

````
57-element Vector{NamedTuple{(:name, :package_name, :is_supervised, :abstract_type, :deep_properties, :docstring, :fit_data_scitype, :hyperparameter_ranges, :hyperparameter_types, :hyperparameters, :implemented_methods, :inverse_transform_scitype, :is_pure_julia, :is_wrapper, :iteration_parameter, :load_path, :package_license, :package_url, :package_uuid, :predict_scitype, :prediction_type, :supports_class_weights, :supports_online, :supports_training_losses, :supports_weights, :transform_scitype, :input_scitype, :target_scitype, :output_scitype), T} where T<:Tuple}:
 (name = ABODDetector, package_name = OutlierDetectionNeighbors, ... )
 (name = ABODDetector, package_name = OutlierDetectionPython, ... )
 (name = AEDetector, package_name = OutlierDetectionNetworks, ... )
 (name = AffinityPropagation, package_name = ScikitLearn, ... )
 (name = AgglomerativeClustering, package_name = ScikitLearn, ... )
 (name = Birch, package_name = ScikitLearn, ... )
 (name = CBLOFDetector, package_name = OutlierDetectionPython, ... )
 (name = COFDetector, package_name = OutlierDetectionNeighbors, ... )
 (name = COFDetector, package_name = OutlierDetectionPython, ... )
 (name = COPODDetector, package_name = OutlierDetectionPython, ... )
 (name = ContinuousEncoder, package_name = MLJModels, ... )
 (name = DBSCAN, package_name = ScikitLearn, ... )
 (name = DNNDetector, package_name = OutlierDetectionNeighbors, ... )
 (name = FactorAnalysis, package_name = MultivariateStats, ... )
 (name = FeatureAgglomeration, package_name = ScikitLearn, ... )
 (name = FeatureSelector, package_name = MLJModels, ... )
 (name = FillImputer, package_name = MLJModels, ... )
 (name = GMMClusterer, package_name = BetaML, ... )
 (name = HBOSDetector, package_name = OutlierDetectionPython, ... )
 (name = ICA, package_name = MultivariateStats, ... )
 (name = IForestDetector, package_name = OutlierDetectionPython, ... )
 (name = KMeans, package_name = BetaML, ... )
 (name = KMeans, package_name = Clustering, ... )
 (name = KMeans, package_name = ParallelKMeans, ... )
 (name = KMeans, package_name = ScikitLearn, ... )
 (name = KMedoids, package_name = BetaML, ... )
 (name = KMedoids, package_name = Clustering, ... )
 (name = KNNDetector, package_name = OutlierDetectionNeighbors, ... )
 (name = KNNDetector, package_name = OutlierDetectionPython, ... )
 (name = KernelPCA, package_name = MultivariateStats, ... )
 (name = LMDDDetector, package_name = OutlierDetectionPython, ... )
 (name = LOCIDetector, package_name = OutlierDetectionPython, ... )
 (name = LODADetector, package_name = OutlierDetectionPython, ... )
 (name = LOFDetector, package_name = OutlierDetectionNeighbors, ... )
 (name = LOFDetector, package_name = OutlierDetectionPython, ... )
 (name = MCDDetector, package_name = OutlierDetectionPython, ... )
 (name = MeanShift, package_name = ScikitLearn, ... )
 (name = MiniBatchKMeans, package_name = ScikitLearn, ... )
 (name = MissingImputator, package_name = BetaML, ... )
 (name = OCSVMDetector, package_name = OutlierDetectionPython, ... )
 (name = OPTICS, package_name = ScikitLearn, ... )
 (name = OneClassSVM, package_name = LIBSVM, ... )
 (name = OneHotEncoder, package_name = MLJModels, ... )
 (name = PCA, package_name = MultivariateStats, ... )
 (name = PCADetector, package_name = OutlierDetectionPython, ... )
 (name = PPCA, package_name = MultivariateStats, ... )
 (name = RODDetector, package_name = OutlierDetectionPython, ... )
 (name = SODDetector, package_name = OutlierDetectionPython, ... )
 (name = SOSDetector, package_name = OutlierDetectionPython, ... )
 (name = SpectralClustering, package_name = ScikitLearn, ... )
 (name = Standardizer, package_name = MLJModels, ... )
 (name = TSVDTransformer, package_name = TSVD, ... )
 (name = UnivariateBoxCoxTransformer, package_name = MLJModels, ... )
 (name = UnivariateDiscretizer, package_name = MLJModels, ... )
 (name = UnivariateFillImputer, package_name = MLJModels, ... )
 (name = UnivariateStandardizer, package_name = MLJModels, ... )
 (name = UnivariateTimeTypeToContinuous, package_name = MLJModels, ... )
````

Some commonly used ones are built-in (do not require `@load`ing):

model type                  | does what?
----------------------------|----------------------------------------------
ContinuousEncoder | transform input table to a table of `Continuous` features (see above)
FeatureSelector | retain or dump selected features
FillImputer | impute missing values
OneHotEncoder | one-hot encoder `Multiclass` (and optionally `OrderedFactor`) features
Standardizer | standardize (whiten) a vector or all `Continuous` features of a table
UnivariateBoxCoxTransformer | apply a learned Box-Cox transformation to a vector
UnivariateDiscretizer | discretize a `Continuous` vector, and hence render its elscitypw `OrderedFactor`

In addition to "dynamic" transformers (ones that learn something
from the data and must be `fit!`) users can wrap ordinary functions
as transformers, and such *static* transformers can depend on
parameters, like the dynamic ones. See
[here](https://alan-turing-institute.github.io/MLJ.jl/dev/transformers/#Static-transformers-1)
for how to define your own static transformers.

### Pipelines

````julia
length(schema(Xcont).names)
````

````
87
````

Let's suppose that additionally we'd like to reduce the dimension of
our data.  A model that will do this is `PCA` from
`MultivariateStats`:

````julia
PCA = @load PCA
reducer = PCA()
````

````
PCA(
    maxoutdim = 0,
    method = :auto,
    pratio = 0.99,
    mean = nothing)
````

Now, rather simply repeating the work-flow above, applying the new
transformation to `Xcont`, we can combine both the encoding and the
dimension-reducing models into a single model, known as a
*pipeline*. While MLJ offers a powerful interface for composing
models in a variety of ways, we'll stick to these simplest class of
composite models for now. The easiest way to construct them is using
the `@pipeline` macro:

````julia
pipe = @pipeline encoder reducer
````

````
Pipeline589(
    continuous_encoder = ContinuousEncoder(
            drop_last = false,
            one_hot_ordered_factors = false),
    pca = PCA(
            maxoutdim = 0,
            method = :auto,
            pratio = 0.99,
            mean = nothing))
````

Notice that `pipe` is an *instance* of an automatically generated
type (called `Pipeline<some digits>`).

The new model behaves like any other transformer:

````julia
mach = machine(pipe, X)
fit!(mach)
Xsmall = transform(mach, X)
schema(Xsmall)
````

````
┌─────────┬─────────┬────────────┐
│ _.names │ _.types │ _.scitypes │
├─────────┼─────────┼────────────┤
│ x1      │ Float64 │ Continuous │
│ x2      │ Float64 │ Continuous │
└─────────┴─────────┴────────────┘
_.nrows = 21613

````

Want to combine this pre-processing with ridge regression?

````julia
RidgeRegressor = @load RidgeRegressor pkg=MLJLinearModels
rgs = RidgeRegressor()
pipe2 = @pipeline encoder reducer rgs
````

````
Pipeline597(
    continuous_encoder = ContinuousEncoder(
            drop_last = false,
            one_hot_ordered_factors = false),
    pca = PCA(
            maxoutdim = 0,
            method = :auto,
            pratio = 0.99,
            mean = nothing),
    ridge_regressor = RidgeRegressor(
            lambda = 1.0,
            fit_intercept = true,
            penalize_intercept = false,
            solver = nothing))
````

Now our pipeline is a supervised model, instead of a transformer,
whose performance we can evaluate:

````julia
mach = machine(pipe2, X, y)
evaluate!(mach, measure=mae, resampling=Holdout()) # CV(nfolds=6) is default
````

````
PerformanceEvaluation object with these fields:
  measure, measurement, operation, per_fold,
  per_observation, fitted_params_per_fold,
  report_per_fold, train_test_pairs
Extract:
┌─────────────────────┬─────────────┬───────────┬────────────┐
│ measure             │ measurement │ operation │ per_fold   │
├─────────────────────┼─────────────┼───────────┼────────────┤
│ MeanAbsoluteError() │ 234000.0    │ predict   │ [234000.0] │
└─────────────────────┴─────────────┴───────────┴────────────┘

````

### Training of composite models is "smart"

Now notice what happens if we train on all the data, then change a
regressor hyper-parameter and retrain:

````julia
fit!(mach)
````

````
Machine{Pipeline597,…} trained 2 times; caches data
  args: 
    1:	Source @822 ⏎ `Table{Union{AbstractVector{Continuous}, AbstractVector{Multiclass{70}}, AbstractVector{OrderedFactor{6}}, AbstractVector{OrderedFactor{13}}, AbstractVector{OrderedFactor{30}}, AbstractVector{OrderedFactor{5}}, AbstractVector{OrderedFactor{12}}, AbstractVector{OrderedFactor{2}}}}`
    2:	Source @258 ⏎ `AbstractVector{Continuous}`

````

````julia
pipe2.ridge_regressor.lambda = 0.1
fit!(mach)
````

````
Machine{Pipeline597,…} trained 3 times; caches data
  args: 
    1:	Source @822 ⏎ `Table{Union{AbstractVector{Continuous}, AbstractVector{Multiclass{70}}, AbstractVector{OrderedFactor{6}}, AbstractVector{OrderedFactor{13}}, AbstractVector{OrderedFactor{30}}, AbstractVector{OrderedFactor{5}}, AbstractVector{OrderedFactor{12}}, AbstractVector{OrderedFactor{2}}}}`
    2:	Source @258 ⏎ `AbstractVector{Continuous}`

````

Second time only the ridge regressor is retrained!

Mutate a hyper-parameter of the `PCA` model and every model except
the `ContinuousEncoder` (which comes before it will be retrained):

````julia
pipe2.pca.pratio = 0.9999
fit!(mach)
````

````
Machine{Pipeline597,…} trained 4 times; caches data
  args: 
    1:	Source @822 ⏎ `Table{Union{AbstractVector{Continuous}, AbstractVector{Multiclass{70}}, AbstractVector{OrderedFactor{6}}, AbstractVector{OrderedFactor{13}}, AbstractVector{OrderedFactor{30}}, AbstractVector{OrderedFactor{5}}, AbstractVector{OrderedFactor{12}}, AbstractVector{OrderedFactor{2}}}}`
    2:	Source @258 ⏎ `AbstractVector{Continuous}`

````

### Inspecting composite models

The dot syntax used above to change the values of *nested*
hyper-parameters is also useful when inspecting the learned
parameters and report generated when training a composite model:

````julia
fitted_params(mach).ridge_regressor
````

````
(coefs = [:x1 => -0.7328956348956878, :x2 => -0.1659056320291739, :x3 => 194.5951589082211, :x4 => 102.71301756136427],
 intercept = 540085.6428739978,)
````

````julia
report(mach).pca
````

````
(indim = 87,
 outdim = 4,
 tprincipalvar = 2.463215246230865e9,
 tresidualvar = 157533.26199626923,
 tvar = 2.4633727794928613e9,
 mean = [4.369869985656781, 8.45912182482765, 2079.8997362698374, 15106.967565816869, 1.988617961412113, 1.0075417572757137, 1.2343034284921113, 3.4094295100171195, 6.6569194466293435, 1788.3906907879516, 291.5090454818859, 1971.0051357978994, 0.01674917873502059, 0.009207421459306898, 0.012955165872391617, 0.01466709850552908, 0.007773099523434969, 0.023041687873039375, 0.006523851385740064, 0.013093971221024384, 0.004626844954425577, 0.009022347661129875, 0.005737287743487716, 0.008791005413408597, 0.010826817193355851, 0.02308795632258363, 0.0037477444130847174, 0.01906260121223338, 0.013093971221024384, 0.014852172303706102, 0.011844723083329478, 0.012677555175126082, 0.005783556193031971, 0.019987970203118495, 0.025216305001619397, 0.027298385231110906, 0.0023134224772127887, 0.013047702771480128, 0.025355110350252164, 0.010225327349280526, 0.02655809003840281, 0.018738722065423586, 0.012399944477860548, 0.018784990514967844, 0.021052144542636375, 0.021653634386711702, 0.01434321935871929, 0.005459677046222181, 0.012631286725581826, 0.020404386249016797, 0.016610373386387822, 0.009161153009762642, 0.016240225790033775, 0.0048581872021468565, 0.027853606625641975, 0.010595474945634571, 0.015499930597325684, 0.012307407578772035, 0.008605931615231573, 0.005043261000323879, 0.012446212927404802, 0.026974506084301113, 0.015268588349604404, 0.02558645259797344, 0.02350437236848193, 0.008513394716143062, 0.013417850367834173, 0.018970064313144866, 0.016379031138666542, 0.02285661407486235, 0.012168602230139268, 0.01587007819367973, 0.013325313468745662, 0.002637301624022579, 0.020635728496738073, 0.011752186184240966, 0.012446212927404802, 0.011798454633785222, 0.012122333780595013, 0.006292509138018785, 0.012955165872391617, 0.01466709850552908, 47.56005251931713, -122.21389640494186, 1986.552491556008, 12768.455651691113, 1.9577106371165502],
 principalvars = [2.1770715510450845e9, 2.8418139726430273e8, 1.6850160830643363e6, 277281.8384131831],)
````

### Incorporating target transformations

Next, suppose that instead of using the raw `:price` as the
training target, we want to use the log-price (a common practice in
dealing with house price data). However, suppose that we still want
to report final *predictions* on the original linear scale (and use
these for evaluation purposes). Then we supply appropriate functions
to key-word arguments `target` and `inverse`.

First we'll overload `log` and `exp` for broadcasting:

````julia
Base.log(v::AbstractArray) = log.(v)
Base.exp(v::AbstractArray) = exp.(v)
````

Now for the new pipeline:

````julia
pipe3 = @pipeline encoder reducer rgs target=log inverse=exp
mach = machine(pipe3, X, y)
evaluate!(mach, measure=mae)
````

````
PerformanceEvaluation object with these fields:
  measure, measurement, operation, per_fold,
  per_observation, fitted_params_per_fold,
  report_per_fold, train_test_pairs
Extract:
┌─────────────────────┬─────────────┬───────────┬──────────────────────────────────────────────────────────────┐
│ measure             │ measurement │ operation │ per_fold                                                     │
├─────────────────────┼─────────────┼───────────┼──────────────────────────────────────────────────────────────┤
│ MeanAbsoluteError() │ 162000.0    │ predict   │ [160000.0, 161000.0, 164000.0, 159000.0, 173000.0, 157000.0] │
└─────────────────────┴─────────────┴───────────┴──────────────────────────────────────────────────────────────┘

````

MLJ will also allow you to insert *learned* target
transformations. For example, we might want to apply
`Standardizer()` to the target, to standardize it, or
`UnivariateBoxCoxTransformer()` to make it look Gaussian. Then
instead of specifying a *function* for `target`, we specify a
unsupervised *model* (or model type). One does not specify `inverse`
because only models implementing `inverse_transform` are
allowed.

Let's see which of these two options results in a better outcome:

````julia
box = UnivariateBoxCoxTransformer(n=20)
stand = Standardizer()

pipe4 = @pipeline encoder reducer rgs target=box
mach = machine(pipe4, X, y)
evaluate!(mach, measure=mae)
````

````
PerformanceEvaluation object with these fields:
  measure, measurement, operation, per_fold,
  per_observation, fitted_params_per_fold,
  report_per_fold, train_test_pairs
Extract:
┌─────────────────────┬─────────────┬───────────┬────────────────────────────────────────────────────────────┐
│ measure             │ measurement │ operation │ per_fold                                                   │
├─────────────────────┼─────────────┼───────────┼────────────────────────────────────────────────────────────┤
│ MeanAbsoluteError() │ 479000.0    │ predict   │ [168000.0, 172000.0, 170000.0, 276000.0, 1.92e6, 167000.0] │
└─────────────────────┴─────────────┴───────────┴────────────────────────────────────────────────────────────┘

````

````julia
pipe4.target = stand
evaluate!(mach, measure=mae)
````

````
PerformanceEvaluation object with these fields:
  measure, measurement, operation, per_fold,
  per_observation, fitted_params_per_fold,
  report_per_fold, train_test_pairs
Extract:
┌─────────────────────┬─────────────┬───────────┬──────────────────────────────────────────────────────────────┐
│ measure             │ measurement │ operation │ per_fold                                                     │
├─────────────────────┼─────────────┼───────────┼──────────────────────────────────────────────────────────────┤
│ MeanAbsoluteError() │ 172000.0    │ predict   │ [173000.0, 171000.0, 172000.0, 172000.0, 176000.0, 166000.0] │
└─────────────────────┴─────────────┴───────────┴──────────────────────────────────────────────────────────────┘

````

### Resources for Part 3

- From the MLJ manual:
    - [Transformers and other unsupervised models](https://alan-turing-institute.github.io/MLJ.jl/dev/transformers/)
    - [Linear pipelines](https://alan-turing-institute.github.io/MLJ.jl/dev/composing_models/#Linear-pipelines-1)
- From Data Science Tutorials:
    - [Composing models](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/composing-models/)

### Exercises for Part 3

#### Exercise 7

Consider again the Horse Colic classification problem considered in
Exercise 6, but with all features, `Finite` and `Infinite`:

````julia
y, X = unpack(horse, ==(:outcome), name -> true);
schema(X)
````

````
┌─────────────────────────┬─────────────────────────────────┬──────────────────┐
│ _.names                 │ _.types                         │ _.scitypes       │
├─────────────────────────┼─────────────────────────────────┼──────────────────┤
│ surgery                 │ CategoricalValue{Int64, UInt32} │ Multiclass{2}    │
│ age                     │ CategoricalValue{Int64, UInt32} │ Multiclass{2}    │
│ rectal_temperature      │ Float64                         │ Continuous       │
│ pulse                   │ Float64                         │ Continuous       │
│ respiratory_rate        │ Float64                         │ Continuous       │
│ temperature_extremities │ CategoricalValue{Int64, UInt32} │ OrderedFactor{4} │
│ mucous_membranes        │ CategoricalValue{Int64, UInt32} │ Multiclass{6}    │
│ capillary_refill_time   │ CategoricalValue{Int64, UInt32} │ Multiclass{3}    │
│ pain                    │ CategoricalValue{Int64, UInt32} │ OrderedFactor{5} │
│ peristalsis             │ CategoricalValue{Int64, UInt32} │ OrderedFactor{4} │
│ abdominal_distension    │ CategoricalValue{Int64, UInt32} │ OrderedFactor{4} │
│ packed_cell_volume      │ Float64                         │ Continuous       │
│ total_protein           │ Float64                         │ Continuous       │
│ surgical_lesion         │ CategoricalValue{Int64, UInt32} │ OrderedFactor{2} │
│ cp_data                 │ CategoricalValue{Int64, UInt32} │ Multiclass{2}    │
└─────────────────────────┴─────────────────────────────────┴──────────────────┘
_.nrows = 366

````

(a) Define a pipeline that:
- uses `Standardizer` to ensure that features that are already
  continuous are centered at zero and have unit variance
- re-encodes the full set of features as `Continuous`, using
  `ContinuousEncoder`
- uses the `KMeans` clustering model from `Clustering.jl`
  to reduce the dimension of the feature space to `k=10`.
- trains a `EvoTreeClassifier` (a gradient tree boosting
  algorithm in `EvoTrees.jl`) on the reduced data, using
  `nrounds=50` and default values for the other
   hyper-parameters

(b) Evaluate the pipeline on all data, using 6-fold cross-validation
and `cross_entropy` loss.

&star;(c) Plot a learning curve which examines the effect on this loss
as the tree booster parameter `max_depth` varies from 2 to 10.

<a id='part-4-tuning-hyper-parameters'></a>

## Part 4 - Tuning Hyper-parameters

### Naive tuning of a single parameter

The most naive way to tune a single hyper-parameter is to use
`learning_curve`, which we already saw in Part 2. Let's see this in
the Horse Colic classification problem, in a case where the parameter
to be tuned is *nested* (because the model is a pipeline):

````julia
y, X = unpack(horse, ==(:outcome), name -> true);

LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels
model = @pipeline Standardizer ContinuousEncoder LogisticClassifier
mach = machine(model, X, y)
````

````
Machine{Pipeline626,…} trained 0 times; caches data
  args: 
    1:	Source @657 ⏎ `Table{Union{AbstractVector{Continuous}, AbstractVector{Multiclass{2}}, AbstractVector{Multiclass{6}}, AbstractVector{Multiclass{3}}, AbstractVector{OrderedFactor{2}}, AbstractVector{OrderedFactor{4}}, AbstractVector{OrderedFactor{5}}}}`
    2:	Source @219 ⏎ `AbstractVector{Multiclass{3}}`

````

````julia
r = range(model, :(logistic_classifier.lambda), lower = 1e-2, upper=100, scale=:log10)
````

````
NumericRange(0.01 ≤ logistic_classifier.lambda ≤ 100.0; origin=50.005, unit=49.995) on log10 scale
````

If you're curious, you can see what `lambda` values this range will
generate for a given resolution:

````julia
iterator(r, 5)
````

````
5-element Vector{Float64}:
   0.01
   0.1
   1.0
  10.0
 100.0
````

````julia
_, _, lambdas, losses = learning_curve(mach,
                                       range=r,
                                       resampling=CV(nfolds=6),
                                       resolution=30, # default
                                       measure=cross_entropy)
plt=plot(lambdas, losses, xscale=:log10)
xlabel!(plt, "lambda")
ylabel!(plt, "cross entropy using 6-fold CV")
````

```@raw html
<script src="https://cdn.plot.ly/plotly-1.57.1.min.js"></script>    <div id="f5ac8668-f4e3-49ea-b6d7-e666fe370ef2" style="width:490px;height:300px;"></div>
    <script>
    
        var PLOT = document.getElementById('f5ac8668-f4e3-49ea-b6d7-e666fe370ef2');
    Plotly.plot(PLOT, [
    {
        "xaxis": "x",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y",
        "x": [
            0.01,
            0.01373823795883263,
            0.018873918221350976,
            0.02592943797404667,
            0.035622478902624426,
            0.04893900918477494,
            0.06723357536499339,
            0.09236708571873861,
            0.12689610031679222,
            0.17433288221999882,
            0.23950266199874853,
            0.3290344562312668,
            0.4520353656360243,
            0.6210169418915615,
            0.8531678524172809,
            1.1721022975334803,
            1.6102620275609394,
            2.2122162910704493,
            3.0391953823131974,
            4.1753189365604015,
            5.736152510448679,
            7.880462815669913,
            10.826367338740546,
            14.87352107293511,
            20.433597178569418,
            28.072162039411772,
            38.56620421163472,
            52.98316906283707,
            72.7895384398315,
            100.0
        ],
        "showlegend": true,
        "mode": "lines",
        "name": "y1",
        "zmin": null,
        "legendgroup": "y1",
        "zmax": null,
        "line": {
            "color": "rgba(0, 154, 250, 1.000)",
            "shape": "linear",
            "dash": "solid",
            "width": 1
        },
        "y": [
            0.7891923451796305,
            0.7844863696840602,
            0.778999822562365,
            0.7727600314717838,
            0.7658754262369921,
            0.7585525943150301,
            0.7511401783949944,
            0.7441599776341977,
            0.7382939645858905,
            0.7332915072986833,
            0.7287462716265534,
            0.7243206491772539,
            0.7198473573335842,
            0.7152827815676913,
            0.7106774825457429,
            0.7061590906270783,
            0.701916755401511,
            0.6981827458551044,
            0.6952109463634933,
            0.6932536230665008,
            0.6925391259976147,
            0.693254492941192,
            0.6955375670021003,
            0.6994813973604733,
            0.7051485338137056,
            0.7125868336911289,
            0.7218359389293543,
            0.7329175692154145,
            0.7458121582971914,
            0.760433698653428
        ],
        "type": "scatter"
    }
]
, {
    "showlegend": true,
    "xaxis": {
        "showticklabels": true,
        "gridwidth": 0.5,
        "tickvals": [
            0.01,
            0.1,
            1.0,
            10.0,
            100.0
        ],
        "range": [
            -2.12,
            2.12
        ],
        "domain": [
            0.11177620654561035,
            0.9919652900530291
        ],
        "mirror": false,
        "tickangle": 0,
        "showline": true,
        "ticktext": [
            "10<sup>−2</sup>",
            "10<sup>−1</sup>",
            "10<sup>0</sup>",
            "10<sup>1</sup>",
            "10<sup>2</sup>"
        ],
        "zeroline": false,
        "tickfont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "zerolinecolor": "rgba(0, 0, 0, 1.000)",
        "anchor": "y",
        "visible": true,
        "ticks": "inside",
        "tickmode": "array",
        "linecolor": "rgba(0, 0, 0, 1.000)",
        "showgrid": true,
        "title": "lambda",
        "gridcolor": "rgba(0, 0, 0, 0.100)",
        "titlefont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 15
        },
        "tickcolor": "rgb(0, 0, 0)",
        "type": "log"
    },
    "paper_bgcolor": "rgba(255, 255, 255, 1.000)",
    "annotations": [],
    "height": 300,
    "margin": {
        "l": 0,
        "b": 20,
        "r": 0,
        "t": 20
    },
    "plot_bgcolor": "rgba(255, 255, 255, 1.000)",
    "yaxis": {
        "showticklabels": true,
        "gridwidth": 0.5,
        "tickvals": [
            0.7000000000000001,
            0.72,
            0.74,
            0.76,
            0.78
        ],
        "range": [
            0.6896395294221542,
            0.792091941755091
        ],
        "domain": [
            0.10108632254301551,
            0.9868766404199475
        ],
        "mirror": false,
        "tickangle": 0,
        "showline": true,
        "ticktext": [
            "0.70",
            "0.72",
            "0.74",
            "0.76",
            "0.78"
        ],
        "zeroline": false,
        "tickfont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "zerolinecolor": "rgba(0, 0, 0, 1.000)",
        "anchor": "x",
        "visible": true,
        "ticks": "inside",
        "tickmode": "array",
        "linecolor": "rgba(0, 0, 0, 1.000)",
        "showgrid": true,
        "title": "cross entropy using 6-fold CV",
        "gridcolor": "rgba(0, 0, 0, 0.100)",
        "titlefont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 15
        },
        "tickcolor": "rgb(0, 0, 0)",
        "type": "-"
    },
    "legend": {
        "yanchor": "auto",
        "xanchor": "auto",
        "bordercolor": "rgba(0, 0, 0, 1.000)",
        "bgcolor": "rgba(255, 255, 255, 1.000)",
        "borderwidth": 1,
        "tracegroupgap": 0,
        "y": 1.0,
        "font": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "title": {
            "font": {
                "color": "rgba(0, 0, 0, 1.000)",
                "family": "sans-serif",
                "size": 15
            },
            "text": ""
        },
        "traceorder": "normal",
        "x": 1.0
    },
    "width": 490
}
);

    
    </script>

```

````julia
best_lambda = lambdas[argmin(losses)]
````

````
5.736152510448679
````

### Self tuning models

A more sophisticated way to view hyper-parameter tuning (inspired by
MLR) is as a model *wrapper*. The wrapped model is a new model in
its own right and when you fit it, it tunes specified
hyper-parameters of the model being wrapped, before training on all
supplied data. Calling `predict` on the wrapped model is like
calling `predict` on the original model, but with the
hyper-parameters already optimized.

In other words, we can think of the wrapped model as a "self-tuning"
version of the original.

We now create a self-tuning version of the pipeline above, adding a
parameter from the `ContinuousEncoder` to the parameters we want
optimized.

First, let's choose a tuning strategy (from [these
options](https://github.com/juliaai/MLJTuning.jl#what-is-provided-here)). MLJ
supports ordinary `Grid` search (query `?Grid` for
details). However, as the utility of `Grid` search is limited to a
small number of parameters, and as `Grid` searches are demonstrated
elsewhere (see the [resources below](#resources-for-part-4)) we'll
demonstrate `RandomSearch` here:

````julia
tuning = RandomSearch(rng=123)
````

````
RandomSearch(
    bounded = Distributions.Uniform,
    positive_unbounded = Distributions.Gamma,
    other = Distributions.Normal,
    rng = MersenneTwister(123))
````

In this strategy each parameter is sampled according to a
pre-specified prior distribution that is fit to the one-dimensional
range object constructed using `range` as before. While one has a
lot of control over the specification of the priors (run
`?RandomSearch` for details) we'll let the algorithm generate these
priors automatically.

#### Unbounded ranges and sampling

In MLJ a range does not have to be bounded. In a `RandomSearch` a
positive unbounded range is sampled using a `Gamma` distribution, by
default:

````julia
r = range(model,
          :(logistic_classifier.lambda),
          lower=0,
          origin=6,
          unit=5,
          scale=:log10)
````

````
NumericRange(0.0 ≤ logistic_classifier.lambda ≤ Inf; origin=6.0, unit=5.0) on log10 scale
````

The `scale` in a range makes no in a `RandomSearch` (unless it is a
function) but this will effect later plots but it does effect the
later plots.

Let's see what sampling using a Gamma distribution is going to mean
for this range:

````julia
import Distributions
sampler_r = sampler(r, Distributions.Gamma)
histogram(rand(sampler_r, 10000), nbins=50)
````

```@raw html
<script src="https://cdn.plot.ly/plotly-1.57.1.min.js"></script>    <div id="1422bacc-5d8b-41d7-9ee1-5bd3f22214ff" style="width:490px;height:300px;"></div>
    <script>
    
        var PLOT = document.getElementById('1422bacc-5d8b-41d7-9ee1-5bd3f22214ff');
    Plotly.plot(PLOT, [
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0
        ],
        "showlegend": true,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            856.0,
            0.0,
            0.0,
            856.0,
            856.0,
            856.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            1.0,
            1.0,
            2.0,
            2.0,
            1.0,
            1.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            1167.0,
            0.0,
            0.0,
            1167.0,
            1167.0,
            1167.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            2.0,
            2.0,
            3.0,
            3.0,
            2.0,
            2.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            1198.0,
            0.0,
            0.0,
            1198.0,
            1198.0,
            1198.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            3.0,
            3.0,
            4.0,
            4.0,
            3.0,
            3.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            1107.0,
            0.0,
            0.0,
            1107.0,
            1107.0,
            1107.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            4.0,
            4.0,
            5.0,
            5.0,
            4.0,
            4.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            966.0,
            0.0,
            0.0,
            966.0,
            966.0,
            966.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            5.0,
            5.0,
            6.0,
            6.0,
            5.0,
            5.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            843.0,
            0.0,
            0.0,
            843.0,
            843.0,
            843.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            6.0,
            6.0,
            7.0,
            7.0,
            6.0,
            6.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            674.0,
            0.0,
            0.0,
            674.0,
            674.0,
            674.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            7.0,
            7.0,
            8.0,
            8.0,
            7.0,
            7.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            591.0,
            0.0,
            0.0,
            591.0,
            591.0,
            591.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            8.0,
            8.0,
            9.0,
            9.0,
            8.0,
            8.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            476.0,
            0.0,
            0.0,
            476.0,
            476.0,
            476.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            9.0,
            9.0,
            10.0,
            10.0,
            9.0,
            9.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            381.0,
            0.0,
            0.0,
            381.0,
            381.0,
            381.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            10.0,
            10.0,
            11.0,
            11.0,
            10.0,
            10.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            326.0,
            0.0,
            0.0,
            326.0,
            326.0,
            326.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            11.0,
            11.0,
            12.0,
            12.0,
            11.0,
            11.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            261.0,
            0.0,
            0.0,
            261.0,
            261.0,
            261.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            12.0,
            12.0,
            13.0,
            13.0,
            12.0,
            12.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            221.0,
            0.0,
            0.0,
            221.0,
            221.0,
            221.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            13.0,
            13.0,
            14.0,
            14.0,
            13.0,
            13.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            183.0,
            0.0,
            0.0,
            183.0,
            183.0,
            183.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            14.0,
            14.0,
            15.0,
            15.0,
            14.0,
            14.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            138.0,
            0.0,
            0.0,
            138.0,
            138.0,
            138.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            15.0,
            15.0,
            16.0,
            16.0,
            15.0,
            15.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            124.0,
            0.0,
            0.0,
            124.0,
            124.0,
            124.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            16.0,
            16.0,
            17.0,
            17.0,
            16.0,
            16.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            108.0,
            0.0,
            0.0,
            108.0,
            108.0,
            108.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            17.0,
            17.0,
            18.0,
            18.0,
            17.0,
            17.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            74.0,
            0.0,
            0.0,
            74.0,
            74.0,
            74.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            18.0,
            18.0,
            19.0,
            19.0,
            18.0,
            18.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            62.0,
            0.0,
            0.0,
            62.0,
            62.0,
            62.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            19.0,
            19.0,
            20.0,
            20.0,
            19.0,
            19.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            54.0,
            0.0,
            0.0,
            54.0,
            54.0,
            54.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            20.0,
            20.0,
            21.0,
            21.0,
            20.0,
            20.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            25.0,
            0.0,
            0.0,
            25.0,
            25.0,
            25.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            21.0,
            21.0,
            22.0,
            22.0,
            21.0,
            21.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            29.0,
            0.0,
            0.0,
            29.0,
            29.0,
            29.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            22.0,
            22.0,
            23.0,
            23.0,
            22.0,
            22.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            31.0,
            0.0,
            0.0,
            31.0,
            31.0,
            31.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            23.0,
            23.0,
            24.0,
            24.0,
            23.0,
            23.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            21.0,
            0.0,
            0.0,
            21.0,
            21.0,
            21.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            24.0,
            24.0,
            25.0,
            25.0,
            24.0,
            24.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            16.0,
            0.0,
            0.0,
            16.0,
            16.0,
            16.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            25.0,
            25.0,
            26.0,
            26.0,
            25.0,
            25.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            16.0,
            0.0,
            0.0,
            16.0,
            16.0,
            16.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            26.0,
            26.0,
            27.0,
            27.0,
            26.0,
            26.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            11.0,
            0.0,
            0.0,
            11.0,
            11.0,
            11.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            27.0,
            27.0,
            28.0,
            28.0,
            27.0,
            27.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            9.0,
            0.0,
            0.0,
            9.0,
            9.0,
            9.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            28.0,
            28.0,
            29.0,
            29.0,
            28.0,
            28.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            6.0,
            0.0,
            0.0,
            6.0,
            6.0,
            6.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            29.0,
            29.0,
            30.0,
            30.0,
            29.0,
            29.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            7.0,
            0.0,
            0.0,
            7.0,
            7.0,
            7.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            30.0,
            30.0,
            31.0,
            31.0,
            30.0,
            30.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            6.0,
            0.0,
            0.0,
            6.0,
            6.0,
            6.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            31.0,
            31.0,
            32.0,
            32.0,
            31.0,
            31.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            32.0,
            32.0,
            33.0,
            33.0,
            32.0,
            32.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            2.0,
            0.0,
            0.0,
            2.0,
            2.0,
            2.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            33.0,
            33.0,
            34.0,
            34.0,
            33.0,
            33.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            1.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            34.0,
            34.0,
            35.0,
            35.0,
            34.0,
            34.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            2.0,
            0.0,
            0.0,
            2.0,
            2.0,
            2.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            35.0,
            35.0,
            36.0,
            36.0,
            35.0,
            35.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            2.0,
            0.0,
            0.0,
            2.0,
            2.0,
            2.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            36.0,
            36.0,
            37.0,
            37.0,
            36.0,
            36.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            1.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            37.0,
            37.0,
            38.0,
            38.0,
            37.0,
            37.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            38.0,
            38.0,
            39.0,
            39.0,
            38.0,
            38.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            1.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            39.0,
            39.0,
            40.0,
            40.0,
            39.0,
            39.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            1.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            40.0,
            40.0,
            41.0,
            41.0,
            40.0,
            40.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            2.0,
            0.0,
            0.0,
            2.0,
            2.0,
            2.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            41.0,
            41.0,
            42.0,
            42.0,
            41.0,
            41.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            42.0,
            42.0,
            43.0,
            43.0,
            42.0,
            42.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            43.0,
            43.0,
            44.0,
            44.0,
            43.0,
            43.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            1.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y",
        "x": [
            0.5,
            1.5,
            2.5,
            3.5,
            4.5,
            5.5,
            6.5,
            7.5,
            8.5,
            9.5,
            10.5,
            11.5,
            12.5,
            13.5,
            14.5,
            15.5,
            16.5,
            17.5,
            18.5,
            19.5,
            20.5,
            21.5,
            22.5,
            23.5,
            24.5,
            25.5,
            26.5,
            27.5,
            28.5,
            29.5,
            30.5,
            31.5,
            32.5,
            33.5,
            34.5,
            35.5,
            36.5,
            37.5,
            38.5,
            39.5,
            40.5,
            41.5,
            42.5,
            43.5
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "y1",
        "zmin": null,
        "legendgroup": "y1",
        "marker": {
            "symbol": "circle",
            "color": "rgba(0, 154, 250, 0.000)",
            "line": {
                "color": "rgba(0, 0, 0, 0.000)",
                "width": 1
            },
            "size": 0
        },
        "zmax": null,
        "y": [
            856.0,
            1167.0,
            1198.0,
            1107.0,
            966.0,
            843.0,
            674.0,
            591.0,
            476.0,
            381.0,
            326.0,
            261.0,
            221.0,
            183.0,
            138.0,
            124.0,
            108.0,
            74.0,
            62.0,
            54.0,
            25.0,
            29.0,
            31.0,
            21.0,
            16.0,
            16.0,
            11.0,
            9.0,
            6.0,
            7.0,
            6.0,
            0.0,
            2.0,
            1.0,
            2.0,
            2.0,
            1.0,
            0.0,
            1.0,
            1.0,
            2.0,
            0.0,
            0.0,
            1.0
        ],
        "type": "scatter"
    }
]
, {
    "showlegend": true,
    "xaxis": {
        "showticklabels": true,
        "gridwidth": 0.5,
        "tickvals": [
            0.0,
            10.0,
            20.0,
            30.0,
            40.0
        ],
        "range": [
            -2.7192,
            46.7192
        ],
        "domain": [
            0.0805970682236149,
            0.991965290053029
        ],
        "mirror": false,
        "tickangle": 0,
        "showline": true,
        "ticktext": [
            "0",
            "10",
            "20",
            "30",
            "40"
        ],
        "zeroline": false,
        "tickfont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "zerolinecolor": "rgba(0, 0, 0, 1.000)",
        "anchor": "y",
        "visible": true,
        "ticks": "inside",
        "tickmode": "array",
        "linecolor": "rgba(0, 0, 0, 1.000)",
        "showgrid": true,
        "title": "",
        "gridcolor": "rgba(0, 0, 0, 0.100)",
        "titlefont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 15
        },
        "tickcolor": "rgb(0, 0, 0)",
        "type": "-"
    },
    "paper_bgcolor": "rgba(255, 255, 255, 1.000)",
    "annotations": [],
    "height": 300,
    "margin": {
        "l": 0,
        "b": 20,
        "r": 0,
        "t": 20
    },
    "plot_bgcolor": "rgba(255, 255, 255, 1.000)",
    "yaxis": {
        "showticklabels": true,
        "gridwidth": 0.5,
        "tickvals": [
            0.0,
            250.0,
            500.0,
            750.0,
            1000.0
        ],
        "range": [
            -35.94,
            1233.94
        ],
        "domain": [
            0.050160396617089535,
            0.9868766404199475
        ],
        "mirror": false,
        "tickangle": 0,
        "showline": true,
        "ticktext": [
            "0",
            "250",
            "500",
            "750",
            "1000"
        ],
        "zeroline": false,
        "tickfont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "zerolinecolor": "rgba(0, 0, 0, 1.000)",
        "anchor": "x",
        "visible": true,
        "ticks": "inside",
        "tickmode": "array",
        "linecolor": "rgba(0, 0, 0, 1.000)",
        "showgrid": true,
        "title": "",
        "gridcolor": "rgba(0, 0, 0, 0.100)",
        "titlefont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 15
        },
        "tickcolor": "rgb(0, 0, 0)",
        "type": "-"
    },
    "legend": {
        "yanchor": "auto",
        "xanchor": "auto",
        "bordercolor": "rgba(0, 0, 0, 1.000)",
        "bgcolor": "rgba(255, 255, 255, 1.000)",
        "borderwidth": 1,
        "tracegroupgap": 0,
        "y": 1.0,
        "font": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "title": {
            "font": {
                "color": "rgba(0, 0, 0, 1.000)",
                "family": "sans-serif",
                "size": 15
            },
            "text": ""
        },
        "traceorder": "normal",
        "x": 1.0
    },
    "width": 490
}
);

    
    </script>

```

The second parameter that we'll add to this is *nominal* (finite) and, by
default, will be sampled uniformly. Since it is nominal, we specify
`values` instead of `upper` and `lower` bounds:

````julia
s  = range(model, :(continuous_encoder.one_hot_ordered_factors),
           values = [true, false])
````

````
NominalRange(continuous_encoder.one_hot_ordered_factors = true, false)
````

#### The tuning wrapper

Now for the wrapper, which is an instance of `TunedModel`:

````julia
tuned_model = TunedModel(model=model,
                         ranges=[r, s],
                         resampling=CV(nfolds=6),
                         measures=cross_entropy,
                         tuning=tuning,
                         n=15)
````

````
ProbabilisticTunedModel(
    model = Pipeline626(
            standardizer = Standardizer,
            continuous_encoder = ContinuousEncoder,
            logistic_classifier = LogisticClassifier),
    tuning = RandomSearch(
            bounded = Distributions.Uniform,
            positive_unbounded = Distributions.Gamma,
            other = Distributions.Normal,
            rng = MersenneTwister(123)),
    resampling = CV(
            nfolds = 6,
            shuffle = false,
            rng = Random._GLOBAL_RNG()),
    measure = LogLoss(tol = 2.220446049250313e-16),
    weights = nothing,
    operation = nothing,
    range = MLJBase.ParamRange[NumericRange(0.0 ≤ logistic_classifier.lambda ≤ Inf; origin=6.0, unit=5.0) on log10 scale, NominalRange(continuous_encoder.one_hot_ordered_factors = true, false)],
    selection_heuristic = MLJTuning.NaiveSelection(nothing),
    train_best = true,
    repeats = 1,
    n = 15,
    acceleration = CPU1{Nothing}(nothing),
    acceleration_resampling = CPU1{Nothing}(nothing),
    check_measure = true,
    cache = true)
````

We can apply the `fit!/predict` work-flow to `tuned_model` just as
for any other model:

````julia
tuned_mach = machine(tuned_model, X, y);
fit!(tuned_mach);
predict(tuned_mach, rows=1:3)
````

````
3-element MLJBase.UnivariateFiniteVector{Multiclass{3}, Int64, UInt32, Float64}:
 UnivariateFinite{Multiclass{3}}(1=>0.631, 2=>0.22, 3=>0.149)
 UnivariateFinite{Multiclass{3}}(1=>0.844, 2=>0.00889, 3=>0.147)
 UnivariateFinite{Multiclass{3}}(1=>0.967, 2=>0.0128, 3=>0.0199)
````

The outcomes of the tuning can be inspected from a detailed
report. For example, we have:

````julia
rep = report(tuned_mach);
rep.best_model
````

````
Pipeline626(
    standardizer = Standardizer(
            features = Symbol[],
            ignore = false,
            ordered_factor = false,
            count = false),
    continuous_encoder = ContinuousEncoder(
            drop_last = false,
            one_hot_ordered_factors = true),
    logistic_classifier = LogisticClassifier(
            lambda = 5.100906526305173,
            gamma = 0.0,
            penalty = :l2,
            fit_intercept = true,
            penalize_intercept = false,
            solver = nothing))
````

By default, sampling of a bounded range is uniform. Lets

In the special case of two-parameters, you can also plot the results:

````julia
plot(tuned_mach)
````

```@raw html
<script src="https://cdn.plot.ly/plotly-1.57.1.min.js"></script>    <div id="07ff1036-908f-47ea-a8c5-540e66fa27e3" style="width:550px;height:500px;"></div>
    <script>
    
        var PLOT = document.getElementById('07ff1036-908f-47ea-a8c5-540e66fa27e3');
    Plotly.plot(PLOT, [
    {
        "xaxis": "x1",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y1",
        "x": [
            0.5,
            0.5,
            1.5,
            0.5,
            1.5,
            1.5,
            0.5,
            0.5,
            1.5,
            0.5,
            1.5,
            0.5,
            1.5,
            1.5,
            0.5
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": null,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(0, 154, 250, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 8
        },
        "zmax": null,
        "y": [
            0.6947237604644253,
            0.6970198418460459,
            0.6762908837231785,
            0.7052765729958649,
            0.683090585511975,
            0.7078735043732424,
            0.6926395376343057,
            0.6960286166265505,
            0.682336182063706,
            0.6948135255623834,
            0.6766358652989815,
            0.6934002697669785,
            0.6837802628953038,
            0.7047604278552547,
            0.6984981284720697
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x4",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y4",
        "x": [
            9.892724685972443,
            2.478099735316854,
            5.100906526305173,
            1.2494270258041364,
            2.863292426229041,
            0.9590656995775001,
            5.118949091319244,
            11.362293052081634,
            2.9982944683087203,
            3.201864395531346,
            4.850035558102979,
            4.041917014605834,
            2.7496839612983868,
            1.081200674187863,
            2.148558024798771
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": null,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(0, 154, 250, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 8
        },
        "zmax": null,
        "y": [
            0.6947237604644253,
            0.6970198418460459,
            0.6762908837231785,
            0.7052765729958649,
            0.683090585511975,
            0.7078735043732424,
            0.6926395376343057,
            0.6960286166265505,
            0.682336182063706,
            0.6948135255623834,
            0.6766358652989815,
            0.6934002697669785,
            0.6837802628953038,
            0.7047604278552547,
            0.6984981284720697
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            0.5
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 0.6762908837231785,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(216, 76, 62, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 20.765030195649757
        },
        "zmax": 0.7078735043732424,
        "y": [
            9.892724685972443
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            0.5
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 0.6762908837231785,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(235, 101, 40, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 23.08496184543145
        },
        "zmax": 0.7078735043732424,
        "y": [
            2.478099735316854
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            1.5
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 0.6762908837231785,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(0, 0, 4, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 2.0
        },
        "zmax": 0.7078735043732424,
        "y": [
            5.100906526305173
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            0.5
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 0.6762908837231785,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(244, 223, 83, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 31.396042476394992
        },
        "zmax": 0.7078735043732424,
        "y": [
            1.2494270258041364
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            1.5
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 0.6762908837231785,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(72, 11, 106, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 8.95165057117062
        },
        "zmax": 0.7078735043732424,
        "y": [
            2.863292426229041
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            1.5
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 0.6762908837231785,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(252, 255, 164, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 34.0
        },
        "zmax": 0.7078735043732424,
        "y": [
            0.9590656995775001
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            0.5
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 0.6762908837231785,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(193, 58, 80, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 18.655836073052697
        },
        "zmax": 0.7078735043732424,
        "y": [
            5.118949091319244
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            0.5
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 0.6762908837231785,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(227, 89, 50, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 22.08390983889309
        },
        "zmax": 0.7078735043732424,
        "y": [
            11.362293052081634
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            1.5
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 0.6762908837231785,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(62, 9, 102, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 8.182100545432792
        },
        "zmax": 0.7078735043732424,
        "y": [
            2.9982944683087203
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            0.5
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 0.6762908837231785,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(217, 77, 62, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 20.855799663387554
        },
        "zmax": 0.7078735043732424,
        "y": [
            3.201864395531346
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            1.5
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 0.6762908837231785,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(1, 1, 8, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 2.3535298958525988
        },
        "zmax": 0.7078735043732424,
        "y": [
            4.850035558102979
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            0.5
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 0.6762908837231785,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(202, 64, 74, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 19.426049893748644
        },
        "zmax": 0.7078735043732424,
        "y": [
            4.041917014605834
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            1.5
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 0.6762908837231785,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(81, 14, 109, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 9.654803256027947
        },
        "zmax": 0.7078735043732424,
        "y": [
            2.7496839612983868
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            1.5
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 0.6762908837231785,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(246, 215, 69, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 30.877930087049783
        },
        "zmax": 0.7078735043732424,
        "y": [
            1.081200674187863
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            0.5
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 0.6762908837231785,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(244, 120, 25, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 24.57658245035529
        },
        "zmax": 0.7078735043732424,
        "y": [
            2.148558024798771
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            0.5
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 0.6762908837231785,
        "marker": {
            "color": [
                0.5
            ],
            "cmin": 0.6762908837231785,
            "opacity": 1.0e-10,
            "size": 1.0e-10,
            "colorscale": [
                [
                    0.0,
                    "rgba(0, 0, 4, 1.000)"
                ],
                [
                    0.00392156862745098,
                    "rgba(1, 0, 5, 1.000)"
                ],
                [
                    0.00784313725490196,
                    "rgba(1, 1, 6, 1.000)"
                ],
                [
                    0.011764705882352941,
                    "rgba(1, 1, 8, 1.000)"
                ],
                [
                    0.01568627450980392,
                    "rgba(2, 1, 10, 1.000)"
                ],
                [
                    0.0196078431372549,
                    "rgba(2, 2, 12, 1.000)"
                ],
                [
                    0.023529411764705882,
                    "rgba(2, 2, 14, 1.000)"
                ],
                [
                    0.027450980392156862,
                    "rgba(3, 2, 16, 1.000)"
                ],
                [
                    0.03137254901960784,
                    "rgba(4, 3, 18, 1.000)"
                ],
                [
                    0.03529411764705882,
                    "rgba(4, 3, 20, 1.000)"
                ],
                [
                    0.0392156862745098,
                    "rgba(5, 4, 23, 1.000)"
                ],
                [
                    0.043137254901960784,
                    "rgba(6, 4, 25, 1.000)"
                ],
                [
                    0.047058823529411764,
                    "rgba(7, 5, 27, 1.000)"
                ],
                [
                    0.050980392156862744,
                    "rgba(8, 5, 29, 1.000)"
                ],
                [
                    0.054901960784313725,
                    "rgba(9, 6, 31, 1.000)"
                ],
                [
                    0.058823529411764705,
                    "rgba(10, 7, 34, 1.000)"
                ],
                [
                    0.06274509803921569,
                    "rgba(11, 7, 36, 1.000)"
                ],
                [
                    0.06666666666666667,
                    "rgba(12, 8, 38, 1.000)"
                ],
                [
                    0.07058823529411765,
                    "rgba(13, 8, 41, 1.000)"
                ],
                [
                    0.07450980392156863,
                    "rgba(14, 9, 43, 1.000)"
                ],
                [
                    0.0784313725490196,
                    "rgba(16, 9, 45, 1.000)"
                ],
                [
                    0.08235294117647059,
                    "rgba(17, 10, 48, 1.000)"
                ],
                [
                    0.08627450980392157,
                    "rgba(18, 10, 50, 1.000)"
                ],
                [
                    0.09019607843137255,
                    "rgba(20, 11, 52, 1.000)"
                ],
                [
                    0.09411764705882353,
                    "rgba(21, 11, 55, 1.000)"
                ],
                [
                    0.09803921568627451,
                    "rgba(22, 11, 57, 1.000)"
                ],
                [
                    0.10196078431372549,
                    "rgba(24, 12, 60, 1.000)"
                ],
                [
                    0.10588235294117647,
                    "rgba(25, 12, 62, 1.000)"
                ],
                [
                    0.10980392156862745,
                    "rgba(27, 12, 65, 1.000)"
                ],
                [
                    0.11372549019607843,
                    "rgba(28, 12, 67, 1.000)"
                ],
                [
                    0.11764705882352941,
                    "rgba(30, 12, 69, 1.000)"
                ],
                [
                    0.12156862745098039,
                    "rgba(31, 12, 72, 1.000)"
                ],
                [
                    0.12549019607843137,
                    "rgba(33, 12, 74, 1.000)"
                ],
                [
                    0.12941176470588237,
                    "rgba(35, 12, 76, 1.000)"
                ],
                [
                    0.13333333333333333,
                    "rgba(36, 12, 79, 1.000)"
                ],
                [
                    0.13725490196078433,
                    "rgba(38, 12, 81, 1.000)"
                ],
                [
                    0.1411764705882353,
                    "rgba(40, 11, 83, 1.000)"
                ],
                [
                    0.1450980392156863,
                    "rgba(41, 11, 85, 1.000)"
                ],
                [
                    0.14901960784313725,
                    "rgba(43, 11, 87, 1.000)"
                ],
                [
                    0.15294117647058825,
                    "rgba(45, 11, 89, 1.000)"
                ],
                [
                    0.1568627450980392,
                    "rgba(47, 10, 91, 1.000)"
                ],
                [
                    0.1607843137254902,
                    "rgba(49, 10, 92, 1.000)"
                ],
                [
                    0.16470588235294117,
                    "rgba(50, 10, 94, 1.000)"
                ],
                [
                    0.16862745098039217,
                    "rgba(52, 10, 95, 1.000)"
                ],
                [
                    0.17254901960784313,
                    "rgba(54, 9, 97, 1.000)"
                ],
                [
                    0.17647058823529413,
                    "rgba(56, 9, 98, 1.000)"
                ],
                [
                    0.1803921568627451,
                    "rgba(57, 9, 99, 1.000)"
                ],
                [
                    0.1843137254901961,
                    "rgba(59, 9, 100, 1.000)"
                ],
                [
                    0.18823529411764706,
                    "rgba(61, 9, 101, 1.000)"
                ],
                [
                    0.19215686274509805,
                    "rgba(62, 9, 102, 1.000)"
                ],
                [
                    0.19607843137254902,
                    "rgba(64, 10, 103, 1.000)"
                ],
                [
                    0.2,
                    "rgba(66, 10, 104, 1.000)"
                ],
                [
                    0.20392156862745098,
                    "rgba(68, 10, 104, 1.000)"
                ],
                [
                    0.20784313725490197,
                    "rgba(69, 10, 105, 1.000)"
                ],
                [
                    0.21176470588235294,
                    "rgba(71, 11, 106, 1.000)"
                ],
                [
                    0.21568627450980393,
                    "rgba(73, 11, 106, 1.000)"
                ],
                [
                    0.2196078431372549,
                    "rgba(74, 12, 107, 1.000)"
                ],
                [
                    0.2235294117647059,
                    "rgba(76, 12, 107, 1.000)"
                ],
                [
                    0.22745098039215686,
                    "rgba(77, 13, 108, 1.000)"
                ],
                [
                    0.23137254901960785,
                    "rgba(79, 13, 108, 1.000)"
                ],
                [
                    0.23529411764705882,
                    "rgba(81, 14, 108, 1.000)"
                ],
                [
                    0.23921568627450981,
                    "rgba(82, 14, 109, 1.000)"
                ],
                [
                    0.24313725490196078,
                    "rgba(84, 15, 109, 1.000)"
                ],
                [
                    0.24705882352941178,
                    "rgba(85, 15, 109, 1.000)"
                ],
                [
                    0.25098039215686274,
                    "rgba(87, 16, 110, 1.000)"
                ],
                [
                    0.2549019607843137,
                    "rgba(89, 16, 110, 1.000)"
                ],
                [
                    0.25882352941176473,
                    "rgba(90, 17, 110, 1.000)"
                ],
                [
                    0.2627450980392157,
                    "rgba(92, 18, 110, 1.000)"
                ],
                [
                    0.26666666666666666,
                    "rgba(93, 18, 110, 1.000)"
                ],
                [
                    0.27058823529411763,
                    "rgba(95, 19, 110, 1.000)"
                ],
                [
                    0.27450980392156865,
                    "rgba(97, 19, 110, 1.000)"
                ],
                [
                    0.2784313725490196,
                    "rgba(98, 20, 110, 1.000)"
                ],
                [
                    0.2823529411764706,
                    "rgba(100, 21, 110, 1.000)"
                ],
                [
                    0.28627450980392155,
                    "rgba(101, 21, 110, 1.000)"
                ],
                [
                    0.2901960784313726,
                    "rgba(103, 22, 110, 1.000)"
                ],
                [
                    0.29411764705882354,
                    "rgba(105, 22, 110, 1.000)"
                ],
                [
                    0.2980392156862745,
                    "rgba(106, 23, 110, 1.000)"
                ],
                [
                    0.30196078431372547,
                    "rgba(108, 24, 110, 1.000)"
                ],
                [
                    0.3058823529411765,
                    "rgba(109, 24, 110, 1.000)"
                ],
                [
                    0.30980392156862746,
                    "rgba(111, 25, 110, 1.000)"
                ],
                [
                    0.3137254901960784,
                    "rgba(113, 25, 110, 1.000)"
                ],
                [
                    0.3176470588235294,
                    "rgba(114, 26, 110, 1.000)"
                ],
                [
                    0.3215686274509804,
                    "rgba(116, 26, 110, 1.000)"
                ],
                [
                    0.3254901960784314,
                    "rgba(117, 27, 110, 1.000)"
                ],
                [
                    0.32941176470588235,
                    "rgba(119, 28, 109, 1.000)"
                ],
                [
                    0.3333333333333333,
                    "rgba(120, 28, 109, 1.000)"
                ],
                [
                    0.33725490196078434,
                    "rgba(122, 29, 109, 1.000)"
                ],
                [
                    0.3411764705882353,
                    "rgba(124, 29, 109, 1.000)"
                ],
                [
                    0.34509803921568627,
                    "rgba(125, 30, 109, 1.000)"
                ],
                [
                    0.34901960784313724,
                    "rgba(127, 30, 108, 1.000)"
                ],
                [
                    0.35294117647058826,
                    "rgba(128, 31, 108, 1.000)"
                ],
                [
                    0.3568627450980392,
                    "rgba(130, 32, 108, 1.000)"
                ],
                [
                    0.3607843137254902,
                    "rgba(132, 32, 107, 1.000)"
                ],
                [
                    0.36470588235294116,
                    "rgba(133, 33, 107, 1.000)"
                ],
                [
                    0.3686274509803922,
                    "rgba(135, 33, 107, 1.000)"
                ],
                [
                    0.37254901960784315,
                    "rgba(136, 34, 106, 1.000)"
                ],
                [
                    0.3764705882352941,
                    "rgba(138, 34, 106, 1.000)"
                ],
                [
                    0.3803921568627451,
                    "rgba(140, 35, 105, 1.000)"
                ],
                [
                    0.3843137254901961,
                    "rgba(141, 35, 105, 1.000)"
                ],
                [
                    0.38823529411764707,
                    "rgba(143, 36, 105, 1.000)"
                ],
                [
                    0.39215686274509803,
                    "rgba(144, 37, 104, 1.000)"
                ],
                [
                    0.396078431372549,
                    "rgba(146, 37, 104, 1.000)"
                ],
                [
                    0.4,
                    "rgba(147, 38, 103, 1.000)"
                ],
                [
                    0.403921568627451,
                    "rgba(149, 38, 103, 1.000)"
                ],
                [
                    0.40784313725490196,
                    "rgba(151, 39, 102, 1.000)"
                ],
                [
                    0.4117647058823529,
                    "rgba(152, 39, 102, 1.000)"
                ],
                [
                    0.41568627450980394,
                    "rgba(154, 40, 101, 1.000)"
                ],
                [
                    0.4196078431372549,
                    "rgba(155, 41, 100, 1.000)"
                ],
                [
                    0.4235294117647059,
                    "rgba(157, 41, 100, 1.000)"
                ],
                [
                    0.42745098039215684,
                    "rgba(159, 42, 99, 1.000)"
                ],
                [
                    0.43137254901960786,
                    "rgba(160, 42, 99, 1.000)"
                ],
                [
                    0.43529411764705883,
                    "rgba(162, 43, 98, 1.000)"
                ],
                [
                    0.4392156862745098,
                    "rgba(163, 44, 97, 1.000)"
                ],
                [
                    0.44313725490196076,
                    "rgba(165, 44, 96, 1.000)"
                ],
                [
                    0.4470588235294118,
                    "rgba(166, 45, 96, 1.000)"
                ],
                [
                    0.45098039215686275,
                    "rgba(168, 46, 95, 1.000)"
                ],
                [
                    0.4549019607843137,
                    "rgba(169, 46, 94, 1.000)"
                ],
                [
                    0.4588235294117647,
                    "rgba(171, 47, 94, 1.000)"
                ],
                [
                    0.4627450980392157,
                    "rgba(173, 48, 93, 1.000)"
                ],
                [
                    0.4666666666666667,
                    "rgba(174, 48, 92, 1.000)"
                ],
                [
                    0.47058823529411764,
                    "rgba(176, 49, 91, 1.000)"
                ],
                [
                    0.4745098039215686,
                    "rgba(177, 50, 90, 1.000)"
                ],
                [
                    0.47843137254901963,
                    "rgba(179, 50, 90, 1.000)"
                ],
                [
                    0.4823529411764706,
                    "rgba(180, 51, 89, 1.000)"
                ],
                [
                    0.48627450980392156,
                    "rgba(182, 52, 88, 1.000)"
                ],
                [
                    0.49019607843137253,
                    "rgba(183, 53, 87, 1.000)"
                ],
                [
                    0.49411764705882355,
                    "rgba(185, 53, 86, 1.000)"
                ],
                [
                    0.4980392156862745,
                    "rgba(186, 54, 85, 1.000)"
                ],
                [
                    0.5019607843137255,
                    "rgba(188, 55, 84, 1.000)"
                ],
                [
                    0.5058823529411764,
                    "rgba(189, 56, 83, 1.000)"
                ],
                [
                    0.5098039215686274,
                    "rgba(191, 57, 82, 1.000)"
                ],
                [
                    0.5137254901960784,
                    "rgba(192, 58, 81, 1.000)"
                ],
                [
                    0.5176470588235295,
                    "rgba(193, 58, 80, 1.000)"
                ],
                [
                    0.5215686274509804,
                    "rgba(195, 59, 79, 1.000)"
                ],
                [
                    0.5254901960784314,
                    "rgba(196, 60, 78, 1.000)"
                ],
                [
                    0.5294117647058824,
                    "rgba(198, 61, 77, 1.000)"
                ],
                [
                    0.5333333333333333,
                    "rgba(199, 62, 76, 1.000)"
                ],
                [
                    0.5372549019607843,
                    "rgba(200, 63, 75, 1.000)"
                ],
                [
                    0.5411764705882353,
                    "rgba(202, 64, 74, 1.000)"
                ],
                [
                    0.5450980392156862,
                    "rgba(203, 65, 73, 1.000)"
                ],
                [
                    0.5490196078431373,
                    "rgba(204, 66, 72, 1.000)"
                ],
                [
                    0.5529411764705883,
                    "rgba(206, 67, 71, 1.000)"
                ],
                [
                    0.5568627450980392,
                    "rgba(207, 68, 70, 1.000)"
                ],
                [
                    0.5607843137254902,
                    "rgba(208, 69, 69, 1.000)"
                ],
                [
                    0.5647058823529412,
                    "rgba(210, 70, 68, 1.000)"
                ],
                [
                    0.5686274509803921,
                    "rgba(211, 71, 67, 1.000)"
                ],
                [
                    0.5725490196078431,
                    "rgba(212, 72, 66, 1.000)"
                ],
                [
                    0.5764705882352941,
                    "rgba(213, 74, 65, 1.000)"
                ],
                [
                    0.5803921568627451,
                    "rgba(215, 75, 63, 1.000)"
                ],
                [
                    0.5843137254901961,
                    "rgba(216, 76, 62, 1.000)"
                ],
                [
                    0.5882352941176471,
                    "rgba(217, 77, 61, 1.000)"
                ],
                [
                    0.592156862745098,
                    "rgba(218, 78, 60, 1.000)"
                ],
                [
                    0.596078431372549,
                    "rgba(219, 80, 59, 1.000)"
                ],
                [
                    0.6,
                    "rgba(221, 81, 58, 1.000)"
                ],
                [
                    0.6039215686274509,
                    "rgba(222, 82, 56, 1.000)"
                ],
                [
                    0.6078431372549019,
                    "rgba(223, 83, 55, 1.000)"
                ],
                [
                    0.611764705882353,
                    "rgba(224, 85, 54, 1.000)"
                ],
                [
                    0.615686274509804,
                    "rgba(225, 86, 53, 1.000)"
                ],
                [
                    0.6196078431372549,
                    "rgba(226, 87, 52, 1.000)"
                ],
                [
                    0.6235294117647059,
                    "rgba(227, 89, 51, 1.000)"
                ],
                [
                    0.6274509803921569,
                    "rgba(228, 90, 49, 1.000)"
                ],
                [
                    0.6313725490196078,
                    "rgba(229, 92, 48, 1.000)"
                ],
                [
                    0.6352941176470588,
                    "rgba(230, 93, 47, 1.000)"
                ],
                [
                    0.6392156862745098,
                    "rgba(231, 94, 46, 1.000)"
                ],
                [
                    0.6431372549019608,
                    "rgba(232, 96, 45, 1.000)"
                ],
                [
                    0.6470588235294118,
                    "rgba(233, 97, 43, 1.000)"
                ],
                [
                    0.6509803921568628,
                    "rgba(234, 99, 42, 1.000)"
                ],
                [
                    0.6549019607843137,
                    "rgba(235, 100, 41, 1.000)"
                ],
                [
                    0.6588235294117647,
                    "rgba(235, 102, 40, 1.000)"
                ],
                [
                    0.6627450980392157,
                    "rgba(236, 103, 38, 1.000)"
                ],
                [
                    0.6666666666666666,
                    "rgba(237, 105, 37, 1.000)"
                ],
                [
                    0.6705882352941176,
                    "rgba(238, 106, 36, 1.000)"
                ],
                [
                    0.6745098039215687,
                    "rgba(239, 108, 35, 1.000)"
                ],
                [
                    0.6784313725490196,
                    "rgba(239, 110, 33, 1.000)"
                ],
                [
                    0.6823529411764706,
                    "rgba(240, 111, 32, 1.000)"
                ],
                [
                    0.6862745098039216,
                    "rgba(241, 113, 31, 1.000)"
                ],
                [
                    0.6901960784313725,
                    "rgba(241, 115, 29, 1.000)"
                ],
                [
                    0.6941176470588235,
                    "rgba(242, 116, 28, 1.000)"
                ],
                [
                    0.6980392156862745,
                    "rgba(243, 118, 27, 1.000)"
                ],
                [
                    0.7019607843137254,
                    "rgba(243, 120, 25, 1.000)"
                ],
                [
                    0.7058823529411765,
                    "rgba(244, 121, 24, 1.000)"
                ],
                [
                    0.7098039215686275,
                    "rgba(245, 123, 23, 1.000)"
                ],
                [
                    0.7137254901960784,
                    "rgba(245, 125, 21, 1.000)"
                ],
                [
                    0.7176470588235294,
                    "rgba(246, 126, 20, 1.000)"
                ],
                [
                    0.7215686274509804,
                    "rgba(246, 128, 19, 1.000)"
                ],
                [
                    0.7254901960784313,
                    "rgba(247, 130, 18, 1.000)"
                ],
                [
                    0.7294117647058823,
                    "rgba(247, 132, 16, 1.000)"
                ],
                [
                    0.7333333333333333,
                    "rgba(248, 133, 15, 1.000)"
                ],
                [
                    0.7372549019607844,
                    "rgba(248, 135, 14, 1.000)"
                ],
                [
                    0.7411764705882353,
                    "rgba(248, 137, 12, 1.000)"
                ],
                [
                    0.7450980392156863,
                    "rgba(249, 139, 11, 1.000)"
                ],
                [
                    0.7490196078431373,
                    "rgba(249, 140, 10, 1.000)"
                ],
                [
                    0.7529411764705882,
                    "rgba(249, 142, 9, 1.000)"
                ],
                [
                    0.7568627450980392,
                    "rgba(250, 144, 8, 1.000)"
                ],
                [
                    0.7607843137254902,
                    "rgba(250, 146, 7, 1.000)"
                ],
                [
                    0.7647058823529411,
                    "rgba(250, 148, 7, 1.000)"
                ],
                [
                    0.7686274509803922,
                    "rgba(251, 150, 6, 1.000)"
                ],
                [
                    0.7725490196078432,
                    "rgba(251, 151, 6, 1.000)"
                ],
                [
                    0.7764705882352941,
                    "rgba(251, 153, 6, 1.000)"
                ],
                [
                    0.7803921568627451,
                    "rgba(251, 155, 6, 1.000)"
                ],
                [
                    0.7843137254901961,
                    "rgba(251, 157, 7, 1.000)"
                ],
                [
                    0.788235294117647,
                    "rgba(252, 159, 7, 1.000)"
                ],
                [
                    0.792156862745098,
                    "rgba(252, 161, 8, 1.000)"
                ],
                [
                    0.796078431372549,
                    "rgba(252, 163, 9, 1.000)"
                ],
                [
                    0.8,
                    "rgba(252, 165, 10, 1.000)"
                ],
                [
                    0.803921568627451,
                    "rgba(252, 166, 12, 1.000)"
                ],
                [
                    0.807843137254902,
                    "rgba(252, 168, 13, 1.000)"
                ],
                [
                    0.8117647058823529,
                    "rgba(252, 170, 15, 1.000)"
                ],
                [
                    0.8156862745098039,
                    "rgba(252, 172, 17, 1.000)"
                ],
                [
                    0.8196078431372549,
                    "rgba(252, 174, 18, 1.000)"
                ],
                [
                    0.8235294117647058,
                    "rgba(252, 176, 20, 1.000)"
                ],
                [
                    0.8274509803921568,
                    "rgba(252, 178, 22, 1.000)"
                ],
                [
                    0.8313725490196079,
                    "rgba(252, 180, 24, 1.000)"
                ],
                [
                    0.8352941176470589,
                    "rgba(251, 182, 26, 1.000)"
                ],
                [
                    0.8392156862745098,
                    "rgba(251, 184, 29, 1.000)"
                ],
                [
                    0.8431372549019608,
                    "rgba(251, 186, 31, 1.000)"
                ],
                [
                    0.8470588235294118,
                    "rgba(251, 188, 33, 1.000)"
                ],
                [
                    0.8509803921568627,
                    "rgba(251, 190, 35, 1.000)"
                ],
                [
                    0.8549019607843137,
                    "rgba(250, 192, 38, 1.000)"
                ],
                [
                    0.8588235294117647,
                    "rgba(250, 194, 40, 1.000)"
                ],
                [
                    0.8627450980392157,
                    "rgba(250, 196, 42, 1.000)"
                ],
                [
                    0.8666666666666667,
                    "rgba(250, 198, 45, 1.000)"
                ],
                [
                    0.8705882352941177,
                    "rgba(249, 199, 47, 1.000)"
                ],
                [
                    0.8745098039215686,
                    "rgba(249, 201, 50, 1.000)"
                ],
                [
                    0.8784313725490196,
                    "rgba(249, 203, 53, 1.000)"
                ],
                [
                    0.8823529411764706,
                    "rgba(248, 205, 55, 1.000)"
                ],
                [
                    0.8862745098039215,
                    "rgba(248, 207, 58, 1.000)"
                ],
                [
                    0.8901960784313725,
                    "rgba(247, 209, 61, 1.000)"
                ],
                [
                    0.8941176470588236,
                    "rgba(247, 211, 64, 1.000)"
                ],
                [
                    0.8980392156862745,
                    "rgba(246, 213, 67, 1.000)"
                ],
                [
                    0.9019607843137255,
                    "rgba(246, 215, 70, 1.000)"
                ],
                [
                    0.9058823529411765,
                    "rgba(245, 217, 73, 1.000)"
                ],
                [
                    0.9098039215686274,
                    "rgba(245, 219, 76, 1.000)"
                ],
                [
                    0.9137254901960784,
                    "rgba(244, 221, 79, 1.000)"
                ],
                [
                    0.9176470588235294,
                    "rgba(244, 223, 83, 1.000)"
                ],
                [
                    0.9215686274509803,
                    "rgba(244, 225, 86, 1.000)"
                ],
                [
                    0.9254901960784314,
                    "rgba(243, 227, 90, 1.000)"
                ],
                [
                    0.9294117647058824,
                    "rgba(243, 229, 93, 1.000)"
                ],
                [
                    0.9333333333333333,
                    "rgba(242, 230, 97, 1.000)"
                ],
                [
                    0.9372549019607843,
                    "rgba(242, 232, 101, 1.000)"
                ],
                [
                    0.9411764705882353,
                    "rgba(242, 234, 105, 1.000)"
                ],
                [
                    0.9450980392156862,
                    "rgba(241, 236, 109, 1.000)"
                ],
                [
                    0.9490196078431372,
                    "rgba(241, 237, 113, 1.000)"
                ],
                [
                    0.9529411764705882,
                    "rgba(241, 239, 117, 1.000)"
                ],
                [
                    0.9568627450980393,
                    "rgba(241, 241, 121, 1.000)"
                ],
                [
                    0.9607843137254902,
                    "rgba(242, 242, 125, 1.000)"
                ],
                [
                    0.9647058823529412,
                    "rgba(242, 244, 130, 1.000)"
                ],
                [
                    0.9686274509803922,
                    "rgba(243, 245, 134, 1.000)"
                ],
                [
                    0.9725490196078431,
                    "rgba(243, 246, 138, 1.000)"
                ],
                [
                    0.9764705882352941,
                    "rgba(244, 248, 142, 1.000)"
                ],
                [
                    0.9803921568627451,
                    "rgba(245, 249, 146, 1.000)"
                ],
                [
                    0.984313725490196,
                    "rgba(246, 250, 150, 1.000)"
                ],
                [
                    0.9882352941176471,
                    "rgba(248, 251, 154, 1.000)"
                ],
                [
                    0.9921568627450981,
                    "rgba(249, 252, 157, 1.000)"
                ],
                [
                    0.996078431372549,
                    "rgba(250, 253, 161, 1.000)"
                ],
                [
                    1.0,
                    "rgba(252, 255, 164, 1.000)"
                ]
            ],
            "cmax": 0.7078735043732424,
            "showscale": false
        },
        "zmax": 0.7078735043732424,
        "y": [
            9.892724685972443
        ],
        "type": "scatter",
        "hoverinfo": "none"
    }
]
, {
    "showlegend": true,
    "paper_bgcolor": "rgba(255, 255, 255, 1.000)",
    "xaxis1": {
        "showticklabels": true,
        "gridwidth": 0.5,
        "tickvals": [
            0.5,
            1.5
        ],
        "range": [
            0.47,
            1.53
        ],
        "domain": [
            0.11574405472043268,
            0.4928418038654259
        ],
        "mirror": true,
        "tickangle": 0,
        "showline": true,
        "ticktext": [
            "false",
            "true"
        ],
        "zeroline": false,
        "tickfont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "zerolinecolor": "rgba(0, 0, 0, 1.000)",
        "anchor": "y1",
        "visible": true,
        "ticks": "inside",
        "tickmode": "array",
        "linecolor": "rgba(0, 0, 0, 1.000)",
        "showgrid": true,
        "title": "",
        "gridcolor": "rgba(0, 0, 0, 0.100)",
        "titlefont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 15
        },
        "tickcolor": "rgb(0, 0, 0)",
        "type": "-"
    },
    "legend": {
        "yanchor": "auto",
        "xanchor": "auto",
        "bordercolor": "rgba(0, 0, 0, 1.000)",
        "bgcolor": "rgba(255, 255, 255, 1.000)",
        "borderwidth": 1,
        "tracegroupgap": 0,
        "y": 1.0,
        "font": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "title": {
            "font": {
                "color": "rgba(0, 0, 0, 1.000)",
                "family": "sans-serif",
                "size": 15
            },
            "text": ""
        },
        "traceorder": "normal",
        "x": 1.0
    },
    "height": 500,
    "yaxis4": {
        "showticklabels": true,
        "gridwidth": 0.5,
        "tickvals": [
            0.68,
            0.685,
            0.6900000000000001,
            0.6950000000000001,
            0.7000000000000001,
            0.705
        ],
        "range": [
            0.6753434051036765,
            0.7088209829927443
        ],
        "domain": [
            0.060651793525809315,
            0.5074037620297464
        ],
        "mirror": true,
        "tickangle": 0,
        "showline": true,
        "ticktext": [
            "0.680",
            "0.685",
            "0.690",
            "0.695",
            "0.700",
            "0.705"
        ],
        "zeroline": false,
        "tickfont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "zerolinecolor": "rgba(0, 0, 0, 1.000)",
        "anchor": "x4",
        "visible": true,
        "ticks": "inside",
        "tickmode": "array",
        "linecolor": "rgba(0, 0, 0, 1.000)",
        "showgrid": true,
        "title": "LogLoss{Float64}",
        "gridcolor": "rgba(0, 0, 0, 0.100)",
        "titlefont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 15
        },
        "tickcolor": "rgb(0, 0, 0)",
        "type": "-"
    },
    "yaxis2": {
        "showticklabels": true,
        "gridwidth": 0.5,
        "tickvals": [
            0.0,
            0.25,
            0.5,
            0.75,
            1.0
        ],
        "range": [
            -0.03,
            1.03
        ],
        "domain": [
            0.5453740157480316,
            0.9921259842519685
        ],
        "mirror": false,
        "tickangle": 0,
        "showline": false,
        "ticktext": [
            "0.00",
            "0.25",
            "0.50",
            "0.75",
            "1.00"
        ],
        "zeroline": false,
        "tickfont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "zerolinecolor": "rgba(0, 0, 0, 1.000)",
        "anchor": "x2",
        "visible": false,
        "ticks": "inside",
        "tickmode": "array",
        "linecolor": "rgba(0, 0, 0, 1.000)",
        "showgrid": true,
        "title": "",
        "gridcolor": "rgba(0, 0, 0, 0.100)",
        "titlefont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 15
        },
        "tickcolor": "rgb(0, 0, 0)",
        "type": "-"
    },
    "xaxis3": {
        "showticklabels": true,
        "gridwidth": 0.5,
        "tickvals": [
            0.5,
            1.5
        ],
        "range": [
            -0.3153436333055711,
            0.19040489669727112
        ],
        "domain": [
            0.11574405472043268,
            0.4928418038654259
        ],
        "mirror": true,
        "tickangle": 0,
        "showline": true,
        "ticktext": [
            "false",
            "true"
        ],
        "zeroline": false,
        "tickfont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "zerolinecolor": "rgba(0, 0, 0, 1.000)",
        "anchor": "y3",
        "visible": true,
        "ticks": "inside",
        "tickmode": "array",
        "linecolor": "rgba(0, 0, 0, 1.000)",
        "showgrid": true,
        "title": "continuous_encoder.one_hot_ordered_factors",
        "gridcolor": "rgba(0, 0, 0, 0.100)",
        "titlefont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 15
        },
        "tickcolor": "rgb(0, 0, 0)",
        "type": "log"
    },
    "yaxis3": {
        "showticklabels": true,
        "gridwidth": 0.5,
        "tickvals": [
            2.0,
            4.0,
            6.0,
            8.0,
            10.0
        ],
        "range": [
            0.6469688790023761,
            11.674389872656759
        ],
        "domain": [
            0.060651793525809315,
            0.5074037620297464
        ],
        "mirror": true,
        "tickangle": 0,
        "showline": true,
        "ticktext": [
            "2",
            "4",
            "6",
            "8",
            "10"
        ],
        "zeroline": false,
        "tickfont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "zerolinecolor": "rgba(0, 0, 0, 1.000)",
        "anchor": "x3",
        "visible": true,
        "ticks": "inside",
        "tickmode": "array",
        "linecolor": "rgba(0, 0, 0, 1.000)",
        "showgrid": true,
        "title": "logistic_classifier.lambda",
        "gridcolor": "rgba(0, 0, 0, 0.100)",
        "titlefont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 15
        },
        "tickcolor": "rgb(0, 0, 0)",
        "type": "-"
    },
    "xaxis4": {
        "showticklabels": true,
        "gridwidth": 0.5,
        "tickvals": [
            1.0,
            10.0
        ],
        "range": [
            -0.050360169836476315,
            1.0876745150671971
        ],
        "domain": [
            0.6157440547204327,
            0.9928418038654259
        ],
        "mirror": true,
        "tickangle": 0,
        "showline": true,
        "ticktext": [
            "10<sup>0</sup>",
            "10<sup>1</sup>"
        ],
        "zeroline": false,
        "tickfont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "zerolinecolor": "rgba(0, 0, 0, 1.000)",
        "anchor": "y4",
        "visible": true,
        "ticks": "inside",
        "tickmode": "array",
        "linecolor": "rgba(0, 0, 0, 1.000)",
        "showgrid": true,
        "title": "logistic_classifier.lambda",
        "gridcolor": "rgba(0, 0, 0, 0.100)",
        "titlefont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 15
        },
        "tickcolor": "rgb(0, 0, 0)",
        "type": "log"
    },
    "yaxis1": {
        "showticklabels": true,
        "gridwidth": 0.5,
        "tickvals": [
            0.68,
            0.685,
            0.6900000000000001,
            0.6950000000000001,
            0.7000000000000001,
            0.705
        ],
        "range": [
            0.6753434051036765,
            0.7088209829927443
        ],
        "domain": [
            0.5453740157480316,
            0.9921259842519685
        ],
        "mirror": true,
        "tickangle": 0,
        "showline": true,
        "ticktext": [
            "0.680",
            "0.685",
            "0.690",
            "0.695",
            "0.700",
            "0.705"
        ],
        "zeroline": false,
        "tickfont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "zerolinecolor": "rgba(0, 0, 0, 1.000)",
        "anchor": "x1",
        "visible": true,
        "ticks": "inside",
        "tickmode": "array",
        "linecolor": "rgba(0, 0, 0, 1.000)",
        "showgrid": true,
        "title": "LogLoss{Float64}",
        "gridcolor": "rgba(0, 0, 0, 0.100)",
        "titlefont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 15
        },
        "tickcolor": "rgb(0, 0, 0)",
        "type": "-"
    },
    "annotations": [],
    "xaxis2": {
        "showticklabels": true,
        "gridwidth": 0.5,
        "tickvals": [
            0.0,
            0.25,
            0.5,
            0.75,
            1.0
        ],
        "range": [
            -0.03,
            1.03
        ],
        "domain": [
            0.6157440547204327,
            0.9928418038654259
        ],
        "mirror": false,
        "tickangle": 0,
        "showline": false,
        "ticktext": [
            "0.00",
            "0.25",
            "0.50",
            "0.75",
            "1.00"
        ],
        "zeroline": false,
        "tickfont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "zerolinecolor": "rgba(0, 0, 0, 1.000)",
        "anchor": "y2",
        "visible": false,
        "ticks": "inside",
        "tickmode": "array",
        "linecolor": "rgba(0, 0, 0, 1.000)",
        "showgrid": true,
        "title": "",
        "gridcolor": "rgba(0, 0, 0, 0.100)",
        "titlefont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 15
        },
        "tickcolor": "rgb(0, 0, 0)",
        "type": "-"
    },
    "plot_bgcolor": "rgba(255, 255, 255, 1.000)",
    "margin": {
        "l": 0,
        "b": 20,
        "r": 0,
        "t": 20
    },
    "width": 550
}
);

    
    </script>

```

Finally, let's compare cross-validation estimate of the performance
of the self-tuning model with that of the original model (an example
of [*nested
resampling*](https://mlr3book.mlr-org.com/nested-resampling.html)
here):

````julia
err = evaluate!(mach, resampling=CV(nfolds=3), measure=cross_entropy)
````

````
PerformanceEvaluation object with these fields:
  measure, measurement, operation, per_fold,
  per_observation, fitted_params_per_fold,
  report_per_fold, train_test_pairs
Extract:
┌────────────────────────────┬─────────────┬───────────┬───────────────────────┐
│ measure                    │ measurement │ operation │ per_fold              │
├────────────────────────────┼─────────────┼───────────┼───────────────────────┤
│ LogLoss(tol = 2.22045e-16) │ 0.736       │ predict   │ [0.759, 0.707, 0.741] │
└────────────────────────────┴─────────────┴───────────┴───────────────────────┘

````

````julia
tuned_err = evaluate!(tuned_mach, resampling=CV(nfolds=3), measure=cross_entropy)
````

````
PerformanceEvaluation object with these fields:
  measure, measurement, operation, per_fold,
  per_observation, fitted_params_per_fold,
  report_per_fold, train_test_pairs
Extract:
┌────────────────────────────┬─────────────┬───────────┬───────────────────────┐
│ measure                    │ measurement │ operation │ per_fold              │
├────────────────────────────┼─────────────┼───────────┼───────────────────────┤
│ LogLoss(tol = 2.22045e-16) │ 0.705       │ predict   │ [0.717, 0.701, 0.697] │
└────────────────────────────┴─────────────┴───────────┴───────────────────────┘

````

<a id='resources-for-part-4'></a>

### Resources for Part 4

- From the MLJ manual:
   - [Learning Curves](https://alan-turing-institute.github.io/MLJ.jl/dev/learning_curves/)
   - [Tuning Models](https://alan-turing-institute.github.io/MLJ.jl/dev/tuning_models/)
- The [MLJTuning repo](https://github.com/juliaai/MLJTuning.jl#who-is-this-repo-for) - mostly for developers

- From Data Science Tutorials:
    - [Tuning a model](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/model-tuning/)
    - [Crabs with XGBoost](https://juliaai.github.io/DataScienceTutorials.jl/end-to-end/crabs-xgb/) `Grid` tuning in stages for a tree-boosting model with many parameters
    - [Boston with LightGBM](https://juliaai.github.io/DataScienceTutorials.jl/end-to-end/boston-lgbm/) -  `Grid` tuning for another popular tree-booster
    - [Boston with Flux](https://juliaai.github.io/DataScienceTutorials.jl/end-to-end/boston-flux/) - optimizing batch size in a simple neural network regressor
- [UCI Horse Colic Data Set](http://archive.ics.uci.edu/ml/datasets/Horse+Colic)

### Exercises for Part 4

#### Exercise 8

This exercise continues our analysis of the King County House price
prediction problem:

````julia
y, X = unpack(house, ==(:price), name -> true, rng=123);
nothing #hide
````

Your task will be to tune the following pipeline regression model,
which includes a gradient tree boosting component:

````julia
EvoTreeRegressor = @load EvoTreeRegressor
tree_booster = EvoTreeRegressor(nrounds = 70)
model = @pipeline ContinuousEncoder tree_booster
````

````
Pipeline639(
    continuous_encoder = ContinuousEncoder(
            drop_last = false,
            one_hot_ordered_factors = false),
    evo_tree_regressor = EvoTreeRegressor(
            loss = EvoTrees.Linear(),
            nrounds = 70,
            λ = 0.0,
            γ = 0.0,
            η = 0.1,
            max_depth = 5,
            min_weight = 1.0,
            rowsample = 1.0,
            colsample = 1.0,
            nbins = 64,
            α = 0.5,
            metric = :mse,
            rng = MersenneTwister(123),
            device = "cpu"))
````

(a) Construct a bounded range `r1` for the `evo_tree_booster`
parameter `max_depth`, varying between 1 and 12.

\star&(b) For the `nbins` parameter of the `EvoTreeRegressor`, define the range

````julia
r2 = range(model,
           :(evo_tree_regressor.nbins),
           lower = 2.5,
           upper= 7.5, scale=x->2^round(Int, x))
````

````
transformed NumericRange(2.5 ≤ evo_tree_regressor.nbins ≤ 7.5; origin=5.0, unit=2.5)
````

Notice that in this case we've specified a *function* instead of a
canned scale, like `:log10`. In this case the `scale` function is
applied after sampling (uniformly) between the limits of `lower` and
`upper`. Perhaps you can guess the outputs of the following lines of
code?

````julia
r2_sampler = sampler(r2, Distributions.Uniform)
samples = rand(r2_sampler, 1000);
histogram(samples, nbins=50)
````

```@raw html
<script src="https://cdn.plot.ly/plotly-1.57.1.min.js"></script>    <div id="7c09a994-1d28-4e5c-a968-b6dfe66d4f94" style="width:490px;height:300px;"></div>
    <script>
    
        var PLOT = document.getElementById('7c09a994-1d28-4e5c-a968-b6dfe66d4f94');
    Plotly.plot(PLOT, [
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            5.0,
            5.0,
            10.0,
            10.0,
            5.0,
            5.0
        ],
        "showlegend": true,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            208.0,
            0.0,
            0.0,
            208.0,
            208.0,
            208.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            10.0,
            10.0,
            15.0,
            15.0,
            10.0,
            10.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            15.0,
            15.0,
            20.0,
            20.0,
            15.0,
            15.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            207.0,
            0.0,
            0.0,
            207.0,
            207.0,
            207.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            20.0,
            20.0,
            25.0,
            25.0,
            20.0,
            20.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            25.0,
            25.0,
            30.0,
            30.0,
            25.0,
            25.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            30.0,
            30.0,
            35.0,
            35.0,
            30.0,
            30.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            184.0,
            0.0,
            0.0,
            184.0,
            184.0,
            184.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            35.0,
            35.0,
            40.0,
            40.0,
            35.0,
            35.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            40.0,
            40.0,
            45.0,
            45.0,
            40.0,
            40.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            45.0,
            45.0,
            50.0,
            50.0,
            45.0,
            45.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            50.0,
            50.0,
            55.0,
            55.0,
            50.0,
            50.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            55.0,
            55.0,
            60.0,
            60.0,
            55.0,
            55.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            60.0,
            60.0,
            65.0,
            65.0,
            60.0,
            60.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            201.0,
            0.0,
            0.0,
            201.0,
            201.0,
            201.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            65.0,
            65.0,
            70.0,
            70.0,
            65.0,
            65.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            70.0,
            70.0,
            75.0,
            75.0,
            70.0,
            70.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            75.0,
            75.0,
            80.0,
            80.0,
            75.0,
            75.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            80.0,
            80.0,
            85.0,
            85.0,
            80.0,
            80.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            85.0,
            85.0,
            90.0,
            90.0,
            85.0,
            85.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            90.0,
            90.0,
            95.0,
            95.0,
            90.0,
            90.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            95.0,
            95.0,
            100.0,
            100.0,
            95.0,
            95.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            100.0,
            100.0,
            105.0,
            105.0,
            100.0,
            100.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            105.0,
            105.0,
            110.0,
            110.0,
            105.0,
            105.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            110.0,
            110.0,
            115.0,
            115.0,
            110.0,
            110.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            115.0,
            115.0,
            120.0,
            120.0,
            115.0,
            115.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            120.0,
            120.0,
            125.0,
            125.0,
            120.0,
            120.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x1",
        "fill": "tozeroy",
        "yaxis": "y1",
        "x": [
            125.0,
            125.0,
            130.0,
            130.0,
            125.0,
            125.0
        ],
        "showlegend": false,
        "mode": "lines",
        "fillcolor": "rgba(0, 154, 250, 1.000)",
        "name": "y1",
        "legendgroup": "y1",
        "line": {
            "color": "rgba(0, 0, 0, 1.000)",
            "dash": "solid",
            "width": 1
        },
        "y": [
            200.0,
            0.0,
            0.0,
            200.0,
            200.0,
            200.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    },
    {
        "xaxis": "x",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y",
        "x": [
            7.5,
            12.5,
            17.5,
            22.5,
            27.5,
            32.5,
            37.5,
            42.5,
            47.5,
            52.5,
            57.5,
            62.5,
            67.5,
            72.5,
            77.5,
            82.5,
            87.5,
            92.5,
            97.5,
            102.5,
            107.5,
            112.5,
            117.5,
            122.5,
            127.5
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "y1",
        "zmin": null,
        "legendgroup": "y1",
        "marker": {
            "symbol": "circle",
            "color": "rgba(0, 154, 250, 0.000)",
            "line": {
                "color": "rgba(0, 0, 0, 0.000)",
                "width": 1
            },
            "size": 0
        },
        "zmax": null,
        "y": [
            208.0,
            0.0,
            207.0,
            0.0,
            0.0,
            184.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            201.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            200.0
        ],
        "type": "scatter"
    }
]
, {
    "showlegend": true,
    "xaxis": {
        "showticklabels": true,
        "gridwidth": 0.5,
        "tickvals": [
            0.0,
            20.0,
            40.0,
            60.0,
            80.0,
            100.0,
            120.0
        ],
        "range": [
            -2.7249999999999996,
            137.725
        ],
        "domain": [
            0.062456478654453904,
            0.991965290053029
        ],
        "mirror": false,
        "tickangle": 0,
        "showline": true,
        "ticktext": [
            "0",
            "20",
            "40",
            "60",
            "80",
            "100",
            "120"
        ],
        "zeroline": false,
        "tickfont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "zerolinecolor": "rgba(0, 0, 0, 1.000)",
        "anchor": "y",
        "visible": true,
        "ticks": "inside",
        "tickmode": "array",
        "linecolor": "rgba(0, 0, 0, 1.000)",
        "showgrid": true,
        "title": "",
        "gridcolor": "rgba(0, 0, 0, 0.100)",
        "titlefont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 15
        },
        "tickcolor": "rgb(0, 0, 0)",
        "type": "-"
    },
    "paper_bgcolor": "rgba(255, 255, 255, 1.000)",
    "annotations": [],
    "height": 300,
    "margin": {
        "l": 0,
        "b": 20,
        "r": 0,
        "t": 20
    },
    "plot_bgcolor": "rgba(255, 255, 255, 1.000)",
    "yaxis": {
        "showticklabels": true,
        "gridwidth": 0.5,
        "tickvals": [
            0.0,
            50.0,
            100.0,
            150.0,
            200.0
        ],
        "range": [
            -6.24,
            214.24
        ],
        "domain": [
            0.050160396617089535,
            0.9868766404199475
        ],
        "mirror": false,
        "tickangle": 0,
        "showline": true,
        "ticktext": [
            "0",
            "50",
            "100",
            "150",
            "200"
        ],
        "zeroline": false,
        "tickfont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "zerolinecolor": "rgba(0, 0, 0, 1.000)",
        "anchor": "x",
        "visible": true,
        "ticks": "inside",
        "tickmode": "array",
        "linecolor": "rgba(0, 0, 0, 1.000)",
        "showgrid": true,
        "title": "",
        "gridcolor": "rgba(0, 0, 0, 0.100)",
        "titlefont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 15
        },
        "tickcolor": "rgb(0, 0, 0)",
        "type": "-"
    },
    "legend": {
        "yanchor": "auto",
        "xanchor": "auto",
        "bordercolor": "rgba(0, 0, 0, 1.000)",
        "bgcolor": "rgba(255, 255, 255, 1.000)",
        "borderwidth": 1,
        "tracegroupgap": 0,
        "y": 1.0,
        "font": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "title": {
            "font": {
                "color": "rgba(0, 0, 0, 1.000)",
                "family": "sans-serif",
                "size": 15
            },
            "text": ""
        },
        "traceorder": "normal",
        "x": 1.0
    },
    "width": 490
}
);

    
    </script>

```

````julia
sort(unique(samples))
````

````
5-element Vector{Int64}:
   8
  16
  32
  64
 128
````

(c) Optimize `model` over these the parameter ranges `r1` and `r2`
using a random search with uniform priors (the default). Use
`Holdout()` resampling, and implement your search by first
constructing a "self-tuning" wrap of `model`, as described
above. Make `mae` (mean absolute error) the loss function that you
optimize, and search over a total of 40 combinations of
hyper-parameters.  If you have time, plot the results of your
search. Feel free to use all available data.

(d) Evaluate the best model found in the search using 3-fold
cross-validation and compare with that of the self-tuning model
(which is different!). Setting data hygiene concerns aside, feel
free to use all available data.

<a id='part-5-advanced-model-composition'>

## Part 5 - Advanced Model Composition

> **Goals:**
> 1. Learn how to build a prototypes of a composite model, called a *learning network*
> 2. Learn how to use the `@from_network` macro to export a learning network as a new stand-alone model type

While `@pipeline` is great for composing models in an unbranching
sequence, for more complicated model composition you'll want to use
MLJ's generic model composition syntax. There are two main steps:

- **Prototype** the composite model by building a *learning
  network*, which can be tested on some (dummy) data as you build
  it.

- **Export** the learning network as a new stand-alone model type.

Like pipeline models, instances of the exported model type behave
like any other model (and are not bound to any data, until you wrap
them in a machine).

### Building a pipeline using the generic composition syntax

To warm up, we'll do the equivalent of

````julia
pipe = @pipeline Standardizer LogisticClassifier;
nothing #hide
````

using the generic syntax.

Here's some dummy data we'll be using to test our learning network:

````julia
X, y = make_blobs(5, 3)
pretty(X)
````

````
┌────────────┬────────────┬────────────┐
│ x1         │ x2         │ x3         │
│ Float64    │ Float64    │ Float64    │
│ Continuous │ Continuous │ Continuous │
├────────────┼────────────┼────────────┤
│ -6.63967   │ 4.28997    │ -0.647029  │
│ 15.9674    │ 9.23765    │ -18.2396   │
│ -8.4269    │ 6.1368     │ -0.761621  │
│ 0.943899   │ 12.2991    │ -9.13964   │
│ -8.46366   │ 4.65309    │ -2.26029   │
└────────────┴────────────┴────────────┘

````

**Step 0** - Proceed as if you were combining the models "by hand",
using all the data available for training, transforming and
prediction:

````julia
stand = Standardizer();
linear = LogisticClassifier();

mach1 = machine(stand, X);
fit!(mach1);
Xstand = transform(mach1, X);

mach2 = machine(linear, Xstand, y);
fit!(mach2);
yhat = predict(mach2, Xstand)
````

````
5-element MLJBase.UnivariateFiniteVector{Multiclass{3}, Int64, UInt32, Float64}:
 UnivariateFinite{Multiclass{3}}(1=>0.0427, 2=>0.0489, 3=>0.908)
 UnivariateFinite{Multiclass{3}}(1=>0.704, 2=>0.198, 3=>0.0981)
 UnivariateFinite{Multiclass{3}}(1=>0.0468, 2=>0.089, 3=>0.864)
 UnivariateFinite{Multiclass{3}}(1=>0.223, 2=>0.544, 3=>0.233)
 UnivariateFinite{Multiclass{3}}(1=>0.0461, 2=>0.0576, 3=>0.896)
````

**Step 1** - Edit your code as follows:

- pre-wrap the data in `Source` nodes

- delete the `fit!` calls

````julia
X = source(X)  # or X = source() if not testing
y = source(y)  # or y = source()

stand = Standardizer();
linear = LogisticClassifier();

mach1 = machine(stand, X);
Xstand = transform(mach1, X);

mach2 = machine(linear, Xstand, y);
yhat = predict(mach2, Xstand)
````

````
Node{Machine{LogisticClassifier,…}}
  args:
    1:	Node{Machine{Standardizer,…}}
  formula:
    predict(
        [0m[1mMachine{LogisticClassifier,…}[22m, 
        transform(
            [0m[1mMachine{Standardizer,…}[22m, 
            Source @185))
````

Now `X`, `y`, `Xstand` and `yhat` are *nodes* ("variables" or
"dynammic data") instead of data. All training, predicting and
transforming is now executed lazily, whenever we `fit!` one of these
nodes. We *call* a node to retrieve the data it represents in the
original manual workflow.

````julia
fit!(Xstand)
Xstand() |> pretty
````

````
[ Info: Training Machine{Standardizer,…}.
┌────────────┬────────────┬────────────┐
│ x1         │ x2         │ x3         │
│ Float64    │ Float64    │ Float64    │
│ Continuous │ Continuous │ Continuous │
├────────────┼────────────┼────────────┤
│ -0.510448  │ -0.892916  │ 0.734333   │
│ 1.66036    │ 0.563516   │ -1.5881    │
│ -0.682063  │ -0.349268  │ 0.719205   │
│ 0.217749   │ 1.46469    │ -0.386797  │
│ -0.685593  │ -0.786025  │ 0.521362   │
└────────────┴────────────┴────────────┘

````

````julia
fit!(yhat);
yhat()
````

````
5-element MLJBase.UnivariateFiniteVector{Multiclass{3}, Int64, UInt32, Float64}:
 UnivariateFinite{Multiclass{3}}(1=>0.0427, 2=>0.0489, 3=>0.908)
 UnivariateFinite{Multiclass{3}}(1=>0.704, 2=>0.198, 3=>0.0981)
 UnivariateFinite{Multiclass{3}}(1=>0.0468, 2=>0.089, 3=>0.864)
 UnivariateFinite{Multiclass{3}}(1=>0.223, 2=>0.544, 3=>0.233)
 UnivariateFinite{Multiclass{3}}(1=>0.0461, 2=>0.0576, 3=>0.896)
````

The node `yhat` is the "descendant" (in an associated DAG we have
defined) of a unique source node:

````julia
sources(yhat)
````

````
2-element Vector{Any}:
 Source @185 ⏎ `Table{AbstractVector{Continuous}}`
 Source @776 ⏎ `AbstractVector{Multiclass{3}}`
````

The data at the source node is replaced by `Xnew` to obtain a
new prediction when we call `yhat` like this:

````julia
Xnew, _ = make_blobs(2, 3);
yhat(Xnew)
````

````
2-element MLJBase.UnivariateFiniteVector{Multiclass{3}, Int64, UInt32, Float64}:
 UnivariateFinite{Multiclass{3}}(1=>0.00136, 2=>1.46e-5, 3=>0.999)
 UnivariateFinite{Multiclass{3}}(1=>0.00614, 2=>0.000723, 3=>0.993)
````

**Step 2** - Export the learning network as a new stand-alone model type

Now, somewhat paradoxically, we can wrap the whole network in a
special machine - called a *learning network machine* - before have
defined the new model type. Indeed doing so is a necessary step in
the export process, for this machine will tell the export macro:

- what kind of model the composite will be (`Deterministic`,
  `Probabilistic` or `Unsupervised`)a

- which source nodes are input nodes and which are for the target

- which nodes correspond to each operation (`predict`, `transform`,
  etc) that we might want to define

````julia
surrogate = Probabilistic()     # a model with no fields!
mach = machine(surrogate, X, y; predict=yhat)
````

````
Machine{ProbabilisticSurrogate,…} trained 0 times; does not cache data
  args: 
    1:	Source @185 ⏎ `Table{AbstractVector{Continuous}}`
    2:	Source @776 ⏎ `AbstractVector{Multiclass{3}}`

````

Although we have no real need to use it, this machine behaves like
you'd expect it to:

````julia
Xnew, _ = make_blobs(2, 3)
fit!(mach)
predict(mach, Xnew)
````

````
2-element MLJBase.UnivariateFiniteVector{Multiclass{3}, Int64, UInt32, Float64}:
 UnivariateFinite{Multiclass{3}}(1=>0.23, 2=>0.0726, 3=>0.698)
 UnivariateFinite{Multiclass{3}}(1=>0.277, 2=>0.0476, 3=>0.675)
````

Now we create a new model type using a Julia `struct` definition
appropriately decorated:

````julia
@from_network mach begin
    mutable struct YourPipe
        standardizer = stand
        classifier = linear::Probabilistic
    end
end
````

Instantiating and evaluating on some new data:

````julia
pipe = YourPipe()
X, y = @load_iris;   # built-in data set
mach = machine(pipe, X, y)
evaluate!(mach, measure=misclassification_rate, operation=predict_mode)
````

````
PerformanceEvaluation object with these fields:
  measure, measurement, operation, per_fold,
  per_observation, fitted_params_per_fold,
  report_per_fold, train_test_pairs
Extract:
┌─────────────────────────┬─────────────┬──────────────┬────────────────────────────────────┐
│ measure                 │ measurement │ operation    │ per_fold                           │
├─────────────────────────┼─────────────┼──────────────┼────────────────────────────────────┤
│ MisclassificationRate() │ 0.08        │ predict_mode │ [0.0, 0.04, 0.08, 0.08, 0.08, 0.2] │
└─────────────────────────┴─────────────┴──────────────┴────────────────────────────────────┘

````

### A composite model to average two regressor predictors

The following is condensed version of
[this](https://github.com/alan-turing-institute/MLJ.jl/blob/master/binder/MLJ_demo.ipynb)
tutorial. We will define a composite model that:

- standardizes the input data

- learns and applies a Box-Cox transformation to the target variable

- blends the predictions of two supervised learning models - a ridge
 regressor and a random forest regressor; we'll blend using a simple
 average (for a more sophisticated stacking example, see
 [here](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/stacking/))

- applies the *inverse* Box-Cox transformation to this blended prediction

````julia
RandomForestRegressor = @load RandomForestRegressor pkg=DecisionTree
````

````
MLJDecisionTreeInterface.RandomForestRegressor
````

**Input layer**

````julia
X = source()
y = source()
````

````
Source @072 ⏎ `Nothing`
````

**First layer and target transformation**

````julia
std_model = Standardizer()
stand = machine(std_model, X)
W = MLJ.transform(stand, X)

box_model = UnivariateBoxCoxTransformer()
box = machine(box_model, y)
z = MLJ.transform(box, y)
````

````
Node{Machine{UnivariateBoxCoxTransformer,…}}
  args:
    1:	Source @072
  formula:
    transform(
        [0m[1mMachine{UnivariateBoxCoxTransformer,…}[22m, 
        Source @072)
````

**Second layer**

````julia
ridge_model = RidgeRegressor(lambda=0.1)
ridge = machine(ridge_model, W, z)

forest_model = RandomForestRegressor(n_trees=50)
forest = machine(forest_model, W, z)

ẑ = 0.5*predict(ridge, W) + 0.5*predict(forest, W)
````

````
Node{Nothing}
  args:
    1:	Node{Nothing}
    2:	Node{Nothing}
  formula:
    +(
        #134(
            predict(
                [0m[1mMachine{RidgeRegressor,…}[22m, 
                transform(
                    [0m[1mMachine{Standardizer,…}[22m, 
                    Source @231))),
        #134(
            predict(
                [0m[1mMachine{RandomForestRegressor,…}[22m, 
                transform(
                    [0m[1mMachine{Standardizer,…}[22m, 
                    Source @231))))
````

**Output**

````julia
ŷ = inverse_transform(box, ẑ)
````

````
Node{Machine{UnivariateBoxCoxTransformer,…}}
  args:
    1:	Node{Nothing}
  formula:
    inverse_transform(
        [0m[1mMachine{UnivariateBoxCoxTransformer,…}[22m, 
        +(
            #134(
                predict(
                    [0m[1mMachine{RidgeRegressor,…}[22m, 
                    transform(
                        [0m[1mMachine{Standardizer,…}[22m, 
                        Source @231))),
            #134(
                predict(
                    [0m[1mMachine{RandomForestRegressor,…}[22m, 
                    transform(
                        [0m[1mMachine{Standardizer,…}[22m, 
                        Source @231)))))
````

With the learning network defined, we're ready to export:

````julia
@from_network machine(Deterministic(), X, y, predict=ŷ) begin
    mutable struct CompositeModel
        rgs1 = ridge_model
        rgs2 = forest_model
    end
end
````

Let's instantiate the new model type and try it out on some data:

````julia
composite = CompositeModel()
````

````
CompositeModel(
    rgs1 = RidgeRegressor(
            lambda = 0.1,
            fit_intercept = true,
            penalize_intercept = false,
            solver = nothing),
    rgs2 = RandomForestRegressor(
            max_depth = -1,
            min_samples_leaf = 1,
            min_samples_split = 2,
            min_purity_increase = 0.0,
            n_subfeatures = -1,
            n_trees = 50,
            sampling_fraction = 0.7,
            pdf_smoothing = 0.0,
            rng = Random._GLOBAL_RNG()))
````

````julia
X, y = @load_boston;
mach = machine(composite, X, y);
evaluate!(mach,
          resampling=CV(nfolds=6, shuffle=true),
          measures=[rms, mae])
````

````
PerformanceEvaluation object with these fields:
  measure, measurement, operation, per_fold,
  per_observation, fitted_params_per_fold,
  report_per_fold, train_test_pairs
Extract:
┌────────────────────────┬─────────────┬───────────┬──────────────────────────────────────┐
│ measure                │ measurement │ operation │ per_fold                             │
├────────────────────────┼─────────────┼───────────┼──────────────────────────────────────┤
│ RootMeanSquaredError() │ 3.83        │ predict   │ [3.49, 3.07, 5.3, 3.22, 3.09, 4.29]  │
│ MeanAbsoluteError()    │ 2.42        │ predict   │ [2.23, 2.21, 2.97, 2.22, 2.21, 2.71] │
└────────────────────────┴─────────────┴───────────┴──────────────────────────────────────┘

````

### Resources for Part 5

- From the MLJ manual:
   - [Learning Networks](https://alan-turing-institute.github.io/MLJ.jl/stable/composing_models/#Learning-Networks-1)
- From Data Science Tutorials:
    - [Learning Networks](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/learning-networks/)
    - [Learning Networks 2](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/learning-networks-2/)

    - [Stacking](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/learning-networks-2/)
       an advanced example of model compostion

    - [Finer Control](https://alan-turing-institute.github.io/MLJ.jl/dev/composing_models/#Method-II:-Finer-control-(advanced)-1)
      exporting learning networks without a macro for finer control

<a id='solutions-to-exercises'></a>

## Solutions to exercises

#### Exercise 2 solution

````julia
quality = coerce(quality, OrderedFactor);
levels!(quality, ["poor", "good", "excellent"]);
elscitype(quality)
````

````
Union{Missing, OrderedFactor{3}}
````

#### Exercise 3 solution

First pass:

````julia
coerce!(house, autotype(house));
schema(house)
````

````
┌───────────────┬───────────────────────────────────┬───────────────────┐
│ _.names       │ _.types                           │ _.scitypes        │
├───────────────┼───────────────────────────────────┼───────────────────┤
│ price         │ Float64                           │ Continuous        │
│ bedrooms      │ CategoricalValue{Int64, UInt32}   │ OrderedFactor{13} │
│ bathrooms     │ CategoricalValue{Float64, UInt32} │ OrderedFactor{30} │
│ sqft_living   │ Float64                           │ Continuous        │
│ sqft_lot      │ Float64                           │ Continuous        │
│ floors        │ CategoricalValue{Float64, UInt32} │ OrderedFactor{6}  │
│ waterfront    │ CategoricalValue{Int64, UInt32}   │ OrderedFactor{2}  │
│ view          │ CategoricalValue{Int64, UInt32}   │ OrderedFactor{5}  │
│ condition     │ CategoricalValue{Int64, UInt32}   │ OrderedFactor{5}  │
│ grade         │ CategoricalValue{Int64, UInt32}   │ OrderedFactor{12} │
│ sqft_above    │ Float64                           │ Continuous        │
│ sqft_basement │ Float64                           │ Continuous        │
│ yr_built      │ Float64                           │ Continuous        │
│ zipcode       │ CategoricalValue{Int64, UInt32}   │ Multiclass{70}    │
│ lat           │ Float64                           │ Continuous        │
│ long          │ Float64                           │ Continuous        │
│ sqft_living15 │ Float64                           │ Continuous        │
│ sqft_lot15    │ Float64                           │ Continuous        │
│ is_renovated  │ CategoricalValue{Bool, UInt32}    │ OrderedFactor{2}  │
└───────────────┴───────────────────────────────────┴───────────────────┘
_.nrows = 21613

````

All the "sqft" fields refer to "square feet" so are
really `Continuous`. We'll regard `:yr_built` (the other `Count`
variable above) as `Continuous` as well. So:

````julia
coerce!(house, Count => Continuous);
nothing #hide
````

And `:zipcode` should not be ordered:

````julia
coerce!(house, :zipcode => Multiclass);
schema(house)
````

````
┌───────────────┬───────────────────────────────────┬───────────────────┐
│ _.names       │ _.types                           │ _.scitypes        │
├───────────────┼───────────────────────────────────┼───────────────────┤
│ price         │ Float64                           │ Continuous        │
│ bedrooms      │ CategoricalValue{Int64, UInt32}   │ OrderedFactor{13} │
│ bathrooms     │ CategoricalValue{Float64, UInt32} │ OrderedFactor{30} │
│ sqft_living   │ Float64                           │ Continuous        │
│ sqft_lot      │ Float64                           │ Continuous        │
│ floors        │ CategoricalValue{Float64, UInt32} │ OrderedFactor{6}  │
│ waterfront    │ CategoricalValue{Int64, UInt32}   │ OrderedFactor{2}  │
│ view          │ CategoricalValue{Int64, UInt32}   │ OrderedFactor{5}  │
│ condition     │ CategoricalValue{Int64, UInt32}   │ OrderedFactor{5}  │
│ grade         │ CategoricalValue{Int64, UInt32}   │ OrderedFactor{12} │
│ sqft_above    │ Float64                           │ Continuous        │
│ sqft_basement │ Float64                           │ Continuous        │
│ yr_built      │ Float64                           │ Continuous        │
│ zipcode       │ CategoricalValue{Int64, UInt32}   │ Multiclass{70}    │
│ lat           │ Float64                           │ Continuous        │
│ long          │ Float64                           │ Continuous        │
│ sqft_living15 │ Float64                           │ Continuous        │
│ sqft_lot15    │ Float64                           │ Continuous        │
│ is_renovated  │ CategoricalValue{Bool, UInt32}    │ OrderedFactor{2}  │
└───────────────┴───────────────────────────────────┴───────────────────┘
_.nrows = 21613

````

`:bathrooms` looks like it has a lot of levels, but on further
inspection we see why, and `OrderedFactor` remains appropriate:

````julia
import StatsBase.countmap
countmap(house.bathrooms)
````

````
Dict{CategoricalValue{Float64, UInt32}, Int64} with 30 entries:
  5.0 => 21
  5.25 => 13
  1.25 => 9
  8.0 => 2
  6.75 => 2
  1.0 => 3852
  5.5 => 10
  0.0 => 10
  6.0 => 6
  6.25 => 2
  4.75 => 23
  3.25 => 589
  3.0 => 753
  2.25 => 2047
  0.5 => 4
  7.5 => 1
  5.75 => 4
  1.5 => 1446
  3.75 => 155
  4.0 => 136
  4.25 => 79
  2.0 => 1930
  2.75 => 1185
  3.5 => 731
  6.5 => 2
  1.75 => 3048
  0.75 => 72
  2.5 => 5380
  4.5 => 100
  7.75 => 1
````

#### Exercise 4 solution

4(a)

There are *no* models that apply immediately:

````julia
models(matching(X4, y4))
````

````
NamedTuple{(:name, :package_name, :is_supervised, :abstract_type, :deep_properties, :docstring, :fit_data_scitype, :hyperparameter_ranges, :hyperparameter_types, :hyperparameters, :implemented_methods, :inverse_transform_scitype, :is_pure_julia, :is_wrapper, :iteration_parameter, :load_path, :package_license, :package_url, :package_uuid, :predict_scitype, :prediction_type, :supports_class_weights, :supports_online, :supports_training_losses, :supports_weights, :transform_scitype, :input_scitype, :target_scitype, :output_scitype), T} where T<:Tuple[]
````

4(b)

````julia
y4 = coerce(y4, Continuous);
models(matching(X4, y4))
````

````
6-element Vector{NamedTuple{(:name, :package_name, :is_supervised, :abstract_type, :deep_properties, :docstring, :fit_data_scitype, :hyperparameter_ranges, :hyperparameter_types, :hyperparameters, :implemented_methods, :inverse_transform_scitype, :is_pure_julia, :is_wrapper, :iteration_parameter, :load_path, :package_license, :package_url, :package_uuid, :predict_scitype, :prediction_type, :supports_class_weights, :supports_online, :supports_training_losses, :supports_weights, :transform_scitype, :input_scitype, :target_scitype, :output_scitype), T} where T<:Tuple}:
 (name = ConstantRegressor, package_name = MLJModels, ... )
 (name = DecisionTreeRegressor, package_name = BetaML, ... )
 (name = DecisionTreeRegressor, package_name = DecisionTree, ... )
 (name = DeterministicConstantRegressor, package_name = MLJModels, ... )
 (name = RandomForestRegressor, package_name = BetaML, ... )
 (name = RandomForestRegressor, package_name = DecisionTree, ... )
````

#### Exercise 6 solution

6(a)

````julia
y, X = unpack(horse,
              ==(:outcome),
              name -> elscitype(Tables.getcolumn(horse, name)) == Continuous);
nothing #hide
````

6(b)(i)

````julia
model = (@load LogisticClassifier pkg=MLJLinearModels)();
model.lambda = 100
mach = machine(model, X, y)
fit!(mach, rows=train)
fitted_params(mach)
````

````
(classes = CategoricalValue{Int64, UInt32}[1, 2, 3],
 coefs = Pair{Symbol, SubArray{Float64, 1, Matrix{Float64}, Tuple{Int64, Base.Slice{Base.OneTo{Int64}}}, true}}[:rectal_temperature => [0.061700165029243595, -0.06507181612449195, 0.003371651095248075], :pulse => [-0.009584825604133507, 0.004022558654903623, 0.005562266949230421], :respiratory_rate => [-0.009584825604133507, 0.004022558654903623, 0.005562266949230421], :packed_cell_volume => [-0.04309372172868995, 0.020859863946466082, 0.022233857782222862], :total_protein => [0.02750875240430624, -0.06317268051316188, 0.03566392810885553]],
 intercept = [0.0008917385182322375, -0.000891738518934035, -4.972412447643922],)
````

````julia
coefs_given_feature = Dict(fitted_params(mach).coefs)
coefs_given_feature[:pulse]

#6(b)(ii)

yhat = predict(mach, rows=test); # or predict(mach, X[test,:])
err = cross_entropy(yhat, y[test]) |> mean
````

````
0.7187276476001894
````

6(b)(iii)

The predicted probabilities of the actual observations in the test
are given by

````julia
p = broadcast(pdf, yhat, y[test]);
nothing #hide
````

The number of times this probability exceeds 50% is:

````julia
n50 = filter(x -> x > 0.5, p) |> length
````

````
30
````

Or, as a proportion:

````julia
n50/length(test)
````

````
0.6666666666666666
````

6(b)(iv)

````julia
misclassification_rate(mode.(yhat), y[test])
````

````
0.28888888888888886
````

6(c)(i)

````julia
model = (@load RandomForestClassifier pkg=DecisionTree)()
mach = machine(model, X, y)
evaluate!(mach, resampling=CV(nfolds=6), measure=cross_entropy)

r = range(model, :n_trees, lower=10, upper=70, scale=:log10)
````

````
NumericRange(10 ≤ n_trees ≤ 70; origin=40.0, unit=30.0) on log10 scale
````

Since random forests are inherently randomized, we generate multiple
curves:

````julia
plt = plot()
for i in 1:4
    one_curve = learning_curve(mach,
                           range=r,
                           resampling=Holdout(),
                           measure=cross_entropy)
    plt=plot!(one_curve.parameter_values, one_curve.measurements)
end
xlabel!(plt, "n_trees")
ylabel!(plt, "cross entropy")
plt
````

```@raw html
<script src="https://cdn.plot.ly/plotly-1.57.1.min.js"></script>    <div id="6e033c6b-fac4-446f-a15e-a7a8f14de7b6" style="width:490px;height:300px;"></div>
    <script>
    
        var PLOT = document.getElementById('6e033c6b-fac4-446f-a15e-a7a8f14de7b6');
    Plotly.plot(PLOT, [
    {
        "xaxis": "x",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y",
        "x": [
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
            17.0,
            18.0,
            20.0,
            21.0,
            22.0,
            24.0,
            26.0,
            27.0,
            29.0,
            31.0,
            33.0,
            36.0,
            38.0,
            41.0,
            44.0,
            47.0,
            50.0,
            54.0,
            57.0,
            61.0,
            65.0,
            70.0
        ],
        "showlegend": true,
        "mode": "lines",
        "name": "y1",
        "zmin": null,
        "legendgroup": "y1",
        "zmax": null,
        "line": {
            "color": "rgba(0, 154, 250, 1.000)",
            "shape": "linear",
            "dash": "solid",
            "width": 1
        },
        "y": [
            2.500592629219947,
            2.4844079381716195,
            1.589419049629833,
            2.201692075243932,
            1.515721068423343,
            1.6347827036241174,
            1.8696531100049416,
            1.5441411231166788,
            1.6126301663014575,
            1.5882631712678525,
            1.5653964891991226,
            1.5587275032786951,
            1.59065542574611,
            1.876681001711597,
            1.604196730762838,
            1.2959188958851218,
            1.243570023099615,
            1.8833384202412176,
            1.5784577020000967,
            0.9112491057047294,
            1.2774566677603563,
            0.9677190607000338,
            0.9315968238231759,
            1.2930135879591567,
            1.2831691326827137,
            1.5670985807057622,
            1.0162385304807988,
            0.9596572079013734,
            1.2695842704709341
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y",
        "x": [
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
            17.0,
            18.0,
            20.0,
            21.0,
            22.0,
            24.0,
            26.0,
            27.0,
            29.0,
            31.0,
            33.0,
            36.0,
            38.0,
            41.0,
            44.0,
            47.0,
            50.0,
            54.0,
            57.0,
            61.0,
            65.0,
            70.0
        ],
        "showlegend": true,
        "mode": "lines",
        "name": "y2",
        "zmin": null,
        "legendgroup": "y2",
        "zmax": null,
        "line": {
            "color": "rgba(227, 111, 71, 1.000)",
            "shape": "linear",
            "dash": "solid",
            "width": 1
        },
        "y": [
            1.883899554238724,
            1.625867734175794,
            2.5022142975220754,
            2.1623448775741125,
            1.9288387074644613,
            1.8788667647156383,
            2.5133822787967963,
            1.6020239405698984,
            1.8671017146352946,
            1.3121601733020927,
            1.872251431920012,
            1.5548314755184725,
            0.9587632687903472,
            1.2664791929730221,
            1.5803187544213393,
            1.6045315999111314,
            1.5531099582741648,
            1.294948084181965,
            1.5694522851356478,
            1.5836030459520019,
            1.305398724633725,
            0.655934689357321,
            0.9854311522748207,
            1.2523588519285673,
            1.2512513479180503,
            0.6805919124383506,
            1.2563422381852816,
            0.9910248432720332,
            0.9695139903590071
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y",
        "x": [
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
            17.0,
            18.0,
            20.0,
            21.0,
            22.0,
            24.0,
            26.0,
            27.0,
            29.0,
            31.0,
            33.0,
            36.0,
            38.0,
            41.0,
            44.0,
            47.0,
            50.0,
            54.0,
            57.0,
            61.0,
            65.0,
            70.0
        ],
        "showlegend": true,
        "mode": "lines",
        "name": "y3",
        "zmin": null,
        "legendgroup": "y3",
        "zmax": null,
        "line": {
            "color": "rgba(62, 164, 78, 1.000)",
            "shape": "linear",
            "dash": "solid",
            "width": 1
        },
        "y": [
            2.80810377562416,
            1.5450100059267853,
            2.1905480590706006,
            2.1567800153555465,
            1.2619388450020796,
            1.8892089921655586,
            2.1880639893892915,
            1.8689806384499592,
            1.8643339415381082,
            0.9893528722295268,
            1.5576115427962824,
            1.5959636566489295,
            1.599671434872608,
            1.882757359621781,
            1.5850257418609048,
            1.275473687534327,
            1.576221418624591,
            1.5650140114892312,
            1.255081080486682,
            1.2541176998342265,
            1.0011295054428586,
            1.2519341823754833,
            1.256250767927983,
            0.9888239260497446,
            0.9569400149262214,
            0.9833664635736361,
            1.2830280209366278,
            1.276077490464742,
            1.5407912364558107
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y",
        "x": [
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
            17.0,
            18.0,
            20.0,
            21.0,
            22.0,
            24.0,
            26.0,
            27.0,
            29.0,
            31.0,
            33.0,
            36.0,
            38.0,
            41.0,
            44.0,
            47.0,
            50.0,
            54.0,
            57.0,
            61.0,
            65.0,
            70.0
        ],
        "showlegend": true,
        "mode": "lines",
        "name": "y4",
        "zmin": null,
        "legendgroup": "y4",
        "zmax": null,
        "line": {
            "color": "rgba(195, 113, 210, 1.000)",
            "shape": "linear",
            "dash": "solid",
            "width": 1
        },
        "y": [
            1.5527906831910534,
            2.175201053397431,
            1.5450866239159031,
            2.4634675656676506,
            1.5762461644329513,
            1.9151838213681003,
            2.2212885663691533,
            1.8972433425266422,
            1.5339495157378469,
            1.536969528935671,
            1.5830413491024007,
            1.2805761659044252,
            1.3037447037152468,
            1.8661014705949202,
            0.9592656068137831,
            1.857110429723102,
            2.1772120865811417,
            1.294893843575161,
            0.955605393559062,
            1.2884121799797748,
            1.5851870710761609,
            0.9498310264888373,
            1.5583453580256361,
            1.3023979744277596,
            1.8745735076348031,
            1.5536463392295161,
            1.280504820345702,
            1.5482822115514197,
            0.9720927318214075
        ],
        "type": "scatter"
    }
]
, {
    "showlegend": true,
    "xaxis": {
        "showticklabels": true,
        "gridwidth": 0.5,
        "tickvals": [
            10.0,
            20.0,
            30.0,
            40.0,
            50.0,
            60.0,
            70.0
        ],
        "range": [
            8.2,
            71.8
        ],
        "domain": [
            0.09363561697644936,
            0.9919652900530291
        ],
        "mirror": false,
        "tickangle": 0,
        "showline": true,
        "ticktext": [
            "10",
            "20",
            "30",
            "40",
            "50",
            "60",
            "70"
        ],
        "zeroline": false,
        "tickfont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "zerolinecolor": "rgba(0, 0, 0, 1.000)",
        "anchor": "y",
        "visible": true,
        "ticks": "inside",
        "tickmode": "array",
        "linecolor": "rgba(0, 0, 0, 1.000)",
        "showgrid": true,
        "title": "n_trees",
        "gridcolor": "rgba(0, 0, 0, 0.100)",
        "titlefont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 15
        },
        "tickcolor": "rgb(0, 0, 0)",
        "type": "-"
    },
    "paper_bgcolor": "rgba(255, 255, 255, 1.000)",
    "annotations": [],
    "height": 300,
    "margin": {
        "l": 0,
        "b": 20,
        "r": 0,
        "t": 20
    },
    "plot_bgcolor": "rgba(255, 255, 255, 1.000)",
    "yaxis": {
        "showticklabels": true,
        "gridwidth": 0.5,
        "tickvals": [
            1.0,
            1.5,
            2.0,
            2.5
        ],
        "range": [
            0.5913696167693159,
            2.8726688482121654
        ],
        "domain": [
            0.10108632254301551,
            0.9868766404199475
        ],
        "mirror": false,
        "tickangle": 0,
        "showline": true,
        "ticktext": [
            "1.0",
            "1.5",
            "2.0",
            "2.5"
        ],
        "zeroline": false,
        "tickfont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "zerolinecolor": "rgba(0, 0, 0, 1.000)",
        "anchor": "x",
        "visible": true,
        "ticks": "inside",
        "tickmode": "array",
        "linecolor": "rgba(0, 0, 0, 1.000)",
        "showgrid": true,
        "title": "cross entropy",
        "gridcolor": "rgba(0, 0, 0, 0.100)",
        "titlefont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 15
        },
        "tickcolor": "rgb(0, 0, 0)",
        "type": "-"
    },
    "legend": {
        "yanchor": "auto",
        "xanchor": "auto",
        "bordercolor": "rgba(0, 0, 0, 1.000)",
        "bgcolor": "rgba(255, 255, 255, 1.000)",
        "borderwidth": 1,
        "tracegroupgap": 0,
        "y": 1.0,
        "font": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "title": {
            "font": {
                "color": "rgba(0, 0, 0, 1.000)",
                "family": "sans-serif",
                "size": 15
            },
            "text": ""
        },
        "traceorder": "normal",
        "x": 1.0
    },
    "width": 490
}
);

    
    </script>

```

6(c)(ii)

````julia
evaluate!(mach, resampling=CV(nfolds=9),
                measure=cross_entropy,
                rows=train).measurement[1]

model.n_trees = 90
````

````
90
````

6(c)(iii)

````julia
err_forest = evaluate!(mach, resampling=Holdout(),
                       measure=cross_entropy).measurement[1]
````

````
0.7026647659355011
````

#### Exercise 7

(a)

````julia
KMeans = @load KMeans pkg=Clustering
EvoTreeClassifier = @load EvoTreeClassifier
pipe = @pipeline(Standardizer,
                 ContinuousEncoder,
                 KMeans(k=10),
                 EvoTreeClassifier(nrounds=50))
````

````
Pipeline665(
    standardizer = Standardizer(
            features = Symbol[],
            ignore = false,
            ordered_factor = false,
            count = false),
    continuous_encoder = ContinuousEncoder(
            drop_last = false,
            one_hot_ordered_factors = false),
    k_means = KMeans(
            k = 10,
            metric = Distances.SqEuclidean(0.0)),
    evo_tree_classifier = EvoTreeClassifier(
            loss = EvoTrees.Softmax(),
            nrounds = 50,
            λ = 0.0,
            γ = 0.0,
            η = 0.1,
            max_depth = 5,
            min_weight = 1.0,
            rowsample = 1.0,
            colsample = 1.0,
            nbins = 64,
            α = 0.5,
            metric = :mlogloss,
            rng = MersenneTwister(123),
            device = "cpu"))
````

(b)

````julia
mach = machine(pipe, X, y)
evaluate!(mach, resampling=CV(nfolds=6), measure=cross_entropy)
````

````
PerformanceEvaluation object with these fields:
  measure, measurement, operation, per_fold,
  per_observation, fitted_params_per_fold,
  report_per_fold, train_test_pairs
Extract:
┌────────────────────────────┬─────────────┬───────────┬───────────────────────────────────────────┐
│ measure                    │ measurement │ operation │ per_fold                                  │
├────────────────────────────┼─────────────┼───────────┼───────────────────────────────────────────┤
│ LogLoss(tol = 2.22045e-16) │ 0.804       │ predict   │ [0.874, 0.99, 0.662, 0.789, 0.776, 0.734] │
└────────────────────────────┴─────────────┴───────────┴───────────────────────────────────────────┘

````

(c)

````julia
r = range(pipe, :(evo_tree_classifier.max_depth), lower=1, upper=10)

curve = learning_curve(mach,
                       range=r,
                       resampling=CV(nfolds=6),
                       measure=cross_entropy)

plt = plot(curve.parameter_values, curve.measurements)
xlabel!(plt, "max_depth")
ylabel!(plt, "CV estimate of cross entropy")
plt
````

```@raw html
<script src="https://cdn.plot.ly/plotly-1.57.1.min.js"></script>    <div id="ec2638a4-ab84-4b52-914d-478ae92af3c0" style="width:490px;height:300px;"></div>
    <script>
    
        var PLOT = document.getElementById('ec2638a4-ab84-4b52-914d-478ae92af3c0');
    Plotly.plot(PLOT, [
    {
        "xaxis": "x",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y",
        "x": [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0
        ],
        "showlegend": true,
        "mode": "lines",
        "name": "y1",
        "zmin": null,
        "legendgroup": "y1",
        "zmax": null,
        "line": {
            "color": "rgba(0, 154, 250, 1.000)",
            "shape": "linear",
            "dash": "solid",
            "width": 1
        },
        "y": [
            0.9238731498849247,
            0.7745626577651471,
            0.7406583158592932,
            0.7946563699634687,
            0.8125338673899253,
            0.8173011861881747,
            0.8765012519612285,
            0.9826805809060014,
            0.9505051671900722,
            0.9639596945260562
        ],
        "type": "scatter"
    }
]
, {
    "showlegend": true,
    "xaxis": {
        "showticklabels": true,
        "gridwidth": 0.5,
        "tickvals": [
            2.0,
            4.0,
            6.0,
            8.0,
            10.0
        ],
        "range": [
            0.73,
            10.27
        ],
        "domain": [
            0.11177620654561035,
            0.9919652900530291
        ],
        "mirror": false,
        "tickangle": 0,
        "showline": true,
        "ticktext": [
            "2",
            "4",
            "6",
            "8",
            "10"
        ],
        "zeroline": false,
        "tickfont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "zerolinecolor": "rgba(0, 0, 0, 1.000)",
        "anchor": "y",
        "visible": true,
        "ticks": "inside",
        "tickmode": "array",
        "linecolor": "rgba(0, 0, 0, 1.000)",
        "showgrid": true,
        "title": "max_depth",
        "gridcolor": "rgba(0, 0, 0, 0.100)",
        "titlefont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 15
        },
        "tickcolor": "rgb(0, 0, 0)",
        "type": "-"
    },
    "paper_bgcolor": "rgba(255, 255, 255, 1.000)",
    "annotations": [],
    "height": 300,
    "margin": {
        "l": 0,
        "b": 20,
        "r": 0,
        "t": 20
    },
    "plot_bgcolor": "rgba(255, 255, 255, 1.000)",
    "yaxis": {
        "showticklabels": true,
        "gridwidth": 0.5,
        "tickvals": [
            0.75,
            0.8,
            0.8500000000000001,
            0.9,
            0.9500000000000001
        ],
        "range": [
            0.733397647907892,
            0.9899412488574026
        ],
        "domain": [
            0.10108632254301551,
            0.9868766404199475
        ],
        "mirror": false,
        "tickangle": 0,
        "showline": true,
        "ticktext": [
            "0.75",
            "0.80",
            "0.85",
            "0.90",
            "0.95"
        ],
        "zeroline": false,
        "tickfont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "zerolinecolor": "rgba(0, 0, 0, 1.000)",
        "anchor": "x",
        "visible": true,
        "ticks": "inside",
        "tickmode": "array",
        "linecolor": "rgba(0, 0, 0, 1.000)",
        "showgrid": true,
        "title": "CV estimate of cross entropy",
        "gridcolor": "rgba(0, 0, 0, 0.100)",
        "titlefont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 15
        },
        "tickcolor": "rgb(0, 0, 0)",
        "type": "-"
    },
    "legend": {
        "yanchor": "auto",
        "xanchor": "auto",
        "bordercolor": "rgba(0, 0, 0, 1.000)",
        "bgcolor": "rgba(255, 255, 255, 1.000)",
        "borderwidth": 1,
        "tracegroupgap": 0,
        "y": 1.0,
        "font": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "title": {
            "font": {
                "color": "rgba(0, 0, 0, 1.000)",
                "family": "sans-serif",
                "size": 15
            },
            "text": ""
        },
        "traceorder": "normal",
        "x": 1.0
    },
    "width": 490
}
);

    
    </script>

```

Here's a second curve using a different random seed for the booster:

````julia
using Random
pipe.evo_tree_classifier.rng = MersenneTwister(123)
curve = learning_curve(mach,
                       range=r,
                       resampling=CV(nfolds=6),
                       measure=cross_entropy)
plot!(curve.parameter_values, curve.measurements)
````

```@raw html
<script src="https://cdn.plot.ly/plotly-1.57.1.min.js"></script>    <div id="c21fffa3-9bf3-4cbb-8de1-9295c2f4cced" style="width:490px;height:300px;"></div>
    <script>
    
        var PLOT = document.getElementById('c21fffa3-9bf3-4cbb-8de1-9295c2f4cced');
    Plotly.plot(PLOT, [
    {
        "xaxis": "x",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y",
        "x": [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0
        ],
        "showlegend": true,
        "mode": "lines",
        "name": "y1",
        "zmin": null,
        "legendgroup": "y1",
        "zmax": null,
        "line": {
            "color": "rgba(0, 154, 250, 1.000)",
            "shape": "linear",
            "dash": "solid",
            "width": 1
        },
        "y": [
            0.9238731498849247,
            0.7745626577651471,
            0.7406583158592932,
            0.7946563699634687,
            0.8125338673899253,
            0.8173011861881747,
            0.8765012519612285,
            0.9826805809060014,
            0.9505051671900722,
            0.9639596945260562
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y",
        "x": [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0
        ],
        "showlegend": true,
        "mode": "lines",
        "name": "y2",
        "zmin": null,
        "legendgroup": "y2",
        "zmax": null,
        "line": {
            "color": "rgba(227, 111, 71, 1.000)",
            "shape": "linear",
            "dash": "solid",
            "width": 1
        },
        "y": [
            0.9238731498849247,
            0.795228936318412,
            0.7727722336927306,
            0.7736236792566343,
            0.8022407995094171,
            0.8171364945813848,
            0.8374830251445303,
            0.9752326407357771,
            1.0029594328432385,
            0.9462890870977344
        ],
        "type": "scatter"
    }
]
, {
    "showlegend": true,
    "xaxis": {
        "showticklabels": true,
        "gridwidth": 0.5,
        "tickvals": [
            2.0,
            4.0,
            6.0,
            8.0,
            10.0
        ],
        "range": [
            0.73,
            10.27
        ],
        "domain": [
            0.11177620654561035,
            0.9919652900530291
        ],
        "mirror": false,
        "tickangle": 0,
        "showline": true,
        "ticktext": [
            "2",
            "4",
            "6",
            "8",
            "10"
        ],
        "zeroline": false,
        "tickfont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "zerolinecolor": "rgba(0, 0, 0, 1.000)",
        "anchor": "y",
        "visible": true,
        "ticks": "inside",
        "tickmode": "array",
        "linecolor": "rgba(0, 0, 0, 1.000)",
        "showgrid": true,
        "title": "max_depth",
        "gridcolor": "rgba(0, 0, 0, 0.100)",
        "titlefont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 15
        },
        "tickcolor": "rgb(0, 0, 0)",
        "type": "-"
    },
    "paper_bgcolor": "rgba(255, 255, 255, 1.000)",
    "annotations": [],
    "height": 300,
    "margin": {
        "l": 0,
        "b": 20,
        "r": 0,
        "t": 20
    },
    "plot_bgcolor": "rgba(255, 255, 255, 1.000)",
    "yaxis": {
        "showticklabels": true,
        "gridwidth": 0.5,
        "tickvals": [
            0.75,
            0.8,
            0.8500000000000001,
            0.9,
            0.9500000000000001,
            1.0
        ],
        "range": [
            0.7327892823497748,
            1.010828466352757
        ],
        "domain": [
            0.10108632254301551,
            0.9868766404199475
        ],
        "mirror": false,
        "tickangle": 0,
        "showline": true,
        "ticktext": [
            "0.75",
            "0.80",
            "0.85",
            "0.90",
            "0.95",
            "1.00"
        ],
        "zeroline": false,
        "tickfont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "zerolinecolor": "rgba(0, 0, 0, 1.000)",
        "anchor": "x",
        "visible": true,
        "ticks": "inside",
        "tickmode": "array",
        "linecolor": "rgba(0, 0, 0, 1.000)",
        "showgrid": true,
        "title": "CV estimate of cross entropy",
        "gridcolor": "rgba(0, 0, 0, 0.100)",
        "titlefont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 15
        },
        "tickcolor": "rgb(0, 0, 0)",
        "type": "-"
    },
    "legend": {
        "yanchor": "auto",
        "xanchor": "auto",
        "bordercolor": "rgba(0, 0, 0, 1.000)",
        "bgcolor": "rgba(255, 255, 255, 1.000)",
        "borderwidth": 1,
        "tracegroupgap": 0,
        "y": 1.0,
        "font": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "title": {
            "font": {
                "color": "rgba(0, 0, 0, 1.000)",
                "family": "sans-serif",
                "size": 15
            },
            "text": ""
        },
        "traceorder": "normal",
        "x": 1.0
    },
    "width": 490
}
);

    
    </script>

```

One can automatic the production of multiple curves with different
seeds in the following way:

````julia
curves = learning_curve(mach,
                        range=r,
                        resampling=CV(nfolds=6),
                        measure=cross_entropy,
                        rng_name=:(evo_tree_classifier.rng),
                        rngs=6) # list of RNGs, or num to auto generate
plot(curves.parameter_values, curves.measurements)
````

```@raw html
<script src="https://cdn.plot.ly/plotly-1.57.1.min.js"></script>    <div id="1beb7da3-98f0-4a46-8ded-f7ef65a9eb7d" style="width:490px;height:300px;"></div>
    <script>
    
        var PLOT = document.getElementById('1beb7da3-98f0-4a46-8ded-f7ef65a9eb7d');
    Plotly.plot(PLOT, [
    {
        "xaxis": "x",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y",
        "x": [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0
        ],
        "showlegend": true,
        "mode": "lines",
        "name": "y1",
        "zmin": null,
        "legendgroup": "y1",
        "zmax": null,
        "line": {
            "color": "rgba(0, 154, 250, 1.000)",
            "shape": "linear",
            "dash": "solid",
            "width": 1
        },
        "y": [
            0.9238731498849247,
            0.767677935754984,
            0.7786128987870696,
            0.7766682353724471,
            0.8447195263531576,
            0.8503118961425608,
            0.871288635946203,
            0.9505346481123973,
            0.8970884689016629,
            0.9985844048714466
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y",
        "x": [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0
        ],
        "showlegend": true,
        "mode": "lines",
        "name": "y2",
        "zmin": null,
        "legendgroup": "y2",
        "zmax": null,
        "line": {
            "color": "rgba(227, 111, 71, 1.000)",
            "shape": "linear",
            "dash": "solid",
            "width": 1
        },
        "y": [
            0.9238731498849247,
            0.7662635496998623,
            0.7751446463692543,
            0.7741287611733841,
            0.8107759398247611,
            0.8154300032658872,
            0.8439963201653328,
            0.9511511319474469,
            0.9197675650490673,
            1.059406337079816
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y",
        "x": [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0
        ],
        "showlegend": true,
        "mode": "lines",
        "name": "y3",
        "zmin": null,
        "legendgroup": "y3",
        "zmax": null,
        "line": {
            "color": "rgba(62, 164, 78, 1.000)",
            "shape": "linear",
            "dash": "solid",
            "width": 1
        },
        "y": [
            0.9238731498849247,
            0.7852084126004854,
            0.7619887546012379,
            0.7880481466883781,
            0.8245855322668941,
            0.8653131947218237,
            0.8609839296904048,
            0.9270419072460047,
            0.9320549766470666,
            0.9671343761402054
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y",
        "x": [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0
        ],
        "showlegend": true,
        "mode": "lines",
        "name": "y4",
        "zmin": null,
        "legendgroup": "y4",
        "zmax": null,
        "line": {
            "color": "rgba(195, 113, 210, 1.000)",
            "shape": "linear",
            "dash": "solid",
            "width": 1
        },
        "y": [
            0.9238731498849247,
            0.8008075646867914,
            0.7643448420396503,
            0.7876663808908777,
            0.8059816452377436,
            0.7725585525129796,
            0.9307285213708983,
            0.9201057700781181,
            0.9540019557709658,
            0.981086136189291
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y",
        "x": [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0
        ],
        "showlegend": true,
        "mode": "lines",
        "name": "y5",
        "zmin": null,
        "legendgroup": "y5",
        "zmax": null,
        "line": {
            "color": "rgba(172, 142, 24, 1.000)",
            "shape": "linear",
            "dash": "solid",
            "width": 1
        },
        "y": [
            0.9238731498849247,
            0.784778431791898,
            0.7695142315970737,
            0.78628742183407,
            0.7993960306571676,
            0.8482626960394204,
            0.8994467957541467,
            0.901331632951137,
            0.9696814349568762,
            1.0473450855016655
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y",
        "x": [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0
        ],
        "showlegend": true,
        "mode": "lines",
        "name": "y6",
        "zmin": null,
        "legendgroup": "y6",
        "zmax": null,
        "line": {
            "color": "rgba(0, 170, 174, 1.000)",
            "shape": "linear",
            "dash": "solid",
            "width": 1
        },
        "y": [
            0.9238731498849247,
            0.7812808505547748,
            0.7844177421806817,
            0.7742691482959025,
            0.8083476992677316,
            0.835725539857429,
            0.8683510886156259,
            0.9076539761822641,
            0.9263029332525514,
            1.0105062542840904
        ],
        "type": "scatter"
    }
]
, {
    "showlegend": true,
    "xaxis": {
        "showticklabels": true,
        "gridwidth": 0.5,
        "tickvals": [
            2.0,
            4.0,
            6.0,
            8.0,
            10.0
        ],
        "range": [
            0.73,
            10.27
        ],
        "domain": [
            0.0805970682236149,
            0.991965290053029
        ],
        "mirror": false,
        "tickangle": 0,
        "showline": true,
        "ticktext": [
            "2",
            "4",
            "6",
            "8",
            "10"
        ],
        "zeroline": false,
        "tickfont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "zerolinecolor": "rgba(0, 0, 0, 1.000)",
        "anchor": "y",
        "visible": true,
        "ticks": "inside",
        "tickmode": "array",
        "linecolor": "rgba(0, 0, 0, 1.000)",
        "showgrid": true,
        "title": "",
        "gridcolor": "rgba(0, 0, 0, 0.100)",
        "titlefont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 15
        },
        "tickcolor": "rgb(0, 0, 0)",
        "type": "-"
    },
    "paper_bgcolor": "rgba(255, 255, 255, 1.000)",
    "annotations": [],
    "height": 300,
    "margin": {
        "l": 0,
        "b": 20,
        "r": 0,
        "t": 20
    },
    "plot_bgcolor": "rgba(255, 255, 255, 1.000)",
    "yaxis": {
        "showticklabels": true,
        "gridwidth": 0.5,
        "tickvals": [
            0.8,
            0.8500000000000001,
            0.9,
            0.9500000000000001,
            1.0,
            1.05
        ],
        "range": [
            0.7530662271268806,
            1.0683288645541733
        ],
        "domain": [
            0.050160396617089535,
            0.9868766404199475
        ],
        "mirror": false,
        "tickangle": 0,
        "showline": true,
        "ticktext": [
            "0.80",
            "0.85",
            "0.90",
            "0.95",
            "1.00",
            "1.05"
        ],
        "zeroline": false,
        "tickfont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "zerolinecolor": "rgba(0, 0, 0, 1.000)",
        "anchor": "x",
        "visible": true,
        "ticks": "inside",
        "tickmode": "array",
        "linecolor": "rgba(0, 0, 0, 1.000)",
        "showgrid": true,
        "title": "",
        "gridcolor": "rgba(0, 0, 0, 0.100)",
        "titlefont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 15
        },
        "tickcolor": "rgb(0, 0, 0)",
        "type": "-"
    },
    "legend": {
        "yanchor": "auto",
        "xanchor": "auto",
        "bordercolor": "rgba(0, 0, 0, 1.000)",
        "bgcolor": "rgba(255, 255, 255, 1.000)",
        "borderwidth": 1,
        "tracegroupgap": 0,
        "y": 1.0,
        "font": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "title": {
            "font": {
                "color": "rgba(0, 0, 0, 1.000)",
                "family": "sans-serif",
                "size": 15
            },
            "text": ""
        },
        "traceorder": "normal",
        "x": 1.0
    },
    "width": 490
}
);

    
    </script>

```

If you have multiple threads available in your julia session, you
can add the option `acceleration=CPUThreads()` to speed up this
computation.

#### Exercise 8

````julia
y, X = unpack(house, ==(:price), name -> true, rng=123);

EvoTreeRegressor = @load EvoTreeRegressor
tree_booster = EvoTreeRegressor(nrounds = 70)
model = @pipeline ContinuousEncoder tree_booster
````

````
Pipeline674(
    continuous_encoder = ContinuousEncoder(
            drop_last = false,
            one_hot_ordered_factors = false),
    evo_tree_regressor = EvoTreeRegressor(
            loss = EvoTrees.Linear(),
            nrounds = 70,
            λ = 0.0,
            γ = 0.0,
            η = 0.1,
            max_depth = 5,
            min_weight = 1.0,
            rowsample = 1.0,
            colsample = 1.0,
            nbins = 64,
            α = 0.5,
            metric = :mse,
            rng = MersenneTwister(123),
            device = "cpu"))
````

(a)

````julia
r1 = range(model, :(evo_tree_regressor.max_depth), lower=1, upper=12)
````

````
NumericRange(1 ≤ evo_tree_regressor.max_depth ≤ 12; origin=6.5, unit=5.5)
````

(c)

````julia
tuned_model = TunedModel(model=model,
                         ranges=[r1, r2],
                         resampling=Holdout(),
                         measures=mae,
                         tuning=RandomSearch(rng=123),
                         n=40)

tuned_mach = machine(tuned_model, X, y) |> fit!
plot(tuned_mach)
````

```@raw html
<script src="https://cdn.plot.ly/plotly-1.57.1.min.js"></script>    <div id="60c64092-b272-45ba-aa6d-61fa11803898" style="width:550px;height:500px;"></div>
    <script>
    
        var PLOT = document.getElementById('60c64092-b272-45ba-aa6d-61fa11803898');
    Plotly.plot(PLOT, [
    {
        "xaxis": "x1",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y1",
        "x": [
            8.0,
            4.0,
            4.0,
            6.0,
            4.0,
            4.0,
            4.0,
            10.0,
            11.0,
            7.0,
            10.0,
            1.0,
            8.0,
            7.0,
            9.0,
            2.0,
            7.0,
            7.0,
            4.0,
            6.0,
            10.0,
            11.0,
            2.0,
            7.0,
            1.0,
            10.0,
            10.0,
            5.0,
            11.0,
            4.0,
            8.0,
            10.0,
            6.0,
            3.0,
            12.0,
            12.0,
            5.0,
            4.0,
            6.0,
            9.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": null,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(0, 154, 250, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 8
        },
        "zmax": null,
        "y": [
            68548.32883032663,
            84191.51882482592,
            95072.80134377317,
            86632.84038714247,
            84191.51882482592,
            85878.81923900853,
            84191.51882482592,
            75655.62745197897,
            70459.29990851681,
            69709.81453229475,
            75655.62745197897,
            235247.7958973312,
            68548.32883032663,
            77007.66977573698,
            84646.0763450026,
            114840.74306390483,
            85262.73635979637,
            71772.0808568759,
            84191.51882482592,
            74970.02478086705,
            84932.16703010538,
            76694.48801937555,
            113372.1344869396,
            85262.73635979637,
            235247.7958973312,
            67710.06227324973,
            67710.06227324973,
            79061.65344832145,
            70459.29990851681,
            82164.16372192024,
            75107.78759995372,
            84932.16703010538,
            86632.84038714247,
            97088.80834333315,
            69606.65176409061,
            71428.11919380794,
            90159.97757256187,
            95072.80134377317,
            74970.02478086705,
            70509.41327146809
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x4",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y4",
        "x": [
            128.0,
            64.0,
            8.0,
            8.0,
            64.0,
            32.0,
            64.0,
            16.0,
            64.0,
            128.0,
            16.0,
            32.0,
            128.0,
            16.0,
            8.0,
            64.0,
            8.0,
            64.0,
            64.0,
            32.0,
            8.0,
            16.0,
            128.0,
            8.0,
            128.0,
            128.0,
            128.0,
            32.0,
            64.0,
            128.0,
            16.0,
            8.0,
            8.0,
            16.0,
            128.0,
            64.0,
            8.0,
            8.0,
            32.0,
            32.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": null,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(0, 154, 250, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 8
        },
        "zmax": null,
        "y": [
            68548.32883032663,
            84191.51882482592,
            95072.80134377317,
            86632.84038714247,
            84191.51882482592,
            85878.81923900853,
            84191.51882482592,
            75655.62745197897,
            70459.29990851681,
            69709.81453229475,
            75655.62745197897,
            235247.7958973312,
            68548.32883032663,
            77007.66977573698,
            84646.0763450026,
            114840.74306390483,
            85262.73635979637,
            71772.0808568759,
            84191.51882482592,
            74970.02478086705,
            84932.16703010538,
            76694.48801937555,
            113372.1344869396,
            85262.73635979637,
            235247.7958973312,
            67710.06227324973,
            67710.06227324973,
            79061.65344832145,
            70459.29990851681,
            82164.16372192024,
            75107.78759995372,
            84932.16703010538,
            86632.84038714247,
            97088.80834333315,
            69606.65176409061,
            71428.11919380794,
            90159.97757256187,
            95072.80134377317,
            74970.02478086705,
            70509.41327146809
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            8.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(1, 0, 5, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 2.228569456923344
        },
        "zmax": 235247.7958973312,
        "y": [
            128.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            4.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(22, 11, 58, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 6.2625746809132234
        },
        "zmax": 235247.7958973312,
        "y": [
            64.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            4.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(50, 10, 93, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 8.850484157318771
        },
        "zmax": 235247.7958973312,
        "y": [
            8.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            6.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(28, 12, 67, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 6.857108830633711
        },
        "zmax": 235247.7958973312,
        "y": [
            8.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            4.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(22, 11, 58, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 6.2625746809132234
        },
        "zmax": 235247.7958973312,
        "y": [
            64.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            4.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(26, 12, 64, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 6.674386777227789
        },
        "zmax": 235247.7958973312,
        "y": [
            32.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            4.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(22, 11, 58, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 6.2625746809132234
        },
        "zmax": 235247.7958973312,
        "y": [
            64.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            10.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(7, 5, 27, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 4.112928241320031
        },
        "zmax": 235247.7958973312,
        "y": [
            16.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            11.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(2, 1, 10, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 2.744463706263933
        },
        "zmax": 235247.7958973312,
        "y": [
            64.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            7.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(1, 1, 8, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 2.5429733890317854
        },
        "zmax": 235247.7958973312,
        "y": [
            128.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            10.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(7, 5, 27, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 4.112928241320031
        },
        "zmax": 235247.7958973312,
        "y": [
            16.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            1.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(252, 255, 164, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 34.0
        },
        "zmax": 235247.7958973312,
        "y": [
            32.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            8.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(1, 0, 5, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 2.228569456923344
        },
        "zmax": 235247.7958973312,
        "y": [
            128.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            7.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(9, 6, 32, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 4.461219725346705
        },
        "zmax": 235247.7958973312,
        "y": [
            16.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            9.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(23, 12, 59, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 6.373919571194207
        },
        "zmax": 235247.7958973312,
        "y": [
            8.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            2.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(99, 20, 110, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 13.19798718881681
        },
        "zmax": 235247.7958973312,
        "y": [
            64.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            7.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(25, 12, 62, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 6.524495025240721
        },
        "zmax": 235247.7958973312,
        "y": [
            8.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            7.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(3, 2, 14, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 3.094823939791908
        },
        "zmax": 235247.7958973312,
        "y": [
            64.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            4.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(22, 11, 58, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 6.2625746809132234
        },
        "zmax": 235247.7958973312,
        "y": [
            64.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            6.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(6, 4, 25, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 3.935124954354717
        },
        "zmax": 235247.7958973312,
        "y": [
            32.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            10.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(24, 12, 60, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 6.443844886932734
        },
        "zmax": 235247.7958973312,
        "y": [
            8.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            11.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(8, 6, 31, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 4.380816711477143
        },
        "zmax": 235247.7958973312,
        "y": [
            16.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            2.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(96, 19, 110, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 12.888563282333715
        },
        "zmax": 235247.7958973312,
        "y": [
            128.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            7.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(25, 12, 62, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 6.524495025240721
        },
        "zmax": 235247.7958973312,
        "y": [
            8.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            1.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(252, 255, 164, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 34.0
        },
        "zmax": 235247.7958973312,
        "y": [
            128.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            10.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(0, 0, 4, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 2.0
        },
        "zmax": 235247.7958973312,
        "y": [
            128.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            10.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(0, 0, 4, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 2.0
        },
        "zmax": 235247.7958973312,
        "y": [
            128.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            5.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(12, 8, 39, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 4.984534801721769
        },
        "zmax": 235247.7958973312,
        "y": [
            32.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            11.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(2, 1, 10, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 2.744463706263933
        },
        "zmax": 235247.7958973312,
        "y": [
            64.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            4.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(18, 10, 50, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 5.762269849018324
        },
        "zmax": 235247.7958973312,
        "y": [
            128.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            8.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(6, 5, 25, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 3.9709172415625194
        },
        "zmax": 235247.7958973312,
        "y": [
            16.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            10.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(24, 12, 60, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 6.443844886932734
        },
        "zmax": 235247.7958973312,
        "y": [
            8.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            6.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(28, 12, 67, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 6.857108830633711
        },
        "zmax": 235247.7958973312,
        "y": [
            8.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            3.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(55, 9, 97, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 9.313376852011107
        },
        "zmax": 235247.7958973312,
        "y": [
            16.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            12.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(1, 1, 8, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 2.515154651194204
        },
        "zmax": 235247.7958973312,
        "y": [
            128.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            12.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(2, 2, 13, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 3.003338038521688
        },
        "zmax": 235247.7958973312,
        "y": [
            64.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            5.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(37, 12, 79, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 7.70146811537132
        },
        "zmax": 235247.7958973312,
        "y": [
            8.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            4.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(50, 10, 93, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 8.850484157318771
        },
        "zmax": 235247.7958973312,
        "y": [
            8.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            6.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(6, 4, 25, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 3.935124954354717
        },
        "zmax": 235247.7958973312,
        "y": [
            32.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            9.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "legendgroup": "",
        "marker": {
            "symbol": "circle",
            "color": "rgba(2, 1, 10, 1.000)",
            "line": {
                "color": "rgba(0, 0, 0, 1.000)",
                "width": 1
            },
            "size": 2.757897782760912
        },
        "zmax": 235247.7958973312,
        "y": [
            32.0
        ],
        "type": "scatter"
    },
    {
        "xaxis": "x3",
        "colorbar": {
            "title": ""
        },
        "yaxis": "y3",
        "x": [
            8.0
        ],
        "showlegend": false,
        "mode": "markers",
        "name": "",
        "zmin": 67710.06227324973,
        "marker": {
            "color": [
                0.5
            ],
            "cmin": 67710.06227324973,
            "opacity": 1.0e-10,
            "size": 1.0e-10,
            "colorscale": [
                [
                    0.0,
                    "rgba(0, 0, 4, 1.000)"
                ],
                [
                    0.00392156862745098,
                    "rgba(1, 0, 5, 1.000)"
                ],
                [
                    0.00784313725490196,
                    "rgba(1, 1, 6, 1.000)"
                ],
                [
                    0.011764705882352941,
                    "rgba(1, 1, 8, 1.000)"
                ],
                [
                    0.01568627450980392,
                    "rgba(2, 1, 10, 1.000)"
                ],
                [
                    0.0196078431372549,
                    "rgba(2, 2, 12, 1.000)"
                ],
                [
                    0.023529411764705882,
                    "rgba(2, 2, 14, 1.000)"
                ],
                [
                    0.027450980392156862,
                    "rgba(3, 2, 16, 1.000)"
                ],
                [
                    0.03137254901960784,
                    "rgba(4, 3, 18, 1.000)"
                ],
                [
                    0.03529411764705882,
                    "rgba(4, 3, 20, 1.000)"
                ],
                [
                    0.0392156862745098,
                    "rgba(5, 4, 23, 1.000)"
                ],
                [
                    0.043137254901960784,
                    "rgba(6, 4, 25, 1.000)"
                ],
                [
                    0.047058823529411764,
                    "rgba(7, 5, 27, 1.000)"
                ],
                [
                    0.050980392156862744,
                    "rgba(8, 5, 29, 1.000)"
                ],
                [
                    0.054901960784313725,
                    "rgba(9, 6, 31, 1.000)"
                ],
                [
                    0.058823529411764705,
                    "rgba(10, 7, 34, 1.000)"
                ],
                [
                    0.06274509803921569,
                    "rgba(11, 7, 36, 1.000)"
                ],
                [
                    0.06666666666666667,
                    "rgba(12, 8, 38, 1.000)"
                ],
                [
                    0.07058823529411765,
                    "rgba(13, 8, 41, 1.000)"
                ],
                [
                    0.07450980392156863,
                    "rgba(14, 9, 43, 1.000)"
                ],
                [
                    0.0784313725490196,
                    "rgba(16, 9, 45, 1.000)"
                ],
                [
                    0.08235294117647059,
                    "rgba(17, 10, 48, 1.000)"
                ],
                [
                    0.08627450980392157,
                    "rgba(18, 10, 50, 1.000)"
                ],
                [
                    0.09019607843137255,
                    "rgba(20, 11, 52, 1.000)"
                ],
                [
                    0.09411764705882353,
                    "rgba(21, 11, 55, 1.000)"
                ],
                [
                    0.09803921568627451,
                    "rgba(22, 11, 57, 1.000)"
                ],
                [
                    0.10196078431372549,
                    "rgba(24, 12, 60, 1.000)"
                ],
                [
                    0.10588235294117647,
                    "rgba(25, 12, 62, 1.000)"
                ],
                [
                    0.10980392156862745,
                    "rgba(27, 12, 65, 1.000)"
                ],
                [
                    0.11372549019607843,
                    "rgba(28, 12, 67, 1.000)"
                ],
                [
                    0.11764705882352941,
                    "rgba(30, 12, 69, 1.000)"
                ],
                [
                    0.12156862745098039,
                    "rgba(31, 12, 72, 1.000)"
                ],
                [
                    0.12549019607843137,
                    "rgba(33, 12, 74, 1.000)"
                ],
                [
                    0.12941176470588237,
                    "rgba(35, 12, 76, 1.000)"
                ],
                [
                    0.13333333333333333,
                    "rgba(36, 12, 79, 1.000)"
                ],
                [
                    0.13725490196078433,
                    "rgba(38, 12, 81, 1.000)"
                ],
                [
                    0.1411764705882353,
                    "rgba(40, 11, 83, 1.000)"
                ],
                [
                    0.1450980392156863,
                    "rgba(41, 11, 85, 1.000)"
                ],
                [
                    0.14901960784313725,
                    "rgba(43, 11, 87, 1.000)"
                ],
                [
                    0.15294117647058825,
                    "rgba(45, 11, 89, 1.000)"
                ],
                [
                    0.1568627450980392,
                    "rgba(47, 10, 91, 1.000)"
                ],
                [
                    0.1607843137254902,
                    "rgba(49, 10, 92, 1.000)"
                ],
                [
                    0.16470588235294117,
                    "rgba(50, 10, 94, 1.000)"
                ],
                [
                    0.16862745098039217,
                    "rgba(52, 10, 95, 1.000)"
                ],
                [
                    0.17254901960784313,
                    "rgba(54, 9, 97, 1.000)"
                ],
                [
                    0.17647058823529413,
                    "rgba(56, 9, 98, 1.000)"
                ],
                [
                    0.1803921568627451,
                    "rgba(57, 9, 99, 1.000)"
                ],
                [
                    0.1843137254901961,
                    "rgba(59, 9, 100, 1.000)"
                ],
                [
                    0.18823529411764706,
                    "rgba(61, 9, 101, 1.000)"
                ],
                [
                    0.19215686274509805,
                    "rgba(62, 9, 102, 1.000)"
                ],
                [
                    0.19607843137254902,
                    "rgba(64, 10, 103, 1.000)"
                ],
                [
                    0.2,
                    "rgba(66, 10, 104, 1.000)"
                ],
                [
                    0.20392156862745098,
                    "rgba(68, 10, 104, 1.000)"
                ],
                [
                    0.20784313725490197,
                    "rgba(69, 10, 105, 1.000)"
                ],
                [
                    0.21176470588235294,
                    "rgba(71, 11, 106, 1.000)"
                ],
                [
                    0.21568627450980393,
                    "rgba(73, 11, 106, 1.000)"
                ],
                [
                    0.2196078431372549,
                    "rgba(74, 12, 107, 1.000)"
                ],
                [
                    0.2235294117647059,
                    "rgba(76, 12, 107, 1.000)"
                ],
                [
                    0.22745098039215686,
                    "rgba(77, 13, 108, 1.000)"
                ],
                [
                    0.23137254901960785,
                    "rgba(79, 13, 108, 1.000)"
                ],
                [
                    0.23529411764705882,
                    "rgba(81, 14, 108, 1.000)"
                ],
                [
                    0.23921568627450981,
                    "rgba(82, 14, 109, 1.000)"
                ],
                [
                    0.24313725490196078,
                    "rgba(84, 15, 109, 1.000)"
                ],
                [
                    0.24705882352941178,
                    "rgba(85, 15, 109, 1.000)"
                ],
                [
                    0.25098039215686274,
                    "rgba(87, 16, 110, 1.000)"
                ],
                [
                    0.2549019607843137,
                    "rgba(89, 16, 110, 1.000)"
                ],
                [
                    0.25882352941176473,
                    "rgba(90, 17, 110, 1.000)"
                ],
                [
                    0.2627450980392157,
                    "rgba(92, 18, 110, 1.000)"
                ],
                [
                    0.26666666666666666,
                    "rgba(93, 18, 110, 1.000)"
                ],
                [
                    0.27058823529411763,
                    "rgba(95, 19, 110, 1.000)"
                ],
                [
                    0.27450980392156865,
                    "rgba(97, 19, 110, 1.000)"
                ],
                [
                    0.2784313725490196,
                    "rgba(98, 20, 110, 1.000)"
                ],
                [
                    0.2823529411764706,
                    "rgba(100, 21, 110, 1.000)"
                ],
                [
                    0.28627450980392155,
                    "rgba(101, 21, 110, 1.000)"
                ],
                [
                    0.2901960784313726,
                    "rgba(103, 22, 110, 1.000)"
                ],
                [
                    0.29411764705882354,
                    "rgba(105, 22, 110, 1.000)"
                ],
                [
                    0.2980392156862745,
                    "rgba(106, 23, 110, 1.000)"
                ],
                [
                    0.30196078431372547,
                    "rgba(108, 24, 110, 1.000)"
                ],
                [
                    0.3058823529411765,
                    "rgba(109, 24, 110, 1.000)"
                ],
                [
                    0.30980392156862746,
                    "rgba(111, 25, 110, 1.000)"
                ],
                [
                    0.3137254901960784,
                    "rgba(113, 25, 110, 1.000)"
                ],
                [
                    0.3176470588235294,
                    "rgba(114, 26, 110, 1.000)"
                ],
                [
                    0.3215686274509804,
                    "rgba(116, 26, 110, 1.000)"
                ],
                [
                    0.3254901960784314,
                    "rgba(117, 27, 110, 1.000)"
                ],
                [
                    0.32941176470588235,
                    "rgba(119, 28, 109, 1.000)"
                ],
                [
                    0.3333333333333333,
                    "rgba(120, 28, 109, 1.000)"
                ],
                [
                    0.33725490196078434,
                    "rgba(122, 29, 109, 1.000)"
                ],
                [
                    0.3411764705882353,
                    "rgba(124, 29, 109, 1.000)"
                ],
                [
                    0.34509803921568627,
                    "rgba(125, 30, 109, 1.000)"
                ],
                [
                    0.34901960784313724,
                    "rgba(127, 30, 108, 1.000)"
                ],
                [
                    0.35294117647058826,
                    "rgba(128, 31, 108, 1.000)"
                ],
                [
                    0.3568627450980392,
                    "rgba(130, 32, 108, 1.000)"
                ],
                [
                    0.3607843137254902,
                    "rgba(132, 32, 107, 1.000)"
                ],
                [
                    0.36470588235294116,
                    "rgba(133, 33, 107, 1.000)"
                ],
                [
                    0.3686274509803922,
                    "rgba(135, 33, 107, 1.000)"
                ],
                [
                    0.37254901960784315,
                    "rgba(136, 34, 106, 1.000)"
                ],
                [
                    0.3764705882352941,
                    "rgba(138, 34, 106, 1.000)"
                ],
                [
                    0.3803921568627451,
                    "rgba(140, 35, 105, 1.000)"
                ],
                [
                    0.3843137254901961,
                    "rgba(141, 35, 105, 1.000)"
                ],
                [
                    0.38823529411764707,
                    "rgba(143, 36, 105, 1.000)"
                ],
                [
                    0.39215686274509803,
                    "rgba(144, 37, 104, 1.000)"
                ],
                [
                    0.396078431372549,
                    "rgba(146, 37, 104, 1.000)"
                ],
                [
                    0.4,
                    "rgba(147, 38, 103, 1.000)"
                ],
                [
                    0.403921568627451,
                    "rgba(149, 38, 103, 1.000)"
                ],
                [
                    0.40784313725490196,
                    "rgba(151, 39, 102, 1.000)"
                ],
                [
                    0.4117647058823529,
                    "rgba(152, 39, 102, 1.000)"
                ],
                [
                    0.41568627450980394,
                    "rgba(154, 40, 101, 1.000)"
                ],
                [
                    0.4196078431372549,
                    "rgba(155, 41, 100, 1.000)"
                ],
                [
                    0.4235294117647059,
                    "rgba(157, 41, 100, 1.000)"
                ],
                [
                    0.42745098039215684,
                    "rgba(159, 42, 99, 1.000)"
                ],
                [
                    0.43137254901960786,
                    "rgba(160, 42, 99, 1.000)"
                ],
                [
                    0.43529411764705883,
                    "rgba(162, 43, 98, 1.000)"
                ],
                [
                    0.4392156862745098,
                    "rgba(163, 44, 97, 1.000)"
                ],
                [
                    0.44313725490196076,
                    "rgba(165, 44, 96, 1.000)"
                ],
                [
                    0.4470588235294118,
                    "rgba(166, 45, 96, 1.000)"
                ],
                [
                    0.45098039215686275,
                    "rgba(168, 46, 95, 1.000)"
                ],
                [
                    0.4549019607843137,
                    "rgba(169, 46, 94, 1.000)"
                ],
                [
                    0.4588235294117647,
                    "rgba(171, 47, 94, 1.000)"
                ],
                [
                    0.4627450980392157,
                    "rgba(173, 48, 93, 1.000)"
                ],
                [
                    0.4666666666666667,
                    "rgba(174, 48, 92, 1.000)"
                ],
                [
                    0.47058823529411764,
                    "rgba(176, 49, 91, 1.000)"
                ],
                [
                    0.4745098039215686,
                    "rgba(177, 50, 90, 1.000)"
                ],
                [
                    0.47843137254901963,
                    "rgba(179, 50, 90, 1.000)"
                ],
                [
                    0.4823529411764706,
                    "rgba(180, 51, 89, 1.000)"
                ],
                [
                    0.48627450980392156,
                    "rgba(182, 52, 88, 1.000)"
                ],
                [
                    0.49019607843137253,
                    "rgba(183, 53, 87, 1.000)"
                ],
                [
                    0.49411764705882355,
                    "rgba(185, 53, 86, 1.000)"
                ],
                [
                    0.4980392156862745,
                    "rgba(186, 54, 85, 1.000)"
                ],
                [
                    0.5019607843137255,
                    "rgba(188, 55, 84, 1.000)"
                ],
                [
                    0.5058823529411764,
                    "rgba(189, 56, 83, 1.000)"
                ],
                [
                    0.5098039215686274,
                    "rgba(191, 57, 82, 1.000)"
                ],
                [
                    0.5137254901960784,
                    "rgba(192, 58, 81, 1.000)"
                ],
                [
                    0.5176470588235295,
                    "rgba(193, 58, 80, 1.000)"
                ],
                [
                    0.5215686274509804,
                    "rgba(195, 59, 79, 1.000)"
                ],
                [
                    0.5254901960784314,
                    "rgba(196, 60, 78, 1.000)"
                ],
                [
                    0.5294117647058824,
                    "rgba(198, 61, 77, 1.000)"
                ],
                [
                    0.5333333333333333,
                    "rgba(199, 62, 76, 1.000)"
                ],
                [
                    0.5372549019607843,
                    "rgba(200, 63, 75, 1.000)"
                ],
                [
                    0.5411764705882353,
                    "rgba(202, 64, 74, 1.000)"
                ],
                [
                    0.5450980392156862,
                    "rgba(203, 65, 73, 1.000)"
                ],
                [
                    0.5490196078431373,
                    "rgba(204, 66, 72, 1.000)"
                ],
                [
                    0.5529411764705883,
                    "rgba(206, 67, 71, 1.000)"
                ],
                [
                    0.5568627450980392,
                    "rgba(207, 68, 70, 1.000)"
                ],
                [
                    0.5607843137254902,
                    "rgba(208, 69, 69, 1.000)"
                ],
                [
                    0.5647058823529412,
                    "rgba(210, 70, 68, 1.000)"
                ],
                [
                    0.5686274509803921,
                    "rgba(211, 71, 67, 1.000)"
                ],
                [
                    0.5725490196078431,
                    "rgba(212, 72, 66, 1.000)"
                ],
                [
                    0.5764705882352941,
                    "rgba(213, 74, 65, 1.000)"
                ],
                [
                    0.5803921568627451,
                    "rgba(215, 75, 63, 1.000)"
                ],
                [
                    0.5843137254901961,
                    "rgba(216, 76, 62, 1.000)"
                ],
                [
                    0.5882352941176471,
                    "rgba(217, 77, 61, 1.000)"
                ],
                [
                    0.592156862745098,
                    "rgba(218, 78, 60, 1.000)"
                ],
                [
                    0.596078431372549,
                    "rgba(219, 80, 59, 1.000)"
                ],
                [
                    0.6,
                    "rgba(221, 81, 58, 1.000)"
                ],
                [
                    0.6039215686274509,
                    "rgba(222, 82, 56, 1.000)"
                ],
                [
                    0.6078431372549019,
                    "rgba(223, 83, 55, 1.000)"
                ],
                [
                    0.611764705882353,
                    "rgba(224, 85, 54, 1.000)"
                ],
                [
                    0.615686274509804,
                    "rgba(225, 86, 53, 1.000)"
                ],
                [
                    0.6196078431372549,
                    "rgba(226, 87, 52, 1.000)"
                ],
                [
                    0.6235294117647059,
                    "rgba(227, 89, 51, 1.000)"
                ],
                [
                    0.6274509803921569,
                    "rgba(228, 90, 49, 1.000)"
                ],
                [
                    0.6313725490196078,
                    "rgba(229, 92, 48, 1.000)"
                ],
                [
                    0.6352941176470588,
                    "rgba(230, 93, 47, 1.000)"
                ],
                [
                    0.6392156862745098,
                    "rgba(231, 94, 46, 1.000)"
                ],
                [
                    0.6431372549019608,
                    "rgba(232, 96, 45, 1.000)"
                ],
                [
                    0.6470588235294118,
                    "rgba(233, 97, 43, 1.000)"
                ],
                [
                    0.6509803921568628,
                    "rgba(234, 99, 42, 1.000)"
                ],
                [
                    0.6549019607843137,
                    "rgba(235, 100, 41, 1.000)"
                ],
                [
                    0.6588235294117647,
                    "rgba(235, 102, 40, 1.000)"
                ],
                [
                    0.6627450980392157,
                    "rgba(236, 103, 38, 1.000)"
                ],
                [
                    0.6666666666666666,
                    "rgba(237, 105, 37, 1.000)"
                ],
                [
                    0.6705882352941176,
                    "rgba(238, 106, 36, 1.000)"
                ],
                [
                    0.6745098039215687,
                    "rgba(239, 108, 35, 1.000)"
                ],
                [
                    0.6784313725490196,
                    "rgba(239, 110, 33, 1.000)"
                ],
                [
                    0.6823529411764706,
                    "rgba(240, 111, 32, 1.000)"
                ],
                [
                    0.6862745098039216,
                    "rgba(241, 113, 31, 1.000)"
                ],
                [
                    0.6901960784313725,
                    "rgba(241, 115, 29, 1.000)"
                ],
                [
                    0.6941176470588235,
                    "rgba(242, 116, 28, 1.000)"
                ],
                [
                    0.6980392156862745,
                    "rgba(243, 118, 27, 1.000)"
                ],
                [
                    0.7019607843137254,
                    "rgba(243, 120, 25, 1.000)"
                ],
                [
                    0.7058823529411765,
                    "rgba(244, 121, 24, 1.000)"
                ],
                [
                    0.7098039215686275,
                    "rgba(245, 123, 23, 1.000)"
                ],
                [
                    0.7137254901960784,
                    "rgba(245, 125, 21, 1.000)"
                ],
                [
                    0.7176470588235294,
                    "rgba(246, 126, 20, 1.000)"
                ],
                [
                    0.7215686274509804,
                    "rgba(246, 128, 19, 1.000)"
                ],
                [
                    0.7254901960784313,
                    "rgba(247, 130, 18, 1.000)"
                ],
                [
                    0.7294117647058823,
                    "rgba(247, 132, 16, 1.000)"
                ],
                [
                    0.7333333333333333,
                    "rgba(248, 133, 15, 1.000)"
                ],
                [
                    0.7372549019607844,
                    "rgba(248, 135, 14, 1.000)"
                ],
                [
                    0.7411764705882353,
                    "rgba(248, 137, 12, 1.000)"
                ],
                [
                    0.7450980392156863,
                    "rgba(249, 139, 11, 1.000)"
                ],
                [
                    0.7490196078431373,
                    "rgba(249, 140, 10, 1.000)"
                ],
                [
                    0.7529411764705882,
                    "rgba(249, 142, 9, 1.000)"
                ],
                [
                    0.7568627450980392,
                    "rgba(250, 144, 8, 1.000)"
                ],
                [
                    0.7607843137254902,
                    "rgba(250, 146, 7, 1.000)"
                ],
                [
                    0.7647058823529411,
                    "rgba(250, 148, 7, 1.000)"
                ],
                [
                    0.7686274509803922,
                    "rgba(251, 150, 6, 1.000)"
                ],
                [
                    0.7725490196078432,
                    "rgba(251, 151, 6, 1.000)"
                ],
                [
                    0.7764705882352941,
                    "rgba(251, 153, 6, 1.000)"
                ],
                [
                    0.7803921568627451,
                    "rgba(251, 155, 6, 1.000)"
                ],
                [
                    0.7843137254901961,
                    "rgba(251, 157, 7, 1.000)"
                ],
                [
                    0.788235294117647,
                    "rgba(252, 159, 7, 1.000)"
                ],
                [
                    0.792156862745098,
                    "rgba(252, 161, 8, 1.000)"
                ],
                [
                    0.796078431372549,
                    "rgba(252, 163, 9, 1.000)"
                ],
                [
                    0.8,
                    "rgba(252, 165, 10, 1.000)"
                ],
                [
                    0.803921568627451,
                    "rgba(252, 166, 12, 1.000)"
                ],
                [
                    0.807843137254902,
                    "rgba(252, 168, 13, 1.000)"
                ],
                [
                    0.8117647058823529,
                    "rgba(252, 170, 15, 1.000)"
                ],
                [
                    0.8156862745098039,
                    "rgba(252, 172, 17, 1.000)"
                ],
                [
                    0.8196078431372549,
                    "rgba(252, 174, 18, 1.000)"
                ],
                [
                    0.8235294117647058,
                    "rgba(252, 176, 20, 1.000)"
                ],
                [
                    0.8274509803921568,
                    "rgba(252, 178, 22, 1.000)"
                ],
                [
                    0.8313725490196079,
                    "rgba(252, 180, 24, 1.000)"
                ],
                [
                    0.8352941176470589,
                    "rgba(251, 182, 26, 1.000)"
                ],
                [
                    0.8392156862745098,
                    "rgba(251, 184, 29, 1.000)"
                ],
                [
                    0.8431372549019608,
                    "rgba(251, 186, 31, 1.000)"
                ],
                [
                    0.8470588235294118,
                    "rgba(251, 188, 33, 1.000)"
                ],
                [
                    0.8509803921568627,
                    "rgba(251, 190, 35, 1.000)"
                ],
                [
                    0.8549019607843137,
                    "rgba(250, 192, 38, 1.000)"
                ],
                [
                    0.8588235294117647,
                    "rgba(250, 194, 40, 1.000)"
                ],
                [
                    0.8627450980392157,
                    "rgba(250, 196, 42, 1.000)"
                ],
                [
                    0.8666666666666667,
                    "rgba(250, 198, 45, 1.000)"
                ],
                [
                    0.8705882352941177,
                    "rgba(249, 199, 47, 1.000)"
                ],
                [
                    0.8745098039215686,
                    "rgba(249, 201, 50, 1.000)"
                ],
                [
                    0.8784313725490196,
                    "rgba(249, 203, 53, 1.000)"
                ],
                [
                    0.8823529411764706,
                    "rgba(248, 205, 55, 1.000)"
                ],
                [
                    0.8862745098039215,
                    "rgba(248, 207, 58, 1.000)"
                ],
                [
                    0.8901960784313725,
                    "rgba(247, 209, 61, 1.000)"
                ],
                [
                    0.8941176470588236,
                    "rgba(247, 211, 64, 1.000)"
                ],
                [
                    0.8980392156862745,
                    "rgba(246, 213, 67, 1.000)"
                ],
                [
                    0.9019607843137255,
                    "rgba(246, 215, 70, 1.000)"
                ],
                [
                    0.9058823529411765,
                    "rgba(245, 217, 73, 1.000)"
                ],
                [
                    0.9098039215686274,
                    "rgba(245, 219, 76, 1.000)"
                ],
                [
                    0.9137254901960784,
                    "rgba(244, 221, 79, 1.000)"
                ],
                [
                    0.9176470588235294,
                    "rgba(244, 223, 83, 1.000)"
                ],
                [
                    0.9215686274509803,
                    "rgba(244, 225, 86, 1.000)"
                ],
                [
                    0.9254901960784314,
                    "rgba(243, 227, 90, 1.000)"
                ],
                [
                    0.9294117647058824,
                    "rgba(243, 229, 93, 1.000)"
                ],
                [
                    0.9333333333333333,
                    "rgba(242, 230, 97, 1.000)"
                ],
                [
                    0.9372549019607843,
                    "rgba(242, 232, 101, 1.000)"
                ],
                [
                    0.9411764705882353,
                    "rgba(242, 234, 105, 1.000)"
                ],
                [
                    0.9450980392156862,
                    "rgba(241, 236, 109, 1.000)"
                ],
                [
                    0.9490196078431372,
                    "rgba(241, 237, 113, 1.000)"
                ],
                [
                    0.9529411764705882,
                    "rgba(241, 239, 117, 1.000)"
                ],
                [
                    0.9568627450980393,
                    "rgba(241, 241, 121, 1.000)"
                ],
                [
                    0.9607843137254902,
                    "rgba(242, 242, 125, 1.000)"
                ],
                [
                    0.9647058823529412,
                    "rgba(242, 244, 130, 1.000)"
                ],
                [
                    0.9686274509803922,
                    "rgba(243, 245, 134, 1.000)"
                ],
                [
                    0.9725490196078431,
                    "rgba(243, 246, 138, 1.000)"
                ],
                [
                    0.9764705882352941,
                    "rgba(244, 248, 142, 1.000)"
                ],
                [
                    0.9803921568627451,
                    "rgba(245, 249, 146, 1.000)"
                ],
                [
                    0.984313725490196,
                    "rgba(246, 250, 150, 1.000)"
                ],
                [
                    0.9882352941176471,
                    "rgba(248, 251, 154, 1.000)"
                ],
                [
                    0.9921568627450981,
                    "rgba(249, 252, 157, 1.000)"
                ],
                [
                    0.996078431372549,
                    "rgba(250, 253, 161, 1.000)"
                ],
                [
                    1.0,
                    "rgba(252, 255, 164, 1.000)"
                ]
            ],
            "cmax": 235247.7958973312,
            "showscale": false
        },
        "zmax": 235247.7958973312,
        "y": [
            128.0
        ],
        "type": "scatter",
        "hoverinfo": "none"
    }
]
, {
    "showlegend": true,
    "paper_bgcolor": "rgba(255, 255, 255, 1.000)",
    "xaxis1": {
        "showticklabels": true,
        "gridwidth": 0.5,
        "tickvals": [
            2.5,
            5.0,
            7.5,
            10.0
        ],
        "range": [
            0.67,
            12.33
        ],
        "domain": [
            0.3581682971446751,
            0.4928418038654259
        ],
        "mirror": true,
        "tickangle": 0,
        "showline": true,
        "ticktext": [
            "2.5",
            "5.0",
            "7.5",
            "10.0"
        ],
        "zeroline": false,
        "tickfont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "zerolinecolor": "rgba(0, 0, 0, 1.000)",
        "anchor": "y1",
        "visible": true,
        "ticks": "inside",
        "tickmode": "array",
        "linecolor": "rgba(0, 0, 0, 1.000)",
        "showgrid": true,
        "title": "",
        "gridcolor": "rgba(0, 0, 0, 0.100)",
        "titlefont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 15
        },
        "tickcolor": "rgb(0, 0, 0)",
        "type": "-"
    },
    "legend": {
        "yanchor": "auto",
        "xanchor": "auto",
        "bordercolor": "rgba(0, 0, 0, 1.000)",
        "bgcolor": "rgba(255, 255, 255, 1.000)",
        "borderwidth": 1,
        "tracegroupgap": 0,
        "y": 1.0,
        "font": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "title": {
            "font": {
                "color": "rgba(0, 0, 0, 1.000)",
                "family": "sans-serif",
                "size": 15
            },
            "text": ""
        },
        "traceorder": "normal",
        "x": 1.0
    },
    "height": 500,
    "yaxis4": {
        "showticklabels": true,
        "gridwidth": 0.5,
        "tickvals": [
            75000.0,
            100000.0,
            125000.0,
            150000.0,
            175000.0,
            200000.0,
            225000.0
        ],
        "range": [
            62683.930264527284,
            240273.92790605364
        ],
        "domain": [
            0.060651793525809315,
            0.5074037620297464
        ],
        "mirror": true,
        "tickangle": 0,
        "showline": true,
        "ticktext": [
            "7.500×10<sup>4</sup>",
            "1.000×10<sup>5</sup>",
            "1.250×10<sup>5</sup>",
            "1.500×10<sup>5</sup>",
            "1.750×10<sup>5</sup>",
            "2.000×10<sup>5</sup>",
            "2.250×10<sup>5</sup>"
        ],
        "zeroline": false,
        "tickfont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "zerolinecolor": "rgba(0, 0, 0, 1.000)",
        "anchor": "x4",
        "visible": true,
        "ticks": "inside",
        "tickmode": "array",
        "linecolor": "rgba(0, 0, 0, 1.000)",
        "showgrid": true,
        "title": "MeanAbsoluteError",
        "gridcolor": "rgba(0, 0, 0, 0.100)",
        "titlefont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 15
        },
        "tickcolor": "rgb(0, 0, 0)",
        "type": "-"
    },
    "yaxis2": {
        "showticklabels": true,
        "gridwidth": 0.5,
        "tickvals": [
            0.0,
            0.25,
            0.5,
            0.75,
            1.0
        ],
        "range": [
            -0.03,
            1.03
        ],
        "domain": [
            0.5453740157480316,
            0.9921259842519685
        ],
        "mirror": false,
        "tickangle": 0,
        "showline": false,
        "ticktext": [
            "0.00",
            "0.25",
            "0.50",
            "0.75",
            "1.00"
        ],
        "zeroline": false,
        "tickfont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "zerolinecolor": "rgba(0, 0, 0, 1.000)",
        "anchor": "x2",
        "visible": false,
        "ticks": "inside",
        "tickmode": "array",
        "linecolor": "rgba(0, 0, 0, 1.000)",
        "showgrid": true,
        "title": "",
        "gridcolor": "rgba(0, 0, 0, 0.100)",
        "titlefont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 15
        },
        "tickcolor": "rgb(0, 0, 0)",
        "type": "-"
    },
    "xaxis3": {
        "showticklabels": true,
        "gridwidth": 0.5,
        "tickvals": [
            2.5,
            5.0,
            7.5,
            10.0
        ],
        "range": [
            0.67,
            12.33
        ],
        "domain": [
            0.3581682971446751,
            0.4928418038654259
        ],
        "mirror": true,
        "tickangle": 0,
        "showline": true,
        "ticktext": [
            "2.5",
            "5.0",
            "7.5",
            "10.0"
        ],
        "zeroline": false,
        "tickfont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "zerolinecolor": "rgba(0, 0, 0, 1.000)",
        "anchor": "y3",
        "visible": true,
        "ticks": "inside",
        "tickmode": "array",
        "linecolor": "rgba(0, 0, 0, 1.000)",
        "showgrid": true,
        "title": "evo_tree_regressor.max_depth",
        "gridcolor": "rgba(0, 0, 0, 0.100)",
        "titlefont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 15
        },
        "tickcolor": "rgb(0, 0, 0)",
        "type": "-"
    },
    "yaxis3": {
        "showticklabels": true,
        "gridwidth": 0.5,
        "tickvals": [
            25.0,
            50.0,
            75.0,
            100.0,
            125.0
        ],
        "range": [
            4.4,
            131.6
        ],
        "domain": [
            0.060651793525809315,
            0.5074037620297464
        ],
        "mirror": true,
        "tickangle": 0,
        "showline": true,
        "ticktext": [
            "25",
            "50",
            "75",
            "100",
            "125"
        ],
        "zeroline": false,
        "tickfont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "zerolinecolor": "rgba(0, 0, 0, 1.000)",
        "anchor": "x3",
        "visible": true,
        "ticks": "inside",
        "tickmode": "array",
        "linecolor": "rgba(0, 0, 0, 1.000)",
        "showgrid": true,
        "title": "evo_tree_regressor.nbins",
        "gridcolor": "rgba(0, 0, 0, 0.100)",
        "titlefont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 15
        },
        "tickcolor": "rgb(0, 0, 0)",
        "type": "-"
    },
    "xaxis4": {
        "showticklabels": true,
        "gridwidth": 0.5,
        "tickvals": [
            25.0,
            50.0,
            75.0,
            100.0,
            125.0
        ],
        "range": [
            4.4,
            131.6
        ],
        "domain": [
            0.858168297144675,
            0.9928418038654259
        ],
        "mirror": true,
        "tickangle": 0,
        "showline": true,
        "ticktext": [
            "25",
            "50",
            "75",
            "100",
            "125"
        ],
        "zeroline": false,
        "tickfont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "zerolinecolor": "rgba(0, 0, 0, 1.000)",
        "anchor": "y4",
        "visible": true,
        "ticks": "inside",
        "tickmode": "array",
        "linecolor": "rgba(0, 0, 0, 1.000)",
        "showgrid": true,
        "title": "evo_tree_regressor.nbins",
        "gridcolor": "rgba(0, 0, 0, 0.100)",
        "titlefont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 15
        },
        "tickcolor": "rgb(0, 0, 0)",
        "type": "-"
    },
    "yaxis1": {
        "showticklabels": true,
        "gridwidth": 0.5,
        "tickvals": [
            75000.0,
            100000.0,
            125000.0,
            150000.0,
            175000.0,
            200000.0,
            225000.0
        ],
        "range": [
            62683.930264527284,
            240273.92790605364
        ],
        "domain": [
            0.5453740157480316,
            0.9921259842519685
        ],
        "mirror": true,
        "tickangle": 0,
        "showline": true,
        "ticktext": [
            "7.500×10<sup>4</sup>",
            "1.000×10<sup>5</sup>",
            "1.250×10<sup>5</sup>",
            "1.500×10<sup>5</sup>",
            "1.750×10<sup>5</sup>",
            "2.000×10<sup>5</sup>",
            "2.250×10<sup>5</sup>"
        ],
        "zeroline": false,
        "tickfont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "zerolinecolor": "rgba(0, 0, 0, 1.000)",
        "anchor": "x1",
        "visible": true,
        "ticks": "inside",
        "tickmode": "array",
        "linecolor": "rgba(0, 0, 0, 1.000)",
        "showgrid": true,
        "title": "MeanAbsoluteError",
        "gridcolor": "rgba(0, 0, 0, 0.100)",
        "titlefont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 15
        },
        "tickcolor": "rgb(0, 0, 0)",
        "type": "-"
    },
    "annotations": [],
    "xaxis2": {
        "showticklabels": true,
        "gridwidth": 0.5,
        "tickvals": [
            0.0,
            0.25,
            0.5,
            0.75,
            1.0
        ],
        "range": [
            -0.03,
            1.03
        ],
        "domain": [
            0.858168297144675,
            0.9928418038654259
        ],
        "mirror": false,
        "tickangle": 0,
        "showline": false,
        "ticktext": [
            "0.00",
            "0.25",
            "0.50",
            "0.75",
            "1.00"
        ],
        "zeroline": false,
        "tickfont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 11
        },
        "zerolinecolor": "rgba(0, 0, 0, 1.000)",
        "anchor": "y2",
        "visible": false,
        "ticks": "inside",
        "tickmode": "array",
        "linecolor": "rgba(0, 0, 0, 1.000)",
        "showgrid": true,
        "title": "",
        "gridcolor": "rgba(0, 0, 0, 0.100)",
        "titlefont": {
            "color": "rgba(0, 0, 0, 1.000)",
            "family": "sans-serif",
            "size": 15
        },
        "tickcolor": "rgb(0, 0, 0)",
        "type": "-"
    },
    "plot_bgcolor": "rgba(255, 255, 255, 1.000)",
    "margin": {
        "l": 0,
        "b": 20,
        "r": 0,
        "t": 20
    },
    "width": 550
}
);

    
    </script>

```

(d)

````julia
best_model = report(tuned_mach).best_model;
best_mach = machine(best_model, X, y);
best_err = evaluate!(best_mach, resampling=CV(nfolds=3), measure=mae)
````

````
PerformanceEvaluation object with these fields:
  measure, measurement, operation, per_fold,
  per_observation, fitted_params_per_fold,
  report_per_fold, train_test_pairs
Extract:
┌─────────────────────┬─────────────┬───────────┬─────────────────────────────┐
│ measure             │ measurement │ operation │ per_fold                    │
├─────────────────────┼─────────────┼───────────┼─────────────────────────────┤
│ MeanAbsoluteError() │ 68700.0     │ predict   │ [68800.0, 67800.0, 69500.0] │
└─────────────────────┴─────────────┴───────────┴─────────────────────────────┘

````

````julia
tuned_err = evaluate!(tuned_mach, resampling=CV(nfolds=3), measure=mae)
````

````
PerformanceEvaluation object with these fields:
  measure, measurement, operation, per_fold,
  per_observation, fitted_params_per_fold,
  report_per_fold, train_test_pairs
Extract:
┌─────────────────────┬─────────────┬───────────┬─────────────────────────────┐
│ measure             │ measurement │ operation │ per_fold                    │
├─────────────────────┼─────────────┼───────────┼─────────────────────────────┤
│ MeanAbsoluteError() │ 69900.0     │ predict   │ [71400.0, 67800.0, 70500.0] │
└─────────────────────┴─────────────┴───────────┴─────────────────────────────┘

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

