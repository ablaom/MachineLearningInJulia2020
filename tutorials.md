```@meta
EditURL = "<unknown>/tutorials.jl"
```

# Machine Learning in Julia, JuliaCon2020

A workshop introducing the machine learning toolbox
[MLJ](https://alan-turing-institute.github.io/MLJ.jl/stable/).

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
color_off()
````

````
false
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
tutorial](https://juliaai.github.io/DataScienceTutorials.jl/data/dataframe/).

### Fixing scientific types in tabular data

To show how we can correct the scientific types of data in tables,
we introduce a cleaned up version of the UCI Horse Colic Data Set
(the cleaning work-flow is described
[here](https://juliaai.github.io/DataScienceTutorials.jl/end-to-end/horse/#dealing_with_missing_values)).

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
- [Summary](https://juliaai.github.io/ScientificTypes.jl/dev/#Summary-of-the-default-convention) of the MLJ convention for representing scientific types
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
 0.214416  0.610352  0.158818
 0.174811  0.121146  0.104297
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
 0.214416  0.610352  0.158818
 0.174811  0.121146  0.104297
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
 0.214416  0.610352  0.158818
 0.174811  0.121146  0.104297
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

````julia
coerce!(iris,
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

<display omitted, as not markdown renderable>

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
AEDetector from OutlierDetectionNetworks.jl.
[Documentation](https://github.com/OutlierDetectionJL/OutlierDetectionNetworks.jl).
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
A neural network model for making probabilistic predictions of a `Multiclass` or `OrderedFactor` target, given a table of `Continuous` features. 
→ based on [MLJFlux](https://github.com/alan-turing-institute/MLJFlux.jl).
→ do `@load NeuralNetworkClassifier pkg="MLJFlux"` to use the model.
→ do `?NeuralNetworkClassifier` for documentation.
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
    1:	Source @264 ⏎ `Table{AbstractVector{Continuous}}`
    2:	Source @596 ⏎ `AbstractVector{Multiclass{3}}`

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
    1:	Source @264 ⏎ `Table{AbstractVector{Continuous}}`
    2:	Source @596 ⏎ `AbstractVector{Multiclass{3}}`

````

... and `predict`:

````julia
yhat = predict(mach, rows=test);  # or `predict(mach, Xnew)`
yhat[1:3]
````

````
3-element MLJBase.UnivariateFiniteVector{Multiclass{3}, String, UInt32, Float64}:
 UnivariateFinite{Multiclass{3}}(Iris-setosa=>0.298, Iris-versicolor=>0.37, Iris-virginica=>0.332)
 UnivariateFinite{Multiclass{3}}(Iris-setosa=>0.282, Iris-versicolor=>0.371, Iris-virginica=>0.347)
 UnivariateFinite{Multiclass{3}}(Iris-setosa=>0.379, Iris-versicolor=>0.357, Iris-virginica=>0.264)
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
(training_losses = [1.2102369904460677, 1.2573466409636107, 1.2012884763249403, 1.159415930492209, 1.1176625852059021, 1.074384151701623, 1.1164166010124859, 1.0904731571087158, 1.088586369749853, 1.062854355958386, 1.0343942781406288, 1.016767866961867, 1.0185323857138837],)
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
 UnivariateFinite{Multiclass{3}}(Iris-setosa=>0.27, Iris-versicolor=>0.369, Iris-virginica=>0.362)
 UnivariateFinite{Multiclass{3}}(Iris-setosa=>0.3, Iris-versicolor=>0.371, Iris-virginica=>0.329)
 UnivariateFinite{Multiclass{3}}(Iris-setosa=>0.263, Iris-versicolor=>0.375, Iris-virginica=>0.362)
````

If you want to fit a retrieved model, you will need to bind some data to it:

````julia
mach3 = machine("neural_net.jlso", X, y)
fit!(mach3)
````

````
Machine{NeuralNetworkClassifier{Short,…},…} trained 2 times; caches data
  args: 
    1:	Source @715 ⏎ `Table{AbstractVector{Continuous}}`
    2:	Source @743 ⏎ `AbstractVector{Multiclass{3}}`

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
    1:	Source @264 ⏎ `Table{AbstractVector{Continuous}}`
    2:	Source @596 ⏎ `AbstractVector{Multiclass{3}}`

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
    1:	Source @264 ⏎ `Table{AbstractVector{Continuous}}`
    2:	Source @596 ⏎ `AbstractVector{Multiclass{3}}`

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
    1:	Source @264 ⏎ `Table{AbstractVector{Continuous}}`
    2:	Source @596 ⏎ `AbstractVector{Multiclass{3}}`

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
UnivariateFinite{Multiclass{3}}(Iris-setosa=>0.101, Iris-versicolor=>0.558, Iris-virginica=>0.341)
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
0.3411491914755774
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
 0.5581346326645432
 0.47812424130633313
 0.11796874746986409
 0.2748136014125683
````

````julia
mode.(yhat[1:4])
````

````
4-element CategoricalArray{String,1,UInt32}:
 "Iris-versicolor"
 "Iris-virginica"
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
 "Iris-virginica"
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
 0.100716    0.558135  0.341149
 0.0254209   0.478124  0.496455
 0.845594    0.117969  0.0364368
 0.00171405  0.274814  0.723472
````

However, in a typical MLJ work-flow, this is not as useful as you
might imagine. In particular, all probabilistic performance measures
in MLJ expect distribution objects in their first slot:

````julia
cross_entropy(yhat, y[test]) |> mean
````

````
0.36798476414020065
````

To apply a deterministic measure, we first need to obtain point-estimates:

````julia
misclassification_rate(mode.(yhat), y[test])
````

````
0.06666666666666667
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
│ LogLoss(tol = 2.22045e-16) │ 0.368       │ predict   │ [0.368]  │
│ BrierScore()               │ -0.191      │ predict   │ [-0.191] │
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
│ LogLoss(tol = 2.22045e-16) │ 0.418       │ predict   │ [0.39, 0.401, 0.468, 0.416, 0.459, 0.372]        │
│ BrierScore()               │ -0.232      │ predict   │ [-0.204, -0.206, -0.263, -0.233, -0.281, -0.207] │
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
┌────────────────────────────┬─────────────┬───────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ measure                    │ measurement │ operation │ per_fold                                                                                                                                         │
├────────────────────────────┼─────────────┼───────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ LogLoss(tol = 2.22045e-16) │ 0.398       │ predict   │ [0.322, 0.305, 0.401, 0.493, 0.474, 0.478, 0.426, 0.372, 0.392, 0.422, 0.364, 0.352, 0.414, 0.381, 0.415, 0.376, 0.381, 0.394]                   │
│ BrierScore()               │ -0.22       │ predict   │ [-0.178, -0.155, -0.226, -0.277, -0.271, -0.269, -0.247, -0.208, -0.215, -0.227, -0.198, -0.187, -0.228, -0.197, -0.225, -0.213, -0.211, -0.225] │
└────────────────────────────┴─────────────┴───────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

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
Optimising neural net:  4%[>                        ]  ETA: 0:00:02[KOptimising neural net:  6%[=>                       ]  ETA: 0:00:02[KOptimising neural net:  8%[=>                       ]  ETA: 0:00:02[KOptimising neural net: 10%[==>                      ]  ETA: 0:00:02[KOptimising neural net: 12%[==>                      ]  ETA: 0:00:03[KOptimising neural net: 14%[===>                     ]  ETA: 0:00:03[KOptimising neural net: 16%[===>                     ]  ETA: 0:00:02[KOptimising neural net: 18%[====>                    ]  ETA: 0:00:02[KOptimising neural net: 20%[====>                    ]  ETA: 0:00:02[KOptimising neural net: 22%[=====>                   ]  ETA: 0:00:02[KOptimising neural net: 24%[=====>                   ]  ETA: 0:00:02[KOptimising neural net: 25%[======>                  ]  ETA: 0:00:02[KOptimising neural net: 27%[======>                  ]  ETA: 0:00:02[KOptimising neural net: 29%[=======>                 ]  ETA: 0:00:02[KOptimising neural net: 31%[=======>                 ]  ETA: 0:00:02[KOptimising neural net: 33%[========>                ]  ETA: 0:00:02[KOptimising neural net: 35%[========>                ]  ETA: 0:00:02[KOptimising neural net: 37%[=========>               ]  ETA: 0:00:02[KOptimising neural net: 39%[=========>               ]  ETA: 0:00:02[KOptimising neural net: 41%[==========>              ]  ETA: 0:00:02[KOptimising neural net: 43%[==========>              ]  ETA: 0:00:02[KOptimising neural net: 45%[===========>             ]  ETA: 0:00:02[KOptimising neural net: 47%[===========>             ]  ETA: 0:00:02[KOptimising neural net: 49%[============>            ]  ETA: 0:00:02[KOptimising neural net: 51%[============>            ]  ETA: 0:00:02[KOptimising neural net: 53%[=============>           ]  ETA: 0:00:01[KOptimising neural net: 55%[=============>           ]  ETA: 0:00:01[KOptimising neural net: 57%[==============>          ]  ETA: 0:00:01[KOptimising neural net: 59%[==============>          ]  ETA: 0:00:01[KOptimising neural net: 61%[===============>         ]  ETA: 0:00:01[KOptimising neural net: 63%[===============>         ]  ETA: 0:00:01[KOptimising neural net: 65%[================>        ]  ETA: 0:00:01[KOptimising neural net: 67%[================>        ]  ETA: 0:00:01[KOptimising neural net: 69%[=================>       ]  ETA: 0:00:01[KOptimising neural net: 71%[=================>       ]  ETA: 0:00:01[KOptimising neural net: 73%[==================>      ]  ETA: 0:00:01[KOptimising neural net: 75%[==================>      ]  ETA: 0:00:01[KOptimising neural net: 76%[===================>     ]  ETA: 0:00:01[KOptimising neural net: 78%[===================>     ]  ETA: 0:00:01[KOptimising neural net: 80%[====================>    ]  ETA: 0:00:01[KOptimising neural net: 82%[====================>    ]  ETA: 0:00:01[KOptimising neural net: 84%[=====================>   ]  ETA: 0:00:00[KOptimising neural net: 86%[=====================>   ]  ETA: 0:00:00[KOptimising neural net: 88%[======================>  ]  ETA: 0:00:00[KOptimising neural net: 90%[======================>  ]  ETA: 0:00:00[KOptimising neural net: 92%[=======================> ]  ETA: 0:00:00[KOptimising neural net: 94%[=======================> ]  ETA: 0:00:00[KOptimising neural net: 96%[========================>]  ETA: 0:00:00[KOptimising neural net: 98%[========================>]  ETA: 0:00:00[KOptimising neural net:100%[=========================] Time: 0:00:03[K

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
gr(size=(490,300))
plt=plot(curve.parameter_values, curve.measurements)
xlabel!(plt, "epochs")
ylabel!(plt, "cross entropy on holdout set")
savefig("learning_curve.png")
````

````
[ Info: Training Machine{ProbabilisticTunedModel{Grid,…},…}.
[ Info: Attempting to evaluate 22 models.
Evaluating over 22 metamodels:   0%[>                        ]  ETA: N/A[KEvaluating over 22 metamodels:   5%[=>                       ]  ETA: 0:00:01[KEvaluating over 22 metamodels:   9%[==>                      ]  ETA: 0:00:01[KEvaluating over 22 metamodels:  14%[===>                     ]  ETA: 0:00:01[KEvaluating over 22 metamodels:  18%[====>                    ]  ETA: 0:00:01[KEvaluating over 22 metamodels:  23%[=====>                   ]  ETA: 0:00:01[KEvaluating over 22 metamodels:  27%[======>                  ]  ETA: 0:00:01[KEvaluating over 22 metamodels:  32%[=======>                 ]  ETA: 0:00:01[KEvaluating over 22 metamodels:  36%[=========>               ]  ETA: 0:00:01[KEvaluating over 22 metamodels:  41%[==========>              ]  ETA: 0:00:01[KEvaluating over 22 metamodels:  45%[===========>             ]  ETA: 0:00:01[KEvaluating over 22 metamodels:  50%[============>            ]  ETA: 0:00:01[KEvaluating over 22 metamodels:  55%[=============>           ]  ETA: 0:00:01[KEvaluating over 22 metamodels:  59%[==============>          ]  ETA: 0:00:01[KEvaluating over 22 metamodels:  64%[===============>         ]  ETA: 0:00:01[KEvaluating over 22 metamodels:  68%[=================>       ]  ETA: 0:00:01[KEvaluating over 22 metamodels:  73%[==================>      ]  ETA: 0:00:01[KEvaluating over 22 metamodels:  77%[===================>     ]  ETA: 0:00:00[KEvaluating over 22 metamodels:  82%[====================>    ]  ETA: 0:00:00[KEvaluating over 22 metamodels:  86%[=====================>   ]  ETA: 0:00:00[KEvaluating over 22 metamodels:  91%[======================>  ]  ETA: 0:00:00[KEvaluating over 22 metamodels:  95%[=======================> ]  ETA: 0:00:00[KEvaluating over 22 metamodels: 100%[=========================] Time: 0:00:03[K

````

![](learning_curve.png)

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
 3
 0
 1
 4
 0
 0
 1
 7
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
│ 1     │ 0.148655   │ 0.802052   │ male                             │
│ 2     │ 0.440145   │ 0.301427   │ female                           │
│ 3     │ 0.157132   │ 0.285097   │ female                           │
│ 4     │ 0.287972   │ 0.839287   │ male                             │
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
│ 0.148655   │ 0.802052   │
│ 0.440145   │ 0.301427   │
│ 0.157132   │ 0.285097   │
│ 0.287972   │ 0.839287   │
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
mean(x) = 0.5306341241001111
std(x) = 0.291180654132626

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
mean(xhat) = -5.2735593669694933e-17
std(xhat) = 0.9999999999999997

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
3](#exercise-3-fixing-scitypes-in-a-table)) into a set of `Continuous`
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
Pipeline694(
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
Pipeline702(
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
Machine{Pipeline702,…} trained 2 times; caches data
  args: 
    1:	Source @982 ⏎ `Table{Union{AbstractVector{Continuous}, AbstractVector{Multiclass{70}}, AbstractVector{OrderedFactor{6}}, AbstractVector{OrderedFactor{13}}, AbstractVector{OrderedFactor{30}}, AbstractVector{OrderedFactor{5}}, AbstractVector{OrderedFactor{12}}, AbstractVector{OrderedFactor{2}}}}`
    2:	Source @236 ⏎ `AbstractVector{Continuous}`

````

````julia
pipe2.ridge_regressor.lambda = 0.1
fit!(mach)
````

````
Machine{Pipeline702,…} trained 3 times; caches data
  args: 
    1:	Source @982 ⏎ `Table{Union{AbstractVector{Continuous}, AbstractVector{Multiclass{70}}, AbstractVector{OrderedFactor{6}}, AbstractVector{OrderedFactor{13}}, AbstractVector{OrderedFactor{30}}, AbstractVector{OrderedFactor{5}}, AbstractVector{OrderedFactor{12}}, AbstractVector{OrderedFactor{2}}}}`
    2:	Source @236 ⏎ `AbstractVector{Continuous}`

````

Second time only the ridge regressor is retrained!

Mutate a hyper-parameter of the `PCA` model and every model except
the `ContinuousEncoder` (which comes before it will be retrained):

````julia
pipe2.pca.pratio = 0.9999
fit!(mach)
````

````
Machine{Pipeline702,…} trained 4 times; caches data
  args: 
    1:	Source @982 ⏎ `Table{Union{AbstractVector{Continuous}, AbstractVector{Multiclass{70}}, AbstractVector{OrderedFactor{6}}, AbstractVector{OrderedFactor{13}}, AbstractVector{OrderedFactor{30}}, AbstractVector{OrderedFactor{5}}, AbstractVector{OrderedFactor{12}}, AbstractVector{OrderedFactor{2}}}}`
    2:	Source @236 ⏎ `AbstractVector{Continuous}`

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
    - [Linear pipelines](https://alan-turing-institute.github.io/MLJ.jl/dev/linear_pipelines/#Linear-Pipelines)
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
Machine{Pipeline731,…} trained 0 times; caches data
  args: 
    1:	Source @004 ⏎ `Table{Union{AbstractVector{Continuous}, AbstractVector{Multiclass{2}}, AbstractVector{Multiclass{6}}, AbstractVector{Multiclass{3}}, AbstractVector{OrderedFactor{2}}, AbstractVector{OrderedFactor{4}}, AbstractVector{OrderedFactor{5}}}}`
    2:	Source @765 ⏎ `AbstractVector{Multiclass{3}}`

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
savefig("learning_curve2.png")
````

````
[ Info: Training Machine{ProbabilisticTunedModel{Grid,…},…}.
[ Info: Attempting to evaluate 30 models.
Evaluating over 30 metamodels:   0%[>                        ]  ETA: N/A[KEvaluating over 30 metamodels:   3%[>                        ]  ETA: 0:01:33[KEvaluating over 30 metamodels:   7%[=>                       ]  ETA: 0:00:54[KEvaluating over 30 metamodels:  10%[==>                      ]  ETA: 0:00:40[KEvaluating over 30 metamodels:  13%[===>                     ]  ETA: 0:00:33[KEvaluating over 30 metamodels:  17%[====>                    ]  ETA: 0:00:28[KEvaluating over 30 metamodels:  20%[=====>                   ]  ETA: 0:00:24[KEvaluating over 30 metamodels:  23%[=====>                   ]  ETA: 0:00:22[KEvaluating over 30 metamodels:  27%[======>                  ]  ETA: 0:00:20[KEvaluating over 30 metamodels:  30%[=======>                 ]  ETA: 0:00:18[KEvaluating over 30 metamodels:  33%[========>                ]  ETA: 0:00:16[KEvaluating over 30 metamodels:  37%[=========>               ]  ETA: 0:00:14[KEvaluating over 30 metamodels:  40%[==========>              ]  ETA: 0:00:13[KEvaluating over 30 metamodels:  43%[==========>              ]  ETA: 0:00:12[KEvaluating over 30 metamodels:  47%[===========>             ]  ETA: 0:00:11[KEvaluating over 30 metamodels:  50%[============>            ]  ETA: 0:00:09[KEvaluating over 30 metamodels:  53%[=============>           ]  ETA: 0:00:08[KEvaluating over 30 metamodels:  57%[==============>          ]  ETA: 0:00:08[KEvaluating over 30 metamodels:  60%[===============>         ]  ETA: 0:00:07[KEvaluating over 30 metamodels:  63%[===============>         ]  ETA: 0:00:06[KEvaluating over 30 metamodels:  67%[================>        ]  ETA: 0:00:05[KEvaluating over 30 metamodels:  70%[=================>       ]  ETA: 0:00:04[KEvaluating over 30 metamodels:  73%[==================>      ]  ETA: 0:00:04[KEvaluating over 30 metamodels:  77%[===================>     ]  ETA: 0:00:03[KEvaluating over 30 metamodels:  80%[====================>    ]  ETA: 0:00:03[KEvaluating over 30 metamodels:  83%[====================>    ]  ETA: 0:00:02[KEvaluating over 30 metamodels:  87%[=====================>   ]  ETA: 0:00:02[KEvaluating over 30 metamodels:  90%[======================>  ]  ETA: 0:00:01[KEvaluating over 30 metamodels:  93%[=======================> ]  ETA: 0:00:01[KEvaluating over 30 metamodels:  97%[========================>]  ETA: 0:00:00[KEvaluating over 30 metamodels: 100%[=========================] Time: 0:00:11[K

````

![](learning_curve2.png)

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
plt = histogram(rand(sampler_r, 10000), nbins=50)
savefig("gamma_sampler.png")
````

![](gamma_sampler.png)

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
    model = Pipeline731(
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
Pipeline731(
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
plt = plot(tuned_mach)
savefig("tuning.png")
````

![](tuning.png)

Finally, let's compare cross-validation estimate of the performance
of the self-tuning model with that of the original model (an example
of [*nested
resampling*]((https://mlr.mlr-org.com/articles/tutorial/nested_resampling.html)
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
Pipeline744(
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
plt = histogram(samples, nbins=50)
savefig("uniform_sampler.png")
````

![](uniform_sampler.png)

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
│ -0.797998  │ -12.135    │ -0.292572  │
│ 10.2709    │ 6.93493    │ -3.00118   │
│ 8.59446    │ 5.10898    │ -3.92894   │
│ 10.7357    │ 5.25086    │ -4.17694   │
│ 13.4137    │ 0.089483   │ -6.42931   │
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
 UnivariateFinite{Multiclass{3}}(1=>0.794, 2=>0.0317, 3=>0.174)
 UnivariateFinite{Multiclass{3}}(1=>0.0647, 2=>0.0908, 3=>0.845)
 UnivariateFinite{Multiclass{3}}(1=>0.0863, 2=>0.135, 3=>0.778)
 UnivariateFinite{Multiclass{3}}(1=>0.0671, 2=>0.166, 3=>0.767)
 UnivariateFinite{Multiclass{3}}(1=>0.0558, 2=>0.508, 3=>0.436)
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
        Machine{LogisticClassifier,…}, 
        transform(
            Machine{Standardizer,…}, 
            Source @707))
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
│ -1.69627   │ -1.6898    │ 1.47352    │
│ 0.335451   │ 0.754247   │ 0.254171   │
│ 0.0277376  │ 0.520228   │ -0.163481  │
│ 0.420759   │ 0.538412   │ -0.275123  │
│ 0.912322   │ -0.123084  │ -1.28908   │
└────────────┴────────────┴────────────┘

````

````julia
fit!(yhat);
yhat()
````

````
5-element MLJBase.UnivariateFiniteVector{Multiclass{3}, Int64, UInt32, Float64}:
 UnivariateFinite{Multiclass{3}}(1=>0.794, 2=>0.0317, 3=>0.174)
 UnivariateFinite{Multiclass{3}}(1=>0.0647, 2=>0.0908, 3=>0.845)
 UnivariateFinite{Multiclass{3}}(1=>0.0863, 2=>0.135, 3=>0.778)
 UnivariateFinite{Multiclass{3}}(1=>0.0671, 2=>0.166, 3=>0.767)
 UnivariateFinite{Multiclass{3}}(1=>0.0558, 2=>0.508, 3=>0.436)
````

The node `yhat` is the "descendant" (in an associated DAG we have
defined) of a unique source node:

````julia
sources(yhat)
````

````
2-element Vector{Any}:
 Source @707 ⏎ `Table{AbstractVector{Continuous}}`
 Source @866 ⏎ `AbstractVector{Multiclass{3}}`
````

The data at the source node is replaced by `Xnew` to obtain a
new prediction when we call `yhat` like this:

````julia
Xnew, _ = make_blobs(2, 3);
yhat(Xnew)
````

````
2-element MLJBase.UnivariateFiniteVector{Multiclass{3}, Int64, UInt32, Float64}:
 UnivariateFinite{Multiclass{3}}(1=>0.331, 2=>0.000216, 3=>0.669)
 UnivariateFinite{Multiclass{3}}(1=>0.176, 2=>0.000586, 3=>0.823)
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
    1:	Source @707 ⏎ `Table{AbstractVector{Continuous}}`
    2:	Source @866 ⏎ `AbstractVector{Multiclass{3}}`

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
 UnivariateFinite{Multiclass{3}}(1=>0.68, 2=>0.11, 3=>0.21)
 UnivariateFinite{Multiclass{3}}(1=>0.477, 2=>0.0994, 3=>0.423)
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
Source @998 ⏎ `Nothing`
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
    1:	Source @998
  formula:
    transform(
        Machine{UnivariateBoxCoxTransformer,…}, 
        Source @998)
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
                Machine{RidgeRegressor,…}, 
                transform(
                    Machine{Standardizer,…}, 
                    Source @696))),
        #134(
            predict(
                Machine{RandomForestRegressor,…}, 
                transform(
                    Machine{Standardizer,…}, 
                    Source @696))))
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
        Machine{UnivariateBoxCoxTransformer,…}, 
        +(
            #134(
                predict(
                    Machine{RidgeRegressor,…}, 
                    transform(
                        Machine{Standardizer,…}, 
                        Source @696))),
            #134(
                predict(
                    Machine{RandomForestRegressor,…}, 
                    transform(
                        Machine{Standardizer,…}, 
                        Source @696)))))
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
│ RootMeanSquaredError() │ 3.82        │ predict   │ [3.67, 3.03, 4.4, 3.71, 4.18, 3.75]  │
│ MeanAbsoluteError()    │ 2.45        │ predict   │ [2.32, 2.35, 2.65, 2.65, 2.39, 2.35] │
└────────────────────────┴─────────────┴───────────┴──────────────────────────────────────┘

````

### Resources for Part 5

- From the MLJ manual:
   - [Learning Networks](https://alan-turing-institute.github.io/MLJ.jl/stable/composing_models/#Learning-Networks-1)
- From Data Science Tutorials:
    - [Learning Networks](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/learning-networks/)
    - [Learning Networks 2](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/learning-networks-2/)

    - [Stacking](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/stacking/): an advanced example of model composition

    - [Finer Control](https://alan-turing-institute.github.io/MLJ.jl/dev/composing_models/#Method-II:-Finer-control-(advanced)-1):
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
    plot!(one_curve.parameter_values, one_curve.measurements)
end
xlabel!(plt, "n_trees")
ylabel!(plt, "cross entropy")
savefig("exercise_6ci.png")
````

````
[ Info: Training Machine{ProbabilisticTunedModel{Grid,…},…}.
[ Info: Attempting to evaluate 29 models.
Evaluating over 29 metamodels:   7%[=>                       ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  10%[==>                      ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  14%[===>                     ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  17%[====>                    ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  21%[=====>                   ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  24%[======>                  ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  28%[======>                  ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  31%[=======>                 ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  34%[========>                ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  38%[=========>               ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  41%[==========>              ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  45%[===========>             ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  48%[============>            ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  52%[============>            ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  55%[=============>           ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  59%[==============>          ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  62%[===============>         ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  66%[================>        ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  69%[=================>       ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  72%[==================>      ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  76%[==================>      ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  79%[===================>     ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  83%[====================>    ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  86%[=====================>   ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  90%[======================>  ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  93%[=======================> ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  97%[========================>]  ETA: 0:00:00[KEvaluating over 29 metamodels: 100%[=========================] Time: 0:00:00[K
[ Info: Training Machine{ProbabilisticTunedModel{Grid,…},…}.
[ Info: Attempting to evaluate 29 models.
Evaluating over 29 metamodels:   7%[=>                       ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  10%[==>                      ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  14%[===>                     ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  17%[====>                    ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  21%[=====>                   ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  24%[======>                  ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  28%[======>                  ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  31%[=======>                 ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  34%[========>                ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  38%[=========>               ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  41%[==========>              ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  45%[===========>             ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  48%[============>            ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  52%[============>            ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  55%[=============>           ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  59%[==============>          ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  62%[===============>         ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  66%[================>        ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  69%[=================>       ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  72%[==================>      ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  76%[==================>      ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  79%[===================>     ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  83%[====================>    ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  86%[=====================>   ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  90%[======================>  ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  93%[=======================> ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  97%[========================>]  ETA: 0:00:00[KEvaluating over 29 metamodels: 100%[=========================] Time: 0:00:00[K
[ Info: Training Machine{ProbabilisticTunedModel{Grid,…},…}.
[ Info: Attempting to evaluate 29 models.
Evaluating over 29 metamodels:   0%[>                        ]  ETA: N/A[KEvaluating over 29 metamodels:   3%[>                        ]  ETA: 0:00:00[KEvaluating over 29 metamodels:   7%[=>                       ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  10%[==>                      ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  14%[===>                     ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  17%[====>                    ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  21%[=====>                   ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  24%[======>                  ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  28%[======>                  ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  31%[=======>                 ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  34%[========>                ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  38%[=========>               ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  41%[==========>              ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  45%[===========>             ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  48%[============>            ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  52%[============>            ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  55%[=============>           ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  59%[==============>          ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  62%[===============>         ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  66%[================>        ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  69%[=================>       ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  72%[==================>      ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  76%[==================>      ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  79%[===================>     ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  83%[====================>    ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  86%[=====================>   ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  90%[======================>  ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  93%[=======================> ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  97%[========================>]  ETA: 0:00:00[KEvaluating over 29 metamodels: 100%[=========================] Time: 0:00:00[K
[ Info: Training Machine{ProbabilisticTunedModel{Grid,…},…}.
[ Info: Attempting to evaluate 29 models.
Evaluating over 29 metamodels:   7%[=>                       ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  10%[==>                      ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  14%[===>                     ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  17%[====>                    ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  21%[=====>                   ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  24%[======>                  ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  28%[======>                  ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  31%[=======>                 ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  34%[========>                ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  38%[=========>               ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  41%[==========>              ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  45%[===========>             ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  48%[============>            ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  52%[============>            ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  55%[=============>           ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  59%[==============>          ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  62%[===============>         ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  66%[================>        ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  69%[=================>       ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  72%[==================>      ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  76%[==================>      ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  79%[===================>     ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  83%[====================>    ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  86%[=====================>   ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  90%[======================>  ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  93%[=======================> ]  ETA: 0:00:00[KEvaluating over 29 metamodels:  97%[========================>]  ETA: 0:00:00[KEvaluating over 29 metamodels: 100%[=========================] Time: 0:00:00[K

````

![](exercise_6ci.png)

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
1.5724544828946136
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
Pipeline770(
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
│ LogLoss(tol = 2.22045e-16) │ 0.809       │ predict   │ [0.833, 1.02, 0.789, 0.733, 0.789, 0.695] │
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
savefig("exercise_7c.png")
````

````
[ Info: Training Machine{ProbabilisticTunedModel{Grid,…},…}.
[ Info: Attempting to evaluate 10 models.
Evaluating over 10 metamodels:   0%[>                        ]  ETA: N/A[KEvaluating over 10 metamodels:  10%[==>                      ]  ETA: 0:00:01[KEvaluating over 10 metamodels:  20%[=====>                   ]  ETA: 0:00:01[KEvaluating over 10 metamodels:  30%[=======>                 ]  ETA: 0:00:01[KEvaluating over 10 metamodels:  40%[==========>              ]  ETA: 0:00:01[KEvaluating over 10 metamodels:  50%[============>            ]  ETA: 0:00:02[KEvaluating over 10 metamodels:  60%[===============>         ]  ETA: 0:00:02[KEvaluating over 10 metamodels:  70%[=================>       ]  ETA: 0:00:02[KEvaluating over 10 metamodels:  80%[====================>    ]  ETA: 0:00:02[KEvaluating over 10 metamodels:  90%[======================>  ]  ETA: 0:00:02[KEvaluating over 10 metamodels: 100%[=========================] Time: 0:00:26[K

````

![](exercise_7c.png)

Here's a second curve using a different random seed for the booster:

````julia
using Random
pipe.evo_tree_classifier.rng = MersenneTwister(123)
curve = learning_curve(mach,
                       range=r,
                       resampling=CV(nfolds=6),
                       measure=cross_entropy)
plot!(curve.parameter_values, curve.measurements)
savefig("exercise_7c_2.png")
````

````
[ Info: Training Machine{ProbabilisticTunedModel{Grid,…},…}.
[ Info: Attempting to evaluate 10 models.
Evaluating over 10 metamodels:  20%[=====>                   ]  ETA: 0:00:01[KEvaluating over 10 metamodels:  30%[=======>                 ]  ETA: 0:00:01[KEvaluating over 10 metamodels:  40%[==========>              ]  ETA: 0:00:01[KEvaluating over 10 metamodels:  50%[============>            ]  ETA: 0:00:01[KEvaluating over 10 metamodels:  60%[===============>         ]  ETA: 0:00:02[KEvaluating over 10 metamodels:  70%[=================>       ]  ETA: 0:00:02[KEvaluating over 10 metamodels:  80%[====================>    ]  ETA: 0:00:02[KEvaluating over 10 metamodels:  90%[======================>  ]  ETA: 0:00:02[KEvaluating over 10 metamodels: 100%[=========================] Time: 0:00:25[K

````

![](exercise_7c_2.png)

One can automatic the production of multiple curves with different
seeds in the following way:

````julia
curves = learning_curve(mach,
                        range=r,
                        resampling=CV(nfolds=6),
                        measure=cross_entropy,
                        rng_name=:(evo_tree_classifier.rng),
                        rngs=6) # list of RNGs, or num to auto generate
plt = plot(curves.parameter_values, curves.measurements)
savefig("exercise_7c_3.png")
````

````
Evaluating Learning curve with 6 rngs:   0%[>                 ]  ETA: N/A[KEvaluating Learning curve with 6 rngs:  17%[===>              ]  ETA: 0:02:05[KEvaluating Learning curve with 6 rngs:  33%[======>           ]  ETA: 0:01:43[KEvaluating Learning curve with 6 rngs:  50%[=========>        ]  ETA: 0:01:18[KEvaluating Learning curve with 6 rngs:  67%[============>     ]  ETA: 0:00:53[KEvaluating Learning curve with 6 rngs:  83%[===============>  ]  ETA: 0:00:26[KEvaluating Learning curve with 6 rngs: 100%[==================] Time: 0:02:42[K

````

![](exercise_7c_3.png)

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
Pipeline779(
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
plt = plot(tuned_mach)
savefig("exercise_8c.png")
````

````
[ Info: Training Machine{DeterministicTunedModel{RandomSearch,…},…}.
[ Info: Attempting to evaluate 40 models.
Evaluating over 40 metamodels:   0%[>                        ]  ETA: N/A[KEvaluating over 40 metamodels:   2%[>                        ]  ETA: 0:06:11[KEvaluating over 40 metamodels:   5%[=>                       ]  ETA: 0:03:30[KEvaluating over 40 metamodels:   8%[=>                       ]  ETA: 0:02:20[KEvaluating over 40 metamodels:  10%[==>                      ]  ETA: 0:01:56[KEvaluating over 40 metamodels:  12%[===>                     ]  ETA: 0:01:33[KEvaluating over 40 metamodels:  15%[===>                     ]  ETA: 0:01:22[KEvaluating over 40 metamodels:  18%[====>                    ]  ETA: 0:01:10[KEvaluating over 40 metamodels:  20%[=====>                   ]  ETA: 0:01:27[KEvaluating over 40 metamodels:  22%[=====>                   ]  ETA: 0:02:49[KEvaluating over 40 metamodels:  25%[======>                  ]  ETA: 0:02:39[KEvaluating over 40 metamodels:  28%[======>                  ]  ETA: 0:02:37[KEvaluating over 40 metamodels:  30%[=======>                 ]  ETA: 0:02:19[KEvaluating over 40 metamodels:  32%[========>                ]  ETA: 0:02:18[KEvaluating over 40 metamodels:  35%[========>                ]  ETA: 0:02:06[KEvaluating over 40 metamodels:  38%[=========>               ]  ETA: 0:01:58[KEvaluating over 40 metamodels:  40%[==========>              ]  ETA: 0:01:47[KEvaluating over 40 metamodels:  42%[==========>              ]  ETA: 0:01:39[KEvaluating over 40 metamodels:  45%[===========>             ]  ETA: 0:01:32[KEvaluating over 40 metamodels:  48%[===========>             ]  ETA: 0:01:24[KEvaluating over 40 metamodels:  50%[============>            ]  ETA: 0:01:17[KEvaluating over 40 metamodels:  52%[=============>           ]  ETA: 0:01:15[KEvaluating over 40 metamodels:  55%[=============>           ]  ETA: 0:01:18[KEvaluating over 40 metamodels:  58%[==============>          ]  ETA: 0:01:10[KEvaluating over 40 metamodels:  60%[===============>         ]  ETA: 0:01:05[KEvaluating over 40 metamodels:  62%[===============>         ]  ETA: 0:00:58[KEvaluating over 40 metamodels:  65%[================>        ]  ETA: 0:01:07[KEvaluating over 40 metamodels:  70%[=================>       ]  ETA: 0:00:54[KEvaluating over 40 metamodels:  72%[==================>      ]  ETA: 0:00:58[KEvaluating over 40 metamodels:  75%[==================>      ]  ETA: 0:00:51[KEvaluating over 40 metamodels:  78%[===================>     ]  ETA: 0:00:46[KEvaluating over 40 metamodels:  80%[====================>    ]  ETA: 0:00:40[KEvaluating over 40 metamodels:  82%[====================>    ]  ETA: 0:00:34[KEvaluating over 40 metamodels:  85%[=====================>   ]  ETA: 0:00:29[KEvaluating over 40 metamodels:  88%[=====================>   ]  ETA: 0:00:36[KEvaluating over 40 metamodels:  90%[======================>  ]  ETA: 0:00:35[KEvaluating over 40 metamodels:  92%[=======================> ]  ETA: 0:00:26[KEvaluating over 40 metamodels:  95%[=======================> ]  ETA: 0:00:17[KEvaluating over 40 metamodels:  98%[========================>]  ETA: 0:00:08[KEvaluating over 40 metamodels: 100%[=========================] Time: 0:05:24[K

````

![](exercise_8c.png)

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

