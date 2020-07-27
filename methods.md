# List of methods introduced in the tutorials

## Part 1

`scitype(object)`, `coerce(vector, SomeSciType)`,
`levels(categorical_vector)`, `levels!(categorical_vector)`,
`schema(table)`, `MLJ.table(matrix)`, `autotype(table)`,
`coerce(table, ...)`, `coerce!(dataframe, ...)`, `elscitype(vector)` 

## Part 2

`OpenML.load(id)`, `unpack(table, ...)`, `models()`, `models(filter)`,
`models(string)`, `@load ModelType pkg=PackageName`, `info(model)`,
`machine(model, X, y)`, `partition(row_indices, ...)`, `fit!(mach,
rows=...)`, `predict(mach, rows=...)`, `predict(mach, Xnew)`,
`fitted_params(mach)`, `report(mach)`, `MLJ.save`,
`machine(filename)`, `machine(filename, X, y)`,
`pdf(single_prediction, class)`, `predict_mode(mach, Xnew)`,
`predict_mean(mach, Xnew)`, `predict_median(mach, Xnew)`,
`measures()`, `evaluate!`, `range(model, :(param.nested_param), ...)`,
`learning_curve(mach, ...)`

## Part 3

`Standardizer`, `transform`, `inverse_transform`, `ContinuousEncoder`, `@pipeline`

## Part 4

`iterator(r, resolution)`, `sampler(r, distribution)`, `RandomSearch`,
`TunedModel`

## Part 5

`source(data)`, `source()`, `Probabilistic()`, `Deterministic()`,
`Unsupervised()`, `@from_network`
