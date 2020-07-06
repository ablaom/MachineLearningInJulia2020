using Pkg;
Pkg.activate(@__DIR__)
Pkg.instantiate()

using MLJ

using HTTP
using CSV
import DataFrames: DataFrame, select!, Not
req1 = HTTP.get("http://archive.ics.uci.edu/ml/machine-learning-databases/horse-colic/horse-colic.data")
req2 = HTTP.get("http://archive.ics.uci.edu/ml/machine-learning-databases/horse-colic/horse-colic.test")
header = ["surgery", "age", "hospital_number",
    "rectal_temperature", "pulse",
    "respiratory_rate", "temperature_extremities",
    "peripheral_pulse", "mucous_membranes",
    "capillary_refill_time", "pain",
    "peristalsis", "abdominal_distension",
    "nasogastric_tube", "nasogastric_reflux",
    "nasogastric_reflux_ph", "feces", "abdomen",
    "packed_cell_volume", "total_protein",
    "abdomcentesis_appearance", "abdomcentesis_total_protein",
    "outcome", "surgical_lesion", "lesion_1", "lesion_2", "lesion_3",
    "cp_data"]
csv_opts = (header=header, delim=' ', missingstring="?",
            ignorerepeated=true)
data_train = CSV.read(req1.body; csv_opts...)
data_test  = CSV.read(req2.body; csv_opts...)
@show size(data_train)
@show size(data_test)

unwanted = [:lesion_1, :lesion_2, :lesion_3]
data = vcat(data_train, data_test)
select!(data, Not(unwanted));

train = 1:nrows(data_train)
test = last(train) .+ (1:nrows(data_test));

datac = coerce(data, autotype(data));

sch0 = schema(data)
sch = schema(datac)

old_scitype_given_name = Dict(
sch0.names[j] => sch0.scitypes[j] for j in eachindex(sch0.names))

length(unique(datac.hospital_number))

datac = select!(datac, Not(:hospital_number));

datac = coerce(datac, autotype(datac, rules=(:discrete_to_continuous,)));

missing_outcome = ismissing.(datac.outcome)
idx_missing_outcome = missing_outcome |> findall

train = setdiff!(train |> collect, idx_missing_outcome)
test = setdiff!(test |> collect, idx_missing_outcome)
datac = datac[.!missing_outcome, :];

for name in names(datac)
    col = datac[:, name]
    ratio_missing = sum(ismissing.(col)) / nrows(datac) * 100
    println(rpad(name, 30), round(ratio_missing, sigdigits=3))
end

unwanted = [:peripheral_pulse, :nasogastric_tube, :nasogastric_reflux,
        :nasogastric_reflux_ph, :feces, :abdomen, :abdomcentesis_appearance, :abdomcentesis_total_protein]
select!(datac, Not(unwanted));

@load FillImputer
filler = machine(FillImputer(), datac)
fit!(filler)
datac = transform(filler, datac)

cat_fields = filter(schema(datac).names) do field
    datac[:, field] isa CategoricalArray
end

for f in cat_fields
    datac[!, f] = get.(datac[:, f])
end

datac.pulse = coerce(datac.pulse, Count)
datac.respiratory_rate = coerce(datac.pulse, Count)

sch1 = schema(datac)

CSV.write("horse.csv", datac)
