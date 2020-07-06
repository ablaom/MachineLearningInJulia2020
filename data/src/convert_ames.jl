using Pkg
Pkg.activate(joinpath(@__DIR__, "convert_ames"))
Pkg.instantiate()

using DataFrames, CSV, MLJBase, CategoricalArrays

df = CSV.read(joinpath(@__DIR__, "reduced_ames.csv"))

schema(df)

price  = df.target
quality = df.OverallQual
area1 = map(df.GrLivArea) do a round(Int, a) end
area2 = map(df.x1stFlrSF) do a round(Int, a) end
area3 = map(df.TotalBsmtSF) do a round(Int, a) end
area4 = map(df.BsmtFinSF1) do a round(Int, a) end
area5 = map(df.GarageArea) do a round(Int, a) end
lot_area = map(df.LotArea) do a round(Int, a) end
garage_cars = map(df.GarageCars) do a round(Int, a) end
suburb = df.Neighborhood
council_code = map(df.MSSubClass) do a parse(Int, a[2:end]) end
year_built = map(df.YearBuilt) do a round(Int, a) end
year_upgraded =  map(df.YearRemodAdd) do a round(Int, a) end
zone = df.MSSubClass

df2 = DataFrame(price=price,
                area1=area1,
                area2=area2,
                area3=area3,
                area4=area4,
                area5=area5,
                lot_area=lot_area,
                year_built=year_built,
                year_upgraded=year_upgraded,
                quality=quality,
                garage_cars=garage_cars,
                suburb=suburb,
                council_code=council_code,
                zone=zone)

CSV.write(joinpath(@__DIR__, "ames.csv"), df)

