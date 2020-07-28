# # State-of-the-art model composition in MLJ (Machine Learning in Julia)

# In this script we use [model
# stacking](https://alan-turing-institute.github.io/DataScienceTutorials.jl/getting-started/stacking/)
# to demonstrate the ease with which machine learning models can be
# combined in sophisticated ways using MLJ. In the future MLJ will
# have a canned version of stacking. For now we show how to stack
# using MLJ's generic model composition syntax, which is an extension
# of the normal fit/predict syntax.

DIR = @__DIR__
include(joinpath(DIR, "setup.jl"))

# ## Stacking is hard

# [Model
# stacking](https://alan-turing-institute.github.io/DataScienceTutorials.jl/getting-started/stacking/),
# popular in Kaggle data science competitions, is a sophisticated way
# to blend the predictions of multiple models.

# With the python toolbox
# [scikit-learn](https://scikit-learn.org/stable/) (or its [julia
# wrap](https://github.com/cstjean/ScikitLearn.jl)) you can use
# pipelines to combine composite models in simple ways but (automated)
# stacking is beyond its capabilities.

# One python alternative is to use
# [vecstack](https://github.com/vecxoz/vecstack). The [core
# algorithm](https://github.com/vecxoz/vecstack/blob/master/vecstack/core.py)
# is about eight pages (without the scikit-learn interface):

# ![](vecstack.png).

# ## Stacking is easy (in MLJ)

# Using MLJ's [generic model composition
# API](https://alan-turing-institute.github.io/MLJ.jl/dev/composing_models/)
# you can build a stack in about a page.

# Here's the complete code needed to define a new model type that
# stacks two base regressors and one adjudicator in MLJ.  Here we use
# three folds to create the base-learner [out-of-sample
# predictions](https://alan-turing-institute.github.io/DataScienceTutorials.jl/getting-started/stacking/)
# to make it easier to read. You can make this generic with little fuss.

using MLJ

folds(data, nfolds) =
    partition(1:nrows(data), (1/nfolds for i in 1:(nfolds-1))...);

model1 = @load LinearRegressor pkg=MLJLinearModels
model2 = @load LinearRegressor pkg=MLJLinearModels
judge = @load LinearRegressor pkg=MLJLinearModels

X = source()
y = source()

folds(X::AbstractNode, nfolds) = node(XX->folds(XX, nfolds), X)
MLJ.restrict(X::AbstractNode, f::AbstractNode, i) =
    node((XX, ff) -> restrict(XX, ff, i), X, f);
MLJ.corestrict(X::AbstractNode, f::AbstractNode, i) =
    node((XX, ff) -> corestrict(XX, ff, i), X, f);

f = folds(X, 3)

m11 = machine(model1, corestrict(X, f, 1), corestrict(y, f, 1))
m12 = machine(model1, corestrict(X, f, 2), corestrict(y, f, 2))
m13 = machine(model1, corestrict(X, f, 3), corestrict(y, f, 3))

y11 = predict(m11, restrict(X, f, 1));
y12 = predict(m12, restrict(X, f, 2));
y13 = predict(m13, restrict(X, f, 3));

m21 = machine(model2, corestrict(X, f, 1), corestrict(y, f, 1))
m22 = machine(model2, corestrict(X, f, 2), corestrict(y, f, 2))
m23 = machine(model2, corestrict(X, f, 3), corestrict(y, f, 3))

y21 = predict(m21, restrict(X, f, 1));
y22 = predict(m22, restrict(X, f, 2));
y23 = predict(m23, restrict(X, f, 3));

y1_oos = vcat(y11, y12, y13);
y2_oos = vcat(y21, y22, y23);

X_oos = MLJ.table(hcat(y1_oos, y2_oos))

m_judge = machine(judge, X_oos, y)

m1 = machine(model1, X, y)
m2 = machine(model2, X, y)

y1 = predict(m1, X);
y2 = predict(m2, X);

X_judge = MLJ.table(hcat(y1, y2))
yhat = predict(m_judge, X_judge)

@from_network machine(Deterministic(), X, y; predict=yhat) begin
    mutable struct MyStack
        regressor1=model1
        regressor2=model2
        judge=judge
    end
end

my_stack = MyStack()

# For the curious: Only the last block defines the new model type. The
# rest defines a *[learning network]()* - a kind of working prototype
# or blueprint for the type. If the source nodes `X` and `y` wrap some
# data (instead of nothing) then the network can be trained and tested
# as you build it.


# ## Composition plays well with other work-flows

# We did not include standardization of inputs and target (with
# post-prediction inversion) in our stack. However, we can add these
# now, using MLJ's canned pipeline composition:

pipe = @pipeline Standardizer my_stack target=Standardizer

# Want to change a base learner and adjudicator?

pipe.my_stack.regressor2 = @load DecisionTreeRegressor pkg=DecisionTree;
pipe.my_stack.judge = @load KNNRegressor;

# Want a CV estimate of performance of the complete model on some data?

X, y = @load_boston;
mach = machine(pipe, X, y)
evaluate!(mach, resampling=CV(), measure=mae)

# Want to inspect the learned parameters of the adjudicator?

fp =  fitted_params(mach);
fp.my_stack.judge

# What about the first base-learner of the stack? There are four sets
# of learned parameters!  One for each fold to make an out-of-sample
# prediction, and one trained on all the data:

fp.my_stack.regressor1

#-

fp.my_stack.regressor1[1].coefs

# Want to tune multiple (nested) hyperparameters in the stack? Tuning is a
# model wrapper (for better composition!):

r1 = range(pipe, :(my_stack.regressor2.max_depth), lower = 1, upper = 25)
r2 = range(pipe, :(my_stack.judge.K), lower=1, origin=10, unit=10)

import Distributions.Poisson

tuned_pipe = TunedModel(model=pipe,
                         ranges=[r1, (r2, Poisson)],
                         tuning=RandomSearch(),
                         resampling=CV(),
                         measure=rms,
                         n=100)
mach = machine(tuned_pipe, X, y) |> fit!
best_model = fitted_params(mach).best_model
K = fitted_params(mach).best_model.my_stack.judge.K;
max_depth = fitted_params(mach).best_model.my_stack.regressor2.max_depth
@show K max_depth;

# Visualize tuning results:

using Plots
pyplot()
plot(mach)

#

using Literate #src
Literate.notebook(@__FILE__, @__DIR__, execute=false) #src
