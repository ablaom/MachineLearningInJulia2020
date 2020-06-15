# Machine Learning in Julia using MLJ

## Housekeeping

### Getting help during the workshop

### Resources to help you

From the MLJ ecosystem:

- The docs

- DataScienceTutorials

From elsewhere:

- Julia specific:

   - ScikitLearn

- General:

   -

   - 
   
## Programme

- An overview of machine learning and MLJ (lecture)

- Workshop scope

- Installing MLJ and the tutorials

- Part 1: Data ingestion and pre-processing 

Break

- Part 2: Basic fit and predict/transform

- Part 3: Evaluating model performance

- Part 4: Tuning model hyper-parameters 

Break

- Part 5: Model pipelines

- Part 6: Learning networks (lecture)

Each Parts 2-6 begins with demonstration on the "teacher's dataset", with
time for participants to carry out a similar exercise on a "student's
datasets" and interact with the instructors in the chat forum. 


## What this workshop won't cover

This workshop assumes at some experience with data and, ideally, some
understanding of machine learning principles.

Lightly covered or not covered

- data wrangling and data cleaning

- feature engineering

- options for parallelism or using GPU's 


## Part 1: Data ingestion and pre-processing 

### What is machine learning?

Supervised learning - show with examples and pictures what the basic
idea and processes are: fitting, evaluating, tuning.

Unsupervised learning - no labels; main use-case is dimension reduction; explain PCA with a picture

Re-enforcement learning - out of scope


### Different machine learning models and paradigms

- machine learning ≠ deep learning

- there are hundreds of machine learning models. All of the following
  are in common use:
  
  - linear models, especially Ridge regression, elastic net, pca (unsupervised)
  
  - Naive Bayes 
  
  - K-nearest neighbours
  
  - K-means clustering (unsupervised)
  
  - random forests
  
  - gradient boosted tree models (e.g., XGBoost)
  
  - support vector machines
  
  - probablistic programming models
  
  - neural networks
  

### What is a (good) machine learning toolbox?

- provides uniform interface to zoo of models scattered everywhere
  (different packages, different languages)

- provides a searchable model registry

- meta-algorithms: 

    - evaluating performance using different performance measures (aka
      metrics, scores, loss functions)
	  
	- tuning (optimizing hyperparmaters)
	
	- facilitates model *composition* (e.g., pipelines)
	
- customizable (getting under the hood)

### MLJ features 


### A short tour of MLJ


## Part 1: Data ingestion and pre-processing 

### Scientific types and type coercion

- inspecting scitypes and coercing them

- working with categorical data 

### Tabular data

- Lots of things can be considered as tabular data; examples: native
  tables, matrices, DataFrames, CSV files

- Lots of ways to grab data; examples:

   - load a canned dataset
   - load from local file (e.g., csv)
   - create a synthetic data set
   - use OpenML
   - use RDatasets
   - use UrlDownload (or is there something better?)
   
### Demo - 


### Exercise






	

