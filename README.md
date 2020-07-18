# Machine Learning in Julia, JuliaCon2020

A workshop introducing the machine learning toolbox
[MLJ](https://alan-turing-institute.github.io/MLJ.jl/stable/)


<div align="center">
	<img src="MLJLogo2.svg" alt="MLJ" width="200">
</div>


## About the tutorials

These tutorials were prepared for use in a 3.5 hour online workshop at
JuliaCon2020. Their main aim is to introuduce the
[MLJ](https://alan-turing-institute.github.io/MLJ.jl/stable/) machine
learning toolbox to working data scientists.

- The tutorials focus on *machine learning* part of the data science
  workflow, and less on exploratory data analysis and other
  conventional "data analytics" methodology

- "Machine learning" is meant here in a broad sense, and is not
  restricted to so-called *deep learning* (neural networks)

- The tutorials are crafted to rapidly familiarize the user with what
  MLJ can do, and are not a substitute for a course on machine
  learning fundamentals. Examples do not necessarily represent best
  practice or the best solution to a problem.


## Options for running the tutorials

**If all else fails**, a static version of the tutorials can be viewed
[here](tutorials.md).


### 1. Plug-and-play

Recommended for users with little Julia experience or users having
problems with the other options.

If you never ran a Julia/Juptyer notebook on your local machine before, never
used a Julia IDE before, or never cloned a GitHub repository, then use
this option.


#### Pros

No need to install Julia or any libraries on your local machine


#### Cons

- The (automatic) setup can take a little while

- If the online notebook crashes or drops your connection, you will have to start over


#### Instructions

Click [here]() BINDER NOTEBOOK LINK


### 2. Clone the repo and choose your preferred interface

Assumes that you have a working installation of
[Julia](https://julialang.org/downloads/) 1.3 or higher and that
either:

- You can run Julia/Juptyer notebooks on your local machine without problems; or

- You are comfortable running Julia scripts from an IDE, such as [Juno](https://junolab.org) or [Emacs](https://github.com/JuliaEditorSupport/julia-emacs) (see [here](https://julialang.org) for a complete list).


#### Pros

More stable option

#### Cons

You need to meet above requirements


#### Instructions

- Clone this repository

- Change to your local repo directory "MachineLearningInJulia2020/"

- Either run the Juptyper notebook called "tutorials.ipynb", or open
  "tutorials.jl" in your favourite IDE
