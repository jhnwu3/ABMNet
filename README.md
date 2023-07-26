# ABMNets - 

Training a neural network to act as a surrogate model for Agent Based Models to speed up the simulation of mechanistic models used in the Das Lab.

## What is Surrogate Modeling?
Simply put, surrogate modeling is when one designs a "surrogate" model, usually with some black-box method, that sufficiently approximates the behavior of a pre-existing model. Generally, speaking, surrogate models approximate models with some "mechanistic" or interpretable behavior such that its fast computations are of something meaningful. In this repository, we explore various different deep learning architectures for surrogate modeling. In particular, we take inspiration from the following paper [here](https://www.sciencedirect.com/science/article/pii/S016926072100153X) where a mapping is formed between cellular automata model parameters and its corresponding outputs. 


![Surrogate Model Figure](./figs/.png)


However, there exists many other approaches where some are more suitable than others. A review of surrogate modeling for finite element method computations can be found [here](https://link.springer.com/article/10.1007/s00500-022-07362-8). 


## Why Surrogate Modeling?
The big question one might ask is why create another model that models a model? While this seems like a very unnecessary roundabout step, especially when one considers the extra error in predictions resulting from adding an additional layer of abstraction, its speed benefits can not be neglected. See below for a 

## How is surrogate modeling applied in this repository? 



< insert picture of super long parameter estimation runs and simulation times>

Here is a flowchart providing some contextual relationships as to how a surrogate relates to an actual mechanistic model and what we are using it for. 
![Context](figs/Surrogate.jpg)

## When does surrogate modeling work? 


## When does it fail? 


## How to use the 


## Potential Future Directions, taking inspiration from the scientific machine learning community. 





