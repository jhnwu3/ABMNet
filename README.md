# ABMNet (title needs some work)

Training a neural network to act as a surrogate model for Agent Based Models to speed up the simulation of mechanistic models used in the Das Lab.

## What is Surrogate Modeling?
Simply put, surrogate modeling is when one designs a "surrogate" model, usually with some black-box method, that sufficiently approximates the behavior of a pre-existing model. Generally, speaking, surrogate models approximate models with some "mechanistic" or interpretable behavior such that its fast computations are of something meaningful. In this repository, we explore various different deep learning architectures for surrogate modeling. In particular, we take inspiration from the following paper [here](https://www.sciencedirect.com/science/article/pii/S016926072100153X) where a mapping is formed between cellular automata model parameters and its corresponding outputs. 


![Surrogate Model Figure](./figs/SurrogateModelANN.drawio.png)


However, there exists many other approaches where some are more suitable than others. A review of surrogate modeling for finite element method computations can be found [here](https://link.springer.com/article/10.1007/s00500-022-07362-8). 


## Why Surrogate Modeling?
The big question one might ask is why create another model that models a model? While this seems like a very unnecessary roundabout step, especially when one considers the extra error in predictions resulting from adding an additional layer of abstraction, its speed benefits can not be neglected. See below for a quick example on the computational time needed to perform numerous simulations given a biochemical model and its surrogate model.

|                       | Nonlinear 6 Protein Model | Surrogate Model |
|-----------------------|---------------------------|-----------------|
| Number of Evaluations | 7,500                     | 2,344           |
| Time (s)              | 230                       | 0.104           |

The surrogate model was approximately 691 times faster than the original model. The surrogate model is capable of scaling up to far larger numbers of evaluations, which is commonly needed when performing parameter estimation and identification (which is just another way of saying optimization or fitting) tasks. For reference, the actual model was evaluated in parallel on a 30-core CPU on a high performance compute cluster whereas the surrogate model was ran on my laptop GPU, one a vastly cheaper compute platform. 


Here is a flowchart providing some contextual relationships as to how a surrogate relates to an actual mechanistic model and what we are using it for. 
![Surrogate Flowchart](figs/SurrogateModelingContext.drawio.png)

## When does surrogate modeling work? 
We started with a simple reaction network of a linear 3 protein model shown below,

![L3P](figs/lin3ExpModel.png)

In this case, the reaction network contained 5 sets of reactions (each with their own reaction rates, the thetas) and 3 abundances values (P1, P2, and P3). Given some set of 5,000 initial conditions of 3 proteins, a new set of 5,000 final conditions of 3 proteins are simulated. Taking these final conditions, we can derive some vector of 9 moments (means, variances, and covariances) of the final conditions. We do these moment computations, because there exists intrinsic noise from the reaction network simulation (i.e Gillespie) and the extrinsic noise resulting from the set of initial conditions. While this toy problem, once examined, is quite simple (as it can be approximated by a system of linear ODEs), it provided a useful frame of reference for this surrogate modeling project, because it incorporates some of the aspects that one might encounter when performing biological simulations (namely noise).

And, so, when training the surrogate model on some 8,500 sets of parameter sets to moment pairs, and then evaluating on 1,500 parameter sets in the test set, we get something that looks like this with a relatively low test average MSE of 0.0301 (please note that the abundance values range on the order of magnitude of 10^2) and an r^2 plot as well as histogram that looks like the figures below. Note that each histogram plot corresponds to some type of moment (E(X), E(X^2) E(XY), etc.). 

![L3PScatter](graphs/l3/readme/l3p_10k_test_og_scatter.png)
![L3PHis](graphs/l3/readme/l3p_10k_test_og.png)


Note: One should in principle do cross validation MSE loss (or some other metric) to get a better idea of how the surrogate model is truly performing (on average), but in this case, even with cross-validation, negligible generalization performance gains were seen (probably because the dataset contained a large number of parameter sets were sampled from some uniform distribution and thus were evenly spread out).

However, this while a useful litmus test . 
## When does it fail? 
As it turns out and this has been a subject well-studied in modeling neural networks after deterministic systems (link [here](https://arxiv.org/abs/2201.05624) for some related work), there is difficulty in neural networks learning from unbounded spaces. 

## Does a successful surrogate model's weights have any meaning? 


## Is there a good rule of thumb for hyperparameters to use?


## How to use the code written in this repository?

To those in the Das lab that might be taking up the flag in developing this project further, for a quickstart, one can simply look in the /tutorials/tutorial_for_indrani.pynb notebook for a quick rundown on how one might use the pieces of code written. There's also a cli interface that I've provided with main.py, please look in the slurm_scripts/ folder for a plethora of shell scripts used to run the code on the cluster. The flow chart below provides a general workflow of the surrogate modeling done in this repository.

![Surrogate Flowchart](figs/SurrogateFlowchart.drawio.png)

Please look in the modules/ folder for relevant pieces of code.

## Potential future directions (that I wish there was time for), taking inspiration from the scientific machine learning community. 





