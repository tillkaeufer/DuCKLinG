# Dust Continuum Kit with Line emission from Gas (DuCKLinG)

<img src="./DuCKLinG_logo.png" width="250" />

This repository gives you all the files to run the DuCKLinG model.  
It can be run as a forward model or in retrieval with MultiNest or UltraNest.

## Install

DuCKLinG is not installed, it is a Python object.
There are several Python packages however, that need to be installed in your environment:

Additionally, it is recommended to install OpenMPI to run the retrieval in parallel.

## Getting started
### Forward model

The notebook forward_model gives a quick introduction to the different functionalities of the model.  
You can run it and see how different molecular conditions and dust species change the resulting output.

### Retrieval

There is an example input file in the Example folder.
You can use this as a test ground if everything works.

## How to run

## How to plot

After running a retrieval you might want to have a look at the results.  
The plot_retrieval_results program does that for you.

You can run it with:  

Additionally, it accepts a large range of options to specify your plotting needs:


## Licence
