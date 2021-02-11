Variables
=======================

In latenta, we build up a graph of how variables are related to eachother.

## Definition

Each variable in latenta is annotated, 

## Variable

## Leafs

### Fixed

### Parameters

## Random

## Distributions

## Computed

- Definition: Annotated tensor
  - Variable: An annotated tensor within a DAG, with components and children
    - Leaf: A variable without any components.
      - Parameter: A leaf that can be optimized.
      - Fixed: A leaf that has a fixed (either observed or given) value.
    - Random: A random variable, i.e. follows a certain distribution.
      - Observed: A random variable that is observed.
      - Latent: A random variable that needs to be inferred.
    - Distribution: A variable that is sampled from a distribution.
    - Computed: A variable that can be determinstically computed from other variables.
      - Transform: An invertible transform of a variable. Does not have to be bijective.
      - Modular: A variable that sums or multiplies different variables. One dimension can be subsetted.

### Transforms

### Modular

All other classes extend from these types of variables.
