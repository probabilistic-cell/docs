# Amortization

Amortization means estimating a variable from observed variables {citel}`gershman_amortized_2014`. This mapping is often done by a flexible (albeit black-box) function, e.g. a neural network. Amortization can have the following advantages:  

- Faster training because of a lower number of parameters.
- Ability to transfer a latent space (e.g. cluster labels) between datasets.
- Less prone of getting stuck in local minima.

It also comes with a cost:

- Prone to underfitting, because an amortized latent space is less expressive than a completely free latent space. Our latent space estimates may thus become less accurate. This is also known as the amortization gap.
- Prone to overfitting, something that is particularly important when transferring between datasets. Our latent space estimates may be very accurate on the training data, but may not well to other datasets.
- The choice of amortization function may affect the results.

```{seealso}
A more in-depth explanation of amortization is provided in its [explanation section](/explanation/amortization).
```

The prime example of amortization is the encoder function inside a variational autoencoder.

## The amortization function

## Pretraining

## Transfer