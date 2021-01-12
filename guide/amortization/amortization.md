# Amortization

Amortization means estimating a variable from observed variables <span style="border-bottom: 1px dotted;cursor: pointer;" data-container="body" data-toggle="popover" data-placement="auto" data-trigger="click" data-content="Ranganath, Rajesh, Sean Gerrish, and David M. Blei. 'Black box variational inference.' arXiv preprint arXiv:1401.0118 (2013).">(Ranganath, 2013)</span>. This mapping is often done by a flexible (albeit black-box) function, e.g. a neural network. Amortization can have the following advantages:

- Faster training because of a lower number of parameters.
- Ability to transfer a latent space (e.g. cluster labels) between datasets.
- Improved training because latent variables of individual samples are less prone to get stuck in local minima.

It also comes with a cost:

- Prone to underfitting, because an amortized latent space is less expressive than a completely free latent space. Our latent space estimates may thus become less accurate.
- Prone to overfitting, something that is particularly important when transferring between datasets. Our latent space estimates may be very accurate on the training data, but may not well to other datasets.
- We need to choose an amortization function, and this choice may affect our results.

```{seealso}
A more in-depth explanation of amortization is provided in its [explanation section](/explanation/amortization).
```

The prime example of amortization is the encoder function inside a variational autoencoder.

<script>
jQuery(function ($) {
  $("[data-toggle='popover']").popover({trigger: "click"}).click(function (event) {
    event.stopPropagation();

  }).on('inserted.bs.popover', function () {
    $(".popover").click(function (event) {
      event.stopPropagation();
    })
  })

  $(document).click(function () {
    $("[data-toggle='popover']").popover('hide')
  })
})
</script>

## The amortization function

## Pretraining

## Transfer