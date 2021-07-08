<div class="container">
<div class="row full-width">
    <div class="col-md-3 col-sm-4 col-lg">
        <div class="card h-100">
            <div class="card-body">
                <h3>🎬 Getting started</h3>
                Installation and first steps
                <a href="/tutorials/getting-started.html" class="stretched-link"></a>
            </div>
        </div>
    </div>
    <div class="col-md-3 col-sm-4 col-lg">
        <div class="card h-100">
            <div class="card-body">
                <h3>👨‍🎓 Tutorials</h3>
                Learning latenta by example
                <a href="/tutorials/overview.html" class="stretched-link"></a>
            </div>
        </div>
    </div>
    <div class="col-md-3 col-sm-4 col-lg">
        <div class="card h-100">
            <div class="card-body">
                <h3>👩‍🍳 User guide</h3>
                How do I do «…»?
                <a href="/guide/overview.html" class="stretched-link"></a>
            </div>
        </div>
    </div>
</div>
<br>
<div class="row full-width">
    <div class="col-md-3 col-sm-4 col-lg">
        <div class="card h-100">
            <div class="card-body">
                <h3>🕵️‍♀️ Explanation</h3>
                Intro into the maths, statistics and machine learning
                <a href="/explanation/overview.html" class="stretched-link"></a>
            </div>
        </div>
    </div>
    <div class="col-md-3 col-sm-4 col-lg">
        <div class="card h-100">
            <div class="card-body">
                <h3>🦸 Reference</h3>
                How does «…» work?
                <a href="/reference/overview.html" class="stretched-link"></a>
            </div>
        </div>
    </div>
    <div class="col-md-3 col-sm-4 col-lg">
        <div class="card h-100">
            <div class="card-body">
                <h3>👩‍💻 Source code</h3>
                <a class="btn btn-outline-secondary" role="button" href="https://github.com/probabilistic-cell/latenta">latenta</a></a>
                <a class="btn btn-outline-secondary" role="button" href="https://github.com/probabilistic-cell/lacell">lacell</a>
            </div>
        </div>
    </div>
</div>
</div>
<br>

# Overview

Latenta models the cell as a combination of different _cellular processes_ (cell types, cell cycle, zonation, genetic perturbations, time, unexplained variation, nuissance variables, ...). Each of these processes influence _observations_ (transcriptome, unspliced transcriptome, surface proteome, chromatin, phenotype, ...). The processes may also interact with each other, and can thus combinatorially affect observations in various ways. Latenta is modular by design, and this is particularly useful when working with multimodal data, data where cells have multiple co-variates (including nuissance variables), or when trying to find interpretable _mechanistic_ models. In this way, it allows you to answer questions such as:

* How does the remaining variation look like once I model («regress out») my time series, batch effect and cell cycle?
* Does my genetic perturbation affect target gene expression in a linear, sigmoid, exponential or more complex manner?
* How is the cell cycle distributed with zonation?
* Are there genes that are induced after 6 hours commonly in all celltypes?
* Does the surgery «interact» with the cell cycle?
* Do my genetic perturbations interact? Is this in an additive, synergistic, or more complex _epistatic_ manner?
* How many cells are cycling in a spatial spot, and in which phases are these cells?
* Is my genetic perturbation inducing two or three cellular states?

Key concepts are:

- **Modular**: Although each biological dataset and question is unique, different processes come up again and again and can thus be reused across models. Latent contains different standard modules to different cellular processes, such as a clustering, a batch effect, a transcriptome, a linear trajectory, ... and these can be combined in ways to suit a particular dataset.
- **Flexible**: Technology and biology is still evolving, and we cannot predict what kind of models will be necessary to model these new data types. Moreover, most technical parameters, such as a transcriptome's library size, can also have a biological meaning. Latenta provides the flexibility to adapt a model to a particular use case, and to extend 
- **Scalable**: Single-cell omics datasets are relatively large, both in number of cells and in number of features. Moreover, a user will likely require several modelling steps and iterations ideally done in an interactive fashion.
- **Statistical significance**: Despite large cell numbers, the actual number of biological replicates (tissues, patients, ...), is relatively low. Latenta therefore contains tools to assess statistical significance of your findings.
- **Model selection**: We typically don't know the underlying biochemical mechanism that drives certain biological processes. In fact, understanding this mechanism is often one desired outcome of the study. Latenta thus contains methods to do model selection, and to check whether some models are more or equally likely than others.
- **Hidden latent variables**: Interesting biological phenomena, such as cell differentiation or the cell cycle, are often unobserved. Latenta allows you to model these hidden latent variables and relate them to observation.
- **Hypothetical latent variables**: If we want to do an unbiased analysis, or want to model any unexplained variation, we can put these into hypothetical latent variables (such as a clustering or an embedding). Latenta contains tools to model these variables and combine them with hidden latent variables (i.e. "regressing out").
- **Interpretability**: Interpretable models are often preferable, both for the latent space (for example, latent variables that denote the cell cycle) and any functions that connect the latent space to observations (for example, a hill equation). Latenta's models therefore often focus on interpretability, and only use black-box functions such as neural networks when extreme flexibility is required.
- **Prior information**: A lot is already known about certain biological processes that we're studying, and it is desirable to include this knowledge so that we can focus on what is unknown. Latenta therefore uses Bayesian modelling to include prior knowledge wherever available.
- **Hierarchical interactions**: Biological processes are tightly integrated and interconnected. For example, a differentiation process almost always affects the activity of the cell cycle. Latenta can therefore not only connect the latent space to observations, but also create connections between latent variables themselves.
- **Minimal analytical mindset**: Any framework that follows the previous principles will likely require considerable mathematical and statistical know-how to implement.

For example, a cell cycle model can be fitted once and then reused across single-cell and spatial datasets. Or, a latent variable can be fit on one modality (e.g. the transcriptome) and then used to model other modalities. But most importantly, having one unified framework allows the simultaneous modelling different processes, whether you are modelling scalar variables (trajectories, zonation, time), circular variables (cell cycle, circadian clock), discrete variables (clustering, cell type, cell state, batch effect), vector variables (embedding, proportions), nuissance variables (batch, ambient mRNA), different modalities (transcriptome, surface proteome) or different ways of sampling (spatial, nuclear, RNA-FISH, ...).

Some variables may be observed (cell types, transcriptome, ...), while other _latent_ variables may need to be inferred (cell cycle, cell state, ...). Latenta provides tools to model these processes properly, include prior information where needed, do biological and statistical interpretation, and transfer models between datasets.