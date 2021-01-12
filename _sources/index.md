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
                <a class="btn btn-outline-secondary" role="button" href="https://github.com/probabilistic-cell/latentacell">latentacell</a>
            </div>
        </div>
    </div>
</div>
<br>

# Overview

Latenta models the cell as a combination of different _cellular processes_ (cell types, cell cycle, zonation, genetic perturbations, time, unexplained variation, ...). Each of these processes influence _observations_ (transcriptome, surface proteome, phenotype, ...). The processes may also interact with each other, and can thus combinatorially affect observations in various ways. Through a Bayesian inference framework, latenta makes it possible to determine which of these processes interact, and select interpretable, mechanistic and statistically significant models of how these interactions affect the cellular state. In this way, it allows you to answer questions such as:

* Does my genetic perturbation affect target gene expression in a linear, sigmoid, exponential or more complex manner?
* Is my genetic perturbation inducing two or three cellular states?
* How is the cell cycle distributed with zonation?
* Are there genes that are induced after 6 hours in all celltypes?
* Does the surgery «interact» with the cell cycle?
* Do my genetic perturbations interact? Is this in an additive, synergistic, or more complex _epistatic_ manner?
* How many cells are cycling in a spatial spot, and in which phases are these cells?
* How does the remaining variation look like once I model («regress out») my time series, batch effect and cell cycle?

Latenta is modular by design, and this is particularly useful when working on larger projects or multimodal datasets. For example, a cell cycle model can be fitted once and then reused across single-cell and spatial datasets. Or, a latent variable can be fit on one modality (e.g. the transcriptome) and then used to model other modalities. But most importantly, having one unified framework allows the simultaneous modelling different processes, whether you are modelling scalar variables (trajectories, zonation, time), circular variables (cell cycle, circadian clock), discrete variables (clustering, cell type, cell state, batch effect), vector variables (embedding, proportions), nuissance variables (batch, ambient mRNA), different modalities (transcriptome, surface proteome) or different ways of sampling (spatial, nuclear, RNA-FISH, ...).

Some variables may be observed (cell types, transcriptome, ...), while other _latent_ variables may need to be inferred (cell cycle, cell state, ...). Latenta provides tools to model these processes properly, include prior information where needed, do biological and statistical interpretation, and transfer models between datasets.