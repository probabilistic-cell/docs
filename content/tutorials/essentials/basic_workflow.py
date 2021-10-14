import laflow as laf
import latenta as la
import lacell as lac


class LinearModel(laf.Flow):
    # sets the default name
    default_name = "linear"

    # lets the flow know that it can contain a link to another flow called dataset
    dataset = laf.FlowObj()

    # lets the flow know that it can expect a model_initial object, that is not persistent
    model_initial = laf.LatentaObj(persistent=False)

    # creates a particular step, that expects adata as input, and that outputs model initial
    # note the way we access the adata from the dataset object
    @laf.Step(laf.Inputs(adata=dataset.adata), laf.Outputs(model_initial))
    def create(self, output, adata):
        # define the model as before
        overexpression = la.Fixed(
            adata.obs["log_overexpression"], label="overexpression"
        )
        transcriptome = lac.transcriptome.TranscriptomeObservation.from_adata(adata)
        foldchange = transcriptome.find("foldchange")

        foldchange.overexpression = la.links.scalar.Linear(
            overexpression, a=True, definition=foldchange.value_definition
        )

        # apart from each input, the step function will also receive an output object
        # you should put any expected outputs in this object either by assigning it
        # or by using the update function
        return output.update(model_initial=transcriptome)

    # the (final) model is persistent by default
    model = laf.LatentaObj()

    # another step, that receives the model_initial as input, and outputs the model
    @laf.Step(laf.Inputs(model_initial), laf.Outputs(model))
    def infer(self, output, model_initial):
        # infer the model as before
        # we first clone the model_initial so that we do not overwrite it
        model = model_initial.clone()
        with model.switch(la.config.device):
            inference = la.infer.svi.SVI(
                model, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=5e-2)
            )
            trainer = la.infer.trainer.Trainer(inference)
            trace = trainer.train(10000)

        return output.update(model=model)

    # define our three posteriors, each one with a "db" argument
    # this will make sure that any model objects are not stored in the posterior,
    # but are rather extracted each time from the model object as defined earlier
    transcriptome_observed = laf.LatentaObj(db={model})
    overexpression_observed = laf.LatentaObj(db={model})
    overexpression_causal = laf.LatentaObj(db={model})

    @laf.Step(
        laf.Inputs(model),
        laf.Outputs(
            transcriptome_observed, overexpression_observed, overexpression_causal
        ),
    )
    def interpret(self, output, model):
        transcriptome_observed = la.posterior.vector.VectorObserved(model)
        transcriptome_observed.sample(5)
        output.transcriptome_observed = transcriptome_observed

        overexpression = model.find("overexpression")

        overexpression_observed = la.posterior.scalar.ScalarObserved(overexpression)
        overexpression_observed.sample(5)
        output.overexpression_observed = overexpression_observed

        overexpression_causal = la.posterior.scalar.ScalarVectorCausal(
            overexpression,
            model,
            interpretable=model.p.mu.expression,
            observed=overexpression_observed,
        )
        overexpression_causal.sample(10)
        overexpression_causal.sample_random(10)
        overexpression_causal.observed
        overexpression_causal.sample_empirical()
        output.overexpression_causal = overexpression_causal

        return output


class ConstantModel(LinearModel):
    # we change the default name, as to make sure this model is put in a different folder
    default_name = "constant"

    def create(self, output, adata):
        # we can access the parent function by adding a "_" at the end
        # without this, we would call the actual step itself, and not the user-defined function
        output = super().create_(output=output, adata=adata)

        # extract the model_initial from the output
        model_initial = output.model_initial

        # now we can further adapt the model to our wishes
        foldchange = model_initial.find("foldchange")
        overexpression = model_initial.find("overexpression")

        foldchange.overexpression = la.links.scalar.Constant(
            overexpression, definition=foldchange.value_definition
        )

        # again return the output
        # because we only adapted the model inplace, we do not need to update the output
        return output


class SplineModel(LinearModel):
    default_name = "spline"

    def create(self, output, adata):
        output = super().create_(output=output, adata=adata)

        model_initial = output.model_initial

        foldchange = model_initial.find("foldchange")
        overexpression = model_initial.find("overexpression")

        foldchange.overexpression = la.links.scalar.Spline(
            overexpression, definition=foldchange.value_definition
        )

        return output
