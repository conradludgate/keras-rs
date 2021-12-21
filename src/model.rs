use std::marker::PhantomData;

use ndarray::{ArrayBase, Data, Dimension, IntoDimension, OwnedRepr};

use crate::{cost::Cost, optimise::Optimiser, GraphBuilder, Layer, Scalar};

pub fn builder<G: GraphBuilder>(graph: G) -> ModelBuilder<G> {
    ModelBuilder { graph }
}

pub struct ModelBuilder<G: GraphBuilder> {
    graph: G,
}

impl<G: GraphBuilder> ModelBuilder<G> {
    pub fn input_shape(
        self,
        input_shape: impl IntoDimension<Dim = G::InputShape>,
    ) -> ModelBuilderShape<G> {
        let layer = self.graph.with_input_shape(input_shape.into_dimension());
        ModelBuilderShape { layer }
    }
}

pub struct ModelBuilderShape<G: GraphBuilder> {
    layer: G::Layer,
}

impl<G: GraphBuilder> ModelBuilderShape<G> {
    pub fn with_cost<C: Cost<G::OutputShape>>(self, cost: C) -> ModelBuilderCost<G, C> {
        let Self { layer } = self;
        ModelBuilderCost { layer, cost }
    }

    // pub fn initialise<F: Scalar>(self, initialiser: ()) -> ModelBuilderCost<F, G> {
    //     let Self { layer } = self;
    //     let size = layer.size();
    //     let zero = F::zero();
    //     let data = vec![zero; size];
    //     ModelBuilderCost {
    //         layer, data
    //     }
    // }
}

pub struct ModelBuilderCost<G: GraphBuilder, C: Cost<G::OutputShape>> {
    layer: G::Layer,
    cost: C,
}

impl<G: GraphBuilder, C: Cost<G::OutputShape>> ModelBuilderCost<G, C> {
    pub fn with_optimiser<F: Scalar, O: Optimiser<F>>(
        self,
        optimiser: O,
    ) -> ModelBuilderOptimiser<F, G, C, O> {
        let Self { layer, cost } = self;
        ModelBuilderOptimiser {
            layer,
            cost,
            optimiser,
            data: PhantomData,
        }
    }
}

pub struct ModelBuilderOptimiser<
    F: Scalar,
    G: GraphBuilder,
    C: Cost<G::OutputShape>,
    O: Optimiser<F>,
> {
    layer: G::Layer,
    cost: C,
    optimiser: O,
    data: PhantomData<Vec<F>>,
}

impl<F: Scalar, G: GraphBuilder, C: Cost<G::OutputShape>, O: Optimiser<F>>
    ModelBuilderOptimiser<F, G, C, O>
{
    pub fn initialise(self) -> Model<F, G, C, O> {
        let Self {
            layer,
            cost,
            optimiser,
            data: _,
        } = self;
        let size = layer.size();
        let zero = F::zero();
        let data = vec![zero; size];
        Model {
            layer,
            cost,
            optimiser,
            data,
        }
    }
}

pub struct Model<F: Scalar, G: GraphBuilder, C: Cost<G::OutputShape>, O: Optimiser<F>> {
    layer: G::Layer,
    cost: C,
    optimiser: O,
    data: Vec<F>,
}

impl<F: Scalar, G: GraphBuilder, C: Cost<G::OutputShape>, O: Optimiser<F>> Model<F, G, C, O> {
    pub fn apply(
        &self,
        input: ArrayBase<impl Data<Elem = F>, <G::InputShape as Dimension>::Larger>,
    ) -> ArrayBase<OwnedRepr<F>, <G::OutputShape as Dimension>::Larger> {
        let state = self
            .layer
            .view(&self.data)
            .expect("size of data should always be a valid view into a layer");
        self.layer.apply(state, input)
    }
}
