use ndarray::{ArrayBase, Data, Dimension, OwnedRepr, IntoDimension};

use crate::{cost::Cost, optimise::Optimiser, GraphBuilder, Layer, Scalar, TrainableLayer};

pub struct Model<F: Scalar, G: GraphBuilder> {
    pub(crate) layer: G::Layer,
    pub(crate) data: Vec<F>,
}

impl<F: Scalar, G: GraphBuilder> Model<F, G> {
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

pub struct ModelTrainer<F: Scalar, G: GraphBuilder, O: Optimiser<F>, C: Cost<G::OutputShape>>
where
    G::Layer: TrainableLayer,
{
    pub model: Model<F, G>,
    pub optimiser: O,
    pub cost: C,
}

impl<F: Scalar, G: GraphBuilder, O: Optimiser<F>, C: Cost<G::OutputShape>> ModelTrainer<F, G, O, C>
where
    G::Layer: TrainableLayer,
{
    pub fn train_batch(
        &mut self,
        input: ArrayBase<impl Data<Elem = F>, <G::InputShape as Dimension>::Larger>,
        expected: ArrayBase<impl Data<Elem = F>, <G::OutputShape as Dimension>::Larger>,
    ) {
        debug_assert_eq!(input.shape()[0], expected.shape()[0], "input and expected batch sizes should be equal")


    }
}
