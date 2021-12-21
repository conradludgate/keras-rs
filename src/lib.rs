#![feature(generic_associated_types)]

use model::ModelBuilder;
use ndarray::{
    ArrayBase, Data, Dimension, LinalgScalar, OwnedRepr, RawData, ScalarOperand, ShapeError,
    ViewRepr,
};
use rand_distr::num_traits::{Float, FromPrimitive};

pub mod activation;
pub(crate) mod array;
pub mod linear;
pub mod network;
pub mod model;
pub mod optimise;
pub mod cost;

/// An abstract representation of a Computation Graph.
pub trait GraphBuilder: Sized {
    type InputShape: Dimension;
    type OutputShape: Dimension;

    /// The state that this builder produces
    type Layer: Layer<InputShape = Self::InputShape, OutputShape = Self::OutputShape>;

    fn with_input_shape(self, input_shape: Self::InputShape) -> Self::Layer;

    fn build_model(self) -> ModelBuilder<Self> {
        model::builder(self)
    }
}

pub trait Scalar: LinalgScalar + ScalarOperand + Float + FromPrimitive {}
impl<S> Scalar for S where S: LinalgScalar + ScalarOperand + Float + FromPrimitive {}

pub trait Layer {
    type InputShape: Dimension;
    type OutputShape: Dimension;
    type State<S: RawData>;

    fn size(&self) -> usize;
    fn output_shape(&self) -> Self::OutputShape;

    fn view<'a, F>(&self, data: &'a [F]) -> Result<Self::State<ViewRepr<&'a F>>, ShapeError>;
    fn view_mut<'a, F>(
        &self,
        data: &'a mut [F],
    ) -> Result<Self::State<ViewRepr<&'a mut F>>, ShapeError>;

    fn apply<F: Scalar>(
        &self,
        state: Self::State<ViewRepr<&F>>,
        input: ArrayBase<impl Data<Elem = F>, <<Self as Layer>::InputShape as Dimension>::Larger>,
    ) -> ArrayBase<OwnedRepr<F>, <<Self as Layer>::OutputShape as Dimension>::Larger>;
}

pub trait TrainableLayer: Layer {
    fn backward<F: Scalar>(
        &self,
        state: Self::State<ViewRepr<&F>>,
        d_state: Self::State<ViewRepr<&mut F>>,
        input: ArrayBase<impl Data<Elem = F>, <<Self as Layer>::InputShape as Dimension>::Larger>,
        output: ArrayBase<impl Data<Elem = F>, <<Self as Layer>::OutputShape as Dimension>::Larger>,
        d_output: ArrayBase<
            impl Data<Elem = F>,
            <<Self as Layer>::OutputShape as Dimension>::Larger,
        >,
    ) -> ArrayBase<OwnedRepr<F>, <<Self as Layer>::InputShape as Dimension>::Larger>;
}
