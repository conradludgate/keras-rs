#![allow(unused_mut, reason = "rust analyzer is broken")]

use std::ops::Neg;

use model::Model;
use named::Named;
use ndarray::{
    ArrayBase, Data, Dimension, IntoDimension, LinalgScalar, RawData, ScalarOperand, ViewRepr,
};
use rand::{thread_rng, Rng};
use rand_distr::num_traits::{Float, FromPrimitive};

pub mod activation;
// pub mod attention;
pub mod cost;
pub mod dense;
pub mod embedding;
pub mod linear;
pub mod model;
pub mod named;
pub mod network;
pub mod optimise;

/// Type representing a simple ArrayBase, but dimension larger to account for batches
pub type Arr<S, D> = ndarray::ArrayBase<S, <D as Dimension>::Larger>;
pub type MutRepr<'f, F> = ViewRepr<&'f mut F>;
pub type OwnedArr<F, D> = Arr<ndarray::OwnedRepr<F>, D>;
pub type ArrView<'a, F, D> = Arr<ViewRepr<&'a F>, D>;
pub type ArrViewMut<'a, F, D> = Arr<MutRepr<'a, F>, D>;

type LayerTrainState<'a, F, L> = <L as TrainableLayer>::TrainState<ViewRepr<&'a F>>;

/// An abstract representation of a Computation Graph.
pub trait GraphBuilder: Sized {
    type InputShape: Dimension;
    type OutputShape: Dimension;

    /// The state that this builder produces
    type Layer: Layer<InputShape = Self::InputShape, OutputShape = Self::OutputShape>;

    fn with_input_shape(self, input_shape: Self::InputShape) -> Self::Layer;

    fn into_model<F: Scalar>(
        self,
        input_shape: impl IntoDimension<Dim = Self::InputShape>,
    ) -> Model<F, Self>
    where
        Self::Layer: Initialise<F>,
    {
        let layer = self.with_input_shape(input_shape.into_dimension());
        let len = layer.size();

        let mut data = vec![F::zero(); len];

        layer.init(&mut thread_rng(), layer.view_state(&mut data[..]));

        Model { layer, data }
    }

    fn named<S: ToString>(self, name: S) -> Named<Self, S> {
        Named { inner: self, name }
    }
}

pub trait Scalar:
    LinalgScalar + ScalarOperand + Float + FromPrimitive + Neg<Output = Self> + std::fmt::Debug
{
}
impl<S> Scalar for S where
    S: LinalgScalar + ScalarOperand + Float + FromPrimitive + Neg<Output = S> + std::fmt::Debug
{
}

pub trait Slice: Sized {
    type Repr: RawData;
    fn split(self, at: usize) -> (Self, Self);
    fn into_array<D: IntoDimension>(self, shape: D) -> ArrayBase<Self::Repr, D::Dim>;
}

impl<'a, T> Slice for &'a mut [T] {
    type Repr = ViewRepr<&'a mut T>;
    fn split(self, at: usize) -> (Self, Self) {
        self.split_at_mut(at)
    }

    fn into_array<D: IntoDimension>(self, shape: D) -> ArrayBase<Self::Repr, D::Dim> {
        ndarray::ArrayViewMut::from_shape(shape, self).unwrap()
    }
}

impl<'a, T> Slice for &'a [T] {
    type Repr = ViewRepr<&'a T>;
    fn split(self, at: usize) -> (Self, Self) {
        self.split_at(at)
    }

    fn into_array<D: IntoDimension>(self, shape: D) -> ArrayBase<Self::Repr, D::Dim> {
        ndarray::ArrayView::from_shape(shape, self).unwrap()
    }
}

pub trait Layer {
    type InputShape: Dimension;
    type OutputShape: Dimension;
    type State<S: RawData>;

    fn size(&self) -> usize;
    fn output_shape(&self) -> Self::OutputShape;
    fn batched_output_shape(&self, batch_size: usize) -> <Self::OutputShape as Dimension>::Larger;
    fn stack_space(&self, batch_size: usize) -> usize;

    fn view_state<S: Slice>(&self, data: S) -> Self::State<S::Repr>;

    /// Apply the layer to the input and get the output
    ///
    /// `state` stores any parameters used in the forward pass. See [`view_state`].
    /// `input` is the input to the layer
    /// `stack` is where the output is written to. The length is specified by [`stack_space`].
    /// If `stack` length is larger than needed for the output, any spare data can be written there and will be discarded later
    fn apply<F: Scalar>(
        &self,
        state: Self::State<ViewRepr<&F>>,
        input: Arr<impl Data<Elem = F>, Self::InputShape>,
        output: ArrViewMut<F, Self::OutputShape>,
        stack: &mut [F],
    );
}

pub trait Initialise<F: Scalar>: Layer {
    fn init(&self, rng: &mut impl Rng, state: Self::State<MutRepr<F>>);
}

pub trait TrainableLayer: Layer {
    type TrainState<S: RawData>;
    fn train_state_size(&self, batch_size: usize) -> usize;
    fn view_train_state<S: Slice>(&self, batch_size: usize, data: S) -> Self::TrainState<S::Repr>;
    fn train_stack_space(&self, batch_size: usize) -> usize;

    fn forward<F: Scalar>(
        &self,
        state: Self::State<ViewRepr<&F>>,
        input: Arr<impl Data<Elem = F>, Self::InputShape>,
        output: ArrViewMut<F, Self::OutputShape>,
        stack: &mut [F],
        train_state: &mut Self::TrainState<MutRepr<F>>,
    );

    fn backward<F: Scalar>(
        &self,
        state: Self::State<ViewRepr<&F>>,
        d_state: Self::State<MutRepr<F>>,
        train_state: Self::TrainState<ViewRepr<&F>>,
        d_output: Arr<impl Data<Elem = F>, Self::OutputShape>,
        d_input: ArrViewMut<F, Self::InputShape>,
        stack: &mut [F],
    );
}
