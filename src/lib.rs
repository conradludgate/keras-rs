#![feature(generic_associated_types)]
#![feature(vec_spare_capacity)]

use std::mem::MaybeUninit;

use model::Model;
use ndarray::{
    ArrayBase, Data, Dimension, IntoDimension, LinalgScalar, OwnedRepr, RawData, ScalarOperand,
    ShapeError, ViewRepr,
};
use rand::{thread_rng, Rng};
use rand_distr::num_traits::{Float, FromPrimitive};

pub mod activation;
pub(crate) mod array;
pub mod cost;
pub mod linear;
pub mod model;
pub mod network;
pub mod optimise;

pub type Arr<S, D> = ArrayBase<S, <D as Dimension>::Larger>;
pub type UninitRepr<'f, F> = ViewRepr<&'f mut MaybeUninit<F>>;
pub type UninitArr<'f, F, D> = Arr<UninitRepr<'f, F>, D>;
pub type OwnedArr<F, D> = Arr<OwnedRepr<F>, D>;

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

        let mut data = Vec::with_capacity(len);
        let mut uninit = layer
            .view_mut(&mut data.spare_capacity_mut()[..len])
            .unwrap();
        layer.init(&mut thread_rng(), &mut uninit);
        drop(uninit);
        unsafe {
            data.set_len(len);
        }

        Model { layer, data }
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
        input: Arr<impl Data<Elem = F>, Self::InputShape>,
        output: UninitArr<F, Self::OutputShape>
    );
}

/// # Safety
/// init should initialise all values in the state
pub unsafe trait Initialise<F: Scalar>: Layer {
    fn init(&self, rng: &mut impl Rng, state: &mut Self::State<ViewRepr<&mut MaybeUninit<F>>>);
}

pub trait TrainableLayer: Layer {
    fn train_state_size(&self) -> usize;

    fn forward<F: Scalar>(
        &self,
        state: Self::State<ViewRepr<&F>>,
        input: Arr<impl Data<Elem = F>, Self::InputShape>,
        output: UninitArr<F, Self::OutputShape>,
        train_state: &mut [MaybeUninit<F>],
    );

    fn backward<F: Scalar>(
        &self,
        state: Self::State<ViewRepr<&F>>,
        d_state: Self::State<UninitRepr<F>>,
        d_input: UninitArr<F, Self::OutputShape>,
        train_state: &[F],
        d_output: Arr<impl Data<Elem = F>, Self::OutputShape>,
    );
}
