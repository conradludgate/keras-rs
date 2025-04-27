#![allow(unused_mut, reason = "rust analyzer is broken")]

use std::ops::Neg;

use model::Model;
use ndarray::{
    ArrayBase, Dimension, IntoDimension, Ix1, Ix2, LinalgScalar, RawData, ScalarOperand, ViewRepr,
};
use rand::{thread_rng, Rng};
use rand_distr::num_traits::{Float, FromPrimitive};

pub mod activation;
// pub mod attention;
pub mod cost;
// pub mod dense;
// pub mod embedding;
pub mod linear;
pub mod model;
// pub mod named;
pub mod network;
pub mod optimise;

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

pub trait Initialise<F: Scalar>: BackpropShape {
    fn init(&self, rng: &mut impl Rng, state: Params<Self, &mut F>);

    fn into_model(self, input_shape: Self::Input) -> Model<F, Self> {
        let shape = self.shape(input_shape);
        let len = shape.params.size();
        let mut data = vec![F::zero(); len];

        self.init(&mut thread_rng(), shape.params.from_slice(&mut data[..]));

        Model {
            input: input_shape,
            layer: self,
            params: data,
            stack: Vec::new(),
        }
    }
}

pub type View<S, R> = <S as Shape>::Base<ViewRepr<R>>;
pub type Batched<S> = <S as BatchShape>::Batched;
pub type Batch<S, R> = View<Batched<S>, R>;

pub type Input<S, R> = Batch<<S as BackpropShape>::Input, R>;
pub type Output<S, R> = Batch<<S as BackpropShape>::Output, R>;
pub type Cache<S, R> = Batch<<S as BackpropShape>::Cache, R>;
pub type Params<S, R> = View<<S as BackpropShape>::Params, R>;

pub trait Shape: Copy {
    type Base<S: RawData>;

    fn size(self) -> usize;
    fn from_slice<S: Slice>(self, slice: S) -> Self::Base<S::Repr>;
}

pub trait BatchShape: Copy {
    type Batched: Shape;

    fn batched(self, batch: usize) -> Self::Batched;

    #[inline]
    fn size(self, batch: usize) -> usize {
        self.batched(batch).size()
    }

    #[inline]
    fn from_slice<S: Slice>(
        self,
        batch: usize,
        slice: S,
    ) -> <Batched<Self> as Shape>::Base<S::Repr> {
        self.batched(batch).from_slice(slice)
    }
}

impl<D: Dimension + Copy> Shape for D {
    type Base<S: RawData> = ArrayBase<S, D>;

    #[inline]
    fn size(self) -> usize {
        Dimension::size(&self)
    }

    fn from_slice<S: Slice>(self, slice: S) -> Self::Base<S::Repr> {
        slice.into_array(self)
    }
}

impl BatchShape for Ix1 {
    type Batched = Ix2;

    #[inline]
    fn batched(self, batch: usize) -> Self::Batched {
        let x0 = self.into_pattern();
        Ix2(batch, x0)
    }
}

pub struct BackpropShapes<B: BackpropShape> {
    pub params: B::Params,
    pub output: B::Output,
    pub cache: B::Cache,
    pub stack_size: usize,
}

pub trait BackpropShape: Copy {
    type Params: Shape;
    type Input: BatchShape;
    type Output: BatchShape;
    type Cache: BatchShape;

    fn shape(self, input: Self::Input) -> BackpropShapes<Self>;
}

pub trait Backprop<F>: BackpropShape {
    /// # Inputs
    /// self: fn f(params, input) -> output
    /// params: the parameters
    /// input: the input
    ///
    /// # Outputs
    /// output = f(params, input)
    /// cache = encode(input)
    fn forward(
        self,
        input_shape: Self::Input,
        batch_size: usize,
        params: Params<Self, &F>,
        input: Input<Self, &F>,
        output: Output<Self, &mut F>,
        cache: Cache<Self, &mut F>,
        stack: &mut [F],
    );

    /// # Inputs
    /// self:
    ///   fn df_dparams(params, input) = derivative of f with respect to params
    ///   fn df_dinput(params, input) = derivative of f with respect to input
    ///
    /// params: the parameters
    /// de_doutput: derivative of the error with respect to the output
    /// input = decode(cache)
    ///
    /// # Outputs
    /// de_dinput = de_doutput * df_dinput(params, input)
    /// de_dparams = de_doutput * df_dparams(params, input)
    fn backward(
        self,
        input_shape: Self::Input,
        batch_size: usize,
        params: Params<Self, &F>,
        de_doutput: Output<Self, &F>,
        de_dinput: Input<Self, &mut F>,
        de_dparams: Params<Self, &mut F>,
        cache: Cache<Self, &F>,
        stack: &mut [F],
    );
}
