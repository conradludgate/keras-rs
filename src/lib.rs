use std::ops::Neg;

use model::Model;
use ndarray::{
    ArrayBase, Axis, Data, DataMut, Dimension, IntoDimension, Ix1, Ix2, NdFloat, RawData, ViewRepr,
};
use rand::{thread_rng, Rng};
use rand_distr::num_traits::FromPrimitive;

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

pub trait Scalar: NdFloat + FromPrimitive + Neg<Output = Self> {}
impl<S> Scalar for S where S: NdFloat + FromPrimitive + Neg<Output = S> {}

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

pub trait Initialise<Input: BatchShape, F: Scalar>: ModelShape<Input> {
    fn init(&self, rng: &mut impl Rng, state: Params<Input, Self, &mut F>);

    fn into_model(self, input_shape: Input) -> Model<F, Input, Self> {
        let (shape, _) = self.shape(input_shape);
        let len = shape.size();
        let mut data = vec![F::zero(); len];

        self.init(&mut thread_rng(), shape.from_slice(&mut data[..]));

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

pub type Input<I, R> = Batch<I, R>;
pub type Output<I, S, R> = Batch<<S as ModelShape<I>>::Output, R>;
pub type TrainingCache<I, S, R> = View<<S as BackpropShape<I>>::TrainingCache, R>;
pub type Params<I, S, R> = View<<S as ModelShape<I>>::Params, R>;

pub trait Shape: Copy {
    type Base<S: RawData>: Array<S, Shape = Self>;

    fn size(self) -> usize;
    fn from_slice<S: Slice>(self, slice: S) -> Self::Base<S::Repr>;
}

pub trait Array<R: RawData> {
    type Shape: Shape;

    fn as_ref<'a>(&'a self) -> View<Self::Shape, &'a R::Elem>
    where
        R: Data;

    fn as_mut<'a>(&'a mut self) -> View<Self::Shape, &'a mut R::Elem>
    where
        R: DataMut;

    fn assign_to(&self, dst: View<Self::Shape, &mut R::Elem>)
    where
        R::Elem: Copy,
        R: Data;
}

pub trait BatchShape: Shape {
    type Batched: Shape;

    fn batched(self, batch: usize) -> Self::Batched;

    #[inline]
    fn batched_size(self, batch: usize) -> usize {
        self.batched(batch).size()
    }

    #[inline]
    fn from_batched_slice<S: Slice>(
        self,
        batch: usize,
        slice: S,
    ) -> <Batched<Self> as Shape>::Base<S::Repr> {
        self.batched(batch).from_slice(slice)
    }

    fn from_single<R: RawData>(self, base: Self::Base<R>) -> <Batched<Self> as Shape>::Base<R>;

    fn get_single<R: RawData>(
        self,
        base: <Batched<Self> as Shape>::Base<R>,
        index: usize,
    ) -> Self::Base<R>;

    fn batches<R: RawData>(self, base: &<Batched<Self> as Shape>::Base<R>) -> usize;
}

impl<D: Dimension + Copy> Shape for D {
    type Base<S: RawData> = ArrayBase<S, D>;

    #[inline]
    fn size(self) -> usize {
        Dimension::size(&self)
    }

    #[inline]
    fn from_slice<S: Slice>(self, slice: S) -> Self::Base<S::Repr> {
        slice.into_array(self)
    }
}

impl<R: RawData, D: Dimension + Copy> Array<R> for ArrayBase<R, D> {
    type Shape = D;

    fn as_ref<'a>(&'a self) -> View<Self::Shape, &'a <R as RawData>::Elem>
    where
        R: Data,
    {
        self.view()
    }

    fn as_mut<'a>(&'a mut self) -> View<Self::Shape, &'a mut <R as RawData>::Elem>
    where
        R: DataMut,
    {
        self.view_mut()
    }

    fn assign_to(&self, mut dst: View<Self::Shape, &mut <R as RawData>::Elem>)
    where
        R: Data,
        R::Elem: Copy,
    {
        dst.assign(self);
    }
}

impl BatchShape for Ix1 {
    type Batched = Ix2;

    #[inline]
    fn batched(self, batch: usize) -> Self::Batched {
        let x0 = self.into_pattern();
        Ix2(batch, x0)
    }

    #[inline]
    fn from_single<R: RawData>(self, base: ArrayBase<R, Ix1>) -> <Batched<Self> as Shape>::Base<R> {
        base.insert_axis(Axis(0))
    }

    #[inline]
    fn get_single<R: RawData>(
        self,
        base: <Batched<Self> as Shape>::Base<R>,
        index: usize,
    ) -> ArrayBase<R, Ix1> {
        base.index_axis_move(Axis(0), index)
    }

    #[inline]
    fn batches<R: RawData>(self, base: &<Batched<Self> as Shape>::Base<R>) -> usize {
        base.dim().0
    }
}

pub struct Stack<'a, F>(&'a mut [F]);

impl<'a, F: Scalar> Stack<'a, F> {
    pub fn new(v: &'a mut Vec<F>, size: usize) -> Self {
        v.resize(size, F::zero());
        Self(&mut *v)
    }
}
impl<'b, F> Stack<'b, F> {
    pub fn take<S: Shape>(self, shape: S) -> (View<S, &'b mut F>, Stack<'b, F>) {
        let size = shape.size();
        let (left, right) = self.0.split_at_mut(size);
        let view = shape.from_slice(left);
        (view, Stack(right))
    }

    pub fn as_mut<'a>(&'a mut self) -> Stack<'a, F> {
        Stack(&mut self.0)
    }
}

pub struct ModelShapes<Input, B: ModelShape<Input>> {
    pub params: B::Params,
    pub output: B::Output,
    pub stack_size: usize,
}

pub trait ModelShape<Input>: Copy {
    type Params: Shape;
    type Output: BatchShape;

    fn shape(self, input: Input) -> (Self::Params, Self::Output);
    fn stack(self, batch_size: usize, input: Input) -> usize;
}

pub trait BackpropShape<Input>: ModelShape<Input> {
    type TrainingCache: Shape;

    fn backprop_shape(self, batch_size: usize, input: Input) -> (Self::TrainingCache, usize);
}

pub trait Inference<Input: BatchShape, F>: ModelShape<Input> {
    /// # Inputs
    /// self: fn f(params, input) -> output
    /// params: the parameters
    /// input: the input
    ///
    /// # Outputs
    /// output = f(params, input)
    fn infer(
        self,
        input_shape: Input,
        batch_size: usize,
        params: Params<Input, Self, &F>,
        input: Batch<Input, &F>,
        output: Output<Input, Self, &mut F>,
        stack: Stack<F>,
    );
}

pub trait Backprop<Input: BatchShape, F>: Inference<Input, F> + BackpropShape<Input> {
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
        input_shape: Input,
        batch_size: usize,
        params: Params<Input, Self, &F>,
        input: Batch<Input, &F>,
        output: Output<Input, Self, &mut F>,
        cache: TrainingCache<Input, Self, &mut F>,
        stack: Stack<F>,
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
        input_shape: Input,
        batch_size: usize,
        params: Params<Input, Self, &F>,
        de_doutput: Output<Input, Self, &F>,
        de_dinput: Batch<Input, &mut F>,
        de_dparams: Params<Input, Self, &mut F>,
        cache: TrainingCache<Input, Self, &F>,
        stack: Stack<F>,
    );
}
