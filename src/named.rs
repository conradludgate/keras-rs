use std::mem::MaybeUninit;

use ndarray::Dimension;

use crate::{Arr, GraphBuilder, Layer, Scalar, Slice, UninitArr};

pub struct Named<I, S: ToString> {
    pub(crate) inner: I,
    pub(crate) name: S,
}

impl<I, S: ToString> Named<I, S> {
    pub fn into_inner(self) -> I {
        self.inner
    }
    pub fn name(&self) -> &S {
        &self.name
    }
}

impl<G: GraphBuilder, S: ToString> GraphBuilder for Named<G, S> {
    type InputShape = G::InputShape;
    type OutputShape = G::OutputShape;
    type Layer = Named<G::Layer, S>;

    fn with_input_shape(self, input_shape: Self::InputShape) -> Self::Layer {
        let Self { inner, name } = self;
        let inner = inner.with_input_shape(input_shape);
        Named { inner, name }
    }
}

impl<L: Layer, S: ToString> Layer for Named<L, S> {
    type InputShape = L::InputShape;
    type OutputShape = L::OutputShape;
    type State<St: ndarray::RawData> = L::State<St>;

    fn size(&self) -> usize {
        self.inner.size()
    }

    fn output_shape(&self) -> Self::OutputShape {
        self.inner.output_shape()
    }

    fn batched_output_shape(&self, batch_size: usize) -> <Self::OutputShape as Dimension>::Larger {
        self.inner.batched_output_shape(batch_size)
    }

    fn stack_space(&self, batch_size: usize) -> usize {
        self.inner.stack_space(batch_size)
    }

    fn view_state<Sl: Slice>(&self, data: Sl) -> Self::State<Sl::Repr> {
        self.inner.view_state(data)
    }

    fn apply<F: Scalar>(
        &self,
        state: Self::State<ndarray::ViewRepr<&F>>,
        input: Arr<impl ndarray::Data<Elem = F>, Self::InputShape>,
        output: UninitArr<F, Self::OutputShape>,
        stack: &mut [MaybeUninit<F>],
    ) {
        self.inner.apply(state, input, output, stack)
    }
}
