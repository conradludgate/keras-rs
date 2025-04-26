use ndarray::{Data, Dimension};

use crate::{Arr, ArrViewMut, GraphBuilder, Initialise, Layer, Scalar};

pub mod relu;
pub mod sigmoid;
pub mod softmax;

pub trait Activation {
    type Shape: Dimension + Clone;

    fn apply<F: crate::Scalar>(
        input: Arr<impl Data<Elem = F>, Self::Shape>,
        output: ArrViewMut<F, Self::Shape>,
    );

    fn batch(shape: Self::Shape, batch_size: usize) -> <Self::Shape as Dimension>::Larger;
}

impl<A: Activation> GraphBuilder for A {
    type InputShape = A::Shape;
    type OutputShape = A::Shape;
    type Layer = ActivationLayer<A>;

    fn with_input_shape(self, input_shape: Self::InputShape) -> Self::Layer {
        ActivationLayer { shape: input_shape }
    }
}

pub struct ActivationLayer<A: Activation> {
    shape: A::Shape,
}

impl<A: Activation> Layer for ActivationLayer<A> {
    type InputShape = A::Shape;
    type OutputShape = A::Shape;
    type State<S: ndarray::RawData> = ();

    fn size(&self) -> usize {
        0
    }

    fn output_shape(&self) -> Self::OutputShape {
        self.shape.clone()
    }

    fn batched_output_shape(&self, batch_size: usize) -> <Self::OutputShape as Dimension>::Larger {
        A::batch(self.output_shape(), batch_size)
    }

    fn stack_space(&self, _batch_size: usize) -> usize {
        0
    }

    fn view_state<S: crate::Slice>(&self, _data: S) -> Self::State<S::Repr> {}

    fn apply<F: crate::Scalar>(
        &self,
        _state: Self::State<ndarray::ViewRepr<&F>>,
        input: Arr<impl Data<Elem = F>, Self::InputShape>,
        output: ArrViewMut<F, Self::OutputShape>,
        stack: &mut [F],
    ) {
        debug_assert_eq!(stack.len(), 0);
        A::apply(input, output)
    }
}

impl<F: Scalar, A: Activation> Initialise<F> for ActivationLayer<A> {
    fn init(&self, _rng: &mut impl rand::Rng, _state: Self::State<ndarray::ViewRepr<&mut F>>) {}
}
