use ndarray::{ArrayBase, Data, Dimension, OwnedRepr};

use crate::{GraphBuilder, Layer, TrainableLayer};

pub mod relu;

pub trait Activation {
    type Shape: Dimension + Clone;

    fn apply<F: crate::Scalar>(
        input: ArrayBase<impl Data<Elem = F>, <Self::Shape as Dimension>::Larger>,
    ) -> ArrayBase<OwnedRepr<F>, <Self::Shape as Dimension>::Larger>;
}

pub trait ActivationTrain: Activation {
    fn backward<F: crate::Scalar>(
        input: ArrayBase<impl Data<Elem = F>, <<Self as Activation>::Shape as Dimension>::Larger>,
        output: ArrayBase<impl Data<Elem = F>, <<Self as Activation>::Shape as Dimension>::Larger>,
        d_output: ArrayBase<
            impl Data<Elem = F>,
            <<Self as Activation>::Shape as Dimension>::Larger,
        >,
    ) -> ArrayBase<OwnedRepr<F>, <<Self as Activation>::Shape as Dimension>::Larger>;
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

    fn view<'a, F>(
        &self,
        _data: &'a [F],
    ) -> Result<Self::State<ndarray::ViewRepr<&'a F>>, ndarray::ShapeError> {
        Ok(())
    }

    fn view_mut<'a, F>(
        &self,
        _data: &'a mut [F],
    ) -> Result<Self::State<ndarray::ViewRepr<&'a mut F>>, ndarray::ShapeError> {
        Ok(())
    }

    fn apply<F: crate::Scalar>(
        &self,
        _state: Self::State<ndarray::ViewRepr<&F>>,
        input: ArrayBase<impl Data<Elem = F>, <<Self as Layer>::InputShape as Dimension>::Larger>,
    ) -> ArrayBase<OwnedRepr<F>, <<Self as Layer>::OutputShape as Dimension>::Larger> {
        A::apply(input)
    }
}

impl<A: ActivationTrain> TrainableLayer for ActivationLayer<A> {
    fn backward<F: crate::Scalar>(
        &self,
        _state: Self::State<ndarray::ViewRepr<&F>>,
        _d_state: Self::State<ndarray::ViewRepr<&mut F>>,
        input: ArrayBase<impl Data<Elem = F>, <<Self as Layer>::InputShape as Dimension>::Larger>,
        output: ArrayBase<impl Data<Elem = F>, <<Self as Layer>::OutputShape as Dimension>::Larger>,
        d_output: ArrayBase<
            impl Data<Elem = F>,
            <<Self as Layer>::OutputShape as Dimension>::Larger,
        >,
    ) -> ArrayBase<OwnedRepr<F>, <<Self as Layer>::InputShape as Dimension>::Larger> {
        A::backward(input, output, d_output)
    }
}
