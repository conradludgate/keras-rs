use ndarray::{ArrayBase, Data, Dimension, OwnedRepr};

use crate::{GraphBuilder, Initialise, Layer, Scalar, OwnedArr, Arr};

pub mod relu;
pub mod sigmoid;

pub trait Activation {
    type Shape: Dimension + Clone;

    fn apply<F: crate::Scalar>(
        input: Arr<impl Data<Elem= F>, Self::Shape>,
    ) -> OwnedArr<F, Self::Shape>;
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

unsafe impl<F: Scalar, A: Activation> Initialise<F> for ActivationLayer<A> {
    fn init(
        &self,
        _rng: &mut impl rand::Rng,
        state: &mut Self::State<ndarray::ViewRepr<&mut std::mem::MaybeUninit<F>>>,
    ) {
        debug_assert_eq!(std::mem::size_of_val(state), 0);
    }
}
