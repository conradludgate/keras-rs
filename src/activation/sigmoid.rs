use ndarray::{ArrayView, ArrayViewMut, Data, Dimension, Ix1};

use crate::{Arr, OwnedArr, Scalar, TrainableLayer, UninitRepr};

use super::{Activation, ActivationLayer};

#[derive(Debug, Copy, Clone)]
pub struct Sigmoid;
impl Activation for Sigmoid {
    type Shape = Ix1;

    fn apply<F: Scalar>(input: Arr<impl Data<Elem = F>, Self::Shape>) -> OwnedArr<F, Self::Shape> {
        let mut y = input.into_owned();
        let one = F::one();
        y.mapv_inplace(|x| (one / (one + (-x).exp())));
        y
    }
}

impl TrainableLayer for ActivationLayer<Sigmoid> {
    fn train_state_size(&self, batch_size: usize) -> usize {
        self.shape.size() * batch_size
    }

    fn forward<F: Scalar>(
        &self,
        _state: Self::State<ndarray::ViewRepr<&F>>,
        input: Arr<impl Data<Elem = F>, Self::InputShape>,
        train_state: &mut [std::mem::MaybeUninit<F>],
    ) -> OwnedArr<F, Self::OutputShape> {
        let y = Sigmoid::apply(input);
        let train_state = ArrayViewMut::from_shape(y.raw_dim(), train_state).unwrap();
        y.assign_to(train_state);
        y
    }

    fn backward<F: Scalar>(
        &self,
        _state: Self::State<ndarray::ViewRepr<&F>>,
        _d_state: Self::State<UninitRepr<F>>,
        train_state: &[F],
        d_output: Arr<impl Data<Elem = F>, Self::OutputShape>,
    ) -> OwnedArr<F, Self::InputShape> {
        let y = ArrayView::from_shape(d_output.raw_dim(), train_state).unwrap();
        d_output.into_owned() * y * (y.mapv(F::neg) + F::one())
    }
}
