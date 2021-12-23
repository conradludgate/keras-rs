use ndarray::{ArrayView, ArrayViewMut, Data, Dimension, Ix1};

use crate::{Arr, OwnedArr, Scalar, TrainableLayer, UninitRepr};

use super::{Activation, ActivationLayer};

#[derive(Debug, Copy, Clone)]
pub struct Relu;
impl Activation for Relu {
    type Shape = Ix1;

    fn apply<F: Scalar>(input: Arr<impl Data<Elem = F>, Self::Shape>) -> OwnedArr<F, Self::Shape> {
        let zero = F::zero();
        input.mapv(|x| x.max(zero))
    }
}

impl TrainableLayer for ActivationLayer<Relu> {
    fn train_state_size(&self, batch_size: usize) -> usize {
        self.shape.size() * batch_size
    }

    fn forward<F: Scalar>(
        &self,
        _state: Self::State<ndarray::ViewRepr<&F>>,
        input: Arr<impl Data<Elem = F>, Self::InputShape>,
        train_state: &mut [std::mem::MaybeUninit<F>],
    ) -> OwnedArr<F, Self::OutputShape> {
        let train_state = ArrayViewMut::from_shape(input.raw_dim(), train_state).unwrap();
        input.assign_to(train_state);

        Relu::apply(input)
    }

    fn backward<F: Scalar>(
        &self,
        _state: Self::State<ndarray::ViewRepr<&F>>,
        _d_state: Self::State<UninitRepr<F>>,
        train_state: &[F],
        d_output: Arr<impl Data<Elem = F>, Self::OutputShape>,
    ) -> OwnedArr<F, Self::InputShape> {
        let (batch_size, output_size) = d_output.raw_dim().into_pattern();
        debug_assert_eq!(
            output_size,
            self.shape.into_pattern(),
            "output size should match specified size for the layer"
        );

        let train_state =
            ArrayView::from_shape([batch_size, self.shape.into_pattern()], train_state).unwrap();
        let sign = train_state.mapv(|f| f.signum().max(F::zero()));
        sign * d_output
    }
}
