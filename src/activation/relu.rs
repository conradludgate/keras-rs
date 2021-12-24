use ndarray::{ArrayBase, Data, Dimension, Ix1, Ix2, ViewRepr};

use crate::{Arr, OwnedArr, Scalar, Slice, TrainableLayer, UninitRepr};

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
    type TrainState<S: ndarray::RawData> = ArrayBase<S, Ix2>;

    fn train_state_size(&self, batch_size: usize) -> usize {
        self.shape.size() * batch_size
    }

    fn view_train_state<S: Slice>(&self, batch_size: usize, data: S) -> Self::TrainState<S::Repr> {
        data.into_array([batch_size, self.shape.into_pattern()])
    }

    fn forward<F: Scalar>(
        &self,
        _state: Self::State<ndarray::ViewRepr<&F>>,
        input: Arr<impl Data<Elem = F>, Self::InputShape>,
        train_state: &mut Self::TrainState<UninitRepr<F>>,
    ) -> OwnedArr<F, Self::OutputShape> {
        input.assign_to(train_state);
        Relu::apply(input)
    }

    fn backward<F: Scalar>(
        &self,
        _state: Self::State<ndarray::ViewRepr<&F>>,
        _d_state: Self::State<UninitRepr<F>>,
        train_state: Self::TrainState<ViewRepr<&F>>,
        d_output: Arr<impl Data<Elem = F>, Self::OutputShape>,
    ) -> OwnedArr<F, Self::InputShape> {
        let sign = train_state.mapv(|f| f.signum().max(F::zero()));
        sign * d_output
    }
}
