use std::mem::MaybeUninit;

use ndarray::{ArrayBase, Data, Dimension, Ix1, Ix2, ViewRepr, IntoDimension};

use crate::{Arr, OwnedArr, Scalar, Slice, TrainableLayer, UninitRepr, UninitArr};

use super::{Activation, ActivationLayer};

#[derive(Debug, Copy, Clone)]
pub struct Sigmoid;
impl Activation for Sigmoid {
    type Shape = Ix1;

    fn apply<F: Scalar>(
        input: Arr<impl Data<Elem = F>, Self::Shape>,
        mut output: UninitArr<F, Self::Shape>,
    ) {
        let one = F::one();
        output.zip_mut_with(&input, |o, &i| {
            o.write(one / (one + (-i).exp()));
        });
    }

    fn batch(shape: Self::Shape, batch_size: usize) -> <Self::Shape as Dimension>::Larger {
        [batch_size, shape.into_pattern()].into_dimension()
    }
}

impl TrainableLayer for ActivationLayer<Sigmoid> {
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
        mut output: UninitArr<F, Self::OutputShape>,
        _stack: &mut [MaybeUninit<F>],
        train_state: &mut Self::TrainState<UninitRepr<F>>,
    ) {
        Sigmoid::apply(input, output.view_mut());
        output.assign_to(train_state);
    }

    fn backward<F: Scalar>(
        &self,
        _state: Self::State<ndarray::ViewRepr<&F>>,
        _d_state: Self::State<UninitRepr<F>>,
        train_state: Self::TrainState<ViewRepr<&F>>,
        d_output: Arr<impl Data<Elem = F>, Self::OutputShape>,
    ) -> OwnedArr<F, Self::InputShape> {
        d_output.into_owned() * train_state * (train_state.mapv(F::neg) + F::one())
    }
}
