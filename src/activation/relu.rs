use ndarray::{ArrayBase, Data, Dimension, IntoDimension, Ix1, Ix2, ViewRepr};

use crate::{Arr, ArrViewMut, MutRepr, Scalar, Slice, TrainableLayer};

use super::{Activation, ActivationLayer};

#[derive(Debug, Copy, Clone)]
pub struct Relu;
impl Activation for Relu {
    type Shape = Ix1;

    fn apply<F: Scalar>(
        input: Arr<impl Data<Elem = F>, Self::Shape>,
        mut output: ArrViewMut<F, Self::Shape>,
    ) {
        let zero = F::zero();
        output.zip_mut_with(&input, |o, i| {
            *o = i.max(zero);
        });
    }

    fn batch(shape: Self::Shape, batch_size: usize) -> <Self::Shape as Dimension>::Larger {
        [batch_size, shape.into_pattern()].into_dimension()
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

    fn train_stack_space(&self, _batch_size: usize) -> usize {
        0
    }

    fn forward<F: Scalar>(
        &self,
        _state: Self::State<ndarray::ViewRepr<&F>>,
        input: Arr<impl Data<Elem = F>, Self::InputShape>,
        output: ArrViewMut<F, Self::OutputShape>,
        _stack: &mut [F],
        train_state: &mut Self::TrainState<MutRepr<F>>,
    ) {
        input.assign_to(train_state);
        Relu::apply(input, output);
    }

    fn backward<F: Scalar>(
        &self,
        _state: Self::State<ndarray::ViewRepr<&F>>,
        _d_state: Self::State<MutRepr<F>>,
        train_state: Self::TrainState<ViewRepr<&F>>,
        d_output: Arr<impl Data<Elem = F>, Self::OutputShape>,
        mut d_input: ArrViewMut<F, Self::InputShape>,
        stack: &mut [F],
    ) {
        debug_assert_eq!(stack.len(), 0);
        d_output.assign_to(d_input.view_mut());
        d_input.zip_mut_with(&train_state, |di, &f| {
            *di = *di * f.signum().max(F::zero());
        });
    }
}
