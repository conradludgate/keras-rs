use std::mem::MaybeUninit;

use ndarray::{
    linalg::{general_mat_mul, general_mat_vec_mul},
    ArrayBase, ArrayViewMut, Axis, Data, Dimension, IntoDimension, Ix1, Ix2, ViewRepr,
};

use crate::{Arr, Scalar, Slice, TrainableLayer, UninitArr, UninitRepr};

use super::{Activation, ActivationLayer};

#[derive(Debug, Copy, Clone)]
pub struct Softmax;
impl Activation for Softmax {
    type Shape = Ix1;

    fn apply<F: Scalar>(
        input: Arr<impl Data<Elem = F>, Self::Shape>,
        mut output: UninitArr<F, Self::Shape>,
    ) {
        output.zip_mut_with(&input, |x, y| {
            x.write(y.exp());
        });
        let mut output = unsafe { output.assume_init() };
        let sum = output.sum_axis(Axis(1));
        output.zip_mut_with(&sum, |x, y| *x = *x / *y);
    }

    fn batch(shape: Self::Shape, batch_size: usize) -> <Self::Shape as Dimension>::Larger {
        [batch_size, shape.into_pattern()].into_dimension()
    }
}

impl TrainableLayer for ActivationLayer<Softmax> {
    type TrainState<S: ndarray::RawData> = ArrayBase<S, Ix2>;
    fn train_state_size(&self, batch_size: usize) -> usize {
        self.shape.size() * batch_size
    }

    fn view_train_state<S: Slice>(&self, batch_size: usize, data: S) -> Self::TrainState<S::Repr> {
        data.into_array([batch_size, self.shape.into_pattern()])
    }

    fn train_stack_space(&self, _batch_size: usize) -> usize {
        self.shape.into_pattern() * self.shape.into_pattern()
    }

    fn forward<F: Scalar>(
        &self,
        _state: Self::State<ndarray::ViewRepr<&F>>,
        input: Arr<impl Data<Elem = F>, Self::InputShape>,
        mut output: UninitArr<F, Self::OutputShape>,
        _stack: &mut [MaybeUninit<F>],
        train_state: &mut Self::TrainState<UninitRepr<F>>,
    ) {
        Softmax::apply(input, output.view_mut());
        output.assign_to(train_state);
    }

    fn backward<F: Scalar>(
        &self,
        _state: Self::State<ndarray::ViewRepr<&F>>,
        _d_state: Self::State<UninitRepr<F>>,
        train_state: Self::TrainState<ViewRepr<&F>>,
        d_output: Arr<impl Data<Elem = F>, Self::OutputShape>,
        d_input: UninitArr<F, Self::InputShape>,
        stack: &mut [MaybeUninit<F>],
    ) {
        let n = self.shape.into_pattern();

        // d_output = dL/d_by = [batch_size, N]
        // d_input = dL/d_bx = d_output * d_by/d_bx = [batch_size, N]
        // where y_b = softmax(x_b)
        // we need dy/dx = [N, N]

        // d_output_b = dL/d_by = [1, N]
        // d_input_b = dL/d_bx = d_output_b * dy_b/dx_b = [1, N]
        // where y_b = softmax(x_b)
        // we need dy_b/dx_b = [N, N]

        // y_bi = softmax(x_bi) = exp(x_bi) / sum(exp(x_bj))
        // dy_bi/dx_bij = y_bj * (Dij - y_bi)

        let mut d_input = unsafe { d_input.assume_init() };

        let output_iter = train_state.axis_iter(Axis(0));
        let d_output_iter = d_output.axis_iter(Axis(0));
        let d_input_iter = d_input.axis_iter_mut(Axis(0));

        let scratch = ArrayViewMut::from_shape([n, n], stack).unwrap();
        let mut scratch = unsafe { scratch.assume_init() };

        // TODO: can we elimintate this for loop?

        for ((output, mut d_input), d_output) in
            std::iter::zip(output_iter, d_input_iter).zip(d_output_iter)
        {
            let s = output.into_shape([output.len(), 1]).unwrap();
            general_mat_mul(-F::one(), &s, &s.t(), F::zero(), &mut scratch);
            scratch
                .diag_mut()
                .zip_mut_with(&output, |x, y| *x = *x - *y);

            general_mat_vec_mul(F::one(), &scratch, &d_output, F::zero(), &mut d_input);
        }
    }
}
