use std::iter::zip;

use ndarray::{
    linalg::{general_mat_mul, general_mat_vec_mul},
    Axis, Dimension, Ix0, Ix1, Ix2,
};

use crate::{Backprop, BackpropShape, Batch, BatchShape, Inference, ModelShape, Scalar, Stack};

#[derive(Debug, Copy, Clone)]
pub struct Softmax;

type Input<R> = Batch<Ix1, R>;
type Output<R> = crate::Output<Ix1, Softmax, R>;
type Cache<R> = crate::TrainingCache<Ix1, Softmax, R>;
type Params<R> = crate::Params<Ix1, Softmax, R>;

impl ModelShape<Ix1> for Softmax {
    type Params = Ix0;
    type Output = Ix1;

    fn shape(self, input: Ix1) -> (Self::Params, Self::Output) {
        (Ix0(), input)
    }

    fn stack(self, _: usize, _: Ix1) -> usize {
        0
    }
}

impl<F: Scalar> Inference<Ix1, F> for Softmax {
    fn infer(
        self,
        input_shape: Ix1,
        _: usize,
        _: Params<&F>,
        x: Input<&F>,
        mut y: Output<&mut F>,
        _: Stack<F>,
    ) {
        for (x, mut y) in zip(x.axis_iter(Axis(0)), y.axis_iter_mut(Axis(0))) {
            debug_assert_eq!(x.raw_dim(), y.raw_dim());
            debug_assert_eq!(x.raw_dim(), input_shape);

            for (x, y) in zip(x.iter(), y.iter_mut()) {
                *y = x.exp()
            }

            let sum = y.sum();
            y /= sum;
        }
    }
}

impl BackpropShape<Ix1> for Softmax {
    type TrainingCache = Ix2;

    fn backprop_shape(self, batch: usize, input: Ix1) -> (Self::TrainingCache, usize) {
        (input.batched(batch), input.size() * input.size())
    }
}

impl<F: Scalar> Backprop<Ix1, F> for Softmax {
    fn forward(
        self,
        input_shape: Ix1,
        _: usize,
        _: Params<&F>,
        x: Batch<Ix1, &F>,
        mut y: Output<&mut F>,
        cache: Cache<&mut F>,
        _: Stack<F>,
    ) {
        for (x, mut y) in zip(x.axis_iter(Axis(0)), y.axis_iter_mut(Axis(0))) {
            debug_assert_eq!(x.raw_dim(), y.raw_dim());
            debug_assert_eq!(x.raw_dim(), input_shape);

            for (x, y) in zip(x.iter(), y.iter_mut()) {
                *y = x.exp()
            }

            let sum = y.sum();
            y /= sum;
        }

        y.assign_to(cache);
    }

    fn backward(
        self,
        input_shape: Ix1,
        _: usize,
        _: Params<&F>,
        dy: Output<&F>,
        mut dx: Batch<Ix1, &mut F>,
        _: Params<&mut F>,
        cache: Cache<&F>,
        stack: Stack<F>,
    ) {
        let n = input_shape.into_pattern();

        // dy = dL/d_by = [batch_size, N]
        // dx = dL/d_bx = dy * d_by/d_bx = [batch_size, N]
        // where y_b = softmax(x_b)
        // we need dy/dx = [N, N]

        // dy_b = dL/d_by = [1, N]
        // dx_b = dL/d_bx = dy_b * dy_b/dx_b = [1, N]
        // where y_b = softmax(x_b)
        // we need dy_b/dx_b = [N, N]

        // y_bi = softmax(x_bi) = exp(x_bi) / sum(exp(x_bj))
        // dy_bi/dx_bij = y_bj * (Dij - y_bi)

        let output_iter = cache.axis_iter(Axis(0));
        let dy_iter = dy.axis_iter(Axis(0));
        let dx_iter = dx.axis_iter_mut(Axis(0));

        let (mut scratch, _) = stack.take(Ix2(n, n));

        // TODO: can we elimintate this for loop?

        for ((output, mut dx), dy) in std::iter::zip(output_iter, dx_iter).zip(dy_iter) {
            let s = output.into_shape_with_order([output.len(), 1]).unwrap();
            general_mat_mul(-F::one(), &s, &s.t(), F::zero(), &mut scratch);
            scratch
                .diag_mut()
                .zip_mut_with(&output, |x, y| *x = *x - *y);

            general_mat_vec_mul(F::one(), &scratch, &dy, F::zero(), &mut dx);
        }
    }
}
