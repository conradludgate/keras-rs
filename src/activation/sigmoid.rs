use ndarray::{Ix0, Ix1, Ix2};

use crate::{
    Backprop, BackpropShape, Batch, BatchShape, Inference, Initialise, ModelShape, Scalar, Stack,
};

type Input<R> = Batch<Ix1, R>;
type Output<R> = crate::Output<Ix1, Sigmoid, R>;
type Cache<R> = crate::TrainingCache<Ix1, Sigmoid, R>;
type Params<R> = crate::Params<Ix1, Sigmoid, R>;

#[derive(Debug, Copy, Clone)]
pub struct Sigmoid;

impl ModelShape<Ix1> for Sigmoid {
    type Params = Ix0;
    type Output = Ix1;

    fn shape(self, input: Ix1) -> (Self::Params, Self::Output) {
        (Ix0(), input)
    }

    fn stack(self, _: usize, _: Ix1) -> usize {
        0
    }
}

impl BackpropShape<Ix1> for Sigmoid {
    type TrainingCache = Ix2;

    fn backprop_shape(self, batch_size: usize, input: Ix1) -> (Self::TrainingCache, usize) {
        (input.batched(batch_size), 0)
    }
}

impl<F: Scalar> Initialise<Ix1, F> for Sigmoid {
    fn init(&self, _: &mut impl rand::Rng, _: Params<&mut F>) {}
}

impl<F: Scalar> Inference<Ix1, F> for Sigmoid {
    fn infer(
        self,
        _: Ix1,
        _: usize,
        _: Params<&F>,
        input: Input<&F>,
        output: Output<&mut F>,
        _: Stack<F>,
    ) {
        let one = F::one();
        ndarray::azip!((input in input, output in output) {
            *output = one / (one + (-*input).exp());
        });
    }
}

impl<F: Scalar> Backprop<Ix1, F> for Sigmoid {
    fn forward(
        self,
        _: Ix1,
        _: usize,
        _: Params<&F>,
        input: Input<&F>,
        output: Output<&mut F>,
        cache: Cache<&mut F>,
        _: Stack<F>,
    ) {
        let one = F::one();
        ndarray::azip!((input in input, output in output, cache in cache) {
            *output = one / (one + (-*input).exp());
            *cache = *output;
        });
    }

    fn backward(
        self,
        _: Ix1,
        _: usize,
        _: Params<&F>,
        de_doutput: Output<&F>,
        de_dinput: Input<&mut F>,
        _: Params<&mut F>,
        cache: Cache<&F>,
        _: Stack<F>,
    ) {
        ndarray::azip!((di in de_dinput, &out in de_doutput, &f in cache) {
            *di = out * f * (-f + F::one())
        });
    }
}
