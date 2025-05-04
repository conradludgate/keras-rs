use ndarray::{Ix0, Ix1, Ix2};

use crate::{
    Backprop, BackpropShape, Batch, BatchShape, Inference, Initialise, ModelShape, Scalar, Stack,
};

type Input<R> = Batch<Ix1, R>;
type Output<R> = crate::Output<Ix1, Relu, R>;
type Cache<R> = crate::TrainingCache<Ix1, Relu, R>;
type Params<R> = crate::Params<Ix1, Relu, R>;

#[derive(Clone, Copy, Debug)]
pub struct Relu;

impl ModelShape<Ix1> for Relu {
    type Params = Ix0;
    type Output = Ix1;

    fn shape(self, input: Ix1) -> (Self::Params, Self::Output) {
        (Ix0(), input)
    }

    fn stack(self, _: usize, _: Ix1) -> usize {
        0
    }
}

impl BackpropShape<Ix1> for Relu {
    type TrainingCache = Ix2;

    fn backprop_shape(self, batch_size: usize, input: Ix1) -> (Self::TrainingCache, usize) {
        (input.batched(batch_size), 0)
    }
}

impl<F: Scalar> Initialise<Ix1, F> for Relu {
    fn init(&self, _: &mut impl rand::Rng, _: Params<&mut F>) {}
}

impl<F: Scalar> Inference<Ix1, F> for Relu {
    fn infer(
        self,
        _: Ix1,
        _: usize,
        _: Params<&F>,
        input: Input<&F>,
        output: Output<&mut F>,
        _: Stack<F>,
    ) {
        let zero = F::zero();
        ndarray::azip!((&i in input, o in output) {
            *o = i.max(zero);
        });
    }
}

impl<F: Scalar> Backprop<Ix1, F> for Relu {
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
        let zero = F::zero();
        ndarray::azip!((&i in input, o in output, c in cache) {
            *o = i.max(zero);
            *c = i;
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
        ndarray::azip!((i in de_dinput, &o in de_doutput, &c in cache) {
            *i = o * c.signum().max(F::zero());
        });
    }
}
