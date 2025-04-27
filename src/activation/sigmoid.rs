use ndarray::{Ix0, Ix1};

use crate::{
    Backprop, BackpropShape, Batch, Inference, Initialise, ModelShape, ModelShapes, Scalar,
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

    fn shape(self, input: Ix1) -> ModelShapes<Ix1, Self> {
        ModelShapes {
            params: Ix0(),
            output: input,
            stack_size: 0,
        }
    }
}

impl BackpropShape<Ix1> for Sigmoid {
    type TrainingCache = Ix1;

    fn shape_with_cache(self, input: Ix1) -> (ModelShapes<Ix1, Self>, Self::TrainingCache) {
        (self.shape(input), input)
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
        _: &mut [F],
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
        _: &mut [F],
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
        _: &mut [F],
    ) {
        ndarray::azip!((di in de_dinput, &out in de_doutput, &f in cache) {
            *di = out * f * (-f + F::one())
        });
    }
}
