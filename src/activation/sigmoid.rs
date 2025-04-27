use ndarray::{Ix0, Ix1};

use crate::{Backprop, BackpropShape, BackpropShapes, Batch, Initialise, Scalar, View};

#[derive(Debug, Copy, Clone)]
pub struct Sigmoid;

impl BackpropShape for Sigmoid {
    type Params = Ix0;
    type Input = Ix1;
    type Output = Ix1;
    type Cache = Ix1;

    fn shape(self, input: Self::Input) -> BackpropShapes<Self> {
        BackpropShapes {
            params: Ix0(),
            output: input,
            cache: input,
            stack_size: 0,
        }
    }
}

impl<F: Scalar> Initialise<F> for Sigmoid {
    fn init(&self, _: &mut impl rand::Rng, _: View<Self::Params, &mut F>) {}
}

impl<F: Scalar> Backprop<F> for Sigmoid {
    fn forward(
        self,
        _: Self::Input,
        _: usize,
        _: View<Self::Params, &F>,
        input: Batch<Self::Input, &F>,
        mut output: Batch<Self::Output, &mut F>,
        cache: Batch<Self::Cache, &mut F>,
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
        _: Self::Input,
        _: usize,
        _: View<Self::Params, &F>,
        de_doutput: Batch<Self::Output, &F>,
        mut de_dinput: Batch<Self::Input, &mut F>,
        _: View<Self::Params, &mut F>,
        cache: Batch<Self::Cache, &F>,
        _: &mut [F],
    ) {
        ndarray::azip!((di in de_dinput, &out in de_doutput, &f in cache) {
            *di = out * f * (-f + F::one())
        });
    }
}
