use ndarray::{Ix0, Ix1};

use crate::{Backprop, BackpropShape, BackpropShapes, Batch, Initialise, Scalar, View};

#[derive(Clone, Copy, Debug)]
pub struct Relu;
impl BackpropShape for Relu {
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

impl<F: Scalar> Initialise<F> for Relu {
    fn init(&self, _: &mut impl rand::Rng, _: View<Self::Params, &mut F>) {}
}

impl<F: Scalar> Backprop<F> for Relu {
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
        let zero = F::zero();
        ndarray::azip!((&i in input, o in output, c in cache) {
            *o = i.max(zero);
            *c = i;
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
        ndarray::azip!((i in de_dinput, &o in de_doutput, &c in cache) {
            *i = o * c.signum().max(F::zero());
        });
    }
}
