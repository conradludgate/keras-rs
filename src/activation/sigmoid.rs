use ndarray::{Ix0, Ix1};

use crate::{Backprop, BackpropShape, BackpropShapes, Initialise, Scalar};

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
    fn init(&self, _: &mut impl rand::Rng, _: crate::View<Self::Params, &mut F>) {}
}

impl<F: Scalar> Backprop<F> for Sigmoid {
    fn forward(
        self,
        _: Self::Input,
        _: usize,
        _: crate::View<Self::Params, &F>,
        input: crate::View<crate::Batched<Self::Input>, &F>,
        mut output: crate::View<crate::Batched<Self::Output>, &mut F>,
        cache: crate::View<crate::Batched<Self::Cache>, &mut F>,
        _: &mut [F],
    ) {
        let one = F::one();
        output.zip_mut_with(&input, |o, &i| {
            *o = one / (one + (-i).exp());
        });
        output.assign_to(cache);
    }

    fn backward(
        self,
        _: Self::Input,
        _: usize,
        _: crate::View<Self::Params, &F>,
        de_doutput: crate::View<crate::Batched<Self::Output>, &F>,
        mut de_dinput: crate::View<crate::Batched<Self::Input>, &mut F>,
        _: crate::View<Self::Params, &mut F>,
        cache: crate::View<crate::Batched<Self::Cache>, &F>,
        _: &mut [F],
    ) {
        de_doutput.assign_to(de_dinput.view_mut());
        de_dinput.zip_mut_with(&cache, |di, &f| *di = *di * f * (-f + F::one()));
    }
}
