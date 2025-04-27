use ndarray::{linalg::general_mat_mul, ArrayBase, Axis, Dimension, Ix1, Ix2, RawData};
use rand::Rng;
use rand_distr::{Distribution, Normal, StandardNormal};

use crate::{
    Backprop, BackpropShape, BackpropShapes, Batched, Initialise, Scalar, Shape, Slice, View,
};

#[derive(Clone, Copy)]
pub struct Linear {
    output_shape: Ix1,
}

impl Linear {
    pub fn output(output_size: usize) -> Self {
        Self {
            output_shape: Ix1(output_size),
        }
    }
}

#[derive(Clone, Copy)]
pub struct Layer {
    input_shape: Ix1,
    output_shape: Ix1,
}

impl<F: Scalar> Initialise<F> for Linear
where
    StandardNormal: Distribution<F>,
{
    fn init(&self, rng: &mut impl Rng, mut state: View<Self::Params, &mut F>) {
        // let inputs = F::from_usize(self.input_shape.into_pattern()).unwrap();

        let inputs = F::from_usize(state.weights.shape()[0]).unwrap();

        let var = F::one() / inputs;
        let dist = Normal::new(F::zero(), var.sqrt()).unwrap();

        state.weights.map_inplace(|w| {
            *w = dist.sample(rng);
        });
        // state.biases.fill(F::zero());
    }
}

pub struct LinearState<S: RawData> {
    weights: ArrayBase<S, Ix2>,
    biases: ArrayBase<S, Ix1>,
}

impl Shape for Layer {
    type Base<S: RawData> = LinearState<S>;

    #[inline]
    fn size(self) -> usize {
        let i = self.input_shape.into_pattern();
        let o = self.output_shape.into_pattern();
        (i + 1) * o
    }

    #[inline]
    fn from_slice<S: Slice>(self, slice: S) -> Self::Base<S::Repr> {
        let i = self.input_shape.into_pattern();
        let o = self.output_shape.into_pattern();

        let (weights, biases) = slice.split(i * o);
        let weights = weights.into_array([i, o]);
        let biases = biases.into_array(o);
        LinearState { weights, biases }
    }
}

impl BackpropShape for Linear {
    type Params = Layer;
    type Input = Ix1;
    type Output = Ix1;
    type Cache = Ix1;

    fn shape(self, input: Self::Input) -> BackpropShapes<Self> {
        BackpropShapes {
            params: Layer {
                input_shape: input,
                output_shape: self.output_shape,
            },
            output: self.output_shape,
            cache: input,
            stack_size: 0,
        }
    }
}

impl<F: Scalar> Backprop<F> for Linear {
    fn forward(
        self,
        _: Self::Input,
        _: usize,
        p: View<Self::Params, &F>,
        in_: View<Batched<Self::Input>, &F>,
        mut out: View<Batched<Self::Output>, &mut F>,
        c: View<Batched<Self::Cache>, &mut F>,
        _: &mut [F],
    ) {
        in_.assign_to(c);

        for out in out.axis_iter_mut(Axis(0)) {
            p.biases.assign_to(out);
        }

        general_mat_mul(F::one(), &in_, &p.weights, F::one(), &mut out);
    }

    fn backward(
        self,
        _: Self::Input,
        _: usize,
        p: View<Self::Params, &F>,
        dout: View<Batched<Self::Output>, &F>,
        mut din_: View<Batched<Self::Input>, &mut F>,
        mut dp: View<Self::Params, &mut F>,
        in_: View<Batched<Self::Cache>, &F>,
        _: &mut [F],
    ) {
        // d_weights = in_.T @ dout
        general_mat_mul(F::one(), &in_.t(), &dout, F::zero(), &mut dp.weights);

        // d_biases = dout.mean(axis = 0)
        for (dout, d_bias) in std::iter::zip(dout.axis_iter(Axis(1)), &mut dp.biases) {
            *d_bias = dout.mean().unwrap();
        }

        // din_ = dout @ weights.T
        general_mat_mul(F::one(), &dout, &p.weights.t(), F::zero(), &mut din_);
    }
}
