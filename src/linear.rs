use ndarray::{linalg::general_mat_mul, ArrayBase, Axis, Dimension, Ix1, Ix2, RawData};
use rand::Rng;
use rand_distr::{Distribution, Normal, StandardNormal};

use crate::{
    Backprop, BackpropShape, Batch, Inference, Initialise, ModelShape, ModelShapes, Scalar, Shape,
    Slice, View,
};

type Input<R> = Batch<Ix1, R>;
type Output<R> = super::Output<Ix1, Linear, R>;
type TrainingCache<R> = super::TrainingCache<Ix1, Linear, R>;
type Params<R> = super::Params<Ix1, Linear, R>;

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

impl<F: Scalar> Initialise<Ix1, F> for Linear
where
    StandardNormal: Distribution<F>,
{
    fn init(&self, rng: &mut impl Rng, mut state: Params<&mut F>) {
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

    #[inline]
    fn assign_to<F: Copy>(
        self,
        mut dst: Self::Base<ndarray::ViewRepr<&mut F>>,
        src: Self::Base<ndarray::ViewRepr<&F>>,
    ) {
        dst.weights.assign(&src.weights);
        dst.biases.assign(&src.biases);
    }

    #[inline]
    fn as_ref<'a, T>(
        self,
        s: &'a Self::Base<ndarray::ViewRepr<&T>>,
    ) -> Self::Base<ndarray::ViewRepr<&'a T>> {
        LinearState {
            weights: s.weights.view(),
            biases: s.biases.view(),
        }
    }

    #[inline]
    fn as_mut<'a, T>(
        self,
        s: &'a mut Self::Base<ndarray::ViewRepr<&mut T>>,
    ) -> Self::Base<ndarray::ViewRepr<&'a mut T>> {
        LinearState {
            weights: s.weights.view_mut(),
            biases: s.biases.view_mut(),
        }
    }
}

impl ModelShape<Ix1> for Linear {
    type Params = Layer;
    type Output = Ix1;

    fn shape(self, input: Ix1) -> ModelShapes<Ix1, Self> {
        ModelShapes {
            params: Layer {
                input_shape: input,
                output_shape: self.output_shape,
            },
            output: self.output_shape,
            stack_size: 0,
        }
    }
}

impl BackpropShape<Ix1> for Linear {
    type TrainingCache = Ix1;

    fn shape_with_cache(self, input: Ix1) -> (ModelShapes<Ix1, Self>, Ix1) {
        (self.shape(input), input)
    }
}

impl<F: Scalar> Inference<Ix1, F> for Linear {
    fn infer(
        self,
        _: Ix1,
        _: usize,
        p: Params<&F>,
        in_: Input<&F>,
        mut out: Output<&mut F>,
        _: &mut [F],
    ) {
        for out in out.axis_iter_mut(Axis(0)) {
            p.biases.assign_to(out);
        }

        general_mat_mul(F::one(), &in_, &p.weights, F::one(), &mut out);
    }
}

impl<F: Scalar> Backprop<Ix1, F> for Linear {
    fn forward(
        self,
        _: Ix1,
        _: usize,
        p: Params<&F>,
        in_: Input<&F>,
        mut out: Output<&mut F>,
        c: TrainingCache<&mut F>,
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
        _: Ix1,
        _: usize,
        p: Params<&F>,
        dout: Output<&F>,
        mut din_: Input<&mut F>,
        mut dp: View<Self::Params, &mut F>,
        in_: TrainingCache<&F>,
        _: &mut [F],
    ) {
        // d_weights = in_.T @ dout
        general_mat_mul(F::one(), &in_.t(), &dout, F::zero(), &mut dp.weights);

        // d_biases = dout.mean(axis = 0)
        for (dout, d_bias) in std::iter::zip(dout.axis_iter(Axis(0)), &mut dp.biases) {
            *d_bias = dout.mean().unwrap();
        }

        // din_ = dout @ weights.T
        general_mat_mul(F::one(), &dout, &p.weights.t(), F::zero(), &mut din_);
    }
}
