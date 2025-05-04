use ndarray::{
    linalg::general_mat_mul, ArrayBase, Axis, Data, DataMut, Dimension, Ix1, Ix2, RawData,
};
use rand::Rng;
use rand_distr::{Distribution, Normal, StandardNormal};

use crate::{
    Array, Backprop, BackpropShape, Batch, BatchShape, Inference, Initialise, ModelShape, Scalar,
    Shape, Slice, Stack, View,
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
}

impl<R: RawData> Array<R> for LinearState<R> {
    type Shape = Layer;

    fn as_ref<'a>(&'a self) -> View<Self::Shape, &'a <R as RawData>::Elem>
    where
        R: Data,
    {
        LinearState {
            weights: self.weights.view(),
            biases: self.biases.view(),
        }
    }

    fn as_mut<'a>(&'a mut self) -> View<Self::Shape, &'a mut <R as RawData>::Elem>
    where
        R: DataMut,
    {
        LinearState {
            weights: self.weights.view_mut(),
            biases: self.biases.view_mut(),
        }
    }
    fn assign_to(&self, mut dst: View<Self::Shape, &mut <R as RawData>::Elem>)
    where
        <R as RawData>::Elem: Copy,
        R: Data,
    {
        dst.weights.assign(&self.weights);
        dst.biases.assign(&self.biases);
    }
}

impl ModelShape<Ix1> for Linear {
    type Params = Layer;
    type Output = Ix1;

    fn shape(self, input: Ix1) -> (Layer, Ix1) {
        (
            Layer {
                input_shape: input,
                output_shape: self.output_shape,
            },
            self.output_shape,
        )
    }

    fn stack(self, _: usize, _: Ix1) -> usize {
        0
    }
}

impl BackpropShape<Ix1> for Linear {
    type TrainingCache = Ix2;

    fn backprop_shape(self, batch_size: usize, input: Ix1) -> (Self::TrainingCache, usize) {
        (input.batched(batch_size), 0)
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
        _: Stack<F>,
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
        _: Stack<F>,
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
        _: Stack<F>,
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
