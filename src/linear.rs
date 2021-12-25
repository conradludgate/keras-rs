use std::mem::MaybeUninit;

use ndarray::{
    linalg::general_mat_mul, ArrayBase, Axis, Data, Dimension, IntoDimension, Ix1, Ix2, RawData,
    ViewRepr,
};
use rand::Rng;
use rand_distr::{Distribution, Normal, StandardNormal};

use crate::{
    Arr, GraphBuilder, Initialise, OwnedArr, Scalar, Slice, TrainableLayer, UninitArr, UninitRepr,
};

impl Layer {
    pub fn output(shape: impl IntoDimension<Dim = Ix1>) -> Builder {
        Builder {
            output_shape: shape.into_dimension(),
        }
    }
}

pub struct Builder {
    output_shape: Ix1,
}

impl GraphBuilder for Builder {
    type InputShape = Ix1;
    type OutputShape = Ix1;

    type Layer = Layer;

    fn with_input_shape(self, input_shape: Self::InputShape) -> Self::Layer {
        let Self { output_shape } = self;
        Layer {
            output_shape,
            input_shape,
        }
    }
}

pub struct Layer {
    input_shape: Ix1,
    output_shape: Ix1,
}

impl crate::Layer for Layer {
    type InputShape = Ix1;
    type OutputShape = Ix1;

    type State<S: RawData> = LinearState<S>;

    fn size(&self) -> usize {
        let i = self.input_shape.into_pattern();
        let o = self.output_shape.into_pattern();
        (i + 1) * o
    }

    fn output_shape(&self) -> Self::OutputShape {
        self.output_shape
    }

    fn batched_output_shape(&self, batch_size: usize) -> Ix2 {
        [batch_size, self.output_shape.into_pattern()].into_dimension()
    }

    fn stack_space(&self, _batch_size: usize) -> usize {
        0
    }

    fn view_state<S: Slice>(&self, data: S) -> Self::State<S::Repr> {
        let i = self.input_shape.into_pattern();
        let o = self.output_shape.into_pattern();

        let (weights, biases) = data.split(i * o);
        let weights = weights.into_array([i, o]);
        let biases = biases.into_array(o);
        LinearState { weights, biases }
    }

    fn apply<F: Scalar>(
        &self,
        state: Self::State<ViewRepr<&F>>,
        input: Arr<impl Data<Elem = F>, Self::InputShape>,
        mut output: UninitArr<F, Self::OutputShape>,
        _stack: &mut [MaybeUninit<F>],
    ) {
        let (batch_size, input_size) = input.raw_dim().into_pattern();
        debug_assert_eq!(
            input_size,
            self.input_shape.into_pattern(),
            "input size should match specified size for the model"
        );

        state
            .biases
            .broadcast([batch_size, self.output_shape.into_pattern()])
            .unwrap()
            .assign_to(output.view_mut());

        let mut output = unsafe { output.assume_init() };

        general_mat_mul(F::one(), &input, &state.weights, F::one(), &mut output);
    }
}

unsafe impl<F: Scalar> Initialise<F> for Layer
where
    StandardNormal: Distribution<F>,
{
    fn init(
        &self,
        rng: &mut impl Rng,
        state: &mut Self::State<ViewRepr<&mut std::mem::MaybeUninit<F>>>,
    ) {
        let inputs = F::from_usize(self.input_shape.into_pattern()).unwrap();
        let var = F::one() / inputs;
        let dist = Normal::new(F::zero(), var.sqrt()).unwrap();

        state.weights.map_inplace(|w| {
            w.write(dist.sample(rng));
        });
        state.biases.fill(MaybeUninit::new(F::zero()));
    }
}

impl TrainableLayer for Layer {
    type TrainState<S: RawData> = ArrayBase<S, Ix2>;

    fn train_state_size(&self, batch_size: usize) -> usize {
        self.input_shape.size() * batch_size
    }

    fn view_train_state<S: Slice>(&self, batch_size: usize, data: S) -> Self::TrainState<S::Repr> {
        data.into_array([batch_size, self.input_shape.into_pattern()])
    }

    fn forward<F: Scalar>(
        &self,
        state: Self::State<ViewRepr<&F>>,
        input: Arr<impl Data<Elem = F>, Self::InputShape>,
        train_state: &mut Self::TrainState<UninitRepr<F>>,
    ) -> OwnedArr<F, Self::OutputShape> {
        use crate::Layer;
        input.assign_to(train_state);
        let batch_size = input.shape()[0];
        let mut output = ndarray::Array2::uninit(self.batched_output_shape(batch_size));
        self.apply(state, input, output.view_mut(), &mut []);
        unsafe { output.assume_init() }
    }

    fn backward<F: Scalar>(
        &self,
        state: Self::State<ViewRepr<&F>>,
        d_state: Self::State<crate::UninitRepr<F>>,
        train_state: Self::TrainState<ViewRepr<&F>>,
        d_output: Arr<impl Data<Elem = F>, Self::OutputShape>,
    ) -> OwnedArr<F, Self::InputShape> {
        // d_weights = input.T @ d_output
        unsafe {
            // this is only safe iff this GEMM function respects beta == 0 and does not try to read
            // from d_weights.
            let mut d_weights = d_state.weights.assume_init();
            general_mat_mul(
                F::one(),
                &train_state.t(),
                &d_output,
                F::zero(),
                &mut d_weights,
            );
        }

        // d_biases = d_output.mean(axis = 0)
        {
            d_output
                .view()
                .mean_axis(Axis(0))
                .unwrap()
                .assign_to(d_state.biases);
        }

        // d_input = d_output @ weights.T
        d_output.dot(&state.weights.t())
    }
}

pub struct LinearState<S: RawData> {
    weights: ArrayBase<S, Ix2>,
    biases: ArrayBase<S, Ix1>,
}
