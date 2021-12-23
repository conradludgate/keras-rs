use std::mem::MaybeUninit;

use ndarray::{
    linalg::general_mat_mul, ArrayBase, ArrayView, ArrayViewMut, Axis, Data, Dimension,
    IntoDimension, Ix1, Ix2, RawData, ViewRepr,
};
use rand::Rng;
use rand_distr::{Distribution, Normal, StandardNormal};

use crate::{Arr, GraphBuilder, Initialise, OwnedArr, Scalar, TrainableLayer};

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

    fn view<'a, F>(&self, data: &'a [F]) -> Self::State<ViewRepr<&'a F>> {
        let i = self.input_shape.into_pattern();
        let o = self.output_shape.into_pattern();

        let (weights, biases) = data.split_at(i * o);
        let weights = ArrayView::from_shape([i, o], weights).unwrap();
        let biases = ArrayView::from_shape(o, biases).unwrap();
        LinearState { weights, biases }
    }

    fn view_mut<'a, F>(&self, data: &'a mut [F]) -> Self::State<ViewRepr<&'a mut F>> {
        let i = self.input_shape.into_pattern();
        let o = self.output_shape.into_pattern();

        let (weights, biases) = data.split_at_mut(i * o);
        let weights = ArrayViewMut::from_shape([i, o], weights).unwrap();
        let biases = ArrayViewMut::from_shape(o, biases).unwrap();
        LinearState { weights, biases }
    }

    fn apply<F: Scalar>(
        &self,
        state: Self::State<ViewRepr<&F>>,
        input: Arr<impl Data<Elem = F>, Self::InputShape>,
    ) -> OwnedArr<F, Self::OutputShape> {
        let (batch_size, input_size) = input.raw_dim().into_pattern();
        debug_assert_eq!(
            input_size,
            self.input_shape.into_pattern(),
            "input size should match specified size for the model"
        );

        let mut output = state
            .biases
            .broadcast([batch_size, self.output_shape.into_pattern()])
            .unwrap()
            .into_owned();

        general_mat_mul(F::one(), &input, &state.weights, F::one(), &mut output);

        output
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
        state.biases.map_inplace(|w| {
            w.write(dist.sample(rng));
        });
    }
}

impl TrainableLayer for Layer {
    fn train_state_size(&self, batch_size: usize) -> usize {
        self.input_shape.size() * batch_size
    }

    fn forward<F: Scalar>(
        &self,
        state: Self::State<ViewRepr<&F>>,
        input: Arr<impl Data<Elem = F>, Self::InputShape>,
        train_state: &mut [MaybeUninit<F>],
    ) -> OwnedArr<F, Self::OutputShape> {
        let train_state = ArrayViewMut::from_shape(input.raw_dim(), train_state).unwrap();
        input.assign_to(train_state);
        crate::Layer::apply(self, state, input)
    }

    fn backward<F: Scalar>(
        &self,
        state: Self::State<ViewRepr<&F>>,
        d_state: Self::State<crate::UninitRepr<F>>,
        train_state: &[F],
        d_output: Arr<impl Data<Elem = F>, Self::OutputShape>,
    ) -> OwnedArr<F, Self::InputShape> {
        let (batch_size, output_size) = d_output.raw_dim().into_pattern();
        debug_assert_eq!(
            output_size,
            self.output_shape.into_pattern(),
            "output size should match specified size for the layer"
        );

        let train_state =
            ArrayView::from_shape([batch_size, self.input_shape.into_pattern()], train_state)
                .unwrap();

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
