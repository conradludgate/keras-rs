use ndarray::{
    linalg::general_mat_mul, ArrayBase, Data, Dimension, IntoDimension, Ix1, Ix2, RawData, ViewRepr,
};
use rand::Rng;
use rand_distr::{Distribution, Normal, StandardNormal};

use crate::{
    Arr, ArrViewMut, GraphBuilder, Initialise, Layer as ModelLayer, MutRepr, Scalar, Slice,
    TrainableLayer,
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

impl ModelLayer for Layer {
    type InputShape = Ix1;
    type OutputShape = Ix1;

    type State<S: RawData> = DenseState<S>;

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

        let weights = data.into_array([i, o]);
        DenseState { weights }
    }

    fn apply<F: Scalar>(
        &self,
        state: Self::State<ViewRepr<&F>>,
        input: Arr<impl Data<Elem = F>, Self::InputShape>,
        mut output: ArrViewMut<F, Self::OutputShape>,
        stack: &mut [F],
    ) {
        let (_, input_size) = input.raw_dim().into_pattern();
        debug_assert_eq!(stack.len(), 0);
        debug_assert_eq!(
            input_size,
            self.input_shape.into_pattern(),
            "input size should match specified size for the model"
        );

        general_mat_mul(F::one(), &input, &state.weights, F::zero(), &mut output);
    }
}

impl<F: Scalar> Initialise<F> for Layer
where
    StandardNormal: Distribution<F>,
{
    fn init(&self, rng: &mut impl Rng, mut state: Self::State<ViewRepr<&mut F>>) {
        let inputs = F::from_usize(self.input_shape.into_pattern()).unwrap();
        let var = F::one() / inputs;
        let dist = Normal::new(F::zero(), var.sqrt()).unwrap();

        state.weights.map_inplace(|w| {
            *w = dist.sample(rng);
        });
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

    fn train_stack_space(&self, batch_size: usize) -> usize {
        self.stack_space(batch_size) + 0
    }

    fn forward<F: Scalar>(
        &self,
        state: Self::State<ViewRepr<&F>>,
        input: Arr<impl Data<Elem = F>, Self::InputShape>,
        output: ArrViewMut<F, Self::OutputShape>,
        stack: &mut [F],
        train_state: &mut Self::TrainState<MutRepr<F>>,
    ) {
        input.assign_to(train_state);
        self.apply(state, input, output, stack);
    }

    fn backward<F: Scalar>(
        &self,
        state: Self::State<ViewRepr<&F>>,
        d_state: Self::State<crate::MutRepr<F>>,
        train_state: Self::TrainState<ViewRepr<&F>>,
        d_output: Arr<impl Data<Elem = F>, Self::OutputShape>,
        d_input: ArrViewMut<F, Self::InputShape>,
        stack: &mut [F],
    ) {
        debug_assert_eq!(stack.len(), 0);

        // d_weights = input.T @ d_output
        {
            // this is only safe iff this GEMM function respects beta == 0 and does not try to read
            // from d_weights.
            let mut d_weights = d_state.weights;
            general_mat_mul(
                F::one(),
                &train_state.t(),
                &d_output,
                F::zero(),
                &mut d_weights,
            );
        }

        // d_input = d_output @ weights.T
        {
            // this is only safe iff this GEMM function respects beta == 0 and does not try to read
            // from d_input.
            let mut d_input = d_input;
            general_mat_mul(
                F::one(),
                &d_output,
                &state.weights.t(),
                F::zero(),
                &mut d_input,
            );
        }
    }
}

pub struct DenseState<S: RawData> {
    weights: ArrayBase<S, Ix2>,
}
