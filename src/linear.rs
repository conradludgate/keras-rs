use std::mem::MaybeUninit;

use ndarray::{
    linalg::general_mat_mul, ArrayBase, ArrayView, ArrayViewMut, Axis, Data, Dimension,
    IntoDimension, Ix1, Ix2, RawData, ShapeError, ViewRepr,
};
use rand::Rng;
use rand_distr::{Distribution, Normal, StandardNormal};

use crate::{
    Arr, GraphBuilder, Initialise, Scalar, TrainableLayer, UninitArr,
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

    fn view<'a, F>(&self, data: &'a [F]) -> Result<Self::State<ViewRepr<&'a F>>, ShapeError> {
        let i = self.input_shape.into_pattern();
        let o = self.output_shape.into_pattern();

        let (weights, biases) = data.split_at(i * o);
        let weights = ArrayView::from_shape([i, o], weights)?;
        let biases = ArrayView::from_shape(o, biases)?;
        Ok(LinearState { weights, biases })
    }

    fn view_mut<'a, F>(
        &self,
        data: &'a mut [F],
    ) -> Result<Self::State<ViewRepr<&'a mut F>>, ShapeError> {
        let i = self.input_shape.into_pattern();
        let o = self.output_shape.into_pattern();

        let (weights, biases) = data.split_at_mut(i * o);
        let weights = ArrayViewMut::from_shape([i, o], weights)?;
        let biases = ArrayViewMut::from_shape(o, biases)?;
        Ok(LinearState { weights, biases })
    }

    fn apply<F: Scalar>(
        &self,
        state: Self::State<ViewRepr<&F>>,
        input: Arr<impl Data<Elem = F>, Self::InputShape>,
        mut output: UninitArr<F, Self::OutputShape>,
    ) {
        state
            .biases
            .broadcast(output.raw_dim())
            .unwrap()
            .assign_to(output);
        // safe since we just assigned it from the biases
        let mut output = unsafe { output.assume_init() };
        general_mat_mul(F::one(), &state.weights, &input, F::one(), &mut output);
    }

    // fn apply<F: Scalar>(
    //     &self,
    //     state: Self::State<ViewRepr<&F>>,
    //     input: Arr<impl Data<Elem = F>, Self::InputShape>,
    // ) -> Arr<OwnedRepr<F>, Self::OutputShape> {
    //     state.weights.dot(&input) + state.biases
    // }
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
    fn train_state_size(&self) -> usize {
        self.input_shape.size()
    }

    fn forward<F: Scalar>(
        &self,
        state: Self::State<ViewRepr<&F>>,
        input: Arr<impl Data<Elem = F>, Self::InputShape>,
        output: UninitArr<F, Self::OutputShape>,
        train_state: &mut [MaybeUninit<F>],
    ) {
        let train_state = ArrayViewMut::from_shape(input.raw_dim(), train_state).unwrap();
        input.assign_to(train_state);
        crate::Layer::apply(self, state, input, output);
    }

    fn backward<F: Scalar>(
        &self,
        state: Self::State<ViewRepr<&F>>,
        d_state: Self::State<crate::UninitRepr<F>>,
        d_input: UninitArr<F, Self::OutputShape>,
        train_state: &[F],
        d_output: Arr<impl Data<Elem = F>, Self::OutputShape>,
    ) {
        let train_state = ArrayView::from_shape(d_input.raw_dim(), train_state).unwrap();

        // d_weights = input.T @ d_output
        unsafe {
            // this is only safe iff this GEMM function respects beta == 0 and does not try to read
            // from d_weights.
            let d_weights = d_state.weights.assume_init();
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
        unsafe {
            // this is only safe iff this GEMM function respects beta == 0 and does not try to read
            // from d_weights.
            let d_input = d_input.assume_init();
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

pub struct LinearState<S: RawData> {
    weights: ArrayBase<S, Ix2>,
    biases: ArrayBase<S, Ix1>,
}
