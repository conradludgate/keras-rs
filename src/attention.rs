//! attention layer <https://en.wikipedia.org/wiki/Attention_(machine_learning)>

use std::mem::MaybeUninit;

use ndarray::{
    linalg::general_mat_mul, s, ArrayBase, ArrayView, ArrayViewMut, Axis, Data, Dimension,
    IntoDimension, Ix1, Ix2, Ix3, LinalgScalar, RawData, ViewRepr,
};
use rand::Rng;
use rand_distr::{Distribution, Normal, StandardNormal};

use crate::{
    activation::{softmax::Softmax, Activation},
    Arr, GraphBuilder, Initialise, Scalar, Slice, TrainableLayer, UninitArr, UninitRepr,
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
    type InputShape = Ix2;
    type OutputShape = Ix2;

    type Layer = Layer;

    fn with_input_shape(self, input_shape: Self::InputShape) -> Self::Layer {
        let Self { output_shape } = self;
        Layer {
            output_shape: [input_shape.into_pattern().0, output_shape.into_pattern()]
                .into_dimension(),
            input_shape,
        }
    }
}

pub struct Layer {
    input_shape: Ix2,
    output_shape: Ix2,
}

fn gemm0b<'c, A, S1, S2>(
    alpha: A,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    c: &'c mut [MaybeUninit<A>],
) -> ArrayViewMut<'c, A, Ix2>
where
    S1: Data<Elem = A>,
    S2: Data<Elem = A>,
    A: LinalgScalar,
{
    let c = ArrayViewMut::from_shape([a.shape()[0], b.shape()[1]], c).unwrap();
    // SAFETY: not sure, but we aren't going to be reading from k :eyes:
    // should not introduce any invalid refs
    let mut c = unsafe { c.assume_init() };
    general_mat_mul(alpha, a, b, A::zero(), &mut c);
    c
}

impl crate::Layer for Layer {
    type InputShape = Ix2;
    type OutputShape = Ix2;

    type State<S: RawData> = AttentionState<S>;

    fn size(&self) -> usize {
        let (_, embedding) = self.input_shape.into_pattern();
        let (_, hidden) = self.output_shape.into_pattern();
        hidden * embedding * 3
    }

    fn output_shape(&self) -> Self::OutputShape {
        self.output_shape
    }

    fn batched_output_shape(&self, batch_size: usize) -> Ix3 {
        let (history, hidden) = self.output_shape.into_pattern();
        [batch_size, history, hidden].into_dimension()
    }

    fn stack_space(&self, _batch_size: usize) -> usize {
        let (history, hidden) = self.output_shape.into_pattern();

        2 * (history * hidden) + hidden + history
    }

    fn view_state<S: Slice>(&self, data: S) -> Self::State<S::Repr> {
        let (_, embedding) = self.input_shape.into_pattern();
        let (_, hidden) = self.output_shape.into_pattern();

        let (query, rest) = data.split(embedding * hidden);
        let (key, value) = rest.split(embedding * hidden);

        let query = query.into_array([embedding, hidden]);
        let key = key.into_array([embedding, hidden]);
        let value = value.into_array([embedding, hidden]);
        AttentionState { query, key, value }
    }

    fn apply<F: Scalar>(
        &self,
        state: Self::State<ViewRepr<&F>>,
        input: Arr<impl Data<Elem = F>, Self::InputShape>,
        mut output: UninitArr<F, Self::OutputShape>,
        stack: &mut [MaybeUninit<F>],
    ) {
        let (batch_size, history_size, embedding_size) = input.raw_dim().into_pattern();
        let (_, hidden_size) = self.output_shape.into_pattern();
        debug_assert_eq!(
            (history_size, embedding_size),
            self.input_shape.into_pattern(),
            "input size should match specified size for the model"
        );

        let (v, rest) = stack.split_at_mut(hidden_size * history_size);
        let (k, rest) = rest.split_at_mut(hidden_size * history_size);
        let (q, soft) = rest.split_at_mut(hidden_size);
        debug_assert_eq!(soft.len(), history_size);

        let x = input
            .view()
            .into_shape([batch_size * history_size, embedding_size])
            .unwrap();

        let v = gemm0b(F::one(), &x, &state.value, v);
        let k = gemm0b(F::one(), &x, &state.key, k);

        let scale = F::from_usize(hidden_size).unwrap().sqrt().recip();
        for i in 0..history_size {
            let xi = input.slice(s![.., i, ..]);
            let q = gemm0b(F::one(), &xi, &state.query, q);
            let mut soft = gemm0b(scale, &q, &k.t(), soft);

            soft.map_inplace(|x| *x = x.exp());
            let sum = soft.sum_axis(Axis(1));
            soft.zip_mut_with(&sum, |x, y| *x = *x / *y);

            let oi = output.slice_mut(s![.., i, ..]);
            let mut oi = unsafe { oi.assume_init() };
            general_mat_mul(F::one(), &soft, &v, F::zero(), &mut oi);
        }
    }
}

unsafe impl<F: Scalar> Initialise<F> for Layer
where
    StandardNormal: Distribution<F>,
{
    fn init(
        &self,
        _rng: &mut impl Rng,
        state: &mut Self::State<ViewRepr<&mut std::mem::MaybeUninit<F>>>,
    ) {
        state.key.fill(MaybeUninit::new(F::zero()));
        state.query.fill(MaybeUninit::new(F::zero()));
        state.value.fill(MaybeUninit::new(F::zero()));
    }
}

// impl TrainableLayer for Layer {
//     type TrainState<S: RawData> = ();

//     fn train_state_size(&self, batch_size: usize) -> usize {
//         self.input_shape.size() * batch_size
//     }

//     fn view_train_state<S: Slice>(&self, batch_size: usize, data: S) -> Self::TrainState<S::Repr> {
//         // data.into_array([batch_size, self.input_shape.into_pattern()])
//     }

//     fn train_stack_space(&self, _batch_size: usize) -> usize {
//         0
//     }

//     fn forward<F: Scalar>(
//         &self,
//         state: Self::State<ViewRepr<&F>>,
//         input: Arr<impl Data<Elem = F>, Self::InputShape>,
//         output: UninitArr<F, Self::OutputShape>,
//         _stack: &mut [MaybeUninit<F>],
//         train_state: &mut Self::TrainState<UninitRepr<F>>,
//     ) {
//         use crate::Layer;
//         // input.assign_to(train_state);
//         self.apply(state, input, output, &mut []);
//     }

//     fn backward<F: Scalar>(
//         &self,
//         state: Self::State<ViewRepr<&F>>,
//         d_state: Self::State<crate::UninitRepr<F>>,
//         train_state: Self::TrainState<ViewRepr<&F>>,
//         d_output: Arr<impl Data<Elem = F>, Self::OutputShape>,
//         d_input: UninitArr<F, Self::InputShape>,
//         stack: &mut [MaybeUninit<F>],
//     ) {
//         // d_weights = input.T @ d_output
//         unsafe {
//             // this is only safe iff this GEMM function respects beta == 0 and does not try to read
//             // from d_weights.
//             let mut d_weights = d_state.weights.assume_init();
//             general_mat_mul(
//                 F::one(),
//                 &train_state.t(),
//                 &d_output,
//                 F::zero(),
//                 &mut d_weights,
//             );
//         }

//         // d_biases = d_output.mean(axis = 0)
//         {
//             d_output
//                 .view()
//                 .mean_axis(Axis(0))
//                 .unwrap()
//                 .assign_to(d_state.biases);
//         }

//         // d_input = d_output @ weights.T
//         unsafe {
//             // this is only safe iff this GEMM function respects beta == 0 and does not try to read
//             // from d_input.
//             let mut d_input = d_input.assume_init();
//             general_mat_mul(
//                 F::one(),
//                 &d_output,
//                 &state.weights.t(),
//                 F::zero(),
//                 &mut d_input,
//             );
//         }
//     }
// }

pub struct AttentionState<S: RawData> {
    query: ArrayBase<S, Ix2>,
    key: ArrayBase<S, Ix2>,
    value: ArrayBase<S, Ix2>,
}
