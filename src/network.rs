use std::mem::MaybeUninit;

use ndarray::{ArrayViewMut, Dimension, ViewRepr};
use rand::Rng;

use crate::{Arr, GraphBuilder, Initialise, Layer, Scalar, TrainableLayer, UninitArr, UninitRepr};

impl<G1, G2> GraphBuilder for (G1, G2)
where
    G1: GraphBuilder,
    G2: GraphBuilder<InputShape = G1::OutputShape>,
{
    type InputShape = G1::InputShape;
    type OutputShape = G2::OutputShape;

    type Layer = (G1::Layer, G2::Layer);

    fn with_input_shape(self, input_shape: Self::InputShape) -> Self::Layer {
        let l1 = self.0.with_input_shape(input_shape);
        let l2 = self.1.with_input_shape(l1.output_shape());
        (l1, l2)
    }
}

impl<L1, L2> Layer for (L1, L2)
where
    L1: Layer,
    L2: Layer<InputShape = L1::OutputShape>,
{
    type InputShape = L1::InputShape;
    type OutputShape = L2::OutputShape;
    type State<S: ndarray::RawData> = (L1::State<S>, L2::State<S>);

    fn size(&self) -> usize {
        self.0.size() + self.1.size()
    }

    fn output_shape(&self) -> Self::OutputShape {
        self.1.output_shape()
    }

    fn batched_output_shape(&self, batch_size: usize) -> <Self::OutputShape as Dimension>::Larger {
        self.1.batched_output_shape(batch_size)
    }

    fn stack_space(&self, batch_size: usize) -> usize {
        // Each path needs some spare stack space
        let sp0 = self.0.stack_space(batch_size);
        let sp1 = self.1.stack_space(batch_size);
        let inner_size = self.0.batched_output_shape(batch_size).size(); // space used by inner output
        inner_size + sp0.max(sp1)
    }

    fn view_state<S: crate::Slice>(&self, data: S) -> Self::State<S::Repr> {
        let i = self.0.size();
        let data = data.split(i);
        (self.0.view_state(data.0), self.1.view_state(data.1))
    }

    fn apply<F: Scalar>(
        &self,
        state: Self::State<ndarray::ViewRepr<&F>>,
        input: Arr<impl ndarray::Data<Elem = F>, Self::InputShape>,
        output: UninitArr<F, Self::OutputShape>,
        stack: &mut [MaybeUninit<F>],
    ) {
        let batch_size = input.shape()[0];
        let inner_size = self.0.batched_output_shape(batch_size).size(); // space used by inner output

        let (inner, stack) = stack.split_at_mut(inner_size);
        let inner_stack_size = self.0.stack_space(batch_size);

        let mut inner =
            ArrayViewMut::from_shape(self.0.batched_output_shape(batch_size), inner).unwrap();

        self.0.apply(
            state.0,
            input,
            inner.view_mut(),
            &mut stack[..inner_stack_size],
        );

        let inner = unsafe { inner.assume_init() };

        let outer_stack_size = self.1.stack_space(batch_size);
        self.1
            .apply(state.1, inner, output, &mut stack[..outer_stack_size])
    }
}

unsafe impl<F: Scalar, L1, L2> Initialise<F> for (L1, L2)
where
    L1: Layer + Initialise<F>,
    L2: Layer<InputShape = L1::OutputShape> + Initialise<F>,
{
    fn init(
        &self,
        rng: &mut impl Rng,
        state: &mut Self::State<ndarray::ViewRepr<&mut std::mem::MaybeUninit<F>>>,
    ) {
        self.0.init(rng, &mut state.0);
        self.1.init(rng, &mut state.1);
    }
}

impl<L1, L2> TrainableLayer for (L1, L2)
where
    L1: TrainableLayer,
    L2: TrainableLayer + Layer<InputShape = L1::OutputShape>,
{
    type TrainState<S: ndarray::RawData> = (L1::TrainState<S>, L2::TrainState<S>);

    fn train_state_size(&self, batch_size: usize) -> usize {
        self.0.train_state_size(batch_size) + self.1.train_state_size(batch_size)
    }

    fn view_train_state<S: crate::Slice>(
        &self,
        batch_size: usize,
        data: S,
    ) -> Self::TrainState<S::Repr> {
        let n0 = self.0.train_state_size(batch_size);
        let (data0, data1) = data.split(n0);
        (
            self.0.view_train_state(batch_size, data0),
            self.1.view_train_state(batch_size, data1),
        )
    }

    fn train_stack_space(&self, batch_size: usize) -> usize {
        let sp0 = self.0.train_stack_space(batch_size);
        let sp1 = self.1.train_stack_space(batch_size);
        let inner_size = self.0.batched_output_shape(batch_size).size(); // space used by inner output
        inner_size + sp0.max(sp1)
    }

    fn forward<F: Scalar>(
        &self,
        state: Self::State<ndarray::ViewRepr<&F>>,
        input: Arr<impl ndarray::Data<Elem = F>, Self::InputShape>,
        output: UninitArr<F, Self::OutputShape>,
        stack: &mut [MaybeUninit<F>],
        train_state: &mut Self::TrainState<UninitRepr<F>>,
    ) {
        let batch_size = input.shape()[0];
        let inner_size = self.0.batched_output_shape(batch_size).size(); // space used by inner output
        debug_assert_eq!(stack.len(), self.stack_space(batch_size));

        let (inner, stack) = stack.split_at_mut(inner_size);
        let inner_stack_size = self.0.stack_space(batch_size);

        let mut inner =
            ArrayViewMut::from_shape(self.0.batched_output_shape(batch_size), inner).unwrap();

        self.0.forward(
            state.0,
            input,
            inner.view_mut(),
            &mut stack[..inner_stack_size],
            &mut train_state.0,
        );

        let inner = unsafe { inner.assume_init() };

        let outer_stack_size = self.1.stack_space(batch_size);
        self.1.forward(
            state.1,
            inner,
            output,
            &mut stack[..outer_stack_size],
            &mut train_state.1,
        );
    }

    fn backward<F: Scalar>(
        &self,
        state: Self::State<ndarray::ViewRepr<&F>>,
        d_state: Self::State<crate::UninitRepr<F>>,
        train_state: Self::TrainState<ViewRepr<&F>>,
        d_output: Arr<impl ndarray::Data<Elem = F>, Self::OutputShape>,
        d_input: UninitArr<F, Self::InputShape>,
        stack: &mut [MaybeUninit<F>],
    ) {
        let batch_size = d_output.shape()[0];
        let inner_size = self.0.batched_output_shape(batch_size).size(); // space used by inner output
        debug_assert_eq!(stack.len(), self.train_stack_space(batch_size));

        let (inner, stack) = stack.split_at_mut(inner_size);
        let inner_stack_size = self.1.train_stack_space(batch_size);

        let mut inner =
            ArrayViewMut::from_shape(self.0.batched_output_shape(batch_size), inner).unwrap();

        self.1.backward(
            state.1,
            d_state.1,
            train_state.1,
            d_output,
            inner.view_mut(),
            &mut stack[..inner_stack_size],
        );

        let inner = unsafe { inner.assume_init() };

        let outer_stack_size = self.0.train_stack_space(batch_size);
        self.0.backward(
            state.0,
            d_state.0,
            train_state.0,
            inner,
            d_input,
            &mut stack[..outer_stack_size],
        );
    }
}

/// Converts the provided values into a nested chain of tuples.
/// Works by taking each pair of expressions, converting them into a tuple,
/// Then pushing all of them into the macro recursively
///
/// ```
/// use linear_networks::net;
///
/// // These two expressions are the same
/// let a = net!(0, 1, 2, 3);
/// let b = net!((0, 1), (2, 3));
/// assert_eq!(a, b);
/// ```
///
/// There's an edge case to handle odd numbered inputs.
/// It leaves the first input and pairs up the rest of them
///
/// ```
/// use linear_networks::net;
///
/// let a = net!(0, 1, 2, 3, 4);
/// let b = net!(0, (1, 2), (3, 4));
/// let c = net!(0, ((1, 2), (3, 4)));
/// assert_eq!(a, b);
/// assert_eq!(a, c);
/// ```
#[macro_export]
macro_rules! net {
    ($g0:expr $(,)?) => {
        $g0
    };
    ($($g0:expr, $g1:expr),* $(,)?) => {
        $crate::net!($(
            ($g0, $g1)
        ),*)
    };
    ($g:expr, $($g0:expr, $g1:expr),* $(,)?) => {
        $crate::net!(
            $g,
            $(
                ($g0, $g1)
            ),*
        )
    };
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_tuple_macro() {
        // single value
        let t = net!(0);
        assert_eq!(t, 0);

        // two values
        let t = net!(0, 1);
        assert_eq!(t, (0, 1));

        // 8 values (balanced nested binary tree)
        let t = net!(0, 1, 2, 3, 4, 5, 6, 7);
        assert_eq!(t, (((0, 1), (2, 3)), ((4, 5), (6, 7))));

        // 7 values (off balance nested binary tree)
        let t = net!(0, 1, 2, 3, 4, 5, 6);
        assert_eq!(t, ((0, (1, 2)), ((3, 4), (5, 6))));

        // 6 values (off balance nested binary tree)
        let t = net!(0, 1, 2, 3, 4, 5);
        assert_eq!(t, ((0, 1), ((2, 3), (4, 5))));
    }
}
