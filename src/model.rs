use ndarray::{ArrayView, ArrayViewMut, Axis, Data, Dimension, ViewRepr};
use rand::{prelude::SliceRandom, thread_rng};
use rand_distr::num_traits::Float;

use crate::{
    cost::Cost, optimise::Optimiser, Arr, ArrView, ArrViewMut, GraphBuilder, Layer,
    LayerTrainState, OwnedArr, Scalar, TrainableLayer,
};

pub struct Model<F: Scalar, G: GraphBuilder> {
    pub(crate) layer: G::Layer,
    pub(crate) data: Vec<F>,
}

impl<F: Scalar, G: GraphBuilder> Model<F, G> {
    fn state(&self) -> <G::Layer as Layer>::State<ViewRepr<&'_ F>> {
        self.layer.view_state(self.data.as_slice())
    }

    pub fn apply_batch(
        &self,
        input: Arr<impl Data<Elem = F>, G::InputShape>,
    ) -> OwnedArr<F, G::OutputShape> {
        let batch_size = input.shape()[0];
        let mut output =
            OwnedArr::<F, G::OutputShape>::zeros(self.layer.batched_output_shape(batch_size));
        let stack_size = self.layer.stack_space(batch_size);
        let mut stack = vec![F::zero(); stack_size];
        self.layer
            .apply(self.state(), input, output.view_mut(), &mut *stack);
        output
    }

    // pub fn apply_single(
    //     &self,
    //     input: ArrayBase<impl Data<Elem = F>, G::InputShape>,
    // ) -> ArrayBase<ndarray::OwnedRepr<F>, G::OutputShape> {
    //     let mut output =
    //         ArrayBase::<ndarray::OwnedRepr<F>, G::OutputShape>::uninit(self.layer.output_shape());
    //     let stack_size = self.layer.stack_space(1);
    //     let mut stack = Vec::with_capacity(stack_size);
    //     let input_shape = input.raw_dim().insert_axis(Axis(0));
    //     let output_shape = output.raw_dim().insert_axis(Axis(0));
    //     self.layer.apply(
    //         self.state(),
    //         input.into_shape(input_shape).unwrap(),
    //         output.view_mut().into_shape(output_shape).unwrap(),
    //         &mut stack.spare_capacity_mut()[..stack_size],
    //     );
    //     unsafe { output.assume_init() }
    // }
}

pub struct Trainer<F: Scalar, G: GraphBuilder, O: Optimiser<F>, C: Cost<G::OutputShape>>
where
    G::Layer: TrainableLayer,
{
    pub model: Model<F, G>,
    pub optimiser: O,
    pub cost: C,
    pub regularisation: Option<Regularisation<F>>,
}

pub struct ModelTrainer<F: Scalar, G: GraphBuilder, O: Optimiser<F>, C: Cost<G::OutputShape>>
where
    G::Layer: TrainableLayer,
{
    trainer: Trainer<F, G, O, C>,
    bufs: Bufs<F>,
}

struct Bufs<F> {
    data: Vec<F>,
    stack: Vec<F>,
    gradiants: Vec<F>,
    train_state: Vec<F>,
    train_stack: Vec<F>,
}

impl<F> Default for Bufs<F> {
    fn default() -> Self {
        Self {
            data: vec![],
            stack: vec![],
            gradiants: vec![],
            train_state: vec![],
            train_stack: vec![],
        }
    }
}

impl<F: Scalar, G: GraphBuilder, O: Optimiser<F>, C: Cost<G::OutputShape>> ModelTrainer<F, G, O, C>
where
    G::Layer: TrainableLayer,
{
    pub fn train_epoch(
        &mut self,
        input: Arr<impl Data<Elem = F>, G::InputShape>,
        expected: Arr<impl Data<Elem = F>, G::OutputShape>,
        batch_size: usize,
    ) -> F {
        self.trainer
            .train_epoch(input, expected, batch_size, &mut self.bufs)
    }

    pub fn test_epoch(
        &mut self,
        input: Arr<impl Data<Elem = F>, G::InputShape>,
        expected: Arr<impl Data<Elem = F>, G::OutputShape>,
    ) -> F {
        self.trainer.test_epoch(input, expected, &mut self.bufs)
    }

    pub fn into_inner(self) -> Trainer<F, G, O, C> {
        self.trainer
    }

    pub fn as_model(&self) -> &Model<F, G> {
        &self.trainer.model
    }
}

impl<F: Scalar, G: GraphBuilder, O: Optimiser<F>, C: Cost<G::OutputShape>> Trainer<F, G, O, C>
where
    G::Layer: TrainableLayer,
{
    pub fn build(mut self) -> ModelTrainer<F, G, O, C> {
        self.optimiser.init(self.model.layer.size());
        ModelTrainer {
            trainer: self,
            bufs: Bufs::default(),
        }
    }

    fn train_epoch(
        &mut self,
        input: Arr<impl Data<Elem = F>, G::InputShape>,
        expected: Arr<impl Data<Elem = F>, G::OutputShape>,
        batch_size: usize,
        bufs: &mut Bufs<F>,
    ) -> F {
        let total_inputs = input.raw_dim()[0];

        let mut rng = thread_rng();
        let mut indices: Vec<_> = (0..total_inputs).collect();
        indices.shuffle(&mut rng);

        let mut cost = F::zero();
        for indices in indices.chunks(batch_size) {
            let (input, expected) =
                self.select_batch(input.view(), expected.view(), indices, &mut bufs.data);
            cost = cost
                + self.train_batch(
                    input,
                    expected,
                    &mut bufs.gradiants,
                    &mut bufs.stack,
                    &mut bufs.train_state,
                    &mut bufs.train_stack,
                );
        }

        cost / F::from_usize((total_inputs + batch_size - 1) / batch_size).unwrap()
    }

    fn test_epoch(
        &mut self,
        input: Arr<impl Data<Elem = F>, G::InputShape>,
        expected: Arr<impl Data<Elem = F>, G::OutputShape>,
        bufs: &mut Bufs<F>,
    ) -> F {
        let total_inputs = input.raw_dim()[0];

        let mut cost = F::zero();
        for indices in 0..total_inputs {
            let (input, expected) =
                self.select_batch(input.view(), expected.view(), &[indices], &mut bufs.data);
            cost = cost + self.test_batch(input, expected, &mut bufs.stack);
        }

        cost / F::from_usize(total_inputs).unwrap()
    }

    fn select_batch<'a>(
        &mut self,
        input: Arr<impl Data<Elem = F>, G::InputShape>,
        expected: Arr<impl Data<Elem = F>, G::OutputShape>,
        indices: &[usize],
        data_buf: &'a mut Vec<F>,
    ) -> (
        ArrView<'a, F, G::InputShape>,
        ArrView<'a, F, G::OutputShape>,
    ) {
        data_buf.clear();

        let mut input_dim = input.raw_dim();
        input_dim.as_array_view_mut()[0] = indices.len();

        let mut expected_dim = expected.raw_dim();
        expected_dim.as_array_view_mut()[0] = indices.len();

        let (_, _, input, expected) = fill2(
            data_buf,
            input_dim.size(),
            expected_dim.size(),
            |input_batch| {
                let mut input_batch =
                    ArrayViewMut::from_shape(input_dim.clone(), input_batch).unwrap();

                input_batch
                    .axis_iter_mut(Axis(0))
                    .zip(indices)
                    .for_each(|(i, &j)| input.index_axis(Axis(0), j).assign_to(i));
            },
            |expected_batch| {
                let mut expected_batch =
                    ArrayViewMut::from_shape(expected_dim.clone(), expected_batch).unwrap();

                expected_batch
                    .axis_iter_mut(Axis(0))
                    .zip(indices)
                    .for_each(|(i, &j)| expected.index_axis(Axis(0), j).assign_to(i));
            },
        );

        let input = ArrayView::from_shape(input_dim, input).unwrap();
        let expected = ArrayView::from_shape(expected_dim, expected).unwrap();
        (input, expected)
    }

    fn test_batch(
        &mut self,
        input: Arr<impl Data<Elem = F>, G::InputShape>,
        expected: Arr<impl Data<Elem = F>, G::OutputShape>,
        stack_buf: &mut Vec<F>,
    ) -> F {
        debug_assert_eq!(
            input.shape()[0],
            expected.shape()[0],
            "input and expected batch sizes should be equal"
        );

        let output = self.test_batch_forward(input, stack_buf);
        self.cost.cost(output, expected)
    }

    fn train_batch(
        &mut self,
        input: Arr<impl Data<Elem = F>, G::InputShape>,
        expected: Arr<impl Data<Elem = F>, G::OutputShape>,
        gradiants_buf: &mut Vec<F>,
        stack_buf: &mut Vec<F>,
        train_state_buf: &mut Vec<F>,
        train_stack_buf: &mut Vec<F>,
    ) -> F {
        debug_assert_eq!(
            input.shape()[0],
            expected.shape()[0],
            "input and expected batch sizes should be equal"
        );

        let input_shape = input.raw_dim();
        let (output, train_state) = self.batch_forward(input, stack_buf, train_state_buf);

        let d_output = self.cost.diff(output.view(), expected.view());

        let grads = self.batch_backward(
            d_output,
            input_shape,
            train_state,
            gradiants_buf,
            train_stack_buf,
        );

        if let Some(r) = self.regularisation {
            r.apply(grads, &self.model.data);
        }

        self.optimiser.optimise(&mut self.model.data, grads);
        self.cost.cost(output, expected)
    }

    fn batch_forward<'a, 'b>(
        &mut self,
        input: Arr<impl Data<Elem = F>, G::InputShape>,
        mut stack_buf: &'a mut Vec<F>,
        mut train_state_buf: &'b mut Vec<F>,
    ) -> (
        ArrViewMut<'a, F, G::OutputShape>,
        LayerTrainState<'b, F, G::Layer>,
    ) {
        // allocate space and get uninit slice
        let batch_size = input.shape()[0];
        let train_state_size = self.model.layer.train_state_size(batch_size);
        if train_state_buf.len() < train_state_size {
            train_state_buf.resize(train_state_size, F::zero());
        }

        let output_shape = self.model.layer.batched_output_shape(batch_size);
        let stack_size = self.model.layer.stack_space(batch_size);
        if stack_buf.len() < output_shape.size() + stack_size {
            stack_buf.resize(output_shape.size() + stack_size, F::zero());
        }

        let (output, mut stack) = stack_buf.split_at_mut(output_shape.size());
        let mut output = ArrViewMut::<F, G::OutputShape>::from_shape(output_shape, output).unwrap();

        {
            let mut train_state = self
                .model
                .layer
                .view_train_state(batch_size, &mut **train_state_buf);
            self.model.layer.forward(
                self.model.state(),
                input,
                output.view_mut(),
                &mut stack[..stack_size],
                &mut train_state,
            );
        }

        let train_state = self
            .model
            .layer
            .view_train_state(batch_size, &**train_state_buf);
        (output, train_state)
    }

    fn test_batch_forward<'a>(
        &mut self,
        input: Arr<impl Data<Elem = F>, G::InputShape>,
        stack_buf: &'a mut Vec<F>,
    ) -> ArrViewMut<'a, F, G::OutputShape> {
        // allocate space and get uninit slice
        let batch_size = input.shape()[0];

        let output_shape = self.model.layer.batched_output_shape(batch_size);
        let stack_size = self.model.layer.stack_space(batch_size);
        if stack_buf.len() < output_shape.size() + stack_size {
            stack_buf.resize(output_shape.size() + stack_size, F::zero());
        }

        let (output, mut stack) = stack_buf.split_at_mut(output_shape.size());
        let mut output = ArrViewMut::<F, G::OutputShape>::from_shape(output_shape, output).unwrap();

        self.model.layer.apply(
            self.model.state(),
            input,
            output.view_mut(),
            &mut stack[..stack_size],
        );
        output
    }

    fn batch_backward<'a>(
        &mut self,
        d_output: Arr<impl Data<Elem = F>, G::OutputShape>,
        input_shape: <G::InputShape as Dimension>::Larger,
        train_state: <G::Layer as TrainableLayer>::TrainState<ViewRepr<&F>>,
        mut gradiants_buf: &'a mut Vec<F>,
        train_stack_buf: &mut Vec<F>,
    ) -> &'a mut [F] {
        // allocate space and get uninit state
        if gradiants_buf.len() < self.model.layer.size() {
            gradiants_buf.resize(self.model.layer.size(), F::zero());
        }

        // allocate space and get uninit slice
        let batch_size = d_output.shape()[0];
        let train_stack_size = self.model.layer.train_stack_space(batch_size);
        if train_stack_buf.len() < input_shape.size() + train_stack_size {
            train_stack_buf.resize(input_shape.size() + train_stack_size, F::zero());
        }

        let (d_input, mut train_stack) = train_stack_buf.split_at_mut(input_shape.size());
        let d_input = ArrViewMut::<F, G::InputShape>::from_shape(input_shape, d_input).unwrap();

        let d_state = self
            .model
            .layer
            .view_state(&mut gradiants_buf[..self.model.layer.size()]);

        // feed model backward, storing grads in uninit state
        self.model.layer.backward(
            self.model.state(),
            d_state,
            train_state,
            d_output,
            d_input,
            &mut train_stack[..train_stack_size],
        );

        &mut gradiants_buf[..self.model.layer.size()]
    }
}

fn fill<T: Float, O>(buf: &mut Vec<T>, n: usize, f: impl FnOnce(&mut [T]) -> O) -> (O, &mut [T]) {
    if buf.len() < n {
        buf.resize(n, T::zero());
    }
    let output = f(&mut buf[..n]);
    (output, &mut buf[..n])
}

fn fill2<T: Float, O1, O2>(
    buf: &mut Vec<T>,
    n1: usize,
    n2: usize,
    f1: impl FnOnce(&mut [T]) -> O1,
    f2: impl FnOnce(&mut [T]) -> O2,
) -> (O1, O2, &[T], &[T]) {
    let ((o1, o2), s) = fill(buf, n1 + n2, |s| {
        let (s1, s2) = s.split_at_mut(n1);
        (f1(s1), f2(s2))
    });

    let (s1, s2) = s.split_at(n1);
    (o1, o2, s1, s2)
}

#[derive(Debug, Clone, Copy)]
pub enum Regularisation<F> {
    L1(F),
    L2(F),
    L1_2(F, F),
}

impl<F: Scalar> Regularisation<F> {
    fn apply(self, grads: &mut [F], graph: &[F]) {
        let zip = grads.iter_mut().zip(graph);
        match self {
            Regularisation::L1(a) => zip.for_each(|(g, &t)| {
                *g = *g + t.signum() * a;
            }),
            Regularisation::L2(a) => zip.for_each(|(g, &t)| {
                *g = *g + (t + t) * a;
            }),
            Regularisation::L1_2(a, b) => zip.for_each(|(g, &t)| {
                *g = *g + t.signum() * a + (t + t) * b;
            }),
        }
    }
}
