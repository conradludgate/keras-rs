use std::mem::MaybeUninit;

use ndarray::{ArrayView, ArrayViewMut, Axis, Data, Dimension, ViewRepr};
use rand::{prelude::SliceRandom, thread_rng};

use crate::{
    cost::Cost, optimise::Optimiser, Arr, ArrView, GraphBuilder, Layer, OwnedArr, Scalar,
    TrainableLayer,
};

pub struct Model<F: Scalar, G: GraphBuilder> {
    pub(crate) layer: G::Layer,
    pub(crate) data: Vec<F>,
}

impl<F: Scalar, G: GraphBuilder> Model<F, G> {
    fn state(&self) -> <G::Layer as Layer>::State<ViewRepr<&'_ F>> {
        self.layer
            .view(&self.data)
            .expect("size of data should always be a valid view into a layer")
    }

    pub fn apply(
        &self,
        input: Arr<impl Data<Elem = F>, G::InputShape>,
    ) -> OwnedArr<F, G::OutputShape> {
        self.layer.apply(self.state(), input)
    }
}

pub struct Trainer<F: Scalar, G: GraphBuilder, O: Optimiser<F>, C: Cost<G::OutputShape>>
where
    G::Layer: TrainableLayer,
{
    pub model: Model<F, G>,
    pub optimiser: O,
    pub cost: C,
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
    gradiants: Vec<F>,
    train_state: Vec<F>,
}

impl<F> Default for Bufs<F> {
    fn default() -> Self {
        Self {
            data: vec![],
            gradiants: vec![],
            train_state: vec![],
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
                + self.train_batch(input, expected, &mut bufs.gradiants, &mut bufs.train_state);
        }

        cost / F::from_usize((total_inputs + batch_size - 1) / batch_size).unwrap()
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
        let mut input_dim = input.raw_dim();
        input_dim.as_array_view_mut()[0] = indices.len();

        let mut expected_dim = expected.raw_dim();
        expected_dim.as_array_view_mut()[0] = indices.len();

        let (_, _, input, expected) = unsafe {
            fill2(
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
            )
        };

        let input = ArrayView::from_shape(input_dim, input).unwrap();
        let expected = ArrayView::from_shape(expected_dim, expected).unwrap();
        (input, expected)
    }

    fn train_batch(
        &mut self,
        input: Arr<impl Data<Elem = F>, G::InputShape>,
        expected: Arr<impl Data<Elem = F>, G::OutputShape>,
        gradiants_buf: &mut Vec<F>,
        train_state_buf: &mut Vec<F>,
    ) -> F {
        debug_assert_eq!(
            input.shape()[0],
            expected.shape()[0],
            "input and expected batch sizes should be equal"
        );

        let (output, train_state) = self.batch_forward(input, train_state_buf);

        let d_output = self.cost.diff(output.view(), expected.view());

        let grads = self.batch_backward(d_output, train_state, gradiants_buf);

        self.optimiser.optimise(&mut self.model.data, grads);

        self.cost.cost(output, expected)
    }

    fn batch_forward<'a>(
        &mut self,
        input: Arr<impl Data<Elem = F>, G::InputShape>,
        train_state_buf: &'a mut Vec<F>,
    ) -> (OwnedArr<F, G::OutputShape>, &'a [F]) {
        // allocate space and get uninit slice
        let train_state_size = self.model.layer.train_state_size(input.shape()[0]);
        train_state_buf.clear();

        // # Safety
        // Train Layer forward should initialise every value
        unsafe {
            fill(train_state_buf, train_state_size, |train_state| {
                self.model
                    .layer
                    .forward(self.model.state(), input, train_state)
            })
        }
    }

    fn batch_backward<'a>(
        &mut self,
        d_output: Arr<impl Data<Elem = F>, G::OutputShape>,
        train_state: &[F],
        gradiants_buf: &'a mut Vec<F>,
    ) -> &'a [F] {
        // allocate space and get uninit state
        gradiants_buf.clear();

        // # Safety
        // Train Layer backward should initialise every value
        unsafe {
            fill(gradiants_buf, self.model.layer.size(), |d_state| {
                let d_state = self.model.layer.view_mut(d_state).unwrap();

                // feed model backward, storing grads in uninit state
                self.model
                    .layer
                    .backward(self.model.state(), d_state, train_state, d_output)
            })
            .1
        }
    }
}

/// # Safety
/// f should initialise every value in the slice for this to be safe
/// otherwise there will be live uninitialised data
unsafe fn fill<T, O>(
    buf: &mut Vec<T>,
    n: usize,
    f: impl FnOnce(&mut [MaybeUninit<T>]) -> O,
) -> (O, &[T]) {
    buf.reserve(n);
    let uninit = &mut buf.spare_capacity_mut()[..n];

    let output = f(uninit);

    let len = buf.len();
    buf.set_len(len + n);
    (output, &buf[len..])
}

/// # Safety
/// f should initialise every value in the slice for this to be safe
/// otherwise there will be live uninitialised data
unsafe fn fill2<T, O1, O2>(
    buf: &mut Vec<T>,
    n1: usize,
    n2: usize,
    f1: impl FnOnce(&mut [MaybeUninit<T>]) -> O1,
    f2: impl FnOnce(&mut [MaybeUninit<T>]) -> O2,
) -> (O1, O2, &[T], &[T]) {
    let ((o1, o2), s) = fill(buf, n1 + n2, |s| {
        let (s1, s2) = s.split_at_mut(n1);
        (f1(s1), f2(s2))
    });

    let (s1, s2) = s.split_at(n1);
    (o1, o2, s1, s2)
}
