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

        data_buf.clear();
        data_buf.reserve(input_dim.size() + expected_dim.size());
        let uninit = &mut data_buf.spare_capacity_mut()[..input_dim.size() + expected_dim.size()];
        let (uninit_input, uninit_expected) = uninit.split_at_mut(input_dim.size());

        let mut uninit_input = ArrayViewMut::from_shape(input_dim.clone(), uninit_input).unwrap();
        let mut uninit_expected =
            ArrayViewMut::from_shape(expected_dim.clone(), uninit_expected).unwrap();

        for (i, &j) in indices.iter().enumerate() {
            input
                .index_axis(Axis(0), j)
                .assign_to(uninit_input.index_axis_mut(Axis(0), i));

            expected
                .index_axis(Axis(0), j)
                .assign_to(uninit_expected.index_axis_mut(Axis(0), i));
        }

        unsafe { data_buf.set_len(input_dim.size() + expected_dim.size()) }

        let (input, expected) = data_buf.split_at_mut(input_dim.size());
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
        train_state_buf.reserve(train_state_size);
        let train_state = &mut train_state_buf.spare_capacity_mut()[..train_state_size];

        // feed model forward, storing train state in uninit slice
        let output = self
            .model
            .layer
            .forward(self.model.state(), input, train_state);

        // # Safety
        // Train Layer forward should initialise every value
        unsafe {
            train_state_buf.set_len(train_state_size);
        }

        (output, train_state_buf.as_slice())
    }

    fn batch_backward<'a>(
        &mut self,
        d_output: Arr<impl Data<Elem = F>, G::OutputShape>,
        train_state: &[F],
        gradiants_buf: &'a mut Vec<F>,
    ) -> &'a [F] {
        // allocate space and get uninit state
        gradiants_buf.clear();
        gradiants_buf.reserve(self.model.layer.size());
        let d_state = &mut gradiants_buf.spare_capacity_mut()[..self.model.layer.size()];
        let d_state = self.model.layer.view_mut(d_state).unwrap();

        // feed model backward, storing grads in uninit state
        self.model
            .layer
            .backward(self.model.state(), d_state, train_state, d_output);

        // # Safety
        // Train Layer backward should initialise every value
        unsafe {
            gradiants_buf.set_len(self.model.layer.size());
        }

        gradiants_buf.as_slice()
    }
}
