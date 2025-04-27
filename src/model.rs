use ndarray::{ArrayView, ArrayView2, ArrayViewMut, Axis, Dimension, Ix1};
use rand::{prelude::SliceRandom, thread_rng};
use rand_distr::num_traits::Float;

use crate::{
    cost::Cost, optimise::Optimiser, Backprop, BackpropShape, BatchShape, Batched, Input, Output, Scalar, Shape, View
};

pub struct Model<F: Scalar, G: BackpropShape> {
    pub(crate) input: G::Input,
    pub(crate) layer: G,
    pub(crate) params: Vec<F>,
    pub(crate) stack: Vec<F>,
}

impl<F: Scalar, G: Backprop<F>> Model<F, G> {
    pub fn apply_batch<'a>(
        &'a mut self,
        batch_size: usize,
        input: Input<G, &F>,
    ) -> View<Batched<G::Output>, &'a mut F> {
        let stack_buf = &mut self.stack;

        let shape = self.layer.shape(self.input);
        let cache_shape = shape.cache.batched(batch_size);
        let output_shape = shape.output.batched(batch_size);

        // stack:
        // 0. output
        // 1. cache (todo: remove)
        // 2. rest

        let stack0 = output_shape.size();
        let stack1 = cache_shape.size();
        let stack2 = shape.stack_size * batch_size;
        let stack_size = stack0 + stack1 + stack2;
        stack_buf.resize(stack_size, F::zero());

        let (output_buf, stack) = stack_buf.split_at_mut(stack0);
        let (cache_buf, stack) = stack.split_at_mut(stack1);

        let params = shape.params.from_slice(&*self.params);
        let output = output_shape.from_slice(&mut *output_buf);
        let cache = cache_shape.from_slice(&mut *cache_buf);

        self.layer.forward(
            self.input,
            batch_size,
            params,
            input,
            output,
            cache,
            &mut *stack,
        );

        output_shape.from_slice(&mut *output_buf)
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

pub struct Trainer<F: Scalar, G: Backprop<F>, O: Optimiser<F>, C: Cost<G::Output>> {
    pub model: Model<F, G>,
    pub optimiser: O,
    pub cost: C,
    pub regularisation: Option<Regularisation<F>>,
}

pub struct ModelTrainer<F: Scalar, G: Backprop<F>, O: Optimiser<F>, C: Cost<G::Output>> {
    trainer: Trainer<F, G, O, C>,
    bufs: Bufs<F>,
}

struct Bufs<F> {
    data: Vec<F>,
    gradiants: Vec<F>,
}

impl<F> Default for Bufs<F> {
    fn default() -> Self {
        Self {
            data: vec![],
            gradiants: vec![],
        }
    }
}

impl<F: Scalar, G: Backprop<F>, O: Optimiser<F>, C: Cost<G::Output>> ModelTrainer<F, G, O, C>
where
    G: BackpropShape<Input = Ix1, Output = Ix1>,
{
    pub fn train_epoch(
        &mut self,
        input: Input<G, &F>,
        expected: Output<G, &F>,
        batch_size: usize,
    ) -> F {
        self.trainer
            .train_epoch(input, expected, batch_size, &mut self.bufs)
    }

    pub fn test_epoch(
        &mut self,
        input: Input<G, &F>,
        expected: Output<G, &F>,
    ) -> F {
        self.trainer.test_epoch(input, expected, &mut self.bufs)
    }

    pub fn into_inner(self) -> Trainer<F, G, O, C> {
        self.trainer
    }

    pub fn as_model(&mut self) -> &mut Model<F, G> {
        &mut self.trainer.model
    }
}

impl<F: Scalar, G: Backprop<F>, O: Optimiser<F>, C: Cost<G::Output>> Trainer<F, G, O, C>
where
    G: BackpropShape<Input = Ix1, Output = Ix1>,
{
    fn train_epoch(
        &mut self,
        input: ArrayView2<F>,
        expected: ArrayView2<F>,
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
            cost = cost + self.train_batch(batch_size, input, expected, &mut bufs.gradiants);
        }

        cost / F::from_usize((total_inputs + batch_size - 1) / batch_size).unwrap()
    }

    fn test_epoch(
        &mut self,
        input: ArrayView2<F>,
        expected: ArrayView2<F>,
        bufs: &mut Bufs<F>,
    ) -> F {
        let total_inputs = input.raw_dim()[0];

        let mut cost = F::zero();
        for indices in 0..total_inputs {
            let (input, expected) =
                self.select_batch(input.view(), expected.view(), &[indices], &mut bufs.data);
            cost = cost + self.test_batch(1, input, expected);
        }

        cost / F::from_usize(total_inputs).unwrap()
    }

    fn select_batch<'a>(
        &mut self,
        input: ArrayView2<F>,
        expected: ArrayView2<F>,
        indices: &[usize],
        data_buf: &'a mut Vec<F>,
    ) -> (ArrayView2<'a, F>, ArrayView2<'a, F>) {
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
}

impl<F: Scalar, G: Backprop<F>, O: Optimiser<F>, C: Cost<G::Output>> Trainer<F, G, O, C> {
    pub fn build(mut self) -> ModelTrainer<F, G, O, C> {
        self.optimiser
            .init(self.model.layer.shape(self.model.input).params.size());
        ModelTrainer {
            bufs: Bufs {
                data: Vec::new(),
                gradiants: self.model.params.clone(),
            },
            trainer: self,
        }
    }

    fn test_batch(
        &mut self,
        batch_size: usize,
        input: Input<G, &F>,
        expected: Output<G, &F>,
    ) -> F {
        let stack_buf = &mut self.model.stack;

        let shape = self.model.layer.shape(self.model.input);
        let output_shape = shape.output.batched(batch_size);
        let cache_shape = shape.cache.batched(batch_size);

        // stack:
        // 0. doutput
        // 1. cache (todo: remove)
        // 2. output
        // 3. rest

        let stack0 = output_shape.size();
        let stack1 = cache_shape.size();
        let stack2 = output_shape.size();
        let stack3 = shape.stack_size * batch_size;
        let stack_size = stack0 + stack1 + stack2 + stack3;
        stack_buf.resize(stack_size, F::zero());

        let (doutput_buf, stack) = stack_buf.split_at_mut(stack0);
        let (cache_buf, stack) = stack.split_at_mut(stack1);
        let (output_buf, stack) = stack.split_at_mut(stack2);

        // forward
        {
            let params = shape.params.from_slice(&*self.model.params);
            let output = output_shape.from_slice(&mut output_buf[..output_shape.size()]);
            // todo: remove
            let cache = cache_shape.from_slice(&mut *cache_buf);

            self.model.layer.forward(
                self.model.input,
                batch_size,
                params,
                input,
                output,
                cache,
                stack,
            );
        }

        // eval
        let cost = {
            let doutput = output_shape.from_slice(&mut *doutput_buf);
            let output = output_shape.from_slice(&output_buf[..output_shape.size()]);
            self.cost.diff(output, expected, doutput)
        };
        cost
    }

    fn train_batch(
        &mut self,
        batch_size: usize,
        input: Input<G, &F>,
        expected: Output<G, &F>,
        grads: &mut [F],
    ) -> F {
        let stack_buf = &mut self.model.stack;

        let shape = self.model.layer.shape(self.model.input);
        let input_shape = self.model.input.batched(batch_size);
        let output_shape = shape.output.batched(batch_size);
        let cache_shape = shape.cache.batched(batch_size);

        // stack:
        // 0. doutput
        // 1. cache
        // 2. output / dinput
        // 3. rest

        let stack0 = output_shape.size();
        let stack1 = cache_shape.size();
        let stack2 = usize::max(output_shape.size(), input_shape.size());
        let stack3 = shape.stack_size * batch_size;
        let stack_size = stack0 + stack1 + stack2 + stack3;
        stack_buf.resize(stack_size, F::zero());

        let (doutput_buf, stack) = stack_buf.split_at_mut(stack0);
        let (cache_buf, stack) = stack.split_at_mut(stack1);
        let (output_buf, stack) = stack.split_at_mut(stack2);

        // forward
        {
            let params = shape.params.from_slice(&*self.model.params);
            let output = output_shape.from_slice(&mut output_buf[..output_shape.size()]);
            let cache = cache_shape.from_slice(&mut *cache_buf);

            self.model.layer.forward(
                self.model.input,
                batch_size,
                params,
                input,
                output,
                cache,
                stack,
            );
        }

        // eval
        let cost = {
            let doutput = output_shape.from_slice(&mut *doutput_buf);
            let output = output_shape.from_slice(&output_buf[..output_shape.size()]);
            self.cost.diff(output, expected, doutput)
        };

        // backwards
        {
            let params = shape.params.from_slice(&*self.model.params);
            let cache = cache_shape.from_slice(&*cache_buf);
            let doutput = output_shape.from_slice(&*doutput_buf);
            let dinput = input_shape.from_slice(&mut output_buf[..input_shape.size()]);
            let dparams = shape.params.from_slice(&mut *grads);

            self.model.layer.backward(
                self.model.input,
                batch_size,
                params,
                doutput,
                dinput,
                dparams,
                cache,
                stack,
            );
        }

        if let Some(r) = self.regularisation {
            r.apply(grads, &self.model.params);
        }

        self.optimiser.optimise(&mut self.model.params, grads);
        cost
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
