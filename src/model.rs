use ndarray::Ix1;
use rand::{prelude::SliceRandom, thread_rng};

use crate::{
    cost::Cost, optimise::Optimiser, Array, Backprop, Batch, BatchShape, Inference, Output, Scalar,
    Shape, Stack, View,
};

pub struct Model<F: Scalar, I, G> {
    pub(crate) input: I,
    pub(crate) layer: G,
    pub(crate) params: Vec<F>,
    pub(crate) stack: Vec<F>,
}

impl<F: Scalar, I: BatchShape, G: Inference<I, F>> Model<F, I, G> {
    pub fn apply_batch<'a>(
        &'a mut self,
        batch_size: usize,
        input: Batch<I, &F>,
    ) -> Output<I, G, &'a mut F> {
        let stack_buf = &mut self.stack;

        let (params_shape, output_shape) = self.layer.shape(self.input);
        let stack_size = self.layer.stack(batch_size, self.input);
        let output_shape = output_shape.batched(batch_size);

        // stack:
        // 0. output
        // 1. rest

        let stack0 = output_shape.size();
        let stack1 = stack_size;
        let stack_size = stack0 + stack1;
        let stack = Stack::new(stack_buf, stack_size);

        let (mut output, stack) = stack.take(output_shape);

        let params = params_shape.from_slice(&*self.params);

        self.layer.infer(
            self.input,
            batch_size,
            params,
            input,
            output.as_mut(),
            stack,
        );

        output
    }

    pub fn apply_single<'a>(&'a mut self, input: View<I, &F>) -> View<G::Output, &'a mut F> {
        let (_, output_shape) = self.layer.shape(self.input);
        let output = self.apply_batch(1, self.input.from_single(input));
        output_shape.get_single(output, 0)
    }
}

pub struct Trainer<F: Scalar, I, G, O, C> {
    pub model: Model<F, I, G>,
    pub optimiser: O,
    pub cost: C,
    pub regularisation: Option<Regularisation<F>>,
}

pub struct ModelTrainer<F: Scalar, I, G, O, C> {
    trainer: Trainer<F, I, G, O, C>,
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

impl<F, I, G, O, C> ModelTrainer<F, I, G, O, C>
where
    F: Scalar,
    I: BatchShape,
    G: Backprop<I, F>,
    O: Optimiser<F>,
    C: Cost<G::Output>,
{
    pub fn train_epoch(
        &mut self,
        input: &Batch<I, &F>,
        expected: &Output<I, G, &F>,
        batch_size: usize,
    ) -> F {
        self.trainer
            .train_epoch(input, expected, batch_size, &mut self.bufs)
    }
}

impl<F, I, G, O, C> ModelTrainer<F, I, G, O, C>
where
    F: Scalar,
    I: BatchShape,
    G: Inference<I, F>,
    O: Optimiser<F>,
    C: Cost<G::Output>,
{
    pub fn test_epoch(&mut self, input: &Batch<I, &F>, expected: &Output<I, G, &F>) -> F {
        self.trainer.test_epoch(input, expected, &mut self.bufs)
    }

    pub fn into_inner(self) -> Trainer<F, I, G, O, C> {
        self.trainer
    }

    pub fn as_model(&mut self) -> &mut Model<F, I, G> {
        &mut self.trainer.model
    }
}

impl<F, G, O, C> Trainer<F, Ix1, G, O, C>
where
    F: Scalar,
    G: Backprop<Ix1, F, Output = Ix1>,
    O: Optimiser<F>,
    C: Cost<G::Output>,
{
}

impl<F, I, G, O, C> Trainer<F, I, G, O, C>
where
    F: Scalar,
    I: BatchShape,
    G: Backprop<I, F>,
    O: Optimiser<F>,
    C: Cost<G::Output>,
{
    pub fn build(mut self) -> ModelTrainer<F, I, G, O, C> {
        self.optimiser
            .init(self.model.layer.shape(self.model.input).0.size());
        ModelTrainer {
            bufs: Bufs {
                data: Vec::new(),
                gradiants: self.model.params.clone(),
            },
            trainer: self,
        }
    }

    fn train_epoch(
        &mut self,
        input: &Batch<I, &F>,
        expected: &Output<I, G, &F>,
        batch_size: usize,
        bufs: &mut Bufs<F>,
    ) -> F {
        let model = &mut self.model;
        let total_inputs = model.input.batches(&input);

        let mut rng = thread_rng();
        let mut indices: Vec<_> = (0..total_inputs).collect();
        indices.shuffle(&mut rng);

        let mut cost = F::zero();
        for indices in indices.chunks(batch_size) {
            let input = input.as_ref();
            let expected = expected.as_ref();

            let (input, expected) = self.select_batch(input, expected, indices, &mut bufs.data);
            cost = cost + self.train_batch(batch_size, input, expected, &mut bufs.gradiants);
        }

        cost / F::from_usize((total_inputs + batch_size - 1) / batch_size).unwrap()
    }

    fn train_batch(
        &mut self,
        batch_size: usize,
        input: Batch<I, &F>,
        expected: Output<I, G, &F>,
        grads: &mut [F],
    ) -> F {
        let Self {
            model,
            optimiser,
            cost,
            regularisation,
        } = self;

        let stack_buf = &mut model.stack;

        let (params_shape, output_shape) = model.layer.shape(model.input);
        let (cache_shape, stack_size) = model.layer.backprop_shape(batch_size, model.input);
        let total_inputs = model.input.batches(&input);
        let input_shape = model.input.batched(total_inputs);
        let output_shape = output_shape.batched(total_inputs);

        // stack:
        // 0. doutput
        // 1. cache
        // 2. output / dinput
        // 3. rest

        let stack0 = output_shape.size();
        let stack1 = cache_shape.size();
        let stack2 = usize::max(output_shape.size(), input_shape.size());
        let stack3 = stack_size;
        let stack_size = stack0 + stack1 + stack2 + stack3;
        let stack = Stack::new(stack_buf, stack_size);

        let (mut doutput, stack) = stack.take(output_shape);
        let (mut cache, mut stack) = stack.take(cache_shape);

        let cost = {
            let params = params_shape.from_slice(&*model.params);
            let cache = cache.as_mut();
            let doutput = doutput.as_mut();
            let (mut output, stack) = stack.as_mut().take(output_shape);

            model.layer.forward(
                model.input,
                batch_size,
                params,
                input,
                output.as_mut(),
                cache,
                stack,
            );

            let output = output.as_ref();
            cost.diff(output, expected, doutput)
        };

        // backwards
        {
            let params = params_shape.from_slice(&*model.params);
            // let cache = cache_shape.from_slice(&*cache_buf);
            // let doutput = output_shape.from_slice(&*doutput_buf);
            // let dinput = input_shape.from_slice(&mut output_buf[..input_shape.size()]);
            let cache = cache.as_ref();
            let doutput = doutput.as_ref();
            let (dinput, stack) = stack.take(input_shape);
            let dparams = params_shape.from_slice(&mut *grads);

            model.layer.backward(
                model.input,
                batch_size,
                params,
                doutput,
                dinput,
                dparams,
                cache,
                stack,
            );
        }

        if let Some(r) = regularisation {
            r.apply(grads, &model.params);
        }

        optimiser.optimise(&mut model.params, grads);
        cost
    }
}

impl<F, I, G, O, C> Trainer<F, I, G, O, C>
where
    F: Scalar,
    I: BatchShape,
    G: Inference<I, F>,
    O: Optimiser<F>,
    C: Cost<G::Output>,
{
    fn test_epoch(
        &mut self,
        input: &Batch<I, &F>,
        expected: &Output<I, G, &F>,
        bufs: &mut Bufs<F>,
    ) -> F {
        let model = &mut self.model;

        let total_inputs = model.input.batches(&input);

        let mut cost = F::zero();
        for indices in 0..total_inputs {
            let input = input.as_ref();
            let expected = expected.as_ref();

            let (input, expected) = self.select_batch(input, expected, &[indices], &mut bufs.data);
            cost = cost + self.test_batch(1, input, expected);
        }

        cost / F::from_usize(total_inputs).unwrap()
    }

    fn test_batch(
        &mut self,
        batch_size: usize,
        input: Batch<I, &F>,
        expected: Output<I, G, &F>,
    ) -> F {
        let Self { model, cost, .. } = self;

        let stack_buf = &mut model.stack;

        let (params_shape, output_shape) = model.layer.shape(model.input);
        let output_shape = output_shape.batched(batch_size);
        let stack_size = model.layer.stack(batch_size, model.input);

        // stack:
        // 0. doutput
        // 1. output
        // 2. rest

        let stack0 = output_shape.size();
        let stack1 = output_shape.size();
        let stack2 = stack_size;
        let stack_size = stack0 + stack1 + stack2;
        let stack = Stack::new(stack_buf, stack_size);

        let (doutput, mut stack) = stack.take(output_shape);

        let params = params_shape.from_slice(&*model.params);
        let (mut output, stack) = stack.as_mut().take(output_shape);

        model.layer.infer(
            model.input,
            batch_size,
            params,
            input,
            output.as_mut(),
            stack,
        );

        let output = output.as_ref();
        cost.diff(output, expected, doutput)
    }

    fn select_batch<'a>(
        &mut self,
        input: Batch<I, &F>,
        expected: Output<I, G, &F>,
        indices: &[usize],
        data_buf: &'a mut Vec<F>,
    ) -> (Batch<I, &'a F>, Output<I, G, &'a F>) {
        data_buf.clear();

        let (_, output_shape) = self.model.layer.shape(self.model.input);

        let input_size = self.model.input.batched_size(indices.len());
        let output_size = output_shape.batched_size(indices.len());
        let total_size = input_size + output_size;
        data_buf.resize(total_size, F::zero());

        let (input_buf, output_buf) = data_buf.split_at_mut(input_size);

        let inputs = fill(self.model.input, indices, input_buf, input);
        let outputs = fill(output_shape, indices, output_buf, expected);

        (inputs, outputs)
    }
}

fn fill<'a, F: Scalar, I: BatchShape>(
    size: I,
    indices: &[usize],
    space: &'a mut [F],
    from: Batch<I, &F>,
) -> Batch<I, &'a F> {
    let batch = size.batched(indices.len());

    {
        let mut space = batch.from_slice(&mut *space);

        for (i, &index) in indices.iter().enumerate() {
            let space = space.as_mut();
            let from = from.as_ref();
            let dst = size.get_single(space, i);
            let src = size.get_single(from, index);
            src.assign_to(dst);
        }
    }

    batch.from_slice(&*space)
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
