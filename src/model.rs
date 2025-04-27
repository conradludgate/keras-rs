use ndarray::Ix1;
use rand::{prelude::SliceRandom, thread_rng};

use crate::{
    cost::Cost, optimise::Optimiser, Backprop, Batch, BatchShape, Inference, Output, Scalar, Shape,
    View,
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

        let shape = self.layer.shape(self.input);
        let output_shape = shape.output.batched(batch_size);

        // stack:
        // 0. output
        // 1. rest

        let stack0 = output_shape.size();
        let stack1 = shape.stack_size * batch_size;
        let stack_size = stack0 + stack1;
        stack_buf.resize(stack_size, F::zero());

        let (output_buf, stack) = stack_buf.split_at_mut(stack0);

        let params = shape.params.from_slice(&*self.params);
        let output = output_shape.from_slice(&mut *output_buf);

        self.layer
            .infer(self.input, batch_size, params, input, output, &mut *stack);

        output_shape.from_slice(&mut *output_buf)
    }

    pub fn apply_single<'a>(&'a mut self, input: View<I, &F>) -> View<G::Output, &'a mut F> {
        let shape = self.layer.shape(self.input);
        let output = self.apply_batch(1, self.input.from_single(input));
        shape.output.get_single(output, 0)
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
            .init(self.model.layer.shape(self.model.input).params.size());
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
        let shape = self.model.layer.shape(self.model.input);
        let total_inputs = self.model.input.batches(&input);
        let input_shape = self.model.input.batched(total_inputs);
        let output_shape = shape.output.batched(total_inputs);

        let mut rng = thread_rng();
        let mut indices: Vec<_> = (0..total_inputs).collect();
        indices.shuffle(&mut rng);

        let mut cost = F::zero();
        for indices in indices.chunks(batch_size) {
            let input = input_shape.as_ref(input);
            let expected = output_shape.as_ref(expected);

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
        let stack_buf = &mut self.model.stack;

        let (shape, cache_shape) = self.model.layer.shape_with_cache(self.model.input);
        let input_shape = self.model.input.batched(batch_size);
        let output_shape = shape.output.batched(batch_size);
        let cache_shape = cache_shape.batched(batch_size);

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
        let shape = self.model.layer.shape(self.model.input);
        let total_inputs = self.model.input.batches(&input);
        let input_shape = self.model.input.batched(total_inputs);
        let output_shape = shape.output.batched(total_inputs);

        let mut cost = F::zero();
        for indices in 0..total_inputs {
            let input = input_shape.as_ref(input);
            let expected = output_shape.as_ref(expected);

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
        let stack_buf = &mut self.model.stack;

        let shape = self.model.layer.shape(self.model.input);
        let output_shape = shape.output.batched(batch_size);

        // stack:
        // 0. doutput
        // 1. output
        // 2. rest

        let stack0 = output_shape.size();
        let stack1 = output_shape.size();
        let stack2 = shape.stack_size * batch_size;
        let stack_size = stack0 + stack1 + stack2;
        stack_buf.resize(stack_size, F::zero());

        let (doutput_buf, stack) = stack_buf.split_at_mut(stack0);
        let (output_buf, stack) = stack.split_at_mut(stack1);

        // infer
        {
            let params = shape.params.from_slice(&*self.model.params);
            let output = output_shape.from_slice(&mut output_buf[..output_shape.size()]);

            self.model
                .layer
                .infer(self.model.input, batch_size, params, input, output, stack);
        }

        // eval
        let cost = {
            let doutput = output_shape.from_slice(&mut *doutput_buf);
            let output = output_shape.from_slice(&output_buf[..output_shape.size()]);
            self.cost.diff(output, expected, doutput)
        };
        cost
    }

    fn select_batch<'a>(
        &mut self,
        input: Batch<I, &F>,
        expected: Output<I, G, &F>,
        indices: &[usize],
        data_buf: &'a mut Vec<F>,
    ) -> (Batch<I, &'a F>, Output<I, G, &'a F>) {
        data_buf.clear();

        let shape = self.model.layer.shape(self.model.input);
        let output_shape = shape.output.batched(indices.len());
        let input_shape = self.model.input.batched(indices.len());

        let input_size = input_shape.size();
        let output_size = output_shape.size();
        let total_size = input_size + output_size;
        data_buf.resize(total_size, F::zero());

        let (input_buf, output_buf) = data_buf.split_at_mut(input_size);

        {
            let mut dst_input = input_shape.from_slice(&mut *input_buf);
            let mut dst_output = output_shape.from_slice(&mut *output_buf);

            for (i, &index) in indices.iter().enumerate() {
                let dst_input = input_shape.as_mut(&mut dst_input);
                let input = input_shape.as_ref(&input);
                let dst = self.model.input.get_single(dst_input, i);
                let src = self.model.input.get_single(input, index);
                self.model.input.assign_to(dst, src);

                let dst_output = output_shape.as_mut(&mut dst_output);
                let expected = output_shape.as_ref(&expected);
                let dst = shape.output.get_single(dst_output, i);
                let src = shape.output.get_single(expected, index);
                shape.output.assign_to(dst, src);
            }
        }

        let dst_input = input_shape.from_slice(&*input_buf);
        let dst_output = output_shape.from_slice(&*output_buf);

        (dst_input, dst_output)
    }
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
