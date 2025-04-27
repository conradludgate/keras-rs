use crate::{BatchShape, Batched, Scalar, View};

pub mod mse;

pub trait Cost<D: BatchShape> {
    fn cost<F: Scalar>(&self, output: View<Batched<D>, &F>, expected: View<Batched<D>, &F>) -> F;

    fn diff<F: Scalar>(
        &self,
        output: View<Batched<D>, &F>,
        expected: View<Batched<D>, &F>,
        diff: View<Batched<D>, &mut F>,
    ) -> F;
}
