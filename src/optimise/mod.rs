use crate::Scalar;

pub mod adam;
pub mod sgd;

pub trait Optimiser<F: Scalar> {
    fn init(&mut self, size: usize);
    fn optimise(&mut self, graph: &mut [F], grads: &[F]);
}
