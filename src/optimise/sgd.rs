use super::Optimiser;
use crate::Scalar;

#[derive(Debug, Copy, Clone)]
pub struct SGD<F>(F);

impl<F> SGD<F> {
    pub const fn new(alpha: F) -> Self {
        Self(alpha)
    }
}

impl<F: Scalar> Optimiser<F> for SGD<F> {
    fn init(&mut self, _size: usize) {}

    fn optimise(&mut self, graph: &mut [F], grads: &[F]) {
        for (theta, &g) in graph.iter_mut().zip(grads) {
            *theta = *theta - self.0 * g;
        }
    }
}
