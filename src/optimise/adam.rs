use crate::Scalar;

use super::Optimiser;

#[derive(Debug, Clone)]
pub struct Adam<F> {
    alpha: F,
    beta1: F,
    beta2: F,
    epsilon: F,
    m: Vec<F>,
    v: Vec<F>,
    t: i32,
}

impl<F> Adam<F> {
    pub fn new(alpha: F, beta1: F, beta2: F, epsilon: F) -> Self {
        Self {
            alpha,
            beta1,
            beta2,
            epsilon,
            m: vec![],
            v: vec![],
            t: 0,
        }
    }
}

impl<F: Scalar> Optimiser<F> for Adam<F> {
    fn init(&mut self, size: usize) {
        let zero = F::zero();
        self.m = vec![zero; size];
        self.v = vec![zero; size];
    }

    fn optimise(&mut self, graph: &mut [F], grads: &[F]) {
        // Algorithm defined on Page 2 of https://arxiv.org/pdf/1412.6980v9.pdf
        // https://mlfromscratch.com/optimizers-explained/#actually-explaining-adam

        self.t += 1;

        let b1 = self.beta1;
        let b2 = self.beta2;
        let e = self.epsilon;
        let a = self.alpha;

        let one = F::one();

        for i in 0..graph.len() {
            self.m[i] = self.m[i] * b1 + grads[i] * (one - b1);
            self.v[i] = self.v[i] * b2 + grads[i].powi(2) * (one - b2);

            let mb = self.m[i] / (one - b1.powi(self.t));
            let vb = self.v[i] / (one - b1.powi(self.t));
            graph[i] = graph[i] - a * mb / (vb.sqrt() + e);
        }
    }
}
