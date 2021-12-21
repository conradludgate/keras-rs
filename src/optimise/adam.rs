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

        // m_t = b1 * m_t-1 + (1 - b1) * g_t
        self.m.iter_mut().zip(grads).for_each(|(m, &g)| {
            *m = *m * b1 + g * (one - b1);
        });

        // v_t = b2 * v_t-1 + (1 - b2) * g_t^2
        self.v.iter_mut().zip(grads).for_each(|(v, &g)| {
            *v = *v * b2 + g.powi(2) * (one - b2);
        });

        // m_t' = m_t / (1 - b1^t)
        let mb = self.m.iter().map(|&m| m / (one - b1.powi(self.t)));

        // v_t' = v_t / (1 - b2^t)
        let vb = self.v.iter().map(|&v| v / (one - b2.powi(self.t)));

        // x_t = a * m_t' / (sqrt(v_t') + e)
        let mb = mb.zip(vb).map(|(m, v)| m * a / (v.sqrt() + e));

        // g_t = g_t-1 - x_t
        graph.iter_mut().zip(mb).for_each(|(g, m)| {
            *g = *g - m;
        });
    }
}
