use crate::Scalar;

use super::Cost;
use ndarray::{ArrayView2, ArrayViewMut2, Ix1};

#[derive(Debug, Copy, Clone)]
/// Mean Squared Error cost function.
pub struct MSE;

impl Cost<Ix1> for MSE {
    fn cost<F: Scalar>(&self, output: ArrayView2<F>, expected: ArrayView2<F>) -> F {
        let (b, o) = output.dim();
        let size = F::from_usize(b * o).unwrap();
        let mut sum = F::zero();

        ndarray::azip!((output in output, expected in expected) {
            let diff = *output - *expected;
            sum = sum + diff * diff;
        });

        sum / size
    }

    fn diff<F: Scalar>(
        &self,
        output: ArrayView2<F>,
        expected: ArrayView2<F>,
        diff: ArrayViewMut2<F>,
    ) -> F {
        debug_assert_eq!(output.shape(), expected.shape());
        debug_assert_eq!(output.shape(), diff.shape());
        let (b, o) = output.dim();
        let size = F::from_usize(b * o).unwrap();

        let one = F::one();
        let two = one + one;

        let mut sum = F::zero();
        ndarray::azip!((diff in diff, output in output, expected in expected) {
            let d = *output - *expected;
            sum = sum + d * d;
            *diff = d * two
        });

        sum / size
    }
}
