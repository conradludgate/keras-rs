use crate::Scalar;

use super::Cost;
use ndarray::{ArrayView2, ArrayViewMut2, Ix1};

#[derive(Debug, Copy, Clone)]
/// Mean Squared Error cost function.
pub struct MSE;

impl Cost<Ix1> for MSE {
    fn cost<F: Scalar>(&self, output: ArrayView2<F>, expected: ArrayView2<F>) -> F {
        let diff = output.into_owned() - expected;
        diff.t().dot(&diff).mean().unwrap()
    }

    fn diff<F: Scalar>(
        &self,
        output: ArrayView2<F>,
        expected: ArrayView2<F>,
        mut diff: ArrayViewMut2<F>,
    ) -> F {
        debug_assert_eq!(output.shape(), expected.shape());
        debug_assert_eq!(output.shape(), diff.shape());
        let (b, o) = output.dim();
        let size = F::from_usize(b * o).unwrap();

        let one = F::one();
        let two = one + one;
        output.assign_to(&mut diff);

        let mut sum = F::zero();

        diff.zip_mut_with(&expected, |output, expected| {
            let diff = *output - *expected;
            sum = sum + diff * diff;
            *output = diff * two
        });

        sum / size
    }
}
