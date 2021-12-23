use crate::{Arr, OwnedArr, Scalar};

use super::Cost;
use ndarray::{Data, Ix1};

#[derive(Debug, Copy, Clone)]
/// Mean Squared Error cost function.
pub struct MSE;

impl Cost<Ix1> for MSE {
    fn cost<F: Scalar>(
        &self,
        output: Arr<impl Data<Elem = F>, Ix1>,
        expected: Arr<impl Data<Elem = F>, Ix1>,
    ) -> F {
        let diff = output.into_owned() - expected;
        diff.t().dot(&diff).mean().unwrap()
    }

    fn diff<F: Scalar>(
        &self,
        output: Arr<impl Data<Elem = F>, Ix1>,
        expected: Arr<impl Data<Elem = F>, Ix1>,
    ) -> OwnedArr<F, Ix1> {
        let one = F::one();
        let two = one + one;
        (output.into_owned() - expected) * two
    }
}
