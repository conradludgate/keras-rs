use crate::Scalar;

use super::Cost;
use ndarray::{Ix1, Ix2};

#[derive(Debug, Copy, Clone)]
/// Mean Squared Error cost function.
pub struct MSE;

impl Cost<Ix1> for MSE {
    fn cost<F: Scalar>(
        &self,
        output: ndarray::ArrayBase<impl ndarray::Data<Elem = F>, Ix2>,
        expected: ndarray::ArrayBase<impl ndarray::Data<Elem = F>, Ix2>,
    ) -> F {
        let diff = output.into_owned() - expected;
        diff.t().dot(&diff).mean().unwrap()
    }

    fn diff<F: Scalar>(
        &self,
        output: ndarray::ArrayBase<impl ndarray::Data<Elem = F>, Ix2>,
        expected: ndarray::ArrayBase<impl ndarray::Data<Elem = F>, Ix2>,
    ) -> ndarray::ArrayBase<ndarray::OwnedRepr<F>, Ix2> {
        let one = F::one();
        let two = one + one;
        (output.into_owned() - expected) * two
    }
}
