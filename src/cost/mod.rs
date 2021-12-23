use crate::{Arr, OwnedArr, Scalar};
use ndarray::{Data, Dimension};

pub mod mse;

pub trait Cost<D: Dimension> {
    fn cost<F: Scalar>(
        &self,
        output: Arr<impl Data<Elem = F>, D>,
        expected: Arr<impl Data<Elem = F>, D>,
    ) -> F;
    fn diff<F: Scalar>(
        &self,
        output: Arr<impl Data<Elem = F>, D>,
        expected: Arr<impl Data<Elem = F>, D>,
    ) -> OwnedArr<F, D>;
}
