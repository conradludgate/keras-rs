use ndarray::{ArrayBase, Data, Dimension, OwnedRepr};

use crate::Scalar;

pub mod mse;

pub trait Cost<D: Dimension> {
    fn cost<F: Scalar>(
        &self,
        output: ArrayBase<impl Data<Elem = F>, D::Larger>,
        expected: ArrayBase<impl Data<Elem = F>, D::Larger>,
    ) -> F;
    fn diff<F: Scalar>(
        &self,
        output: ArrayBase<impl Data<Elem = F>, D::Larger>,
        expected: ArrayBase<impl Data<Elem = F>, D::Larger>,
    ) -> ArrayBase<OwnedRepr<F>, D::Larger>;
}
