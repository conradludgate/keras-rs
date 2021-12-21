use std::ops::{Div, Mul};

use ndarray::{Dimension, Ix1};

use super::{Activation, ActivationTrain};

#[derive(Debug, Copy, Clone)]
pub struct Relu;
impl Activation for Relu {
    type Shape = Ix1;

    fn apply<F: crate::Scalar>(
        input: ndarray::ArrayBase<
            impl ndarray::Data<Elem = F>,
            <<Self as Activation>::Shape as Dimension>::Larger,
        >,
    ) -> ndarray::ArrayBase<ndarray::OwnedRepr<F>, <<Self as Activation>::Shape as Dimension>::Larger>
    {
        let zero = F::zero();
        input.mapv(|x| x.max(zero))
    }
}

impl ActivationTrain for Relu {
    fn backward<F: crate::Scalar>(
        input: ndarray::ArrayBase<
            impl ndarray::Data<Elem = F>,
            <<Self as Activation>::Shape as Dimension>::Larger,
        >,
        output: ndarray::ArrayBase<
            impl ndarray::Data<Elem = F>,
            <<Self as Activation>::Shape as Dimension>::Larger,
        >,
        d_output: ndarray::ArrayBase<
            impl ndarray::Data<Elem = F>,
            <<Self as Activation>::Shape as Dimension>::Larger,
        >,
    ) -> ndarray::ArrayBase<ndarray::OwnedRepr<F>, <<Self as Activation>::Shape as Dimension>::Larger>
    {
        output.div(input.into_owned()).mul(d_output)
    }
}
