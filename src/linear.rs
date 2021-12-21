use ndarray::{
    ArrayBase, ArrayView, ArrayViewMut, Axis, Data, Dimension, Ix1, Ix2, OwnedRepr, RawData,
    ShapeError, ViewRepr,
};

use crate::{
    array::{compact_front, dot_front, dot_inner},
    GraphBuilder, Layer, Scalar, TrainableLayer,
};

pub struct LinearBuilder {
    output_shape: Ix1,
}

impl GraphBuilder for LinearBuilder {
    type InputShape = Ix1;
    type OutputShape = Ix1;

    type Layer = LinearLayer;

    fn with_input_shape(self, input_shape: Self::InputShape) -> Self::Layer {
        let Self { output_shape } = self;
        LinearLayer {
            output_shape,
            input_shape,
        }
    }
}

pub struct LinearLayer {
    input_shape: Ix1,
    output_shape: Ix1,
}

impl Layer for LinearLayer {
    type InputShape = Ix1;
    type OutputShape = Ix1;

    type State<S: RawData> = LinearState<S>;

    fn size(&self) -> usize {
        let i = self.input_shape.into_pattern();
        let o = self.output_shape.into_pattern();
        (i + 1) * o
    }

    fn output_shape(&self) -> Self::OutputShape {
        self.output_shape
    }

    fn view<'a, F>(&self, data: &'a [F]) -> Result<Self::State<ViewRepr<&'a F>>, ShapeError> {
        let i = self.input_shape.into_pattern();
        let o = self.output_shape.into_pattern();

        let (weights, biases) = data.split_at(i * o);
        let weights = ArrayView::from_shape([i, o], weights)?;
        let biases = ArrayView::from_shape(o, biases)?;
        Ok(LinearState { weights, biases })
    }

    fn view_mut<'a, F>(
        &self,
        data: &'a mut [F],
    ) -> Result<Self::State<ViewRepr<&'a mut F>>, ShapeError> {
        let i = self.input_shape.into_pattern();
        let o = self.output_shape.into_pattern();

        let (weights, biases) = data.split_at_mut(i * o);
        let weights = ArrayViewMut::from_shape([i, o], weights)?;
        let biases = ArrayViewMut::from_shape(o, biases)?;
        Ok(LinearState { weights, biases })
    }

    fn apply<F: Scalar>(
        &self,
        state: Self::State<ViewRepr<&F>>,
        input: ArrayBase<impl Data<Elem = F>, <<Self as Layer>::InputShape as Dimension>::Larger>,
    ) -> ArrayBase<OwnedRepr<F>, <<Self as Layer>::OutputShape as Dimension>::Larger> {
        state.weights.dot(&input) + state.biases
    }
}

impl TrainableLayer for LinearLayer {
    fn backward<F: Scalar>(
        &self,
        state: Self::State<ViewRepr<&F>>,
        d_state: Self::State<ViewRepr<&mut F>>,
        input: ArrayBase<impl Data<Elem = F>, <<Self as Layer>::InputShape as Dimension>::Larger>,
        _output: ArrayBase<
            impl Data<Elem = F>,
            <<Self as Layer>::OutputShape as Dimension>::Larger,
        >,
        d_output: ArrayBase<
            impl Data<Elem = F>,
            <<Self as Layer>::OutputShape as Dimension>::Larger,
        >,
    ) -> ArrayBase<OwnedRepr<F>, <<Self as Layer>::InputShape as Dimension>::Larger> {
        dot_front(input, d_output.view()).assign_to(d_state.weights);
        compact_front(d_output.view())
            .mean_axis(Axis(0))
            .unwrap()
            .assign_to(d_state.biases);

        dot_inner(d_output, &state.weights.t())
    }
}

pub struct LinearState<S: RawData> {
    weights: ArrayBase<S, Ix2>,
    biases: ArrayBase<S, Ix1>,
}
