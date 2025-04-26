use ndarray::{ArrayBase, Axis, Data, Dimension, IntoDimension, Ix1, Ix2, ViewRepr};

use crate::{Arr, ArrViewMut, GraphBuilder, Layer, Scalar, Slice};

pub struct Embedding {
    pub input_dim: usize,
    pub output_dim: usize,
}

impl GraphBuilder for Embedding {
    type InputShape = Ix1;
    type OutputShape = Ix2;
    type Layer = EmbeddingLayer;

    fn with_input_shape(self, input_shape: Self::InputShape) -> Self::Layer {
        let Self {
            input_dim,
            output_dim,
        } = self;
        let timesteps = input_shape.into_pattern();
        EmbeddingLayer {
            timesteps,
            input_dim,
            output_dim,
        }
    }
}

pub struct EmbeddingLayer {
    timesteps: usize,
    input_dim: usize,
    output_dim: usize,
}

impl Layer for EmbeddingLayer {
    type InputShape = Ix1;
    type OutputShape = Ix2;

    type State<S: ndarray::RawData> = ArrayBase<S, Ix2>;

    fn size(&self) -> usize {
        self.input_dim * self.output_dim
    }

    fn output_shape(&self) -> Self::OutputShape {
        [self.timesteps, self.output_dim].into_dimension()
    }

    fn batched_output_shape(&self, batch_size: usize) -> <Self::OutputShape as Dimension>::Larger {
        [batch_size, self.timesteps, self.output_dim].into_dimension()
    }

    fn stack_space(&self, _batch_size: usize) -> usize {
        0
    }

    fn view_state<S: Slice>(&self, data: S) -> Self::State<S::Repr> {
        data.into_array([self.input_dim, self.output_dim])
    }

    fn apply<F: Scalar>(
        &self,
        state: Self::State<ViewRepr<&F>>,
        input: Arr<impl Data<Elem = F>, Self::InputShape>,
        mut output: ArrViewMut<F, Self::OutputShape>,
        _stack: &mut [F],
    ) {
        let (batch_size, timesteps) = input.raw_dim().into_pattern();
        debug_assert_eq!(timesteps, self.timesteps);

        for i in 0..batch_size {
            let input = input.index_axis(Axis(0), i);
            let mut output = output.index_axis_mut(Axis(0), i);
            for j in 0..timesteps {
                let input = input[j].to_usize().unwrap();
                let output = output.index_axis_mut(Axis(0), j);
                state.index_axis(Axis(0), input).assign_to(output)
            }
        }
    }
}
