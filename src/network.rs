use ndarray::Dimension;

use crate::{GraphBuilder, Layer};

impl<G1, G2> GraphBuilder for (G1, G2)
where
    G1: GraphBuilder,
    G2: GraphBuilder<InputShape = G1::OutputShape>,
{
    type InputShape = G1::InputShape;
    type OutputShape = G2::OutputShape;

    type Layer = (G1::Layer, G2::Layer);

    fn with_input_shape(self, input_shape: Self::InputShape) -> Self::Layer {
        let l1 = self.0.with_input_shape(input_shape);
        let l2 = self.1.with_input_shape(l1.output_shape());
        (l1, l2)
    }
}

impl<L1, L2> Layer for (L1, L2)
where
    L1: Layer,
    L2: Layer<InputShape = L1::OutputShape>,
{
    type InputShape = L1::InputShape;
    type OutputShape = L2::OutputShape;
    type State<S: ndarray::RawData> = (L1::State<S>, L2::State<S>);

    fn size(&self) -> usize {
        self.0.size() + self.1.size()
    }

    fn output_shape(&self) -> Self::OutputShape {
        self.1.output_shape()
    }

    fn view<'a, F>(
        &self,
        data: &'a [F],
    ) -> Result<Self::State<ndarray::ViewRepr<&'a F>>, ndarray::ShapeError> {
        let i = self.0.size();
        let data = data.split_at(i);
        Ok((self.0.view(data.0)?, self.1.view(data.1)?))
    }

    fn view_mut<'a, F>(
        &self,
        data: &'a mut [F],
    ) -> Result<Self::State<ndarray::ViewRepr<&'a mut F>>, ndarray::ShapeError> {
        let i = self.0.size();
        let data = data.split_at_mut(i);
        Ok((self.0.view_mut(data.0)?, self.1.view_mut(data.1)?))
    }

    fn apply<F: crate::Scalar>(
        &self,
        state: Self::State<ndarray::ViewRepr<&F>>,
        input: ndarray::ArrayBase<
            impl ndarray::Data<Elem = F>,
            <<Self as Layer>::InputShape as Dimension>::Larger,
        >,
    ) -> ndarray::ArrayBase<
        ndarray::OwnedRepr<F>,
        <<Self as Layer>::OutputShape as Dimension>::Larger,
    > {
        let inner = self.0.apply(state.0, input);
        self.1.apply(state.1, inner)
    }
}

/// Converts the provided values into a nested chain of tuples.
/// Works by taking each pair of expressions, converting them into a tuple,
/// Then pushing all of them into the macro recursively
///
/// ```
/// use linear_networks::net;
///
/// // These two expressions are the same
/// let a = net!(0, 1, 2, 3);
/// let b = net!((0, 1), (2, 3));
/// assert_eq!(a, b);
/// ```
///
/// There's an edge case to handle odd numbered inputs.
/// It leaves the first input and pairs up the rest of them
///
/// ```
/// use linear_networks::net;
///
/// let a = net!(0, 1, 2, 3, 4);
/// let b = net!(0, (1, 2), (3, 4));
/// let c = net!(0, ((1, 2), (3, 4)));
/// assert_eq!(a, b);
/// assert_eq!(a, c);
/// ```
#[macro_export]
macro_rules! net {
    ($g0:expr) => {
        $g0
    };
    ($($g0:expr, $g1:expr),*) => {
        $crate::net!($(
            ($g0, $g1)
        ),*)
    };
    ($g:expr, $($g0:expr, $g1:expr),*) => {
        $crate::net!(
            $g,
            $(
                ($g0, $g1)
            ),*
        )
    };
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_tuple_macro() {
        // single value
        let t = net!(0);
        assert_eq!(t, 0);

        // two values
        let t = net!(0, 1);
        assert_eq!(t, (0, 1));

        // 8 values (balanced nested binary tree)
        let t = net!(0, 1, 2, 3, 4, 5, 6, 7);
        assert_eq!(t, (((0, 1), (2, 3)), ((4, 5), (6, 7))));

        // 7 values (off balance nested binary tree)
        let t = net!(0, 1, 2, 3, 4, 5, 6);
        assert_eq!(t, ((0, (1, 2)), ((3, 4), (5, 6))));

        // 6 values (off balance nested binary tree)
        let t = net!(0, 1, 2, 3, 4, 5);
        assert_eq!(t, ((0, 1), ((2, 3), (4, 5))));
    }
}
