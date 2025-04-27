use ndarray::RawData;
use rand::Rng;

use crate::{
    Backprop, BackpropShape, BackpropShapes, Batch, Batched, Initialise, Scalar, Shape, Slice, View,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Net<B0, B1>(pub B0, pub B1);

impl<S0, S1> Shape for Net<S0, S1>
where
    S0: Shape,
    S1: Shape,
{
    type Base<S: RawData> = Net<S0::Base<S>, S1::Base<S>>;

    #[inline]
    fn size(self) -> usize {
        self.0.size() + self.1.size()
    }

    #[inline]
    fn from_slice<S: Slice>(self, slice: S) -> Self::Base<S::Repr> {
        let mid = self.0.size();
        let (s0, s1) = slice.split(mid);
        Net(self.0.from_slice(s0), self.1.from_slice(s1))
    }
}

impl<F: Scalar, L0, L1> Initialise<F> for Net<L0, L1>
where
    L0: Initialise<F>,
    L1: Initialise<F, Input = L0::Output>,
{
    fn init(&self, rng: &mut impl Rng, state: View<Self::Params, &mut F>) {
        self.0.init(rng, state.0);
        self.1.init(rng, state.1);
    }
}

impl<S0, S1> Batch for Net<S0, S1>
where
    S0: Batch,
    S1: Batch,
{
    type Batched = Net<S0::Batched, S1::Batched>;

    #[inline]
    fn batched(self, batch: usize) -> Self::Batched {
        Net(self.0.batched(batch), self.1.batched(batch))
    }
}

impl<B0, B1> BackpropShape for Net<B0, B1>
where
    B0: BackpropShape,
    B1: BackpropShape<Input = B0::Output>,
{
    type Params = Net<B0::Params, B1::Params>;
    type Input = B0::Input;
    type Output = B1::Output;
    type Cache = Net<B0::Cache, B1::Cache>;

    fn shape(self, input: Self::Input) -> BackpropShapes<Self> {
        let shape0 = self.0.shape(input);
        let shape1 = self.1.shape(shape0.output);
        BackpropShapes {
            params: Net(shape0.params, shape1.params),
            output: shape1.output,
            cache: Net(shape0.cache, shape1.cache),
            stack_size: shape0.output.batched(1).size()
                + usize::max(shape0.stack_size, shape1.stack_size),
        }
    }
}

impl<F, B0, B1> Backprop<F> for Net<B0, B1>
where
    B0: Backprop<F>,
    B1: Backprop<F, Input = B0::Output>,
{
    fn forward(
        self,
        i: Self::Input,
        b: usize,
        p: View<Self::Params, &F>,
        in_: View<Batched<Self::Input>, &F>,
        out: View<Batched<Self::Output>, &mut F>,
        c: View<Batched<Self::Cache>, &mut F>,
        s: &mut [F],
    ) {
        let shape0 = self.0.shape(i);
        let m = shape0.output;
        let mid_shape = m.batched(b);
        let (mid_slice, s) = s.split_at_mut(mid_shape.size());

        let mid = mid_shape.from_slice(&mut *mid_slice);
        self.0.forward(i, b, p.0, in_, mid, c.0, s);

        let mid = mid_shape.from_slice(&*mid_slice);
        self.1.forward(m, b, p.1, mid, out, c.1, s);
    }

    fn backward(
        self,
        i: Self::Input,
        b: usize,
        p: View<Self::Params, &F>,
        dout: View<Batched<Self::Output>, &F>,
        din_: View<Batched<Self::Input>, &mut F>,
        dp: View<Self::Params, &mut F>,
        c: View<Batched<Self::Cache>, &F>,
        s: &mut [F],
    ) {
        let shape0 = self.0.shape(i);
        let m = shape0.output;
        let mid_shape = m.batched(b);
        let (mid_slice, s) = s.split_at_mut(mid_shape.size());

        let dmid = mid_shape.from_slice(&mut *mid_slice);
        self.1.backward(m, b, p.1, dout, dmid, dp.1, c.1, s);

        let dmid = mid_shape.from_slice(&*mid_slice);
        self.0.backward(i, b, p.0, dmid, din_, dp.0, c.0, s);
    }
}

/// Converts the provided values into a nested chain of tuples.
/// Works by taking each pair of expressions, converting them into a tuple,
/// Then pushing all of them into the macro recursively
///
/// ```
/// use keras_rs::net;
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
/// use keras_rs::net;
///
/// let a = net!(0, 1, 2, 3, 4);
/// let b = net!(0, (1, 2), (3, 4));
/// let c = net!(0, ((1, 2), (3, 4)));
/// assert_eq!(a, b);
/// assert_eq!(a, c);
/// ```
#[macro_export]
macro_rules! net {
    ($g0:expr $(,)?) => {
        $g0
    };
    ($($g0:expr, $g1:expr),* $(,)?) => {
        $crate::net!($(
            $crate::network::Net($g0, $g1)
        ),*)
    };
    ($g:expr, $($g0:expr, $g1:expr),* $(,)?) => {
        $crate::net!(
            $g,
            $(
                $crate::network::Net($g0, $g1)
            ),*
        )
    };
}

#[cfg(test)]
mod tests {
    use crate::network::Net;

    #[test]
    fn test_tuple_macro() {
        // single value
        let t = net!(0);
        assert_eq!(t, 0);

        // two values
        let t = net!(0, 1);
        assert_eq!(t, Net(0, 1));

        // 8 values (balanced nested binary tree)
        let t = net!(0, 1, 2, 3, 4, 5, 6, 7);
        assert_eq!(t, Net(Net(Net(0, 1), Net(2, 3)), Net(Net(4, 5), Net(6, 7))));

        // 7 values (off balance nested binary tree)
        let t = net!(0, 1, 2, 3, 4, 5, 6);
        assert_eq!(t, Net(Net(0, Net(1, 2)), Net(Net(3, 4), Net(5, 6))));

        // 6 values (off balance nested binary tree)
        let t = net!(0, 1, 2, 3, 4, 5);
        assert_eq!(t, Net(Net(0, 1), Net(Net(2, 3), Net(4, 5))));
    }
}
