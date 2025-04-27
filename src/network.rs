use ndarray::RawData;
use rand::Rng;

use crate::{
    Backprop, BackpropShape, BatchShape, Batched, Inference, Initialise, ModelShape, ModelShapes,
    Scalar, Shape, Slice, View,
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

    #[inline]
    fn assign_to<F: Copy>(
        self,
        dst: Self::Base<ndarray::ViewRepr<&mut F>>,
        src: Self::Base<ndarray::ViewRepr<&F>>,
    ) {
        self.0.assign_to(dst.0, src.0);
        self.1.assign_to(dst.1, src.1);
    }

    #[inline]
    fn as_ref<'a, T>(
        self,
        s: &'a Self::Base<ndarray::ViewRepr<&T>>,
    ) -> Self::Base<ndarray::ViewRepr<&'a T>> {
        Net(self.0.as_ref(&s.0), self.1.as_ref(&s.1))
    }

    #[inline]
    fn as_mut<'a, T>(
        self,
        s: &'a mut Self::Base<ndarray::ViewRepr<&mut T>>,
    ) -> Self::Base<ndarray::ViewRepr<&'a mut T>> {
        Net(self.0.as_mut(&mut s.0), self.1.as_mut(&mut s.1))
    }
}

impl<F, I, L0, L1> Initialise<I, F> for Net<L0, L1>
where
    F: Scalar,
    I: BatchShape,
    L0: Initialise<I, F>,
    L1: Initialise<L0::Output, F>,
{
    #[inline]
    fn init(&self, rng: &mut impl Rng, state: View<Self::Params, &mut F>) {
        self.0.init(rng, state.0);
        self.1.init(rng, state.1);
    }
}

impl<S0, S1> BatchShape for Net<S0, S1>
where
    S0: BatchShape,
    S1: BatchShape,
{
    type Batched = Net<S0::Batched, S1::Batched>;

    #[inline]
    fn batched(self, batch: usize) -> Self::Batched {
        Net(self.0.batched(batch), self.1.batched(batch))
    }

    #[inline]
    fn from_single<R: RawData>(
        self,
        base: <Self as Shape>::Base<R>,
    ) -> <Batched<Self> as Shape>::Base<R> {
        Net(self.0.from_single(base.0), self.1.from_single(base.1))
    }

    #[inline]
    fn get_single<R: RawData>(
        self,
        base: <Batched<Self> as Shape>::Base<R>,
        index: usize,
    ) -> Self::Base<R> {
        Net(
            self.0.get_single(base.0, index),
            self.1.get_single(base.1, index),
        )
    }

    #[inline]
    fn batches<R: RawData>(self, base: &<Batched<Self> as Shape>::Base<R>) -> usize {
        self.0.batches(&base.0)
    }
}

impl<I, B0, B1> ModelShape<I> for Net<B0, B1>
where
    B0: ModelShape<I>,
    B1: ModelShape<B0::Output>,
{
    type Params = Net<B0::Params, B1::Params>;
    type Output = B1::Output;

    fn shape(self, input: I) -> ModelShapes<I, Self> {
        let shape0 = self.0.shape(input);
        let shape1 = self.1.shape(shape0.output);
        ModelShapes {
            params: Net(shape0.params, shape1.params),
            output: shape1.output,
            stack_size: shape0.output.batched(1).size()
                + usize::max(shape0.stack_size, shape1.stack_size),
        }
    }
}

impl<I, B0, B1> BackpropShape<I> for Net<B0, B1>
where
    B0: BackpropShape<I>,
    B1: BackpropShape<B0::Output>,
{
    type TrainingCache = Net<B0::TrainingCache, B1::TrainingCache>;

    fn shape_with_cache(self, input: I) -> (ModelShapes<I, Self>, Self::TrainingCache) {
        let (shape0, cache0) = self.0.shape_with_cache(input);
        let (shape1, cache1) = self.1.shape_with_cache(shape0.output);
        (
            ModelShapes {
                params: Net(shape0.params, shape1.params),
                output: shape1.output,
                stack_size: shape0.output.batched(1).size()
                    + usize::max(shape0.stack_size, shape1.stack_size),
            },
            Net(cache0, cache1),
        )
    }
}

impl<F, I, B0, B1> Inference<I, F> for Net<B0, B1>
where
    I: BatchShape,
    B0: Inference<I, F>,
    B1: Inference<B0::Output, F>,
{
    fn infer(
        self,
        i: I,
        b: usize,
        p: View<Self::Params, &F>,
        in_: View<Batched<I>, &F>,
        out: View<Batched<Self::Output>, &mut F>,
        s: &mut [F],
    ) {
        let shape0 = self.0.shape(i);
        let m = shape0.output;
        let mid_shape = m.batched(b);
        let (mid_slice, s) = s.split_at_mut(mid_shape.size());

        let mid = mid_shape.from_slice(&mut *mid_slice);
        self.0.infer(i, b, p.0, in_, mid, s);

        let mid = mid_shape.from_slice(&*mid_slice);
        self.1.infer(m, b, p.1, mid, out, s);
    }
}

impl<F, I, B0, B1> Backprop<I, F> for Net<B0, B1>
where
    I: BatchShape,
    B0: Backprop<I, F>,
    B1: Backprop<B0::Output, F>,
{
    fn forward(
        self,
        i: I,
        b: usize,
        p: View<Self::Params, &F>,
        in_: View<Batched<I>, &F>,
        out: View<Batched<Self::Output>, &mut F>,
        c: View<Batched<Self::TrainingCache>, &mut F>,
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
        i: I,
        b: usize,
        p: View<Self::Params, &F>,
        dout: View<Batched<Self::Output>, &F>,
        din_: View<Batched<I>, &mut F>,
        dp: View<Self::Params, &mut F>,
        c: View<Batched<Self::TrainingCache>, &F>,
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
