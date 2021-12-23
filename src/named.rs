use crate::{GraphBuilder, Layer};

pub struct Named<I, S: ToString> {
    pub(crate) inner: I,
    pub(crate) name: S,
}

impl<I, S: ToString> Named<I, S> {
    pub fn into_inner(self) -> I {
        self.inner
    }
    pub fn name(&self) -> &S {
        &self.name
    }
}

impl<G: GraphBuilder, S: ToString> GraphBuilder for Named<G, S> {
    type InputShape = G::InputShape;
    type OutputShape = G::OutputShape;
    type Layer = Named<G::Layer, S>;

    fn with_input_shape(self, input_shape: Self::InputShape) -> Self::Layer {
        let Self { inner, name } = self;
        let inner = inner.with_input_shape(input_shape);
        Named { inner, name }
    }
}

impl<L: Layer, S: ToString> Layer for Named<L, S> {
    type InputShape = L::InputShape;
    type OutputShape = L::OutputShape;
    type State<St: ndarray::RawData> = L::State<St>;

    fn size(&self) -> usize {
        self.inner.size()
    }

    fn output_shape(&self) -> Self::OutputShape {
        self.inner.output_shape()
    }

    fn view<Sl: crate::Slice>(&self, data: Sl) -> Self::State<Sl::Repr> {
        self.inner.view(data)
    }

    fn apply<F: crate::Scalar>(
        &self,
        state: Self::State<ndarray::ViewRepr<&F>>,
        input: crate::Arr<impl ndarray::Data<Elem = F>, Self::InputShape>,
    ) -> crate::OwnedArr<F, Self::OutputShape> {
        self.inner.apply(state, input)
    }
}
