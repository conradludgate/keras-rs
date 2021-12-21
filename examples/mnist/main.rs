mod parse;

use keras_rs::{
    activation::relu::Relu, cost::mse::MSE, linear, net, optimise::adam::Adam, GraphBuilder,
};
use ndarray::{Array2, AssignElem};

fn main() {
    // Load MNIST data set
    let data = parse::load_data();
    let training_data = process_data(&data.training);

    // Create a new compute graph which uses three Dense components
    // With the input having size 28*28 and the output having size 10
    // Initialise it with uniform random data
    let network = net![
        linear::Layer::output(16),
        Relu,
        linear::Layer::output(16),
        Relu,
        linear::Layer::output(10),
        Relu
    ];

    let optimiser = Adam::<f64>::new(0.001, 0.9, 0.99, 1e-8);
    let model = network
        .build_model()
        .input_shape(28 * 28)
        .with_cost(MSE)
        .with_optimiser(optimiser)
        .initialise();

    // let graph = network.input_shape(28 * 28);

    // New trainer with mean squared error cost function and
    // stochastic gradient descent optimisation (alpha=0.1)
    // let mut trainer = Train::new(network, MSE, SGD::new(0.01));

    // let mut trainer = Train {
    //     graph,
    //     optimiser,
    //     cost: MSE,
    //     regularisation: Some(Regularisation::L2(0.01)),
    //     dropout: 0.2,
    // };

    // let mut costs = vec![];

    // const BATCH_SIZE: usize = 120;

    // for _ in 0..20 {
    //     let cost =
    //         trainer.perform_epoch(&training_data.0.view(), &training_data.1.view(), BATCH_SIZE);

    //     costs.push(dbg!(cost));
    // }

    // let graph = trainer.graph;

    // // let file = hdf5::File::create("mnist.h5").unwrap();
    // // network.save(&graph, &file).unwrap();
    // // file.close().unwrap();

    // // println!("network: {:?}", network);

    // let input = training_data.0.index_axis(Axis(0), 0);
    // let expected = training_data.1.index_axis(Axis(0), 0);

    // let output = graph.exec(input);
    // println!("output: {:?}", output);
    // println!("expected: {:?}", expected);
}

fn process_data(data: &parse::DataSet) -> (Array2<f32>, Array2<f32>) {
    let data_len = data.images.len();
    assert_eq!(data_len, data.labels.len());

    let mut input = Array2::uninit((data_len, 28 * 28));
    let mut expected = Array2::zeros((data_len, 10));

    for (i, image) in data.images.iter().enumerate() {
        for (j, &b) in image.iter().enumerate() {
            input[(i, j)].assign_elem((b as f32) / 255.0);
        }
    }
    for (i, &label) in data.labels.iter().enumerate() {
        expected[(i, label as usize)] = 1.0;
    }

    unsafe { (input.assume_init(), expected) }
}
