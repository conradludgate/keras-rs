mod parse;

use keras_rs::{
    activation::{relu::Relu, sigmoid::Sigmoid},
    cost::mse::MSE,
    linear,
    model::{Regularisation, Trainer},
    net,
    optimise::adam::Adam,
    Initialise,
};
use ndarray::{Array2, AssignElem, Axis, Ix1};

// enable optimised accelerate backend on macos
#[cfg(target_os = "macos")]
extern crate blas_src;

fn main() {
    // Load MNIST data set
    let data = parse::load_data();
    let training_data = process_data(&data.training);
    let testing_data = process_data(&data.testing);

    // Create a new compute graph which uses three Dense components
    // With the input having size 28*28 and the output having size 10
    // Initialise it with uniform random data
    let network = net![
        linear::Linear::output(16),
        Relu,
        linear::Linear::output(16),
        Relu,
        linear::Linear::output(10),
        Sigmoid,
    ];

    let model = network.into_model(Ix1(28 * 28));

    let optimiser = Adam::new(0.001, 0.9, 0.99, 1e-8);
    // let optimiser = SGD::new(0.01);
    let mut trainer = Trainer {
        model,
        optimiser,
        cost: MSE,
        regularisation: Some(Regularisation::L2(0.01)),
    }
    .build();

    let mut costs = vec![];

    const BATCH_SIZE: usize = 120;

    for _ in 0..20 {
        let cost =
            trainer.train_epoch(&training_data.0.view(), &training_data.1.view(), BATCH_SIZE);

        costs.push(dbg!(cost));
    }

    let cost = trainer.test_epoch(&testing_data.0.view(), &testing_data.1.view());
    dbg!(cost);

    // let graph = trainer.graph;

    // // let file = hdf5::File::create("mnist.h5").unwrap();
    // // network.save(&graph, &file).unwrap();
    // // file.close().unwrap();

    // // println!("network: {:?}", network);

    let input = training_data.0.index_axis(Axis(0), 0);
    let expected = training_data.1.index_axis(Axis(0), 0);

    let output = trainer.as_model().apply_single(input);
    println!("output: {:?}", output);
    println!("expected: {:?}", expected);
}

fn process_data(data: &parse::DataSet) -> (Array2<f32>, Array2<f32>) {
    let data_len = data.images.len();
    assert_eq!(data_len, data.labels.len());

    let mut input = Array2::<f32>::zeros((data_len, 28 * 28));
    let mut expected = Array2::<f32>::zeros((data_len, 10));

    for (i, image) in data.images.iter().enumerate() {
        for (j, &b) in image.iter().enumerate() {
            input[(i, j)].assign_elem((b as f32) / 255.0);
        }
    }
    for (i, &label) in data.labels.iter().enumerate() {
        expected[(i, label as usize)] = 1.0;
    }

    (input, expected)
}
