use nom::{self, bytes::complete::take, multi::count, number::complete::be_u32, IResult};

use std::path::PathBuf;
fn find_project_path() -> PathBuf {
    let exe_path = std::env::current_exe().expect("could not find exe path");
    let mut path = exe_path.parent().expect("exe has no parent directory");
    while let Some(parent) = path.parent() {
        path = parent;
        if path.join("Cargo").with_extension("toml").exists() {
            return path.to_path_buf();
        }
    }
    panic!("not found");
}

pub struct DataSet {
    pub images: Vec<Vec<u8>>,
    pub labels: Vec<u8>,
}

pub struct Data {
    pub training: DataSet,
    pub testing: DataSet,
}

pub fn load_data() -> Data {
    let path = find_project_path().join("examples/mnist/data");
    Data {
        training: DataSet {
            images: images(&read_file(path.join("train-images-idx3-ubyte")))
                .unwrap()
                .1,
            labels: labels(&read_file(path.join("train-labels-idx1-ubyte")))
                .unwrap()
                .1,
        },
        testing: DataSet {
            images: images(&read_file(path.join("t10k-images-idx3-ubyte")))
                .unwrap()
                .1,
            labels: labels(&read_file(path.join("t10k-labels-idx1-ubyte")))
                .unwrap()
                .1,
        },
    }
}

use core::panic;
use std::path::Path;
fn read_file<P: AsRef<Path>>(path: P) -> Vec<u8> {
    use std::fs::File;
    use std::io::prelude::*;

    let mut file = File::open(path).expect("could not open file");
    let mut input = vec![];
    file.read_to_end(&mut input).expect("could not read file");
    input
}

fn labels(s: &[u8]) -> IResult<&[u8], Vec<u8>> {
    let (s, _) = be_u32(s)?;
    let (s, length) = be_u32(s)?;
    let (s, labels) = take(length)(s)?;
    Ok((s, labels.to_vec()))
}

fn image(s: &[u8]) -> IResult<&[u8], Vec<u8>> {
    let (s, image) = take(28 * 28usize)(s)?;
    Ok((s, image.to_vec()))
}

fn images(s: &[u8]) -> IResult<&[u8], Vec<Vec<u8>>> {
    let (s, _) = be_u32(s)?;
    let (s, length) = be_u32(s)?;
    let (s, w) = be_u32(s)?;
    let (s, h) = be_u32(s)?;
    assert_eq!(w, 28);
    assert_eq!(h, 28);
    count(image, length as usize)(s)
}
