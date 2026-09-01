use anyhow::Result;
use candle_core::{Device, Tensor};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Record {
    age: f32,
    sex: String,
    bmi: f32,
    children: f32,
    smoker: String,
    region: String,
    charges: f32,
}

struct LinearRegression {
    weights: Tensor,
    bias: Tensor,
    device: Device,
}

impl LinearRegression {
    fn new(features_size: usize, device: Device) -> Result<Self> {
        let weights = Tensor::randn(0.0, 1.0, (1, features_size), &device)?;
        let bias = Tensor::new(0.0f32, &device)?;

        Ok(Self {
            weights,
            bias,
            device,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let tensor = Tensor::new(1, &self.device)?;
        Ok(tensor)
    }
}

fn load_data(file_path: &str, device: &Device) -> Result<Tensor> {
    let mut reader = csv::Reader::from_path(file_path)?;

    let mut dataset: Vec<f32> = Vec::new();

    // println!("-----DEBUG load_data-----");
    for line in reader.deserialize() {
        let mut row: Vec<f32> = Vec::new();
        let record: Record = line?;
        row.push(record.age);

        if record.sex == "male" {
            row.push(1.0);
        } else if record.sex == "female" {
            row.push(0.0);
        } else {
            panic!("Invalid sex")
        }

        row.push(record.bmi);
        row.push(record.children);

        if record.smoker == "yes" {
            row.push(1.0);
        } else if record.smoker == "no" {
            row.push(0.0);
        } else {
            panic!("wrong data format for smoker");
        }

        if record.region == "northwest" {
            row.extend(vec![1.0f32, 0.0f32, 0.0f32, 0.0f32]);
        } else if record.region == "northeast" {
            row.extend(vec![0.0f32, 1.0f32, 0.0f32, 0.0f32]);
        } else if record.region == "southeast" {
            row.extend(vec![0.0f32, 0.0f32, 1.0f32, 0.0f32]);
        } else if record.region == "southwest" {
            row.extend(vec![0.0f32, 0.0f32, 0.0f32, 1.0f32]);
        } else {
            panic!("Invalid region");
        }

        row.push(record.charges);

        // println!("{:?}", row);
        dataset.extend(row);
    }
    let row = dataset.len();
    // println!("row: {:?}", row);
    //
    // println!("dataset: {:?}", dataset);

    let tensor: Tensor = Tensor::from_vec(dataset, (1338, 10), device)?;
    Ok(tensor)
}

fn main() -> Result<()> {
    println!("Hello, world!");
    println!("[CONFIG]");
    let device = Device::Cpu;
    println!("device :{:?}", device);
    println!("[END OF CONFIG]");
    let data = load_data("src/insurance.csv", &device)?;

    println!("data {:?}", data);

    let rows = data.shape().dims2()?.0;
    let columns = data.shape().dims2()?.1;
    println!("row: {:?}, columns: {:?}", rows, columns);
    let model = LinearRegression::new(columns, device)?;

    // println!("model {:?}", model);
    Ok(())
}
