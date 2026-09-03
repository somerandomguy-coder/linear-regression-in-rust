use std::ops::Mul;

use anyhow::Result;
use candle_core::{Device, IndexOp, Tensor};
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
trait ZNormable {
    fn z_norm(&self) -> Result<Tensor>;
}
impl ZNormable for Tensor {
    fn z_norm(&self) -> Result<Tensor> {
        let mean = self.mean(0)?;
        println!("mean: {:}", mean);
        let diff = self.broadcast_sub(&mean)?;
        println!("diff: {:}", diff);
        let variance = diff.sqr()?.mean(0)?;
        println!("variance: {:}", variance);
        let stdev = (variance + 1e-8)?.sqrt()?;
        println!("stdev: {:}", stdev);

        Ok(diff.broadcast_div(&stdev)?)
    }
}

impl LinearRegression {
    fn new(features_size: usize, device: Device) -> Result<Self> {
        let weights = Tensor::randn(0.0f32, 1.0f32, (features_size), &device)?;
        let bias = Tensor::new(0.0f32, &device)?;

        Ok(Self {
            weights,
            bias,
            device,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let result = x
            .matmul(&self.weights.unsqueeze(1)?)?
            .squeeze(1)?
            .broadcast_add(&self.bias)?;
        Ok(result)
    }

    fn loss(&self, pred: &Tensor, label: &Tensor) -> Result<f32> {
        // MSE pred (batch, 1), (batch, 1) -> (1)
        // MSE = (mean((pred-label)**2))
        println!("prediction: {:}, label: {}", pred, label);
        let loss = pred.sub(label)?.sqr()?;
        let mean = loss.mean_all()?.to_scalar()?;
        Ok(mean)
    }
    // mut self because we will change the weight
    fn train_1_epoch(
        &mut self,
        x: &Tensor,
        y: &Tensor,
        learning_rate: f32,
        regularization: f32,
    ) -> Result<()> {
        let batch_size = x.shape().dims2()?.0;
        println!("batch_size: {:?}", batch_size);

        let pred = self.forward(x)?;
        println!("pred: {:?}", pred);

        let loss = self.loss(&pred.unsqueeze(1)?, &y)?;
        println!("loss: {:?}", loss);

        let weight_sum = self.weights.sum_all()?.to_scalar::<f32>()?;
        let weighted_sum = weight_sum / (batch_size as f32);
        let regularize_term = weighted_sum.mul(regularization);

        let loss_with_regular = loss + regularize_term;

        Ok(())
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
    // println!("dataset: {:?}", dataset);

    let tensor: Tensor = Tensor::from_vec(dataset, (1338, 10), device)?;
    Ok(tensor)
}

fn main() -> Result<()> {
    println!("Hello, world!");
    println!("[CONFIG]");
    let device = Device::Cpu;
    let lr = 1e-3;
    let regularization = 1e-3;
    println!("device :{:?}", device);
    println!("lr :{:?}", lr);
    println!("regularization :{:?}", regularization);
    println!("[END OF CONFIG]");
    let data = load_data("src/insurance.csv", &device)?;
    println!("data {:?}", data);

    let norm_data = data.z_norm()?;
    println!("norm_data: {:}", norm_data);

    let batch = norm_data.i(..3)?;
    let x = batch.i((.., ..9))?;
    let y = batch.i((.., 9..10))?;
    println!("x: {:}, y: {:}", x, y);

    let rows = x.shape().dims2()?.0;
    let columns = x.shape().dims2()?.1;
    println!("row: {:?}, columns: {:?}", rows, columns);

    let mut model = LinearRegression::new(columns, device)?;

    model.train_1_epoch(&x, &y, lr, regularization)?;

    let result = model.forward(&x)?;
    println!("result: {:?}", result);

    let loss = model.loss(&result.unsqueeze(1)?, &y)?;
    println!("loss: {:?}", loss);

    // println!("model {:?}", model);
    Ok(())
}
