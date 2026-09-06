use anyhow::{Context, Result};
use candle_core::{D, Device, IndexOp, Tensor};
use nalgebra::DMatrix;
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
        let diff = self.broadcast_sub(&mean)?;
        let variance = diff.sqr()?.mean(0)?;
        let stdev = (variance + 1e-8)?.sqrt()?;

        Ok(diff.broadcast_div(&stdev)?)
    }
}

fn invert_tensor(a: &Tensor) -> Result<Tensor> {
    let n = a.dim(0).context("Failed to get tensor dimension")?;

    // 1. Pull data to CPU and parse into nalgebra
    let flat_data = a.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
    let matrix = DMatrix::from_row_slice(n, n, &flat_data);

    // 2. Invert using nalgebra
    let inv_matrix = matrix
        .try_inverse()
        .ok_or_else(|| anyhow::anyhow!("Matrix is singular and cannot be inverted"))?;

    // 3. Extract the underlying slice directly
    let inv_slice = inv_matrix.as_slice();

    // 4. Create the tensor, transpose it, and target the original device
    let result = Tensor::from_slice(inv_slice, (n, n), &Device::Cpu)?
        .t()?
        .to_device(a.device())?;

    Ok(result)
}

impl LinearRegression {
    fn new(features_size: usize, device: Device) -> Result<Self> {
        let weights = Tensor::randn(0.0f32, 1.0f32, features_size, &device)?;
        let bias = Tensor::new(0.5f32, &device)?;

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
        let loss = pred.sub(label)?.sqr()?;
        let mean = loss.mean_all()?.to_scalar()?;
        Ok(mean)
    }

    fn fit(&mut self, x: Tensor, y: Tensor) -> Result<()> {
        let xm = x.mean_all()?;
        let ym = y.mean_all()?;

        let X = x.broadcast_sub(&xm)?;
        let Y = y.broadcast_sub(&ym)?;

        let first = X.t()?.matmul(&X)?;
        let second = X.t()?.matmul(&Y);

        let weight = first * second;

        self.weights = x.clone();
        self.bias = y.clone();

        Ok(())
    }

    // mut self because we will change the weight
    fn train_1_epoch(
        &mut self,
        x: &Tensor,
        y: &Tensor,
        learning_rate: f32,
        regularization: f32,
    ) -> Result<f32> {
        let batch_size = x.shape().dims2()?.0;
        // println!("batch_size: {:?}", batch_size);

        let pred = self.forward(x)?.unsqueeze(1)?;
        // println!("pred: {:?}", pred);

        // println!("first");
        //
        let deltas = pred.sub(y)?;

        let loss = self.loss(&pred, &y)?;

        let regularization = self.weights.broadcast_mul(&Tensor::new(
            regularization / batch_size as f32,
            &self.device,
        )?)?;

        // (x.T * (y^ - y))/m
        let gradient = x
            .t()?
            .matmul(&deltas)?
            .broadcast_div(&Tensor::new(batch_size as f32, &self.device)?)?;

        let gradient = gradient
            .squeeze(D::Minus1)?
            .squeeze(D::Minus1)?
            .add(&regularization)?;

        self.weights = self
            .weights
            .sub(&gradient.broadcast_mul(&Tensor::new(learning_rate, &self.device)?)?)?;

        let gradient = deltas.mean_all()?;
        // println!("gradient: {:?}", gradient);

        self.bias = self
            .bias
            .sub(&gradient.broadcast_mul(&Tensor::new(learning_rate, &self.device)?)?)?;

        Ok(loss)
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

fn r2_score(predictions: &Tensor, labels: &Tensor) -> Result<f32> {
    // y.mean()
    let mean = labels.mean_all()?;
    // (y - y^)^2.sum()
    let ssr = labels.sub(predictions)?;
    let ssr = ssr.mul(&ssr)?.sum_all()?;

    // (y - y.mean())^2.sum()
    let sst = labels.broadcast_sub(&mean)?;
    let sst = sst.mul(&sst)?.sum_all()?;

    // (y - y^)^2.sum() / (y - y.mean())^2.sum()
    let total_ssr = ssr.to_scalar::<f32>()?;
    let total_sst = sst.to_scalar::<f32>()?;

    let r2 = 1.0 - (total_ssr / total_sst);

    Ok(r2)
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
    // println!("data: {:}", data);

    let norm_data = data.z_norm()?;
    // println!("norm_data: {:}", norm_data);

    let batch = norm_data.i(..1200)?;
    let x = batch.i((.., ..9))?;
    let y = batch.i((.., 9..10))?;

    let rows = x.shape().dims2()?.0;
    let columns = x.shape().dims2()?.1;
    // println!("row: {:?}, columns: {:?}", rows, columns);

    let mut model = LinearRegression::new(columns, device)?;

    // backprop train

    // let test = norm_data.i(1200..1201)?;
    // let x_test = test.i((.., ..9))?;
    // let y_test = test.i((.., 9..10))?;
    //
    // let pred = model.forward(&x_test)?.unsqueeze(1)?;
    // println!("y: {y_test}, prediction: {pred}");
    //
    // let epochs = 1000;
    //
    // for i in 0..epochs {
    //     let loss = model.train_1_epoch(&x, &y, lr, regularization)?;
    //     // print!("[EPOCH {i}]: ");
    //     // println!("  loss: {}", loss);
    //     if i % 100 == 0 {
    //         let predictions = model.forward(&x)?.unsqueeze(1)?;
    //         let r2 = r2_score(&predictions, &y)?;
    //         println!("[EPOCH {i}] accuracy is {r2}, loss is {loss}");
    //     }
    // }

    // Test

    // faster fit
    model.fit(x, y)?;

    let batch = norm_data.i(1200..1201)?;
    let x = batch.i((.., ..9))?;
    let y = batch.i((.., 9..10))?;

    let pred = model.forward(&x)?.unsqueeze(1)?;
    println!("y: {y}, prediction: {pred}");

    Ok(())
}
