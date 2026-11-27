use crate::{error::error::ForgeError, utils::calcul_linear::get_coefficient};

pub struct LinearRegression{
    weights: f32,
    bias:f32
}


impl LinearRegression {

    pub fn new() -> Self{
        LinearRegression { weights: 0.0, bias: 0.0 }
    }


    pub fn fit(&mut self, x : &[f32], y:&[f32])->Result<(f32, f32), ForgeError>{
       (self.weights, self.bias) = get_coefficient(x, y)?;
       Ok((self.weights, self.bias))
    }


    pub fn predict(&self, x:&[f32])-> Vec<f32>{
        let mut prediction: Vec<f32> = Vec::new();

        for item in x{
            let y = self.weights * item+self.bias;
            prediction.push(y);
        }

        prediction
    }

    pub fn slope(&self)->f32{
        self.weights
    }

    pub fn intercept(&self)->f32{
        self.bias
    }
}

