use crate::utils::calcul_linear::get_coefficient;

pub struct LinearRegression{
    m: f32,
    b:f32
}


impl LinearRegression {

    pub fn new() -> Self{
        LinearRegression { m: 0.0, b: 0.0 }
    }


    pub fn fit(&mut self, x : &[f32], y:&[f32])->(f32, f32){
       (self.m, self.b) = get_coefficient(x, y);
       (self.m, self.b)
    }


    pub fn predict(&self, x:&[f32])-> Vec<f32>{
        let mut prediction: Vec<f32> = Vec::new();

        for item in x{
            let y = self.m * item+self.b;
            prediction.push(y);
        }

        prediction
    }
}

