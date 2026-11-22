use crate::utils::calcul_linear::get_coefficient;
use crate::metrics::error::{mae, mse, rmse, r2};

pub fn fit(x_train : &[f32], y_train:&[f32], x_test:&[f32], y_test:&[f32]){
    let mut prediction: Vec<f32> = Vec::new();

    let (m, b)  = get_coefficient(x_train, y_train);

    for x in x_test{
        let y = m*x + b;
        prediction.push(y);
    }

    let mae = mae::mae(y_test, &prediction);
    let mse = mse::mean_squared_error(y_test, &prediction);
    let rmse = rmse::rmse(mse);
    let r2 = r2::r2(y_test, &prediction);


    println!("Prediction : {:?}", prediction);
    println!("MAE : {:?}", mae);
    println!("MSE : {:?}", mse);
    println!("RMSE : {:?}", rmse);
    println!("R2 : {:?}", r2);
}