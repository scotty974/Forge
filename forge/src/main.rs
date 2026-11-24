use std::{error::Error, fs::File};

use ml::{linear::simple_linear::LinearRegression, metrics::error::r2::r2};

fn main() -> Result<(), Box<dyn Error>> {
    let file = File::open(r"../data/Salary_Data.csv")?;
    let mut rdr = csv::Reader::from_reader(file);

    let mut x_train: Vec<f32> = Vec::new();
    let mut y_train: Vec<f32> = Vec::new();
    let mut x_test: Vec<f32> = Vec::new();
    let mut y_test: Vec<f32> = Vec::new();


    let mut model = LinearRegression::new();
    model.fit(&x_train, &y_train);
    let pred = model.predict(&x_test);
    let score = r2(&y_test, &pred);

    println!("{:?}", pred);
    println!("{:?}", score);

    Ok(())
}