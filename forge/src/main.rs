use std::{error::Error, fs::File};

use ml::linear;

fn main() -> Result<(), Box<dyn Error>> {
    let file = File::open(r"../data/Salary_Data.csv")?;
    let mut rdr = csv::Reader::from_reader(file);

    let mut x_train: Vec<f32> = Vec::new();
    let mut y_train: Vec<f32> = Vec::new();
    let mut x_test: Vec<f32> = Vec::new();
    let mut y_test: Vec<f32> = Vec::new();

    for (i, result) in rdr.records().enumerate() {
        let record = result?;
        let xi: f32 = record[0].parse()?;
        let yi: f32 = record[1].parse()?;

        if i % 5 == 0 {
            x_test.push(xi);
            y_test.push(yi);
        } else {
            x_train.push(xi);
            y_train.push(yi);
        }
    }

    linear::simple_linear::fit(&x_train, &y_train, &x_test, &y_test);

    Ok(())
}